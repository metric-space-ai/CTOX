// Guards contracts/text_matching_budget.txt.
//
// A decision read out of prose changes silently when someone rephrases the
// prose. ST3 removed one of those: whether a queue task counted as finished
// was decided by searching a human-readable note for `" completed."` together
// with `"changed "`. This ratchet keeps the remaining ones visible and makes
// them expensive to add to.
//
// Like the module boundary guard, it fails in both directions: a budget that
// sits above what the file actually costs is as red as one that is exceeded,
// so a number cannot quietly stop ratcheting.

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::Path;

    const MANIFEST: &str = include_str!("../../contracts/text_matching_budget.txt");

    /// A control-flow keyword — the line decides something.
    const DECIDES: [&str; 9] = [
        "if ",
        "while ",
        "&&",
        "||",
        "matches!",
        ".filter(",
        ".any(",
        ".is_some_and(",
        "return ",
    ];

    /// An error- or status-shaped receiver. Matching on a payload field or a
    /// user-supplied string is a different thing and not counted here.
    ///
    /// Only what stands to the LEFT of `.contains(` counts as the receiver.
    /// The first version of this guard searched the whole line and disagreed
    /// with the script the manifest was built from — two definitions of one
    /// thing, which is the circular-inventory shape this campaign removes.
    /// The guard is the definition now; the manifest is derived from it.
    /// KNOWN BLIND SPOT (01.08.): a rename hides the decision from this guard.
    ///
    ///     let lower = error.to_string().to_ascii_lowercase();
    ///     lower.contains("sqlite_busy")
    ///
    /// The receiver is `lower`, which carries none of these tokens, so the
    /// line is not counted. Two production functions in service.rs already
    /// classify SQLite failures exactly this way — found while checking
    /// whether SF8 had really replaced the SQLite text matching. It had
    /// replaced the ones the guard could see.
    ///
    /// I tried to close it by tracking `let` bindings whose right-hand side
    /// reads an error, and stopped: file-wide tracking reported 139 decisions
    /// against a budget of 3, and even scoped to the function it turned
    /// chat_native.rs from 1 to 42, because there a "message" is the subject
    /// matter rather than a failure. A ratchet that overcounts is as harmful
    /// as one that undercounts — it teaches people to raise the number.
    ///
    /// Closing this needs the receiver's type, not its name, which means
    /// asking the compiler rather than reading lines. Until then the blind
    /// spot is written down here rather than papered over, and the two known
    /// sites are named in the plan.
    const ERROR_SHAPED: [&str; 8] = [
        "err", "error", "status", "note", "message", "msg", "reason", "stderr",
    ];

    fn budgets() -> Vec<(String, usize)> {
        MANIFEST
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| {
                let (path, max) = line
                    .split_once('=')
                    .unwrap_or_else(|| panic!("text matching budget line has no budget: {line}"));
                (
                    path.trim().to_string(),
                    max.trim()
                        .parse()
                        .unwrap_or_else(|err| panic!("bad budget in {line:?}: {err}")),
                )
            })
            .collect()
    }

    /// Counts lines that decide control flow from a substring of an
    /// error/status text. Deliberately narrow: `set.contains(&x)` and
    /// `text.contains('\n')` are not this debt, so only `.contains("` with a
    /// string literal counts.
    fn text_decisions(source: &str) -> usize {
        let mut count = 0;
        for line in source.lines() {
            // The test module carries its own fixtures and assertions; the
            // budget is about production decisions.
            if line.trim_start().starts_with("mod tests")
                || line.trim_start().starts_with("pub(crate) mod tests")
                || line.trim_start().starts_with("pub(super) mod tests")
            {
                break;
            }
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                continue;
            }
            let Some(before) = line
                .split(".contains(\"")
                .next()
                .filter(|_| line.contains(".contains(\""))
            else {
                continue;
            };
            if !DECIDES.iter().any(|kw| line.contains(kw)) {
                continue;
            }
            // Only the receiver decides, not the needle: `path.contains("error")`
            // is a path check, `err.contains("timeout")` is this debt.
            let receiver = before.to_ascii_lowercase();
            if ERROR_SHAPED.iter().any(|needle| receiver.contains(needle)) {
                count += 1;
            }
        }
        count
    }

    #[test]
    fn text_matching_stays_within_its_declared_budget() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut over = Vec::new();
        let mut slack = BTreeMap::new();
        for (path, max) in budgets() {
            let full = repo.join(&path);
            let source = std::fs::read_to_string(&full)
                .unwrap_or_else(|err| panic!("text matching budget names {path}: {err}"));
            let count = text_decisions(&source);
            if count > max {
                over.push(format!("{path}: {count} text decisions, budget {max}"));
            } else if count < max {
                slack.insert(path, (count, max));
            }
        }
        assert!(
            over.is_empty(),
            "these files now decide more often from error text than declared. Read the \
             decision from a field, or say why the text is the only signal available — \
             do not raise a budget to make this pass:\n  {}",
            over.join("\n  ")
        );
        assert!(
            slack.is_empty(),
            "these files now cost less than contracts/text_matching_budget.txt claims. \
             Lower the budget in the same commit so the ratchet cannot slip back:\n  {}",
            slack
                .iter()
                .map(|(path, (count, max))| format!("{path}: {count} decisions, budget {max}"))
                .collect::<Vec<_>>()
                .join("\n  ")
        );
    }
}
