// Guards contracts/module_size_budget.txt.
//
// A split that nothing holds grows back: store.rs fell from roughly 69,000 to
// 27,300 production lines in July 2026 and was already growing again weeks
// later. This ratchet makes every monitored file's current size visible and
// expensive to increase.
//
// Like the module-boundary and text-matching guards, it fails in both
// directions. Growth is red, and so is a budget left above the current count
// after a cut: the contract must ratchet down in the same change.

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::Path;

    const MANIFEST: &str = include_str!("../../contracts/module_size_budget.txt");

    fn budgets() -> Vec<(String, usize)> {
        MANIFEST
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| {
                let (path, max) = line
                    .split_once('=')
                    .unwrap_or_else(|| panic!("module size budget line has no budget: {line}"));
                (
                    path.trim().to_string(),
                    max.trim()
                        .parse()
                        .unwrap_or_else(|err| panic!("bad budget in {line:?}: {err}")),
                )
            })
            .collect()
    }

    /// Counts physical source lines before the last standalone `#[cfg(test)]`.
    /// Blank lines and comments count. `str::lines()` defines physical lines,
    /// and trimming is used only to recognize a marker whose entire content is
    /// exactly `#[cfg(test)]`. The marker itself and every line after it do not
    /// count; without such a marker, the whole file counts.
    ///
    /// The LAST marker is deliberate. Large modules can have earlier markers
    /// on isolated test-only imports or helpers embedded among production
    /// items; stopping at the first one would hide most of store.rs. The final
    /// marker starts the terminal test section in the monitored files. Counting
    /// earlier test-only snippets conservatively gives one stable textual rule
    /// instead of a second, configuration-dependent estimate.
    fn production_lines(source: &str) -> usize {
        let mut line_count = 0;
        let mut last_test_marker = None;

        for line in source.lines() {
            if line.trim() == "#[cfg(test)]" {
                last_test_marker = Some(line_count);
            }
            line_count += 1;
        }

        last_test_marker.unwrap_or(line_count)
    }

    #[test]
    fn module_size_stays_at_its_declared_budget() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut over = Vec::new();
        let mut slack = BTreeMap::new();

        for (path, max) in budgets() {
            let full = repo.join(&path);
            let source = std::fs::read_to_string(&full)
                .unwrap_or_else(|err| panic!("module size budget names {path}: {err}"));
            let count = production_lines(&source);
            if count > max {
                over.push(format!("{path}: {count} production lines, budget {max}"));
            } else if count < max {
                slack.insert(path, (count, max));
            }
        }

        assert!(
            over.is_empty(),
            "these modules now exceed contracts/module_size_budget.txt. Split the file or \
             remove production lines — do not raise a budget to make this pass:\n  {}",
            over.join("\n  ")
        );
        assert!(
            slack.is_empty(),
            "these modules are smaller than contracts/module_size_budget.txt claims. Lower \
             the budget in the same commit so the cut cannot grow back:\n  {}",
            slack
                .iter()
                .map(|(path, (count, max))| {
                    format!("{path}: {count} production lines, budget {max}")
                })
                .collect::<Vec<_>>()
                .join("\n  ")
        );
    }
}
