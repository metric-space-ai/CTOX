// Guards the module boundaries declared in contracts/module_boundaries.txt.
//
// The campaign kept finding the same shape: a core reaching sideways or upward
// until nobody could say which module owned a decision. Enforced edges (budget
// 0) keep the boundaries that already hold; the rest carry the count they cost
// today and may only shrink. A ratchet that is lowered has to be lowered in
// this file too, so the number shows up in review.

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    const MANIFEST: &str = include_str!("../../contracts/module_boundaries.txt");

    struct Budget {
        from: String,
        to: String,
        max: usize,
    }

    fn budgets() -> Vec<Budget> {
        MANIFEST
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| {
                let (edge, max) = line
                    .split_once('=')
                    .unwrap_or_else(|| panic!("module boundary line has no budget: {line}"));
                let (from, to) = edge
                    .split_once("->")
                    .unwrap_or_else(|| panic!("module boundary line has no arrow: {line}"));
                Budget {
                    from: from.trim().to_string(),
                    to: to.trim().to_string(),
                    max: max
                        .trim()
                        .parse()
                        .unwrap_or_else(|err| panic!("bad budget in {line:?}: {err}")),
                }
            })
            .collect()
    }

    fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                rust_sources(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                out.push(path);
            }
        }
    }

    fn edge_count(module_root: &Path, target: &str) -> usize {
        let needle = format!("crate::{target}");
        let mut files = Vec::new();
        rust_sources(module_root, &mut files);
        files
            .iter()
            .filter_map(|path| std::fs::read_to_string(path).ok())
            .map(|source| {
                source
                    .match_indices(&needle)
                    .filter(|(index, _)| {
                        // `crate::mission` must not also count `crate::missionary`.
                        source[index + needle.len()..]
                            .chars()
                            .next()
                            .is_none_or(|next| !next.is_alphanumeric() && next != '_')
                    })
                    .count()
            })
            .sum()
    }

    #[test]
    fn module_boundaries_hold_within_their_declared_budget() {
        let core = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/core");
        let mut over = Vec::new();
        let mut slack = BTreeMap::new();
        for budget in budgets() {
            let module_root = core.join(&budget.from);
            assert!(
                module_root.is_dir(),
                "module boundary names a directory that does not exist: src/core/{}",
                budget.from
            );
            let count = edge_count(&module_root, &budget.to);
            if count > budget.max {
                over.push(format!(
                    "{} -> {}: {} references, budget {}",
                    budget.from, budget.to, count, budget.max
                ));
            } else if count < budget.max {
                slack.insert(
                    format!("{} -> {}", budget.from, budget.to),
                    (count, budget.max),
                );
            }
        }
        assert!(
            over.is_empty(),
            "module boundaries exceeded. Either route through the owning module, \
             or lower the coupling and say so — do not raise a budget to make this \
             pass:\n  {}",
            over.join("\n  ")
        );
        assert!(
            slack.is_empty(),
            "these edges now cost less than contracts/module_boundaries.txt claims. \
             Lower the budget in the same commit so the ratchet cannot slip back:\n  {}",
            slack
                .iter()
                .map(|(edge, (count, max))| format!("{edge}: {count} references, budget {max}"))
                .collect::<Vec<_>>()
                .join("\n  ")
        );
    }

    /// Every `*_tests.rs` beside `main.rs` must be declared there.
    ///
    /// On 2026-08-06 a commit dropped a single line — `mod module_size_tests;` —
    /// and the size ratchet stopped running. Nothing failed, because an unlisted
    /// module is not compiled: `cargo test module_size_tests` reported "0 tests"
    /// and read as success. The guard stayed dark for days and hid a budget
    /// violation that was already committed.
    ///
    /// A guard cannot detect its own absence — if this file were dropped from
    /// `main.rs`, this test would not run either. That case is not solvable from
    /// the inside; what is solvable is the one that actually happened, where one
    /// guard is removed while the others still run. Every remaining guard then
    /// reports the loss.
    #[test]
    fn every_guard_module_stays_registered() {
        let core_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/core");
        let main_rs = std::fs::read_to_string(core_dir.join("main.rs"))
            .expect("src/core/main.rs must be readable");

        let mut unregistered = Vec::new();
        for entry in std::fs::read_dir(&core_dir).expect("src/core must be readable") {
            let path = entry.expect("directory entry must be readable").path();
            let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
                continue;
            }
            if !stem.ends_with("_tests") {
                continue;
            }
            if !main_rs
                .lines()
                .any(|line| line.trim() == format!("mod {stem};"))
            {
                unregistered.push(stem.to_owned());
            }
        }
        unregistered.sort();

        assert!(
            unregistered.is_empty(),
            "these guard modules exist but are not declared in src/core/main.rs, so they \
             never run and their failures are invisible. Add `mod <name>;` — do not delete \
             the file to make this pass:\n  {}",
            unregistered.join("\n  ")
        );
    }
}
