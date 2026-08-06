// ref: internal/pluginstore/version.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::cmp::Ordering;

pub fn update_available(installed: &str, latest: &str) -> bool {
    let installed = normalize_version(installed);
    let latest = normalize_version(latest);
    if installed.is_empty() || latest.is_empty() || installed == latest {
        return false;
    }
    match compare_versions(installed, latest) {
        Some(Ordering::Less) => true,
        Some(Ordering::Equal | Ordering::Greater) => false,
        None => true,
    }
}

fn normalize_version(version: &str) -> &str {
    let version = version.trim();
    if version.len() > 1 && matches!(version.as_bytes()[0], b'v' | b'V') {
        &version[1..]
    } else {
        version
    }
}

fn compare_versions(left: &str, right: &str) -> Option<Ordering> {
    let mut left = left.split('.');
    let mut right = right.split('.');
    loop {
        match (left.next(), right.next()) {
            (None, None) => return Some(Ordering::Equal),
            (left, right) => {
                let left = version_segment(left.unwrap_or("0"))?;
                let right = version_segment(right.unwrap_or("0"))?;
                match left.cmp(&right) {
                    Ordering::Equal => {}
                    ordering => return Some(ordering),
                }
            }
        }
    }
}

fn version_segment(segment: &str) -> Option<i64> {
    segment.parse::<i64>().ok().filter(|number| *number >= 0)
}
