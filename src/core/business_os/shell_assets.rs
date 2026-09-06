//! Immutable addresses and reads for files in a verified Business OS release.
//! Signature and complete inventory admission remain owned by `shell_update`.
//! A request never falls back to the active release or an installed app.
use anyhow::{ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub(super) const URL_PREFIX: &str = "_shell/";

/// Pin every relative document URL to the admitted release before the browser
/// encounters any stylesheet or module. Asset cache busters are not release IDs.
pub(super) fn pin_document(html: String, version: &str) -> Result<String> {
    validate_version(version)?;
    // Published shells already carry the canonical mutable Business OS base.
    // Replace that one known declaration; never retain a second competing base.
    static LEGACY_BASE: OnceLock<regex::Regex> = OnceLock::new();
    let legacy_base = LEGACY_BASE.get_or_init(|| {
        regex::Regex::new(r#"(?i)<base\s+href\s*=\s*(?:"/business-os/"|'/business-os/')\s*/?>"#)
            .expect("constant legacy shell base expression")
    });
    ensure!(
        legacy_base.find_iter(&html).count() <= 1,
        "shell document has multiple base URLs"
    );
    let html = legacy_base.replace(&html, "").into_owned();
    let lower = html.to_ascii_lowercase();
    ensure!(
        !lower.contains("<base"),
        "shell document has an unexpected base URL"
    );
    let head = lower.find("<head").context("shell document has no head")?;
    let end = head
        + lower[head..]
            .find('>')
            .context("shell head is not closed")?
        + 1;
    let mut pinned = html;
    pinned.insert_str(
        end,
        &format!("<base href=\"/business-os/_shell/{version}/\">"),
    );
    Ok(pinned)
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct ReleaseAssetRequest {
    pub version: String,
    pub relative: String,
}

pub(super) fn validate_version(version: &str) -> Result<()> {
    static VERSION: OnceLock<regex::Regex> = OnceLock::new();
    let expression = VERSION.get_or_init(|| regex::Regex::new(
        r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-((?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
    ).expect("constant shell version expression"));
    ensure!(
        version.len() <= 128 && expression.is_match(version),
        "invalid shell release version"
    );
    Ok(())
}

pub(super) fn parse_request(relative: &str) -> Result<Option<ReleaseAssetRequest>> {
    ensure!(
        relative != "_shell",
        "shell release URL requires a version and asset path"
    );
    let Some(rest) = relative.strip_prefix(URL_PREFIX) else {
        return Ok(None);
    };
    let (version, encoded) = rest
        .split_once('/')
        .context("shell release URL requires a version and asset path")?;
    validate_version(version)?;
    let relative = decode_path(encoded)?;
    ensure!(
        relative
            .split('/')
            .all(|part| !part.starts_with('.') && !part.contains('\\') && !part.contains(':')),
        "unsafe shell release asset path"
    );
    ensure!(
        !relative.starts_with('/') && !relative.contains("//"),
        "unsafe shell release asset path"
    );
    Ok(Some(ReleaseAssetRequest {
        version: version.to_owned(),
        relative: if relative.is_empty() {
            "index.html".to_owned()
        } else {
            relative
        },
    }))
}

fn decode_path(encoded: &str) -> Result<String> {
    let mut bytes = Vec::with_capacity(encoded.len());
    let mut source = encoded.as_bytes().iter().copied();
    while let Some(byte) = source.next() {
        let decoded = if byte == b'%' {
            let high = source
                .next()
                .and_then(|byte| (byte as char).to_digit(16))
                .context("invalid shell asset URL escape")?;
            let low = source
                .next()
                .and_then(|byte| (byte as char).to_digit(16))
                .context("invalid shell asset URL escape")?;
            let value = ((high << 4) | low) as u8;
            // Separators must be structural, not concealed in encoded segments.
            ensure!(
                value != b'/' && value != b'\\',
                "encoded shell asset separator"
            );
            value
        } else {
            byte
        };
        ensure!(
            decoded >= 0x20 && decoded != 0x7f,
            "control byte in shell asset URL"
        );
        bytes.push(decoded);
    }
    String::from_utf8(bytes).context("shell asset URL is not UTF-8")
}

#[derive(Clone, Debug)]
pub(super) struct InventoryFile {
    pub size: u64,
    pub sha256: String,
}

#[derive(Debug)]
pub(super) struct VerifiedRelease {
    pub root: PathBuf,
    pub version: String,
    pub source_commit: String,
    pub artifact_sha256: String,
    files: BTreeMap<String, InventoryFile>,
}

impl VerifiedRelease {
    /// Called only after signature, compatibility and complete slot verification.
    pub(super) fn admitted(
        root: &Path,
        version: String,
        source_commit: String,
        artifact_sha256: String,
        files: BTreeMap<String, InventoryFile>,
    ) -> Result<Self> {
        validate_version(&version)?;
        ensure!(
            source_commit.len() == 40 && source_commit.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "invalid shell source commit"
        );
        ensure!(
            artifact_sha256.len() == 64
                && artifact_sha256.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "invalid shell artifact hash"
        );
        Ok(Self {
            root: fs::canonicalize(root)?,
            version,
            source_commit,
            artifact_sha256,
            files,
        })
    }

    /// Missing names return None. Missing/corrupt bytes of an admitted file fail.
    /// Rechecking the exact bytes prevents a verified-cache hit hiding later damage.
    pub(super) fn read(&self, relative: &str) -> Result<Option<(String, Vec<u8>)>> {
        let name = if self.files.contains_key(relative) {
            relative.to_owned()
        } else {
            let index = format!("{}/index.html", relative.trim_end_matches('/'));
            if !self.files.contains_key(&index) {
                return Ok(None);
            }
            index
        };
        let file = self
            .files
            .get(&name)
            .context("shell inventory entry disappeared")?;
        let requested = self.root.join(&name);
        let resolved = fs::canonicalize(&requested).context("verified shell file missing")?;
        ensure!(
            resolved.starts_with(&self.root),
            "verified shell file escaped its release"
        );
        let metadata = fs::symlink_metadata(&requested)?;
        ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink() && metadata.len() == file.size,
            "verified shell file size or type changed"
        );
        let bytes = fs::read(&requested)?;
        ensure!(
            bytes.len() as u64 == file.size
                && format!("{:x}", Sha256::digest(&bytes)) == file.sha256,
            "verified shell file integrity mismatch"
        );
        Ok(Some((name, bytes)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_base_pins_all_relative_assets_before_their_first_load() -> Result<()> {
        let html =
            "<!doctype html><html><head><script src='app.js?v=asset-only'></script></head></html>";
        assert_eq!(
            pin_document(html.to_owned(), "1.2.3-beta.1")?,
            "<!doctype html><html><head><base href=\"/business-os/_shell/1.2.3-beta.1/\"><script src='app.js?v=asset-only'></script></head></html>"
        );
        assert_eq!(
            pin_document("<html><head><base href=\"/business-os/\"><script src='app.js'></script></head></html>".into(), "1.2.3")?,
            "<html><head><base href=\"/business-os/_shell/1.2.3/\"><script src='app.js'></script></head></html>"
        );
        assert!(pin_document(
            "<head><base href='/business-os/'><base href='/business-os/'></head>".into(),
            "1.2.3"
        )
        .is_err());
        assert!(pin_document("<head><BASE href='/other/'></head>".into(), "1.2.3").is_err());
        assert!(pin_document(html.into(), "../other").is_err());
        assert!(pin_document("<html></html>".into(), "1.2.3").is_err());
        Ok(())
    }

    #[test]
    fn immutable_address_preserves_full_release_identity() -> Result<()> {
        let request =
            parse_request("_shell/1.2.3-beta.1+build.7/rxdb/dist/ctox-rxdb-js.mjs")?.unwrap();
        assert_eq!(request.version, "1.2.3-beta.1+build.7");
        assert_eq!(request.relative, "rxdb/dist/ctox-rxdb-js.mjs");
        assert_eq!(
            parse_request("_shell/1.2.3/")?.unwrap().relative,
            "index.html"
        );
        assert_eq!(
            parse_request("_shell/1.2.3/assets/caf%C3%A9.png")?
                .unwrap()
                .relative,
            "assets/café.png"
        );
        assert!(parse_request("installed-modules/app/index.js")?.is_none());
        for invalid in [
            "_shell/1.2.3",
            "_shell/01.2.3/app.js",
            "_shell/1.2.3-01/app.js",
            "_shell/1.2.3/../app.js",
            "_shell/1.2.3/%2e%2e/app.js",
            "_shell/1.2.3/a%2fb",
            "_shell/1.2.3/a%5cb",
            "_shell/1.2.3/%00",
            "_shell/1.2.3/%ff",
            "_shell/1.2.3/%",
            "_shell/1.2.3//app.js",
            "_shell/1.2.3/.ctox-shell-release.v2.json",
        ] {
            assert!(parse_request(invalid).is_err(), "accepted {invalid}");
        }
        Ok(())
    }

    #[test]
    fn admitted_release_detects_damage_after_a_successful_read() -> Result<()> {
        let temp = tempfile::tempdir()?;
        fs::write(temp.path().join("app.js"), b"first")?;
        let release = VerifiedRelease::admitted(
            temp.path(),
            "1.2.3".into(),
            "a".repeat(40),
            "b".repeat(64),
            BTreeMap::from([(
                "app.js".into(),
                InventoryFile {
                    size: 5,
                    sha256: format!("{:x}", Sha256::digest(b"first")),
                },
            )]),
        )?;
        assert_eq!(release.read("app.js")?.unwrap().1, b"first");
        assert!(release.read("installed-modules/app/index.js")?.is_none());
        fs::write(temp.path().join("app.js"), b"other")?;
        assert!(
            release.read("app.js").is_err(),
            "same-size damage must not be hidden by admission caching"
        );
        fs::remove_file(temp.path().join("app.js"))?;
        assert!(
            release.read("app.js").is_err(),
            "missing signed file must not fall back"
        );
        Ok(())
    }
}
