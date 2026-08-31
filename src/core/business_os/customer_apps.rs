// Origin: CTOX
// License: AGPL-3.0-only

//! Fail-closed admission for customer-bound Business OS applications.
//!
//! Customer applications are runtime state, never release content. A package
//! is admitted only when its detached binding is signed, hashes the exact
//! package contents, and names the current CTOX instance explicitly.

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use ring::signature::{UnparsedPublicKey, ED25519};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;

pub(crate) const CUSTOMER_APP_BINDING_FILE: &str = "customer-app-binding.json";
const CUSTOMER_APP_BINDING_TYPE: &str = "ctox.business-os.customer-app-binding.v1";
const CURRENT_KEY: &str = "MCowBQYDK2VwAyEAZECH2XB0VlZWQ7zUzoChyiRkKtfGNK9HmSMvZQuwGjk=";
const NEXT_KEY: &str = "MCowBQYDK2VwAyEAdAgcqbHB2Sr86KzrWcdYxKCxb6Ofz4sVxhkEhTgvo7s=";
const MAX_BINDING_BYTES: u64 = 64 * 1024;
const MAX_PACKAGE_FILES: usize = 20_000;
const MAX_PACKAGE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CustomerAppBindingPayload {
    r#type: String,
    customer_id: String,
    module_id: String,
    allowed_instance_ids: Vec<String>,
    package_version: String,
    package_sha256: String,
    signing_key_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CustomerAppBinding {
    #[serde(flatten)]
    payload: CustomerAppBindingPayload,
    signature: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CustomerAppAuditEntry {
    pub module_id: String,
    pub source: String,
    pub status: String,
    pub reason: Option<String>,
}

fn classify_customer_manifest(manifest: &Value) -> Result<bool> {
    let id = manifest
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let mut declared_customer = false;
    for key in ["distribution", "audience", "visibility"] {
        let Some(raw) = manifest.get(key) else {
            continue;
        };
        let value = raw
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("customer-app-manifest-invalid-scope"))?
            .to_ascii_lowercase();
        match value.as_str() {
            "public" | "global" | "system" | "store" | "internal" | "shared" => {}
            // Unknown distribution vocabulary is private until explicitly
            // reviewed. This prevents a new spelling from silently bypassing
            // the customer boundary.
            _ => declared_customer = true,
        }
    }
    let customer_id = manifest
        .get("customer_id")
        .or_else(|| manifest.get("customerId"));
    let has_customer_id = match customer_id {
        None => false,
        Some(value) => {
            value
                .as_str()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| anyhow!("customer-app-manifest-invalid-customer"))?;
            true
        }
    };
    Ok(declared_customer || has_customer_id || id.starts_with("rem-") || id.starts_with("thesen-"))
}

pub(crate) fn manifest_requires_customer_binding(manifest: &Value) -> bool {
    classify_customer_manifest(manifest).unwrap_or(true)
}

fn reject_module_root_symlink(module_dir: &Path) -> Result<()> {
    let metadata =
        fs::symlink_metadata(module_dir).map_err(|_| anyhow!("customer-app-package-unreadable"))?;
    anyhow::ensure!(metadata.is_dir(), "customer-app-package-invalid-root");
    anyhow::ensure!(
        !metadata.file_type().is_symlink(),
        "customer-app-package-symlink"
    );
    Ok(())
}

/// Global source and marketplace trees are release content. A customer marker
/// or detached binding is therefore always a placement error, even if the
/// binding would be valid for the current instance.
pub(crate) fn authorize_global_module(module_dir: &Path, manifest: &Value) -> Result<()> {
    reject_module_root_symlink(module_dir)?;
    anyhow::ensure!(
        !classify_customer_manifest(manifest)?
            && !module_dir.join(CUSTOMER_APP_BINDING_FILE).exists(),
        "customer-app-global-placement-denied"
    );
    Ok(())
}

pub(crate) fn authorize_runtime_module(
    root: &Path,
    module_dir: &Path,
    manifest: &Value,
) -> Result<()> {
    reject_module_root_symlink(module_dir)?;
    let binding_path = module_dir.join(CUSTOMER_APP_BINDING_FILE);
    if !classify_customer_manifest(manifest)? && !binding_path.is_file() {
        return Ok(());
    }
    anyhow::ensure!(binding_path.is_file(), "customer-app-binding-required");
    let metadata =
        fs::metadata(&binding_path).with_context(|| "customer-app-binding-unreadable")?;
    anyhow::ensure!(
        metadata.len() > 0 && metadata.len() <= MAX_BINDING_BYTES,
        "customer-app-binding-invalid-size"
    );
    let binding: CustomerAppBinding = serde_json::from_slice(
        &fs::read(&binding_path).with_context(|| "customer-app-binding-unreadable")?,
    )
    .map_err(|_| anyhow!("customer-app-binding-invalid"))?;
    validate_binding(root, module_dir, manifest, &binding)
}

fn validate_binding(
    root: &Path,
    module_dir: &Path,
    manifest: &Value,
    binding: &CustomerAppBinding,
) -> Result<()> {
    let payload = &binding.payload;
    anyhow::ensure!(
        payload.r#type == CUSTOMER_APP_BINDING_TYPE,
        "customer-app-binding-unsupported"
    );
    let module_id = manifest
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    let package_version = manifest
        .get("version")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    anyhow::ensure!(
        !payload.customer_id.trim().is_empty()
            && !module_id.is_empty()
            && payload.module_id == module_id,
        "customer-app-binding-module-mismatch"
    );
    anyhow::ensure!(
        !package_version.is_empty() && payload.package_version == package_version,
        "customer-app-binding-version-mismatch"
    );
    anyhow::ensure!(
        payload.package_sha256.len() == 64
            && payload
                .package_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "customer-app-binding-invalid-hash"
    );
    let allowed = payload
        .allowed_instance_ids
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .collect::<BTreeSet<_>>();
    anyhow::ensure!(
        !allowed.is_empty() && allowed.len() == payload.allowed_instance_ids.len(),
        "customer-app-binding-invalid-instance-list"
    );
    let instance_id = read_instance_id(root)?;
    anyhow::ensure!(
        allowed.contains(instance_id.as_str()),
        "customer-app-binding-instance-denied"
    );
    anyhow::ensure!(
        customer_package_sha256(module_dir)? == payload.package_sha256,
        "customer-app-binding-package-mismatch"
    );
    verify_signature(payload, &binding.signature, &payload.signing_key_id)
}

fn read_instance_id(root: &Path) -> Result<String> {
    let path = root.join("runtime/business-os-instance-id");
    let metadata =
        fs::symlink_metadata(&path).map_err(|_| anyhow!("customer-app-instance-id-missing"))?;
    anyhow::ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "customer-app-instance-id-insecure"
    );
    #[cfg(unix)]
    anyhow::ensure!(
        metadata.permissions().mode() & 0o022 == 0,
        "customer-app-instance-id-insecure"
    );
    let value =
        fs::read_to_string(path).map_err(|_| anyhow!("customer-app-instance-id-missing"))?;
    let value = value.trim();
    anyhow::ensure!(!value.is_empty(), "customer-app-instance-id-missing");
    Ok(value.to_owned())
}

fn trusted_key(key_id: &str) -> Result<Vec<u8>> {
    let encoded = match key_id {
        "customer-app-current-2026-08" => CURRENT_KEY,
        "customer-app-next-2026-08" => NEXT_KEY,
        _ => bail!("customer-app-binding-unknown-key"),
    };
    let spki = base64::engine::general_purpose::STANDARD.decode(encoded)?;
    anyhow::ensure!(spki.len() == 44, "customer-app-binding-invalid-key");
    Ok(spki[12..].to_vec())
}

fn decode_signature(signature: &str) -> Result<Vec<u8>> {
    anyhow::ensure!(
        signature.len() == 128
            && signature
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "customer-app-binding-invalid-signature"
    );
    (0..signature.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&signature[index..index + 2], 16))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| anyhow!("customer-app-binding-invalid-signature"))
}

fn verify_signature(
    payload: &CustomerAppBindingPayload,
    signature: &str,
    key_id: &str,
) -> Result<()> {
    verify_with_public_key(payload, signature, &trusted_key(key_id)?)
}

fn verify_with_public_key(
    payload: &CustomerAppBindingPayload,
    signature: &str,
    public_key: &[u8],
) -> Result<()> {
    let signature = decode_signature(signature)?;
    UnparsedPublicKey::new(&ED25519, public_key)
        .verify(&serde_json::to_vec(payload)?, &signature)
        .map_err(|_| anyhow!("customer-app-binding-invalid-signature"))
}

pub(crate) fn customer_package_sha256(module_dir: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    let mut file_count = 0usize;
    let mut total_bytes = 0u64;
    hash_package_tree(
        module_dir,
        module_dir,
        &mut hasher,
        &mut file_count,
        &mut total_bytes,
    )?;
    anyhow::ensure!(file_count > 0, "customer-app-package-empty");
    Ok(format!("{:x}", hasher.finalize()))
}

fn hash_package_tree(
    root: &Path,
    current: &Path,
    hasher: &mut Sha256,
    file_count: &mut usize,
    total_bytes: &mut u64,
) -> Result<()> {
    let mut entries = fs::read_dir(current)
        .with_context(|| "customer-app-package-unreadable")?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if current == root && name == CUSTOMER_APP_BINDING_FILE {
            continue;
        }
        let file_type = entry.file_type()?;
        anyhow::ensure!(!file_type.is_symlink(), "customer-app-package-symlink");
        let path = entry.path();
        if file_type.is_dir() {
            hash_package_tree(root, &path, hasher, file_count, total_bytes)?;
            continue;
        }
        anyhow::ensure!(
            file_type.is_file(),
            "customer-app-package-unsupported-entry"
        );
        *file_count += 1;
        anyhow::ensure!(
            *file_count <= MAX_PACKAGE_FILES,
            "customer-app-package-too-many-files"
        );
        let bytes = fs::read(&path).with_context(|| "customer-app-package-unreadable")?;
        *total_bytes = total_bytes.saturating_add(bytes.len() as u64);
        anyhow::ensure!(
            *total_bytes <= MAX_PACKAGE_BYTES,
            "customer-app-package-too-large"
        );
        let rel = path
            .strip_prefix(root)?
            .to_string_lossy()
            .replace('\\', "/");
        hasher.update((rel.len() as u64).to_le_bytes());
        hasher.update(rel.as_bytes());
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    Ok(())
}

pub(crate) fn audit_runtime_customer_apps(root: &Path) -> Result<Vec<CustomerAppAuditEntry>> {
    let runtime_root = super::store::resolve_business_os_installed_app_root(root);
    let mut report = Vec::new();
    for source in ["installed-modules", "local-modules"] {
        let modules_root = runtime_root.join(source);
        if !modules_root.is_dir() {
            continue;
        }
        let mut entries =
            fs::read_dir(&modules_root)?.collect::<std::result::Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.path());
        for entry in entries {
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let module_dir = entry.path();
            let manifest_path = module_dir.join("module.json");
            if !manifest_path.is_file() {
                continue;
            }
            let manifest: Value = match serde_json::from_slice(&fs::read(&manifest_path)?) {
                Ok(value) => value,
                Err(_) => continue,
            };
            if !manifest_requires_customer_binding(&manifest)
                && !module_dir.join(CUSTOMER_APP_BINDING_FILE).is_file()
            {
                continue;
            }
            let module_id = manifest
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_owned();
            match authorize_runtime_module(root, &module_dir, &manifest) {
                Ok(()) => report.push(CustomerAppAuditEntry {
                    module_id,
                    source: source.to_owned(),
                    status: "authorized".to_owned(),
                    reason: None,
                }),
                Err(error) => report.push(CustomerAppAuditEntry {
                    module_id,
                    source: source.to_owned(),
                    status: "blocked".to_owned(),
                    reason: Some(error.to_string()),
                }),
            }
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ring::rand::SystemRandom;
    use ring::signature::{Ed25519KeyPair, KeyPair};
    use tempfile::TempDir;

    fn fixture() -> Result<(TempDir, PathBuf, Value)> {
        let root = TempDir::new()?;
        fs::create_dir_all(root.path().join("runtime"))?;
        fs::write(
            root.path().join("runtime/business-os-instance-id"),
            "biz_allowed\n",
        )?;
        let module_dir = root
            .path()
            .join("runtime/business-os/installed-modules/rem-private");
        fs::create_dir_all(&module_dir)?;
        let manifest = serde_json::json!({
            "id": "rem-private",
            "title": "Private app",
            "version": "1.2.3",
            "distribution": "customer"
        });
        fs::write(
            module_dir.join("module.json"),
            serde_json::to_vec(&manifest)?,
        )?;
        fs::write(module_dir.join("index.html"), b"private")?;
        Ok((root, module_dir, manifest))
    }

    #[test]
    fn customer_apps_require_a_binding() -> Result<()> {
        let (root, module_dir, manifest) = fixture()?;
        let error = authorize_runtime_module(root.path(), &module_dir, &manifest).unwrap_err();
        assert_eq!(error.to_string(), "customer-app-binding-required");
        Ok(())
    }

    #[test]
    fn signature_and_instance_are_bound_fail_closed() -> Result<()> {
        let (root, module_dir, manifest) = fixture()?;
        let random = SystemRandom::new();
        let pkcs8 = Ed25519KeyPair::generate_pkcs8(&random)
            .map_err(|error| anyhow::anyhow!("generate test key: {error:?}"))?;
        let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref())
            .map_err(|error| anyhow::anyhow!("load test key: {error:?}"))?;
        let payload = CustomerAppBindingPayload {
            r#type: CUSTOMER_APP_BINDING_TYPE.to_owned(),
            customer_id: "customer-opaque".to_owned(),
            module_id: "rem-private".to_owned(),
            allowed_instance_ids: vec!["biz_allowed".to_owned()],
            package_version: "1.2.3".to_owned(),
            package_sha256: customer_package_sha256(&module_dir)?,
            signing_key_id: "test".to_owned(),
        };
        let signature = key_pair.sign(&serde_json::to_vec(&payload)?);
        let signature = signature
            .as_ref()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        verify_with_public_key(&payload, &signature, key_pair.public_key().as_ref())?;

        let denied = CustomerAppBinding {
            payload: CustomerAppBindingPayload {
                allowed_instance_ids: vec!["biz_other".to_owned()],
                ..payload
            },
            signature,
        };
        let error = validate_binding(root.path(), &module_dir, &manifest, &denied).unwrap_err();
        assert_eq!(error.to_string(), "customer-app-binding-instance-denied");
        Ok(())
    }

    #[test]
    fn package_hash_excludes_only_the_detached_binding() -> Result<()> {
        let (_root, module_dir, _manifest) = fixture()?;
        let before = customer_package_sha256(&module_dir)?;
        fs::write(module_dir.join(CUSTOMER_APP_BINDING_FILE), b"binding")?;
        assert_eq!(customer_package_sha256(&module_dir)?, before);
        fs::write(module_dir.join(".runtime-config"), b"hidden but executable")?;
        assert_ne!(customer_package_sha256(&module_dir)?, before);
        fs::remove_file(module_dir.join(".runtime-config"))?;
        fs::write(module_dir.join("index.html"), b"tampered")?;
        assert_ne!(customer_package_sha256(&module_dir)?, before);
        Ok(())
    }

    #[test]
    fn unknown_or_malformed_scope_is_never_public_by_accident() -> Result<()> {
        let (root, module_dir, mut manifest) = fixture()?;
        manifest["id"] = Value::String("ordinary-name".to_owned());
        manifest["distribution"] = Value::String("new-private-spelling".to_owned());
        assert_eq!(
            authorize_runtime_module(root.path(), &module_dir, &manifest)
                .unwrap_err()
                .to_string(),
            "customer-app-binding-required"
        );
        manifest["distribution"] = serde_json::json!(["public"]);
        assert_eq!(
            authorize_runtime_module(root.path(), &module_dir, &manifest)
                .unwrap_err()
                .to_string(),
            "customer-app-manifest-invalid-scope"
        );
        Ok(())
    }

    #[test]
    fn global_source_tree_rejects_customer_markers_and_bindings() -> Result<()> {
        let (_root, module_dir, mut manifest) = fixture()?;
        assert_eq!(
            authorize_global_module(&module_dir, &manifest)
                .unwrap_err()
                .to_string(),
            "customer-app-global-placement-denied"
        );
        manifest["id"] = Value::String("public-app".to_owned());
        manifest.as_object_mut().unwrap().remove("distribution");
        fs::write(module_dir.join(CUSTOMER_APP_BINDING_FILE), b"binding")?;
        assert_eq!(
            authorize_global_module(&module_dir, &manifest)
                .unwrap_err()
                .to_string(),
            "customer-app-global-placement-denied"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn instance_identity_rejects_symlinks_and_writable_files() -> Result<()> {
        use std::os::unix::fs::{symlink, PermissionsExt};

        let (root, _module_dir, _manifest) = fixture()?;
        let instance_id = root.path().join("runtime/business-os-instance-id");
        let target = root.path().join("runtime/other-instance-id");
        fs::write(&target, "biz_allowed\n")?;
        fs::remove_file(&instance_id)?;
        symlink(&target, &instance_id)?;
        assert_eq!(
            read_instance_id(root.path()).unwrap_err().to_string(),
            "customer-app-instance-id-insecure"
        );

        fs::remove_file(&instance_id)?;
        fs::write(&instance_id, "biz_allowed\n")?;
        fs::set_permissions(&instance_id, fs::Permissions::from_mode(0o666))?;
        assert_eq!(
            read_instance_id(root.path()).unwrap_err().to_string(),
            "customer-app-instance-id-insecure"
        );
        Ok(())
    }
}
