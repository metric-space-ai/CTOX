// ref: internal/pluginstore/direct.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use sha2::{Digest, Sha256};

use crate::sdk::pluginstore::{select_artifact, Artifact, Client, InstallPlan, Result};

use super::github::{store_error, SafePluginStoreIo};

pub(crate) fn download_selected(
    io: &SafePluginStoreIo,
    client: &Client,
    plan: &InstallPlan,
    goos: &str,
    goarch: &str,
) -> Result<Vec<u8>> {
    let artifact = select_artifact(plan, goos, goarch)?;
    let data = io.download_artifact(client, &artifact.url, artifact.size)?;
    verify_artifact_checksum(&artifact, &data)?;
    Ok(data)
}

pub(crate) fn verify_artifact_checksum(artifact: &Artifact, data: &[u8]) -> Result<()> {
    let actual = format!("{:x}", Sha256::digest(data));
    let expected = artifact.sha256.trim().to_ascii_lowercase();
    if subtle::ConstantTimeEq::ct_eq(actual.as_bytes(), expected.as_bytes()).into() {
        Ok(())
    } else {
        Err(store_error(format!(
            "checksum mismatch for artifact {}/{}",
            artifact.goos, artifact.goarch
        )))
    }
}
