// ref: sdk/pluginstore/pluginstore_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::*;

const SHA256: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn manifest_validate_requires_pinned_release_tag() {
    let mut manifest = valid_test_manifest();
    manifest.release_tag.clear();

    let error = manifest.validate().unwrap_err();
    assert!(error.to_string().contains("release-tag"), "{error}");
}

#[test]
fn manifest_validate_rejects_release_tag_version_mismatch() {
    let mut manifest = valid_test_manifest();
    manifest.release_tag = "v0.3.0".to_owned();

    let error = manifest.validate().unwrap_err();
    assert!(error.to_string().contains("resolves version"), "{error}");
}

#[test]
fn manifest_from_release_builds_pinned_manifest() {
    let manifest = manifest_from_release(
        &default_source(),
        &sample_plugin(),
        &Release {
            tag_name: "v0.2.0".to_owned(),
            assets: Vec::new(),
        },
    )
    .unwrap();

    manifest.validate().unwrap();
    assert_eq!(manifest.version, "0.2.0");
    assert_eq!(manifest.release_tag, "v0.2.0");
}

#[test]
fn manifest_from_plugin_builds_direct_manifest() {
    let plugin = Plugin {
        version: "0.4.0".to_owned(),
        install: InstallPlan {
            install_type: INSTALL_TYPE_DIRECT.to_owned(),
            artifacts: vec![Artifact {
                goos: "linux".to_owned(),
                goarch: "amd64".to_owned(),
                url: "https://downloads.example/sample-provider.zip".to_owned(),
                sha256: SHA256.to_owned(),
                size: 0,
            }],
        },
        ..sample_plugin()
    };

    let manifest = manifest_from_plugin(&default_source(), &plugin).unwrap();
    manifest.validate().unwrap();
    assert_eq!(manifest.schema_version, SCHEMA_VERSION_V2);
    assert_eq!(manifest.install_type(), INSTALL_TYPE_DIRECT);
    assert!(manifest.release_tag.is_empty());
    assert_eq!(manifest.source_url, DEFAULT_REGISTRY_URL);
    assert_eq!(manifest.install.artifacts.len(), 1);
    let artifact = &manifest.install.artifacts[0];
    assert_eq!(artifact.goos, "linux");
    assert_eq!(artifact.goarch, "amd64");
    assert_eq!(
        artifact.url,
        "https://downloads.example/sample-provider.zip"
    );
}

#[test]
fn manifest_from_plugin_rejects_artifact_query_without_leaking_it() {
    let plugin = Plugin {
        version: "1.0.0".to_owned(),
        install: InstallPlan {
            install_type: INSTALL_TYPE_DIRECT.to_owned(),
            artifacts: vec![Artifact {
                goos: "linux".to_owned(),
                goarch: "amd64".to_owned(),
                url: "https://downloads.example/sample.zip?X-Amz-Signature=secret".to_owned(),
                sha256: SHA256.to_owned(),
                size: 0,
            }],
        },
        ..sample_plugin()
    };

    let error = manifest_from_plugin(&default_source(), &plugin).unwrap_err();
    assert!(!error.to_string().contains("secret"), "{error}");
}

#[test]
fn plugin_artifacts_includes_version_artifacts_and_normalizes_platforms() {
    let plugin = Plugin {
        version: "0.4.0".to_owned(),
        install: InstallPlan {
            install_type: INSTALL_TYPE_DIRECT.to_owned(),
            artifacts: vec![Artifact {
                goos: "windows".to_owned(),
                goarch: "x64".to_owned(),
                url: "https://downloads.example/sample-provider.zip".to_owned(),
                sha256: SHA256.to_owned(),
                size: 0,
            }],
        },
        versions: vec![Version {
            version: "0.3.0".to_owned(),
            install: InstallPlan {
                install_type: INSTALL_TYPE_DIRECT.to_owned(),
                artifacts: vec![Artifact {
                    goos: "linux".to_owned(),
                    goarch: "aarch64".to_owned(),
                    url: "https://downloads.example/sample-provider-0.3.0.zip".to_owned(),
                    sha256: SHA256.to_owned(),
                    size: 0,
                }],
            },
        }],
        ..sample_plugin()
    };

    let artifacts = plugin_artifacts(&plugin);
    assert_eq!(artifacts.len(), 2);
    assert_eq!(artifacts[0].goarch, "amd64");
    assert_eq!(artifacts[1].goarch, "arm64");
}

#[test]
fn resolved_auth_selection_clones_owned_secret_material() {
    let mut auth = vec![ResolvedAuthConfig {
        match_url: "https://api.github.com/repos/acme/tool".to_owned(),
        apply_to: vec![REQUEST_KIND_METADATA.to_owned()],
        auth_type: AUTH_TYPE_BEARER.to_owned(),
        token: Secret::new(b"token".to_vec()),
        ..ResolvedAuthConfig::default()
    }];

    let selected = resolved_auth_for_request(
        &auth,
        "https://api.github.com/repos/acme/tool/releases/latest",
        REQUEST_KIND_METADATA,
    )
    .unwrap();
    clear_resolved_auth_configs(&mut auth);
    assert_eq!(selected.token.expose(), b"token");
    assert!(auth[0].token.expose().is_empty());
}

#[test]
fn secret_debug_is_strictly_redacted_through_auth_config() {
    let auth = ResolvedAuthConfig {
        token: Secret::new(b"credential-must-not-leak".to_vec()),
        ..ResolvedAuthConfig::default()
    };

    let debug = format!("{auth:?}");
    assert!(debug.contains("[REDACTED]"), "{debug}");
    assert!(!debug.contains("credential-must-not-leak"), "{debug}");
    assert!(!debug.contains("99, 114, 101, 100"), "{debug}");
}

#[test]
fn source_and_update_helpers_match_upstream() {
    assert_eq!(
        source_id("https://example.com/registry.json"),
        "source-71e3eb8cdfb7"
    );
    assert!(update_available("v1.2.3", "1.3.0"));
    assert!(!update_available("2.0", "1.99.99"));
    assert!(!update_available("1.2", "1.2.0"));
    assert!(update_available("dev-a", "dev-b"));
}

fn sample_plugin() -> Plugin {
    Plugin {
        id: "sample-provider".to_owned(),
        name: "Sample Provider".to_owned(),
        description: "Adds sample provider support.".to_owned(),
        author: "author-name".to_owned(),
        repository: "https://github.com/author-name/sample-provider".to_owned(),
        ..Plugin::default()
    }
}

fn valid_test_manifest() -> Manifest {
    Manifest {
        id: "sample-provider".to_owned(),
        name: "Sample Provider".to_owned(),
        description: "Adds sample provider support.".to_owned(),
        author: "author-name".to_owned(),
        version: "0.2.0".to_owned(),
        release_tag: "v0.2.0".to_owned(),
        repository: "https://github.com/author-name/sample-provider".to_owned(),
        ..Manifest::default()
    }
}
