use ctox_sync::authority::auth::SigningIdentity;
use std::{fs, process::Command};

#[test]
fn imports_the_existing_node_mesh_key_without_creating_another_identity() {
    let root = tempfile::tempdir().unwrap();
    // The same Node API and DER format used by WorkjetMeshIdentity. Test keys
    // stay inside the temporary private directory and never reach tool output.
    let status = Command::new("node")
        .arg("-e")
        .arg(r#"
            const crypto = require('node:crypto');
            const fs = require('node:fs');
            const path = require('node:path');
            const root = process.argv[1];
            for (const curve of ['ed25519', 'x25519']) {
                const pair = crypto.generateKeyPairSync(curve, {
                    privateKeyEncoding: {format: 'der', type: 'pkcs8'},
                    publicKeyEncoding: {format: 'der', type: 'spki'},
                });
                fs.writeFileSync(path.join(root, curve + '.der'), pair.privateKey, {mode: 0o600, flag: 'wx'});
                if (curve === 'ed25519') {
                    const key = crypto.createPrivateKey({key: pair.privateKey, format: 'der', type: 'pkcs8'});
                    const publicKey = crypto.createPublicKey(key).export({format: 'jwk'}).x;
                    fs.writeFileSync(path.join(root, 'identity.txt'), 'ed25519:' + Buffer.from(publicKey, 'base64url').toString('hex'));
                }
            }
        "#)
        .arg(root.path())
        .status()
        .expect("Node is required to verify the actual Workjet key encoding");
    assert!(status.success());
    let key = fs::read(root.path().join("ed25519.der")).unwrap();
    let expected = fs::read_to_string(root.path().join("identity.txt")).unwrap();
    // This is precisely the incompatibility: strict v2-only import rejects it.
    assert!(SigningIdentity::from_pkcs8(&key).is_err());
    let imported = SigningIdentity::from_existing_pkcs8(&key, &expected).unwrap();
    assert_eq!(imported.public_identity(), expected);
    let other_bytes = SigningIdentity::generate_pkcs8().unwrap();
    let other = SigningIdentity::from_pkcs8(&other_bytes).unwrap();
    assert!(SigningIdentity::from_existing_pkcs8(&key, &other.public_identity()).is_err());
    assert!(SigningIdentity::from_existing_pkcs8(&key[..key.len() - 1], &expected).is_err());
    assert!(SigningIdentity::from_existing_pkcs8(
        &fs::read(root.path().join("x25519.der")).unwrap(),
        &expected
    )
    .is_err());
    assert!(SigningIdentity::from_existing_pkcs8(&other_bytes, &other.public_identity()).is_ok());
}
