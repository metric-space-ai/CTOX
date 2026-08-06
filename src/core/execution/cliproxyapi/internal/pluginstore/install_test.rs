// ref: internal/pluginstore/install_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::io::{Cursor, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;

use chrono::{TimeZone, Utc};
use sha2::{Digest, Sha256};

use crate::sdk::pluginstore::{
    Artifact, Client, InstallOptions, InstallPlan, Plugin, INSTALL_TYPE_DIRECT,
};

use super::auth::UrlPolicy;
use super::github::{HttpRequest, HttpResponse, PluginStoreTransport, SafePluginStoreIo};
use super::install::install_archive;

struct LoopbackTransport;

impl PluginStoreTransport for LoopbackTransport {
    fn get(&self, request: &HttpRequest) -> std::result::Result<HttpResponse, String> {
        let host = request.url.host_str().ok_or("missing host")?;
        let port = request.url.port_or_known_default().ok_or("missing port")?;
        let mut stream = TcpStream::connect((host, port)).map_err(|error| error.to_string())?;
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(2)))
            .map_err(|error| error.to_string())?;
        write!(
            stream,
            "GET {} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n",
            request.url.path()
        )
        .map_err(|error| error.to_string())?;
        for (name, value) in &request.headers {
            write!(stream, "{name}: {value}\r\n").map_err(|error| error.to_string())?;
        }
        stream
            .write_all(b"\r\n")
            .map_err(|error| error.to_string())?;
        let mut bytes = Vec::new();
        while !bytes.windows(4).any(|window| window == b"\r\n\r\n") {
            if bytes.len() == 64 * 1024 {
                return Err("response head too large".to_owned());
            }
            let mut byte = [0_u8; 1];
            stream
                .read_exact(&mut byte)
                .map_err(|error| error.to_string())?;
            bytes.push(byte[0]);
        }
        let head =
            std::str::from_utf8(&bytes[..bytes.len() - 4]).map_err(|error| error.to_string())?;
        let status = head
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|value| value.parse().ok())
            .ok_or("invalid status")?;
        let mut headers = BTreeMap::new();
        for line in head.lines().skip(1) {
            if let Some((name, value)) = line.split_once(':') {
                headers.insert(name.trim().to_owned(), value.trim().to_owned());
            }
        }
        Ok(HttpResponse {
            status,
            headers,
            body: Box::new(stream),
        })
    }
}

#[test]
fn loopback_download_checksum_and_atomic_filesystem_install_are_end_to_end() {
    let archive = plugin_zip("sample.so", b"native-plugin");
    let digest = format!("{:x}", Sha256::digest(&archive));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let expected = archive.clone();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = Vec::new();
        let mut chunk = [0_u8; 1024];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let size = stream.read(&mut chunk).unwrap();
            assert!(size > 0);
            request.extend_from_slice(&chunk[..size]);
        }
        assert!(String::from_utf8_lossy(&request).starts_with("GET /artifact.zip HTTP/1.1"));
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            expected.len()
        )
        .unwrap();
        stream.write_all(&expected).unwrap();
    });
    let origin = format!("http://{address}");
    let now = Utc.with_ymd_and_hms(2026, 8, 4, 0, 0, 0).unwrap();
    let io = Arc::new(SafePluginStoreIo::new(
        Arc::new(LoopbackTransport),
        UrlPolicy::default().allow_http_origin(&origin),
        Arc::new(move || now),
    ));
    let client = Client::new(io, format!("{origin}/registry.json"));
    let root = tempfile::tempdir().unwrap();
    let plugin = Plugin {
        id: "sample".into(),
        name: "Sample".into(),
        description: "Plugin".into(),
        author: "Acme".into(),
        version: "1.0.0".into(),
        install: InstallPlan {
            install_type: INSTALL_TYPE_DIRECT.into(),
            artifacts: vec![Artifact {
                goos: "linux".into(),
                goarch: "amd64".into(),
                url: format!("{origin}/artifact.zip"),
                sha256: digest,
                size: archive.len() as i64,
            }],
        },
        ..Plugin::default()
    };
    let result = client
        .install(
            &plugin,
            &InstallOptions {
                plugins_dir: root.path().into(),
                goos: "linux".into(),
                goarch: "amd64".into(),
                ..InstallOptions::default()
            },
        )
        .unwrap();
    server.join().unwrap();
    assert_eq!(std::fs::read(&result.path).unwrap(), b"native-plugin");
    assert!(!result.overwritten);
    assert!(result.path.ends_with("linux/amd64/sample-v1.0.0.so"));
}

#[test]
fn archive_rejects_nested_library_and_symlinked_target_directory() {
    let plugin = Plugin {
        id: "sample".into(),
        version: "1.0.0".into(),
        ..Plugin::default()
    };
    let root = tempfile::tempdir().unwrap();
    let options = InstallOptions {
        plugins_dir: root.path().into(),
        goos: "linux".into(),
        goarch: "amd64".into(),
        ..InstallOptions::default()
    };
    assert!(
        install_archive(&plugin_zip("nested/sample.so", b"bad"), &plugin, &options)
            .unwrap_err()
            .to_string()
            .contains("zip root")
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        let outside = tempfile::tempdir().unwrap();
        symlink(outside.path(), root.path().join("linux")).unwrap();
        assert!(
            install_archive(&plugin_zip("sample.so", b"bad"), &plugin, &options)
                .unwrap_err()
                .to_string()
                .contains("symlink")
        );
    }
}

fn plugin_zip(name: &str, data: &[u8]) -> Vec<u8> {
    let cursor = Cursor::new(Vec::new());
    let mut writer = zip::ZipWriter::new(cursor);
    writer
        .start_file(
            name,
            zip::write::FileOptions::default().unix_permissions(0o755),
        )
        .unwrap();
    writer.write_all(data).unwrap();
    writer.finish().unwrap().into_inner()
}
