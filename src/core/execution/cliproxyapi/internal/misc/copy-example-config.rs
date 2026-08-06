// ref: internal/misc/copy-example-config.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::Path;

/// Copies a configuration template to a private, durable destination.
///
/// The source is opened before destination directories are created, matching
/// upstream's mutation order. On Unix, newly created directories and the file
/// receive 0700 and 0600 respectively. The final `sync_all` is the durable
/// completion boundary.
pub fn copy_config_template(
    source: impl AsRef<Path>,
    destination: impl AsRef<Path>,
) -> io::Result<()> {
    let mut source = File::open(source)?;
    let destination = destination.as_ref();
    let parent = destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    create_private_directories(parent)?;

    let mut options = OpenOptions::new();
    options.create(true).write(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut output = options.open(destination)?;
    io::copy(&mut source, &mut output)?;
    output.sync_all()
}

fn create_private_directories(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;

        let mut builder = fs::DirBuilder::new();
        builder.recursive(true).mode(0o700);
        builder.create(path)?;
    }
    #[cfg(not(unix))]
    {
        fs::create_dir_all(path)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_exact_bytes_truncates_and_creates_private_destination() {
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("template.yaml");
        let destination = root.path().join("nested/config.yaml");
        fs::write(&source, b"model: claude\n").unwrap();

        copy_config_template(&source, &destination).unwrap();
        fs::write(&destination, b"stale trailing content").unwrap();

        copy_config_template(&source, &destination).unwrap();
        assert_eq!(fs::read(&destination).unwrap(), b"model: claude\n");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(destination.parent().unwrap())
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o700
            );
            assert_eq!(
                fs::metadata(&destination).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn missing_source_does_not_create_destination_parent() {
        let root = tempfile::tempdir().unwrap();
        let parent = root.path().join("not-created");
        let error =
            copy_config_template(root.path().join("missing"), parent.join("config")).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert!(!parent.exists());
    }
}
