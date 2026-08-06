// ref: internal/pluginstore/install.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fs::{self, OpenOptions};
use std::io::{Cursor, Read, Write};
use std::path::{Component, Path, PathBuf};

use crate::sdk::pluginstore::{
    plugin_install_type, release_version, select_artifact, validate_plugin, Client, InstallOptions,
    InstallResult, Manifest, Plugin, PluginStoreIo, Result, ERR_LOADED_PLUGIN_LOCKED,
    INSTALL_TYPE_DIRECT, INSTALL_TYPE_GITHUB_RELEASE,
};

use super::checksum::{parse_checksums, verify_checksum};
use super::direct::download_selected;
use super::github::{select_release_assets, store_error, SafePluginStoreIo};
use super::registry::{
    direct_plugin_version, normalize_goarch, normalize_goos, normalize_version, valid_id,
    valid_version,
};

const MAX_EXTRACTED_LIBRARY_SIZE: u64 = 256 * 1024 * 1024;

pub(crate) fn install_plugin(
    io: &SafePluginStoreIo,
    client: &Client,
    plugin: &Plugin,
    options: &InstallOptions,
) -> Result<InstallResult> {
    validate_plugin(plugin)?;
    let normalized = normalized_options(options);
    if plugin_install_type(plugin) == INSTALL_TYPE_DIRECT {
        return install_direct(io, client, plugin.clone(), &plugin.install, &normalized);
    }
    let release = io.fetch_latest_release(client, plugin)?;
    let version = release_version(&release)?;
    install_release(io, client, plugin.clone(), release, &version, &normalized)
}

pub(crate) fn install_version(
    io: &SafePluginStoreIo,
    client: &Client,
    plugin: &Plugin,
    release_tag: &str,
    version: &str,
    options: &InstallOptions,
) -> Result<InstallResult> {
    validate_plugin(plugin)?;
    let version = normalize_version(version);
    if !valid_version(&version) {
        return Err(store_error(format!("invalid plugin version {version:?}")));
    }
    let tag = if release_tag.trim().is_empty() {
        version.as_str()
    } else {
        release_tag.trim()
    };
    let release = io.fetch_release_by_tag(client, plugin, tag)?;
    let actual = release_version(&release)?;
    if actual != version {
        return Err(store_error(format!(
            "release tag {tag:?} resolved version {actual:?}, want {version:?}"
        )));
    }
    install_release(
        io,
        client,
        plugin.clone(),
        release,
        &version,
        &normalized_options(options),
    )
}

pub(crate) fn install_manifest(
    io: &SafePluginStoreIo,
    client: &Client,
    manifest: &Manifest,
    options: &InstallOptions,
) -> Result<InstallResult> {
    manifest.validate()?;
    match manifest.install_type() {
        INSTALL_TYPE_GITHUB_RELEASE => install_version(
            io,
            client,
            &manifest.plugin(),
            &manifest.release_tag,
            &manifest.version,
            options,
        ),
        INSTALL_TYPE_DIRECT => {
            let mut plugin = manifest.plugin();
            plugin.version = normalize_version(&manifest.version);
            plugin.install.install_type = INSTALL_TYPE_DIRECT.to_owned();
            if plugin.install.artifacts.is_empty() {
                let source_url = if manifest.source_url.trim().is_empty() {
                    client.registry_url()
                } else {
                    manifest.source_url.trim()
                };
                let registry = io.fetch_registry_at(client, source_url)?;
                let resolved = registry
                    .plugin_by_id(&manifest.id)
                    .cloned()
                    .ok_or_else(|| {
                        store_error(format!(
                            "direct install plugin {:?} not found in source",
                            manifest.id.trim()
                        ))
                    })?;
                plugin = direct_plugin_version(resolved, &manifest.id, &manifest.version)?;
            }
            let plan = plugin.install.clone();
            install_direct(io, client, plugin, &plan, &normalized_options(options))
        }
        other => Err(store_error(format!("unsupported install type {other:?}"))),
    }
}

fn install_release(
    io: &SafePluginStoreIo,
    client: &Client,
    mut plugin: Plugin,
    release: crate::sdk::pluginstore::Release,
    version: &str,
    options: &InstallOptions,
) -> Result<InstallResult> {
    let (archive, checksums_asset) = select_release_assets(
        &release,
        &plugin.id,
        version,
        &options.goos,
        &options.goarch,
    )?;
    let archive_data = io
        .download_asset(client, &archive)
        .map_err(|error| store_error(format!("download {}: {error}", archive.name)))?;
    let checksum_data = io
        .download_asset(client, &checksums_asset)
        .map_err(|error| store_error(format!("download checksums.txt: {error}")))?;
    verify_checksum(
        &archive.name,
        &archive_data,
        &parse_checksums(&checksum_data)?,
    )?;
    plugin.version = version.to_owned();
    let mut result = install_archive(&archive_data, &plugin, options)?;
    result.install_type = INSTALL_TYPE_GITHUB_RELEASE.to_owned();
    result.release_tag = release.tag_name.trim().to_owned();
    Ok(result)
}

fn install_direct(
    io: &SafePluginStoreIo,
    client: &Client,
    mut plugin: Plugin,
    plan: &crate::sdk::pluginstore::InstallPlan,
    options: &InstallOptions,
) -> Result<InstallResult> {
    plugin.id = plugin.id.trim().to_owned();
    plugin.version = normalize_version(&plugin.version);
    let _ = select_artifact(plan, &options.goos, &options.goarch)?;
    let data = download_selected(io, client, plan, &options.goos, &options.goarch)?;
    let mut result = install_archive(&data, &plugin, options)?;
    result.install_type = INSTALL_TYPE_DIRECT.to_owned();
    Ok(result)
}

pub fn install_archive(
    archive_data: &[u8],
    plugin: &Plugin,
    options: &InstallOptions,
) -> Result<InstallResult> {
    let options = normalized_options(options);
    let id = plugin.id.trim();
    let version = normalize_version(&plugin.version);
    if !valid_id(id) {
        return Err(store_error(format!("invalid plugin id {:?}", plugin.id)));
    }
    if !valid_version(&version) {
        return Err(store_error(format!(
            "invalid plugin version {:?}",
            plugin.version
        )));
    }
    let (library, mode) = read_target_library(archive_data, id, &version, &options.goos)?;
    let target = target_path(&options, id, &version)?;
    reject_symlink_path(&options.plugins_dir, &target)?;
    let overwritten = match fs::symlink_metadata(&target) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(store_error("target plugin must not be a symlink"))
        }
        Ok(metadata) if metadata.is_file() => true,
        Ok(_) => return Err(store_error("target plugin is not a regular file")),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => return Err(store_error(format!("stat target plugin: {error}"))),
    };
    if overwritten {
        let existing = fs::read(&target)
            .map_err(|error| store_error(format!("read target plugin: {error}")))?;
        if existing == library {
            return Ok(InstallResult {
                id: id.to_owned(),
                version,
                path: target,
                overwritten: true,
                skipped: true,
                ..InstallResult::default()
            });
        }
        if let Some(before_write) = &options.before_write {
            before_write()
                .map_err(|error| store_error(format!("prepare plugin write: {error}")))?;
        }
        if options.goos == "windows"
            && options
                .plugin_loaded
                .as_ref()
                .is_some_and(|loaded| loaded())
        {
            return Err(ERR_LOADED_PLUGIN_LOCKED);
        }
    }
    write_atomic(&target, &library, mode)?;
    Ok(InstallResult {
        id: id.to_owned(),
        version,
        path: target,
        overwritten,
        ..InstallResult::default()
    })
}

fn read_target_library(
    archive_data: &[u8],
    id: &str,
    version: &str,
    goos: &str,
) -> Result<(Vec<u8>, u32)> {
    let mut archive = zip::ZipArchive::new(Cursor::new(archive_data))
        .map_err(|error| store_error(format!("open zip: {error}")))?;
    let plain = format!("{id}{}", plugin_extension(goos));
    let versioned = versioned_name(id, version, goos);
    let mut selected: Option<(usize, u32)> = None;
    for index in 0..archive.len() {
        let file = archive
            .by_index(index)
            .map_err(|error| store_error(format!("read zip directory: {error}")))?;
        let name = clean_zip_name(file.name())?;
        if file.is_dir() {
            continue;
        }
        if file
            .unix_mode()
            .is_some_and(|mode| mode & 0o170000 == 0o120000)
        {
            return Err(store_error(format!(
                "zip entry {} is not a regular file",
                file.name()
            )));
        }
        if !dynamic_library(&name) {
            continue;
        }
        if name != plain && name != versioned {
            if Path::new(&name)
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name == plain || name == versioned)
            {
                return Err(store_error("target dynamic library must be at zip root"));
            }
            return Err(store_error(format!(
                "dynamic library filename must be {plain} or {versioned}"
            )));
        }
        if selected.is_some() {
            return Err(store_error(
                "zip contains multiple target dynamic libraries",
            ));
        }
        selected = Some((index, file.unix_mode().unwrap_or(0o755) & 0o777));
    }
    let (index, mode) =
        selected.ok_or_else(|| store_error(format!("zip does not contain {plain}")))?;
    let file = archive
        .by_index(index)
        .map_err(|error| store_error(format!("open {plain}: {error}")))?;
    if file.size() > MAX_EXTRACTED_LIBRARY_SIZE {
        return Err(store_error(format!(
            "plugin library exceeds maximum allowed size of {MAX_EXTRACTED_LIBRARY_SIZE} bytes"
        )));
    }
    let mut data = Vec::with_capacity(
        usize::try_from(file.size())
            .unwrap_or_default()
            .min(MAX_EXTRACTED_LIBRARY_SIZE as usize),
    );
    file.take(MAX_EXTRACTED_LIBRARY_SIZE + 1)
        .read_to_end(&mut data)
        .map_err(|error| store_error(format!("read {plain}: {error}")))?;
    if data.len() as u64 > MAX_EXTRACTED_LIBRARY_SIZE {
        return Err(store_error(format!(
            "plugin library exceeds maximum allowed size of {MAX_EXTRACTED_LIBRARY_SIZE} bytes"
        )));
    }
    Ok((data, if mode == 0 { 0o755 } else { mode }))
}

fn target_path(options: &InstallOptions, id: &str, version: &str) -> Result<PathBuf> {
    if !valid_version(version) {
        return Err(store_error(format!("invalid plugin version {version:?}")));
    }
    Ok(options
        .plugins_dir
        .join(&options.goos)
        .join(&options.goarch)
        .join(versioned_name(id, version, &options.goos)))
}

fn normalized_options(options: &InstallOptions) -> InstallOptions {
    let plugins_dir = if options.plugins_dir.as_os_str().is_empty() {
        PathBuf::from("plugins")
    } else {
        options.plugins_dir.clone()
    };
    let goos = normalize_goos(if options.goos.trim().is_empty() {
        std::env::consts::OS
    } else {
        &options.goos
    });
    let goarch = normalize_goarch(if options.goarch.trim().is_empty() {
        std::env::consts::ARCH
    } else {
        &options.goarch
    });
    InstallOptions {
        plugins_dir,
        goos,
        goarch,
        plugin_loaded: options.plugin_loaded.clone(),
        before_write: options.before_write.clone(),
    }
}

fn versioned_name(id: &str, version: &str, goos: &str) -> String {
    format!(
        "{id}-v{}{extension}",
        normalize_version(version),
        extension = plugin_extension(goos)
    )
}
fn plugin_extension(goos: &str) -> &'static str {
    match normalize_goos(goos).as_str() {
        "darwin" => ".dylib",
        "windows" => ".dll",
        _ => ".so",
    }
}
fn dynamic_library(name: &str) -> bool {
    [".dylib", ".so", ".dll"]
        .iter()
        .any(|extension| name.to_ascii_lowercase().ends_with(extension))
}

fn clean_zip_name(name: &str) -> Result<String> {
    if name.trim().is_empty() {
        return Err(store_error("zip entry has empty name"));
    }
    if name.contains('\\') {
        return Err(store_error(format!(
            "zip entry {name} uses backslash path separators"
        )));
    }
    let path = Path::new(name);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(store_error(format!(
            "zip entry {name} escapes archive root"
        )));
    }
    Ok(name.trim_end_matches('/').to_owned())
}

fn reject_symlink_path(root: &Path, target: &Path) -> Result<()> {
    if !root.exists() {
        fs::create_dir_all(root)
            .map_err(|error| store_error(format!("create plugin directory: {error}")))?;
    }
    let metadata = fs::symlink_metadata(root)
        .map_err(|error| store_error(format!("inspect plugin directory: {error}")))?;
    if metadata.file_type().is_symlink() {
        return Err(store_error(format!(
            "plugin path contains symlink: {}",
            root.display()
        )));
    }
    if !metadata.is_dir() {
        return Err(store_error("configured plugin root is not a directory"));
    }
    let mut current = root.to_path_buf();
    if let Some(parent) = target.parent() {
        let relative = parent
            .strip_prefix(root)
            .map_err(|_| store_error("plugin target escapes configured directory"))?;
        for component in relative.components() {
            current.push(component.as_os_str());
            match fs::symlink_metadata(&current) {
                Ok(metadata) if metadata.file_type().is_symlink() => {
                    return Err(store_error(format!(
                        "plugin path contains symlink: {}",
                        current.display()
                    )))
                }
                Ok(metadata) if !metadata.is_dir() => {
                    return Err(store_error("plugin path component is not a directory"))
                }
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    fs::create_dir(&current)
                        .map_err(|error| store_error(format!("create plugin directory: {error}")))?
                }
                Err(error) => {
                    return Err(store_error(format!("inspect plugin directory: {error}")))
                }
            }
        }
    }
    Ok(())
}

fn write_atomic(target: &Path, data: &[u8], mode: u32) -> Result<()> {
    #[cfg(windows)]
    let _ = mode;
    let parent = target
        .parent()
        .ok_or_else(|| store_error("plugin target has no parent"))?;
    let temp = parent.join(format!(
        ".{}.tmp-{}",
        target
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("plugin"),
        uuid::Uuid::new_v4()
    ));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|error| store_error(format!("create temp plugin file: {error}")))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(fs::Permissions::from_mode(mode))
                .map_err(|error| store_error(format!("chmod temp plugin file: {error}")))?;
        }
        file.write_all(data)
            .map_err(|error| store_error(format!("write temp plugin file: {error}")))?;
        file.sync_all()
            .map_err(|error| store_error(format!("sync temp plugin file: {error}")))?;
        drop(file);
        replace_file_atomically(&temp, target)
            .map_err(|error| store_error(format!("install plugin file: {error}")))?;
        #[cfg(unix)]
        OpenOptions::new()
            .read(true)
            .open(parent)
            .and_then(|dir| dir.sync_all())
            .map_err(|error| store_error(format!("sync plugin directory: {error}")))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

#[cfg(not(windows))]
fn replace_file_atomically(source: &Path, destination: &Path) -> std::io::Result<()> {
    fs::rename(source, destination)
}

#[cfg(windows)]
fn replace_file_atomically(source: &Path, destination: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let source = source
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let destination = destination
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    // SAFETY: both buffers are owned, NUL-terminated UTF-16 paths and remain
    // alive for the duration of the Win32 call.
    let moved = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if moved == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}
