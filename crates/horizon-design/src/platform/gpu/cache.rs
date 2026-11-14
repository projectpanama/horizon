use std::{
    ffi::OsStr,
    io::Write,
    os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use vello::wgpu;

const CACHE_HEADER: &[u8; 8] = b"HZGPU001";
const CACHE_HEADER_SIZE: usize = 24;
const MAX_CACHE_SIZE: u64 = 64 * 1024 * 1024;

static NEXT_TEMPORARY_FILE: AtomicU64 = AtomicU64::new(0);

/// A persistent pipeline cache associated with a specific GPU adapter.
///
/// The cache is disabled when the device does not support pipeline caching or
/// when no suitable per-user cache directory can be determined.
pub(super) struct PipelineCache {
    pub(super) handle: Option<wgpu::PipelineCache>,
    path: Option<PathBuf>,
}

impl PipelineCache {
    /// Loads a compatible cache for `adapter_info`, if available, and creates
    /// a pipeline-cache handle for `device`.
    ///
    /// Missing, invalid, or incompatible cache data is ignored so pipeline
    /// creation can fall back to compiling pipelines normally.
    pub(super) fn new(device: &wgpu::Device, adapter_info: &wgpu::AdapterInfo) -> Self {
        if !device.features().contains(wgpu::Features::PIPELINE_CACHE) {
            return Self {
                handle: None,
                path: None,
            };
        }

        let path = wgpu::util::pipeline_cache_key(adapter_info).and_then(|key| {
            cache_directory(
                std::env::var_os("XDG_CACHE_HOME").as_deref(),
                std::env::var_os("HOME").as_deref(),
            )
            .map(|directory| directory.join(key))
        });
        let data = path.as_ref().and_then(|path| match read_cache(path) {
            Ok(data) => Some(data),
            Err(error) => {
                if error.kind() != std::io::ErrorKind::NotFound {
                    log::warn!("ignoring GPU pipeline cache {}: {error}", path.display());
                }
                None
            }
        });

        // SAFETY: this per-user cache stores only bytes returned by get_data.
        // Atomic replacement and a checksum reject incomplete or accidentally
        // corrupted writes. Its directory is trusted user-controlled storage;
        // the checksum is not authentication against malicious modification.
        // The key groups vendor/device IDs; wgpu checks driver compatibility.
        // fallback handles incompatibility, not arbitrary untrusted cache bytes.
        let handle = unsafe {
            device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("horizon"),
                data: data.as_deref(),
                fallback: true,
            })
        };

        Self {
            handle: Some(handle),
            path,
        }
    }

    /// Saves the current pipeline cache using an atomic replacement.
    ///
    /// Errors are logged and do not affect the running device.
    pub(super) fn save(&self) {
        if let (Some(handle), Some(path)) = (&self.handle, &self.path)
            && let Some(data) = handle.get_data()
            && let Err(error) = write_cache(path, &data)
        {
            log::warn!("can't save GPU pipeline cache {}: {error}", path.display());
        }
    }
}

/// Reads and validates a cache file, returning only its pipeline-cache payload.
///
/// Validation checks the file type, permissions, size, format header, payload
/// length, and checksum before the data is passed to wgpu.
fn read_cache(path: &Path) -> std::io::Result<Vec<u8>> {
    validate_cache_directory(
        path.parent()
            .ok_or_else(|| std::io::Error::other("cache path has no parent"))?,
    )?;

    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.permissions().mode() & 0o022 != 0 {
        return Err(std::io::Error::other(
            "cache must be a regular file writable only by its owner",
        ));
    }
    if metadata.len() > MAX_CACHE_SIZE + CACHE_HEADER_SIZE as u64 {
        return Err(std::io::Error::other(
            "pipeline cache exceeds the storage limit",
        ));
    }

    let mut data = std::fs::read(path)?;
    if data.len() < CACHE_HEADER_SIZE || &data[..8] != CACHE_HEADER {
        return Err(std::io::Error::other("unrecognized pipeline cache header"));
    }

    let length = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let expected_checksum = u64::from_le_bytes(data[16..24].try_into().unwrap());
    if length != (data.len() - CACHE_HEADER_SIZE) as u64
        || expected_checksum != checksum(&data[CACHE_HEADER_SIZE..])
    {
        return Err(std::io::Error::other(
            "pipeline cache is incomplete or corrupted",
        ));
    }
    data.drain(..CACHE_HEADER_SIZE);

    Ok(data)
}

/// Writes a validated pipeline-cache payload with a format header and checksum.
///
/// The payload is written to a new owner-only temporary file and then renamed
/// over the destination so readers never observe a partial cache.
fn write_cache(path: &Path, data: &[u8]) -> std::io::Result<()> {
    if data.len() as u64 > MAX_CACHE_SIZE {
        return Err(std::io::Error::other(
            "pipeline cache exceeds the storage limit",
        ));
    }

    let directory = path
        .parent()
        .ok_or_else(|| std::io::Error::other("cache path has no parent"))?;
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(directory)?;
    validate_cache_directory(directory)?;

    let (temporary_path, mut temporary_file) = loop {
        let sequence = NEXT_TEMPORARY_FILE.fetch_add(1, Ordering::Relaxed);
        let temporary_path =
            directory.join(format!(".pipeline-{}-{sequence}.tmp", std::process::id()));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary_path)
        {
            Ok(file) => break (temporary_path, file),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    };

    let result = (|| {
        temporary_file.write_all(CACHE_HEADER)?;
        temporary_file.write_all(&(data.len() as u64).to_le_bytes())?;
        temporary_file.write_all(&checksum(data).to_le_bytes())?;
        temporary_file.write_all(data)?;
        drop(temporary_file);
        std::fs::rename(&temporary_path, path)
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(temporary_path);
    }
    result
}

/// Resolves the per-user directory used for persistent GPU cache files.
///
/// An absolute `XDG_CACHE_HOME` takes precedence over `$HOME/.cache`.
fn cache_directory(xdg_cache_home: Option<&OsStr>, home: Option<&OsStr>) -> Option<PathBuf> {
    let absolute_path = |value: &OsStr| {
        let path = PathBuf::from(value);
        path.is_absolute().then_some(path)
    };
    xdg_cache_home
        .and_then(absolute_path)
        .or_else(|| home.and_then(absolute_path).map(|path| path.join(".cache")))
        .map(|path| path.join("horizon"))
}

/// Verifies that a cache directory is a private, existing directory.
fn validate_cache_directory(directory: &Path) -> std::io::Result<()> {
    let metadata = std::fs::symlink_metadata(directory)?;
    if !metadata.is_dir() || metadata.permissions().mode() & 0o022 != 0 {
        return Err(std::io::Error::other(
            "cache directory must be a directory writable only by its owner",
        ));
    }
    Ok(())
}

/// Computes the FNV-1a checksum used to detect accidental cache corruption.
///
/// This is an integrity check, not authentication. Cache provenance still
/// relies on trusted per-user storage as required by wgpu's unsafe cache API.
fn checksum(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            loop {
                let sequence = NEXT_TEMPORARY_FILE.fetch_add(1, Ordering::Relaxed);
                let directory = std::env::temp_dir().join(format!(
                    "horizon-gpu-cache-test-{}-{sequence}",
                    std::process::id()
                ));
                match std::fs::DirBuilder::new().mode(0o700).create(&directory) {
                    Ok(()) => return Self(directory),
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(error) => panic!("can't create test directory: {error}"),
                }
            }
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn ignores_empty_and_relative_cache_environment_paths() {
        for value in ["", "relative"] {
            assert_eq!(
                cache_directory(Some(OsStr::new(value)), Some(OsStr::new("/home/test"))),
                Some(PathBuf::from("/home/test/.cache/horizon"))
            );
        }
        assert_eq!(
            cache_directory(Some(OsStr::new("/cache")), None),
            Some(PathBuf::from("/cache/horizon"))
        );
        assert_eq!(cache_directory(None, Some(OsStr::new("relative"))), None);
    }

    #[test]
    fn rejects_corrupt_truncated_and_legacy_cache_data() {
        let directory = TestDirectory::new();
        let path = directory.0.join("cache");
        let payload = vec![42; 512];
        write_cache(&path, &payload).unwrap();
        assert_eq!(read_cache(&path).unwrap(), payload);
        let stored = std::fs::read(&path).unwrap();
        let mut corrupted = stored.clone();
        corrupted[CACHE_HEADER_SIZE + 10] ^= 1;
        std::fs::write(&path, corrupted).unwrap();
        assert!(read_cache(&path).is_err());
        std::fs::write(&path, &stored[..stored.len() - 1]).unwrap();
        assert!(read_cache(&path).is_err());
        std::fs::write(&path, &payload).unwrap();
        assert!(read_cache(&path).is_err());
    }

    #[test]
    fn concurrent_writers_publish_complete_private_files() {
        let directory = TestDirectory::new();
        let path = directory.0.join("cache");
        let barrier = std::sync::Barrier::new(8);
        std::thread::scope(|scope| {
            for writer_index in 0..8_u8 {
                let path = &path;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    for _ in 0..8 {
                        write_cache(path, &vec![writer_index; 4096 + usize::from(writer_index)])
                            .unwrap();
                        let published = read_cache(path).unwrap();
                        assert!(published.iter().all(|byte| *byte == published[0]));
                        assert_eq!(published.len(), 4096 + usize::from(published[0]));
                    }
                });
            }
        });
        assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 1);
        assert_eq!(
            std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

    #[test]
    fn rejects_symlinked_or_shared_writable_storage() {
        let directory = TestDirectory::new();
        let path = directory.0.join("cache");
        write_cache(&path, &[1, 2, 3]).unwrap();
        let link = directory.0.join("link");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert!(read_cache(&link).is_err());
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o666)).unwrap();
        assert!(read_cache(&path).is_err());
        std::fs::set_permissions(&directory.0, std::fs::Permissions::from_mode(0o777)).unwrap();
        assert!(write_cache(&path, &[4]).is_err());
    }
}
