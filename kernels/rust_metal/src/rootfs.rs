//! Rootfs loader — populate a GpuVfs from a boot image's rootfs region.

use crate::boot_image::unpack_rootfs;
use crate::vfs::GpuVfs;

/// Load rootfs entries from a packed boot image rootfs region into a VFS.
///
/// Creates all necessary parent directories automatically.
pub fn load_rootfs_into_vfs(vfs: &mut GpuVfs, rootfs_data: &[u8]) -> Result<usize, String> {
    let entries =
        unpack_rootfs(rootfs_data).map_err(|e| format!("failed to unpack rootfs: {}", e))?;

    let count = entries.len();
    for entry in &entries {
        // Ensure all parent directories exist
        let path = &entry.path;
        let mut dir_parts = Vec::new();
        for part in path.split('/').filter(|p| !p.is_empty()) {
            dir_parts.push(part);
            // Skip the last part (the file/dir itself)
        }
        // All but last component are directories
        if dir_parts.len() > 1 {
            for i in 1..dir_parts.len() {
                let dir_path = format!("/{}", dir_parts[..i].join("/"));
                vfs.directories.insert(dir_path);
            }
        }

        // If content is empty and path ends with '/', treat as directory
        if entry.content.is_empty() && path.ends_with('/') {
            let dir_path = path.trim_end_matches('/').to_string();
            if !dir_path.is_empty() {
                vfs.directories.insert(dir_path);
            }
        } else {
            // Regular file
            vfs.files.insert(path.clone(), entry.content.clone());
        }
    }

    Ok(count)
}

/// Populate a VFS with the standard Alpine-like rootfs directory structure.
///
/// This mirrors the directories created by `create_alpine_rootfs()` in Python.
pub fn create_standard_dirs(vfs: &mut GpuVfs) {
    let dirs = [
        "/",
        "/bin",
        "/sbin",
        "/usr",
        "/usr/bin",
        "/usr/sbin",
        "/usr/lib",
        "/etc",
        "/etc/init.d",
        "/etc/profile.d",
        "/etc/ssl",
        "/etc/ssl/certs",
        "/etc/network",
        "/etc/network/if-up.d",
        "/etc/network/if-down.d",
        "/home",
        "/home/user",
        "/root",
        "/tmp",
        "/var",
        "/var/log",
        "/var/run",
        "/var/tmp",
        "/var/cache",
        "/var/cache/apk",
        "/dev",
        "/proc",
        "/sys",
        "/mnt",
        "/opt",
        "/srv",
        "/run",
        "/lib",
    ];
    for d in &dirs {
        vfs.directories.insert(d.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boot_image::{pack_rootfs, RootfsEntry};

    #[test]
    fn load_rootfs_basic() {
        let entries = vec![
            RootfsEntry {
                path: "/etc/hostname".to_string(),
                content: b"ncpu\n".to_vec(),
            },
            RootfsEntry {
                path: "/bin/sh".to_string(),
                content: vec![0xFF; 16],
            },
            RootfsEntry {
                path: "/etc/passwd".to_string(),
                content: b"root:x:0:0::/root:/bin/sh\n".to_vec(),
            },
        ];
        let packed = pack_rootfs(&entries);

        let mut vfs = GpuVfs::new();
        let count = load_rootfs_into_vfs(&mut vfs, &packed).unwrap();

        assert_eq!(count, 3);
        assert_eq!(vfs.read_file("/etc/hostname"), Some(b"ncpu\n".as_slice()));
        assert_eq!(vfs.read_file("/bin/sh").unwrap().len(), 16);
        assert!(vfs.directories.contains("/etc"));
        assert!(vfs.directories.contains("/bin"));
    }

    #[test]
    fn standard_dirs() {
        let mut vfs = GpuVfs::new();
        create_standard_dirs(&mut vfs);
        assert!(vfs.directories.contains("/bin"));
        assert!(vfs.directories.contains("/etc/init.d"));
        assert!(vfs.directories.contains("/proc"));
        assert!(vfs.directories.contains("/home/user"));
    }
}
