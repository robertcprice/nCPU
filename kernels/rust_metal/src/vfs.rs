//! In-memory VFS for the GPU boot runtime — Rust port of `ncpu/os/gpu/filesystem.py`.
//!
//! Provides file, directory, symlink, pipe, and fd-table management for ARM64
//! programs running on the Metal GPU. Syscall handlers read/write through this VFS.

use std::collections::{HashMap, HashSet, VecDeque};
use std::net::{TcpListener, TcpStream, UdpSocket};
use std::sync::{Arc, Mutex};

// ── Limits ───────────────────────────────────────────────────────────────

pub const MAX_FD: i32 = 1024;
pub const MAX_PATH_LEN: usize = 4096;
pub const MAX_FILE_SIZE: usize = 16 * 1024 * 1024;
pub const MAX_SYMLINK_DEPTH: u32 = 8;

// ── Open flags (Linux aarch64 values) ────────────────────────────────────

pub const O_RDONLY: i32 = 0;
pub const O_WRONLY: i32 = 1;
pub const O_RDWR: i32 = 2;
pub const O_CREAT: i32 = 64;
pub const O_TRUNC: i32 = 512;
pub const O_APPEND: i32 = 1024;

// ── Pipe buffer ──────────────────────────────────────────────────────────

/// Shared buffer for inter-process pipe communication.
pub struct PipeBuffer {
    pub buffer: VecDeque<u8>,
    pub capacity: usize,
    pub readers: u32,
    pub writers: u32,
}

impl PipeBuffer {
    pub fn new(capacity: usize) -> Self {
        PipeBuffer {
            buffer: VecDeque::new(),
            capacity,
            readers: 1,
            writers: 1,
        }
    }

    pub fn write(&mut self, data: &[u8]) -> i32 {
        if self.readers == 0 {
            return -1; // EPIPE
        }
        let available = self.capacity.saturating_sub(self.buffer.len());
        if available == 0 {
            return 0; // would block
        }
        let n = data.len().min(available);
        self.buffer.extend(&data[..n]);
        n as i32
    }

    /// Returns Some(data) on success, Some(empty) on EOF, None on would-block.
    pub fn read(&mut self, count: usize) -> Option<Vec<u8>> {
        if !self.buffer.is_empty() {
            let n = count.min(self.buffer.len());
            let data: Vec<u8> = self.buffer.drain(..n).collect();
            Some(data)
        } else if self.writers == 0 {
            Some(vec![]) // EOF
        } else {
            None // would block
        }
    }

    pub fn close_reader(&mut self) {
        self.readers = self.readers.saturating_sub(1);
    }

    pub fn close_writer(&mut self) {
        self.writers = self.writers.saturating_sub(1);
    }
}

// ── Fd entry types ───────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum FdKind {
    File,
    Directory,
    PipeRead(usize),  // index into GpuVfs.pipes
    PipeWrite(usize), // index into GpuVfs.pipes
    HostSocket(Arc<Mutex<HostSocketState>>),
    Virtual, // stdin/stdout/stderr
}

#[derive(Debug)]
pub enum HostSocketState {
    PendingTcp {
        bind_host: Option<String>,
        bind_port: Option<u16>,
    },
    PendingUdp {
        bind_host: Option<String>,
        bind_port: Option<u16>,
    },
    Stream(TcpStream),
    Listener(TcpListener),
    UdpSocket(UdpSocket),
}

#[derive(Clone, Debug)]
pub struct FdEntry {
    pub kind: FdKind,
    pub path: String,
    pub offset: usize,
    pub flags: i32,
}

// ── LRU File Cache ───────────────────────────────────────────────────────

/// LRU cache for frequently accessed file contents
pub struct FileCache {
    pub cache: HashMap<String, Vec<u8>>,
    pub access_order: VecDeque<String>,
    pub max_size: usize,
    pub hits: u64,
    pub misses: u64,
}

impl FileCache {
    pub fn new(max_size: usize) -> Self {
        FileCache {
            cache: HashMap::new(),
            access_order: VecDeque::new(),
            max_size,
            hits: 0,
            misses: 0,
        }
    }

    pub fn get(&mut self, path: &str) -> Option<Vec<u8>> {
        if let Some(data) = self.cache.get(path) {
            // Move to front (most recently used)
            self.access_order.retain(|p| p != path);
            self.access_order.push_front(path.to_string());
            self.hits += 1;
            return Some(data.clone());
        }
        self.misses += 1;
        None
    }

    pub fn put(&mut self, path: String, data: Vec<u8>) {
        let size = data.len();

        // Evict if necessary
        while self.cache.len() >= self.max_size {
            if let Some(lru) = self.access_order.pop_back() {
                self.cache.remove(&lru);
            }
        }

        // Add new entry
        self.cache.insert(path.clone(), data);
        self.access_order.retain(|p| p != &path);
        self.access_order.push_front(path);
    }

    pub fn invalidate(&mut self, path: &str) {
        self.cache.remove(path);
        self.access_order.retain(|p| p != path);
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
}

// ── Mmap Region ───────────────────────────────────────────────────────────

/// Memory-mapped region
#[derive(Clone, Debug)]
pub struct MmapRegion {
    pub addr: u64,
    pub length: u64,
    pub prot: i32,      // PROT_READ=1, PROT_WRITE=2, PROT_EXEC=4
    pub flags: i32,    // MAP_SHARED=0x01, MAP_PRIVATE=0x02, MAP_ANONYMOUS=0x20
    pub fd: Option<i32>,  // File descriptor for file-backed mapping
    pub offset: u64,   // File offset
    pub data: Vec<u8>,  // Actual mapped data (for anonymous mappings)
}

// ── GpuVfs ───────────────────────────────────────────────────────────────

/// In-memory filesystem for GPU programs.
pub struct GpuVfs {
    pub files: HashMap<String, Vec<u8>>,
    pub directories: HashSet<String>,
    pub symlinks: HashMap<String, String>,
    pub fd_table: HashMap<i32, FdEntry>,
    pub pipes: Vec<PipeBuffer>,
    pub cwd: String,
    pub stdin_data: Option<Vec<u8>>,  // Piped stdin data
    pub stdin_offset: usize,          // Current read position for stdin
    pub file_cache: Option<FileCache>, // LRU cache for file reads
    pub mmap_regions: Vec<MmapRegion>, // Memory-mapped regions
}

impl GpuVfs {
    pub fn new() -> Self {
        let mut vfs = GpuVfs {
            files: HashMap::new(),
            directories: HashSet::new(),
            symlinks: HashMap::new(),
            fd_table: HashMap::new(),
            pipes: Vec::new(),
            cwd: "/".to_string(),
            stdin_data: None,
            stdin_offset: 0,
            file_cache: Some(FileCache::new(64)), // Cache up to 64 files
            mmap_regions: Vec::new(), // Track mmap regions
        };
        vfs.bootstrap();
        vfs
    }

    /// mmap: map memory or file into address space
    /// Returns the mapped address, or -1 on error
    pub fn mmap(&mut self, addr: u64, length: u64, prot: i32, flags: i32, fd: i32, offset: u64) -> i64 {
        let is_anonymous = (flags & 0x20) != 0; // MAP_ANONYMOUS

        // For now, only support anonymous mappings or read-only file mappings
        // Full file-backed mmap would require more complex fd handling
        if !is_anonymous && fd >= 0 {
            // File-backed mapping - check if we can handle it
            if let Some(entry) = self.fd_table.get(&fd) {
                if let FdKind::File = &entry.kind {
                    // Check if file exists and we can read it
                    if let Some(content) = self.files.get(&entry.path) {
                        let mut region = MmapRegion {
                            addr: 0, // Will be assigned by caller
                            length,
                            prot,
                            flags,
                            fd: Some(fd),
                            offset,
                            data: content.clone(),
                        };
                        self.mmap_regions.push(region);
                        return 0; // Success, address assigned by caller
                    }
                }
            }
        }

        // Anonymous mapping - allocate zeroed memory
        if is_anonymous {
            let data = vec![0u8; length as usize];
            let region = MmapRegion {
                addr: 0,
                length,
                prot,
                flags,
                fd: None,
                offset: 0,
                data,
            };
            self.mmap_regions.push(region);
            return 0;
        }

        // Unsupported mapping type
        -1 // ENOSYS
    }

    /// munmap: unmap memory region
    pub fn munmap(&mut self, addr: u64, length: u64) -> i32 {
        // Find and remove the region
        let initial_len = self.mmap_regions.len();
        self.mmap_regions.retain(|r| !(r.addr <= addr && r.addr + r.length > addr));

        if self.mmap_regions.len() < initial_len {
            0 // Success
        } else {
            -1 // EINVAL - no region found
        }
    }

    /// Read from mmap region
    pub fn mmap_read(&self, addr: u64, length: usize) -> Option<Vec<u8>> {
        for region in &self.mmap_regions {
            if region.addr <= addr && addr < region.addr + region.length {
                let offset = (addr - region.addr) as usize;
                let end = (offset + length).min(region.data.len());
                return Some(region.data[offset..end].to_vec());
            }
        }
        None
    }

    /// Write to mmap region
    pub fn mmap_write(&mut self, addr: u64, data: &[u8]) -> Option<usize> {
        for region in &mut self.mmap_regions {
            if region.addr <= addr && addr < region.addr + region.length {
                // Check PROT_WRITE
                if (region.prot & 2) == 0 {
                    return None; // EACCES
                }
                let offset = (addr - region.addr) as usize;
                let end = (offset + data.len()).min(region.data.len());
                region.data[offset..end].copy_from_slice(&data[..end - offset]);
                return Some(end - offset);
            }
        }
        None
    }

    /// Set stdin data to be served when fd 0 is read
    pub fn set_stdin(&mut self, data: Vec<u8>) {
        self.stdin_data = Some(data);
        self.stdin_offset = 0;
        // Pre-open stdin (fd 0) so the process can read immediately
        self.fd_table.insert(
            0,
            FdEntry {
                kind: FdKind::Virtual,
                path: "<stdin>".to_string(),
                offset: 0,
                flags: O_RDONLY,
            },
        );
    }

    /// Add a file to the VFS
    pub fn add_file(&mut self, path: &str, content: Vec<u8>) {
        let path = self.resolve_path(path);
        // Ensure parent directory exists
        let parent = Self::parent_dir(&path);
        if !parent.is_empty() && parent != "/" {
            self.directories.insert(parent);
        }
        self.files.insert(path, content);
    }

    /// Add a directory to the VFS
    pub fn add_directory(&mut self, path: &str) {
        let path = self.resolve_path(path);
        self.directories.insert(path);
    }

    fn bootstrap(&mut self) {
        for d in &["/", "/bin", "/home", "/tmp", "/etc", "/var", "/usr"] {
            self.directories.insert(d.to_string());
        }
        self.files.insert(
            "/etc/motd".to_string(),
            b"Welcome to GPU-Native UNIX OS v1.0\nRunning on Apple Silicon Metal\n".to_vec(),
        );
        self.files
            .insert("/etc/hostname".to_string(), b"gpu0\n".to_vec());
    }

    // ── Path resolution ──────────────────────────────────────────────────

    pub fn resolve_path(&self, path: &str) -> String {
        if path.is_empty() {
            return self.cwd.clone();
        }
        let truncated = if path.len() > MAX_PATH_LEN {
            &path[..MAX_PATH_LEN]
        } else {
            path
        };

        let full = if truncated.starts_with('/') {
            truncated.to_string()
        } else if self.cwd == "/" {
            format!("/{}", truncated)
        } else {
            format!("{}/{}", self.cwd, truncated)
        };

        let mut parts: Vec<&str> = Vec::new();
        for p in full.split('/') {
            match p {
                "" | "." => continue,
                ".." => {
                    parts.pop();
                }
                _ => parts.push(p),
            }
        }
        if parts.is_empty() {
            "/".to_string()
        } else {
            format!("/{}", parts.join("/"))
        }
    }

    fn parent_dir(path: &str) -> String {
        if path == "/" {
            return "/".to_string();
        }
        match path.rfind('/') {
            Some(0) | None => "/".to_string(),
            Some(idx) => path[..idx].to_string(),
        }
    }

    fn is_child_of(child: &str, parent: &str) -> bool {
        let prefix = if parent == "/" {
            "/".to_string()
        } else {
            format!("{}/", parent)
        };
        if !child.starts_with(&prefix) {
            return false;
        }
        let rest = &child[prefix.len()..];
        !rest.is_empty() && !rest.contains('/')
    }

    pub fn follow_symlink(&self, path: &str, depth: u32) -> String {
        if depth > MAX_SYMLINK_DEPTH {
            return path.to_string();
        }
        if let Some(target) = self.symlinks.get(path) {
            let resolved = self.resolve_path(target);
            self.follow_symlink(&resolved, depth + 1)
        } else {
            path.to_string()
        }
    }

    // ── Fd allocation ────────────────────────────────────────────────────

    fn allocate_fd(&self) -> i32 {
        let mut fd = 3i32;
        while self.fd_table.contains_key(&fd) {
            fd += 1;
            if fd >= MAX_FD {
                return -1;
            }
        }
        fd
    }

    fn access_mode(flags: i32) -> i32 {
        flags & 3
    }

    // ── File operations ──────────────────────────────────────────────────

    pub fn open(&mut self, path: &str, flags: i32) -> i32 {
        let path = self.resolve_path(path);
        let path = self.follow_symlink(&path, 0);

        let fd = self.allocate_fd();
        if fd < 0 {
            return -1;
        } // EMFILE

        // Directory open
        if self.directories.contains(&path) {
            self.fd_table.insert(
                fd,
                FdEntry {
                    kind: FdKind::Directory,
                    path,
                    offset: 0,
                    flags,
                },
            );
            return fd;
        }

        // Check parent exists
        let parent = Self::parent_dir(&path);
        if parent != "/" && !self.directories.contains(&parent) {
            return -1; // ENOENT
        }

        // Create if O_CREAT
        if (flags & O_CREAT) != 0 && !self.files.contains_key(&path) {
            self.files.insert(path.clone(), Vec::new());
        }
        if !self.files.contains_key(&path) {
            return -1; // ENOENT
        }

        // Truncate if O_TRUNC
        if (flags & O_TRUNC) != 0 {
            self.files.insert(path.clone(), Vec::new());
        }

        let offset = if (flags & O_APPEND) != 0 {
            self.files.get(&path).map_or(0, |f| f.len())
        } else {
            0
        };

        self.fd_table.insert(
            fd,
            FdEntry {
                kind: FdKind::File,
                path,
                offset,
                flags,
            },
        );
        fd
    }

    pub fn close(&mut self, fd: i32) -> i32 {
        let entry = match self.fd_table.remove(&fd) {
            Some(e) => e,
            None => return -1,
        };
        // Handle pipe cleanup
        match entry.kind {
            FdKind::PipeRead(idx) => {
                if let Some(pb) = self.pipes.get_mut(idx) {
                    pb.close_reader();
                }
            }
            FdKind::PipeWrite(idx) => {
                if let Some(pb) = self.pipes.get_mut(idx) {
                    pb.close_writer();
                }
            }
            _ => {}
        }
        0
    }

    pub fn read(&mut self, fd: i32, count: usize) -> Option<Vec<u8>> {
        // Clone entry data we need to avoid borrow conflicts
        let (kind, path, offset, flags) = {
            let entry = self.fd_table.get(&fd)?;
            (entry.kind.clone(), entry.path.clone(), entry.offset, entry.flags)
        };
        match &kind {
            FdKind::PipeRead(idx) => {
                let idx = *idx;
                self.pipes.get_mut(idx).and_then(|pb| pb.read(count))
            }
            FdKind::Directory => None, // EISDIR
            FdKind::File => {
                if Self::access_mode(flags) == O_WRONLY {
                    return None;
                }

                // Try cache first
                if let Some(ref mut cache) = self.file_cache {
                    if let Some(cached) = cache.get(&path) {
                        let end = (offset + count).min(cached.len());
                        let result = cached[offset..end].to_vec();
                        let len = result.len();
                        self.fd_table.get_mut(&fd).unwrap().offset += len;
                        return Some(result);
                    }
                }

                // Cache miss - read from files and populate cache
                let data = self.files.get(&path)?;
                let end = (offset + count).min(data.len());
                let result = data[offset..end].to_vec();
                let len = result.len();
                self.fd_table.get_mut(&fd).unwrap().offset += len;

                // Cache the full file content for future reads
                if let Some(ref mut cache) = self.file_cache {
                    if data.len() < 1024 * 1024 { // Only cache files < 1MB
                        cache.put(path.clone(), data.clone());
                    }
                }

                Some(result)
            }
            FdKind::HostSocket(socket) => {
                // Read from TCP stream
                use std::io::Read;
                use crate::vfs::HostSocketState;
                let mut state = socket.lock().unwrap();
                if let HostSocketState::Stream(stream) = &mut *state {
                    let mut buf = vec![0u8; count.min(4096)];
                    match stream.read(&mut buf) {
                        Ok(n) => {
                            if n > 0 {
                                self.fd_table.get_mut(&fd).unwrap().offset += n;
                            }
                            buf.truncate(n);
                            Some(buf)
                        }
                        Err(_) => None,
                    }
                } else {
                    None // Not connected
                }
            }
            FdKind::Virtual => {
                // Handle stdin specially - serve piped stdin_data
                if path == "<stdin>" {
                    if let Some(ref stdin_data) = self.stdin_data {
                        if self.stdin_offset >= stdin_data.len() {
                            return Some(vec![]); // EOF
                        }
                        let end = (self.stdin_offset + count).min(stdin_data.len());
                        let result = stdin_data[self.stdin_offset..end].to_vec();
                        self.stdin_offset = end;
                        return Some(result);
                    }
                }
                Some(vec![]) // EOF for other virtual fds (stdout, stderr)
            }
            _ => None,
        }
    }

    pub fn write(&mut self, fd: i32, data: &[u8]) -> i32 {
        let entry = match self.fd_table.get(&fd) {
            Some(e) => e.clone(),
            None => return -1,
        };
        match entry.kind {
            FdKind::PipeWrite(idx) => {
                if let Some(pb) = self.pipes.get_mut(idx) {
                    pb.write(data)
                } else {
                    -1
                }
            }
            FdKind::Directory => -1,
            FdKind::File => {
                if Self::access_mode(entry.flags) == O_RDONLY {
                    return -1;
                }
                let path = entry.path.clone();
                let offset = entry.offset;

                if offset + data.len() > MAX_FILE_SIZE {
                    return -1;
                }

                let mut content = self.files.get(&path).cloned().unwrap_or_default();
                // Extend if needed
                if offset > content.len() {
                    content.resize(offset, 0);
                }
                // Write at offset
                let end = offset + data.len();
                if end > content.len() {
                    content.resize(end, 0);
                }
                content[offset..end].copy_from_slice(data);
                self.files.insert(path.clone(), content);
                // Invalidate cache on write
                if let Some(ref mut cache) = self.file_cache {
                    cache.invalidate(&path);
                }
                self.fd_table.get_mut(&fd).unwrap().offset = end;
                data.len() as i32
            }
            FdKind::HostSocket(socket) => {
                // Write to TCP stream
                use std::sync::Mutex;
                use crate::vfs::HostSocketState;
                let mut state = socket.lock().unwrap();
                if let HostSocketState::Stream(stream) = &mut *state {
                    use std::io::Write;
                    match stream.write(data) {
                        Ok(n) => {
                            self.fd_table.get_mut(&fd).unwrap().offset += n;
                            n as i32
                        }
                        Err(_) => -1,
                    }
                } else {
                    -1 // Not connected
                }
            }
            _ => -1,
        }
    }

    pub fn lseek(&mut self, fd: i32, offset: i64, whence: i32) -> i64 {
        let entry = match self.fd_table.get(&fd) {
            Some(e) => e,
            None => return -1,
        };
        let size = self.files.get(&entry.path).map_or(0, |f| f.len()) as i64;
        let current = entry.offset as i64;
        let new_off = match whence {
            0 => offset,           // SEEK_SET
            1 => current + offset, // SEEK_CUR
            2 => size + offset,    // SEEK_END
            _ => return -1,
        };
        if new_off < 0 {
            return -1;
        }
        self.fd_table.get_mut(&fd).unwrap().offset = new_off as usize;
        new_off
    }

    // ── Stat ─────────────────────────────────────────────────────────────

    pub fn stat(&self, path: &str) -> Option<StatResult> {
        let path = self.resolve_path(path);
        if self.symlinks.contains_key(&path) {
            Some(StatResult {
                kind: StatKind::Symlink,
                size: self.files.get(&path).map_or(0, |f| f.len()),
                path,
            })
        } else if let Some(data) = self.files.get(&path) {
            Some(StatResult {
                kind: StatKind::File,
                size: data.len(),
                path,
            })
        } else if self.directories.contains(&path) {
            Some(StatResult {
                kind: StatKind::Dir,
                size: 0,
                path,
            })
        } else {
            None
        }
    }

    pub fn fstat(&self, fd: i32) -> Option<StatResult> {
        let entry = self.fd_table.get(&fd)?;
        self.stat(&entry.path)
    }

    /// Pack a stat result into a 128-byte Linux aarch64 stat64 struct.
    pub fn pack_stat64(info: &StatResult) -> [u8; 128] {
        let mut buf = [0u8; 128];
        let st_ino = {
            let mut h: u64 = 5381;
            for b in info.path.bytes() {
                h = h.wrapping_mul(33).wrapping_add(b as u64);
            }
            h
        };
        let (st_mode, st_nlink) = match info.kind {
            StatKind::Symlink => (0o120777u32, 1u32),
            StatKind::Dir => (0o040755, 2),
            StatKind::File => (0o100755, 1),
        };
        let st_size = if matches!(info.kind, StatKind::Dir) {
            0i64
        } else {
            info.size as i64
        };
        let st_blocks = (st_size + 511) / 512;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0i64, |d| d.as_secs() as i64);

        let mut off = 0;
        macro_rules! w_u64 {
            ($v:expr) => {
                buf[off..off + 8].copy_from_slice(&($v as u64).to_le_bytes());
                off += 8;
            };
        }
        macro_rules! w_u32 {
            ($v:expr) => {
                buf[off..off + 4].copy_from_slice(&($v as u32).to_le_bytes());
                off += 4;
            };
        }
        macro_rules! w_i64 {
            ($v:expr) => {
                buf[off..off + 8].copy_from_slice(&($v as i64).to_le_bytes());
                off += 8;
            };
        }
        macro_rules! w_i32 {
            ($v:expr) => {
                buf[off..off + 4].copy_from_slice(&($v as i32).to_le_bytes());
                off += 4;
            };
        }

        w_u64!(1u64); // st_dev
        w_u64!(st_ino); // st_ino
        w_u32!(st_mode); // st_mode
        w_u32!(st_nlink); // st_nlink
        w_u32!(0u32); // st_uid
        w_u32!(0u32); // st_gid
        w_u64!(0u64); // st_rdev
        w_i64!(0i64); // __pad1
        w_i64!(st_size); // st_size
        w_i32!(4096i32); // st_blksize
        w_i32!(0i32); // __pad2
        w_i64!(st_blocks); // st_blocks
        w_i64!(now); // st_atime
        w_i64!(0i64); // st_atime_nsec
        w_i64!(now); // st_mtime
        w_i64!(0i64); // st_mtime_nsec
        w_i64!(now); // st_ctime
        w_i64!(0i64); // st_ctime_nsec
        w_i32!(0i32); // __unused4
        w_i32!(0i32); // __unused5
        debug_assert_eq!(off, 128);
        buf
    }

    // ── Directory operations ─────────────────────────────────────────────

    pub fn mkdir(&mut self, path: &str) -> i32 {
        let path = self.resolve_path(path);
        if self.directories.contains(&path) || self.files.contains_key(&path) {
            return -1; // EEXIST
        }
        let parent = Self::parent_dir(&path);
        if parent != "/" && !self.directories.contains(&parent) {
            return -1; // ENOENT
        }
        self.directories.insert(path);
        0
    }

    pub fn unlink(&mut self, path: &str) -> i32 {
        let path = self.resolve_path(path);
        if self.files.remove(&path).is_some() {
            self.symlinks.remove(&path);
            let to_close: Vec<i32> = self
                .fd_table
                .iter()
                .filter(|(_, e)| e.path == path)
                .map(|(&fd, _)| fd)
                .collect();
            for fd in to_close {
                self.fd_table.remove(&fd);
            }
            0
        } else {
            -1
        }
    }

    pub fn rmdir(&mut self, path: &str) -> i32 {
        let path = self.resolve_path(path);
        if !self.directories.contains(&path) || path == "/" {
            return -1;
        }
        // Check empty
        for f in self.files.keys() {
            if Self::is_child_of(f, &path) {
                return -1;
            }
        }
        for d in &self.directories {
            if d != &path && Self::is_child_of(d, &path) {
                return -1;
            }
        }
        self.directories.remove(&path);
        0
    }

    pub fn rename(&mut self, old_path: &str, new_path: &str) -> i32 {
        let old_path = self.resolve_path(old_path);
        let new_path = self.resolve_path(new_path);
        let new_parent = Self::parent_dir(&new_path);
        if new_parent != "/" && !self.directories.contains(&new_parent) {
            return -1;
        }

        if self.files.contains_key(&old_path) {
            let content = self.files.remove(&old_path).unwrap();
            self.files.insert(new_path.clone(), content);
            for entry in self.fd_table.values_mut() {
                if entry.path == old_path {
                    entry.path = new_path.clone();
                }
            }
            0
        } else if self.directories.contains(&old_path) {
            self.directories.remove(&old_path);
            self.directories.insert(new_path.clone());
            let prefix = format!("{}/", old_path);
            let to_move: Vec<(String, Vec<u8>)> = self
                .files
                .iter()
                .filter(|(k, _)| k.starts_with(&prefix))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            for (k, v) in to_move {
                self.files.remove(&k);
                self.files
                    .insert(format!("{}{}", new_path, &k[old_path.len()..]), v);
            }
            let dir_moves: Vec<String> = self
                .directories
                .iter()
                .filter(|d| d.starts_with(&prefix))
                .cloned()
                .collect();
            for d in dir_moves {
                self.directories.remove(&d);
                self.directories
                    .insert(format!("{}{}", new_path, &d[old_path.len()..]));
            }
            0
        } else {
            -1
        }
    }

    pub fn getcwd(&self) -> &str {
        &self.cwd
    }

    pub fn chdir(&mut self, path: &str) -> i32 {
        let path = self.resolve_path(path);
        if self.directories.contains(&path) {
            self.cwd = path;
            0
        } else {
            -1
        }
    }

    pub fn listdir(&self, path: &str) -> Option<Vec<String>> {
        let path = self.resolve_path(path);
        if !self.directories.contains(&path) {
            return None;
        }
        let mut entries: HashSet<String> = HashSet::new();
        for f in self.files.keys() {
            if Self::is_child_of(f, &path) {
                if let Some(name) = f.rsplit('/').next() {
                    if !name.is_empty() {
                        entries.insert(name.to_string());
                    }
                }
            }
        }
        for d in &self.directories {
            if Self::is_child_of(d, &path) {
                if let Some(name) = d.rsplit('/').next() {
                    if !name.is_empty() {
                        entries.insert(name.to_string());
                    }
                }
            }
        }
        let mut sorted: Vec<String> = entries.into_iter().collect();
        sorted.sort();
        Some(sorted)
    }

    // ── Symlinks ─────────────────────────────────────────────────────────

    pub fn symlink(&mut self, target: &str, link_path: &str) -> i32 {
        let link_path = self.resolve_path(link_path);
        if self.files.contains_key(&link_path) || self.directories.contains(&link_path) {
            return -1;
        }
        let parent = Self::parent_dir(&link_path);
        if parent != "/" && !self.directories.contains(&parent) {
            return -1;
        }
        self.symlinks.insert(link_path.clone(), target.to_string());
        self.files.insert(link_path, target.as_bytes().to_vec());
        0
    }

    pub fn readlink(&self, path: &str) -> Option<&str> {
        let path = self.resolve_path(path);
        self.symlinks.get(&path).map(|s| s.as_str())
    }

    pub fn is_symlink(&self, path: &str) -> bool {
        let path = self.resolve_path(path);
        self.symlinks.contains_key(&path)
    }

    // ── Pipes ────────────────────────────────────────────────────────────

    pub fn create_pipe(&mut self) -> (i32, i32) {
        let read_fd = self.allocate_fd();
        if read_fd < 0 {
            return (-1, -1);
        }

        let pipe_idx = self.pipes.len();
        self.pipes.push(PipeBuffer::new(4096));

        self.fd_table.insert(
            read_fd,
            FdEntry {
                kind: FdKind::PipeRead(pipe_idx),
                path: "<pipe>".to_string(),
                offset: 0,
                flags: O_RDONLY,
            },
        );

        let write_fd = self.allocate_fd();
        if write_fd < 0 {
            self.fd_table.remove(&read_fd);
            self.pipes.pop();
            return (-1, -1);
        }

        self.fd_table.insert(
            write_fd,
            FdEntry {
                kind: FdKind::PipeWrite(pipe_idx),
                path: "<pipe>".to_string(),
                offset: 0,
                flags: O_WRONLY,
            },
        );

        (read_fd, write_fd)
    }

    pub fn insert_host_socket(&mut self, socket: HostSocketState, flags: i32, path: &str) -> i32 {
        let fd = self.allocate_fd();
        if fd < 0 {
            return -1;
        }

        self.fd_table.insert(
            fd,
            FdEntry {
                kind: FdKind::HostSocket(Arc::new(Mutex::new(socket))),
                path: path.to_string(),
                offset: 0,
                flags,
            },
        );
        fd
    }

    pub fn dup2(&mut self, old_fd: i32, new_fd: i32) -> i32 {
        if !self.fd_table.contains_key(&old_fd) {
            // Handle virtual fds
            if (0..=2).contains(&old_fd) {
                let name = match old_fd {
                    0 => "<stdin>",
                    1 => "<stdout>",
                    2 => "<stderr>",
                    _ => "<unknown>",
                };
                self.fd_table.insert(
                    old_fd,
                    FdEntry {
                        kind: FdKind::Virtual,
                        path: name.to_string(),
                        offset: 0,
                        flags: O_RDWR,
                    },
                );
            } else {
                return -1;
            }
        }
        // Close new_fd if open
        if self.fd_table.contains_key(&new_fd) {
            self.close(new_fd);
        }
        let mut entry = self.fd_table[&old_fd].clone();
        // Increment pipe refcounts
        match &entry.kind {
            FdKind::PipeRead(idx) => {
                if let Some(pb) = self.pipes.get_mut(*idx) {
                    pb.readers += 1;
                }
            }
            FdKind::PipeWrite(idx) => {
                if let Some(pb) = self.pipes.get_mut(*idx) {
                    pb.writers += 1;
                }
            }
            _ => {}
        }
        self.fd_table.insert(new_fd, entry);
        new_fd
    }

    pub fn clone_fd_table(&mut self) -> HashMap<i32, FdEntry> {
        let mut cloned = HashMap::new();
        for (&fd, entry) in &self.fd_table {
            let new_entry = entry.clone();
            match &new_entry.kind {
                FdKind::PipeRead(idx) => {
                    if let Some(pb) = self.pipes.get_mut(*idx) {
                        pb.readers += 1;
                    }
                }
                FdKind::PipeWrite(idx) => {
                    if let Some(pb) = self.pipes.get_mut(*idx) {
                        pb.writers += 1;
                    }
                }
                _ => {}
            }
            cloned.insert(fd, new_entry);
        }
        cloned
    }

    // ── Convenience ──────────────────────────────────────────────────────

    pub fn write_file(&mut self, path: &str, content: &[u8]) {
        let path = self.resolve_path(path);
        let parent = Self::parent_dir(&path);
        if parent != "/" {
            self.directories.insert(parent);
        }
        self.files.insert(path, content.to_vec());
    }

    pub fn read_file(&self, path: &str) -> Option<&[u8]> {
        let path = self.resolve_path(path);
        self.files.get(&path).map(|v| v.as_slice())
    }

    pub fn exists(&self, path: &str) -> bool {
        let path = self.resolve_path(path);
        self.files.contains_key(&path) || self.directories.contains(&path)
    }

    // ── Cache control ───────────────────────────────────────────────────────

    /// Enable or disable the file cache
    pub fn set_cache_enabled(&mut self, enabled: bool) {
        if enabled && self.file_cache.is_none() {
            self.file_cache = Some(FileCache::new(64));
        } else if !enabled {
            self.file_cache = None;
        }
    }

    /// Get cache statistics (hits, misses, hit_rate, cached_files)
    pub fn cache_stats(&self) -> Option<(u64, u64, f64, usize)> {
        self.file_cache.as_ref().map(|c| {
            (c.hits, c.misses, c.hit_rate(), c.cache.len())
        })
    }

    /// Clear the file cache
    pub fn clear_cache(&mut self) {
        if let Some(ref mut cache) = self.file_cache {
            cache.cache.clear();
            cache.access_order.clear();
        }
    }

    /// Pack getdents64 entries for a directory fd.
    /// Returns (packed_bytes, entries_consumed).
    pub fn pack_getdents64(
        &self,
        dir_path: &str,
        buf_size: usize,
        consumed: usize,
    ) -> (Vec<u8>, usize) {
        let entries = match self.listdir(dir_path) {
            Some(e) => e,
            None => return (vec![], 0),
        };
        let mut all_entries: Vec<&str> = vec![".", ".."];
        let entry_refs: Vec<&str> = entries.iter().map(|s| s.as_str()).collect();
        all_entries.extend(entry_refs);

        let mut buf = Vec::new();
        let mut count = 0usize;
        for (i, name) in all_entries.iter().enumerate().skip(consumed) {
            let name_bytes = name.as_bytes();
            // d_ino(8) + d_off(8) + d_reclen(2) + d_type(1) + name + NUL
            let reclen_raw = 8 + 8 + 2 + 1 + name_bytes.len() + 1;
            let reclen = (reclen_raw + 7) & !7; // 8-byte align

            if buf.len() + reclen > buf_size {
                break;
            }

            let d_ino: u64 = (i as u64).wrapping_add(1);
            let d_off: u64 = (i as u64).wrapping_add(1);
            let full_path = if *name == "." || *name == ".." {
                dir_path.to_string()
            } else if dir_path == "/" {
                format!("/{}", name)
            } else {
                format!("{}/{}", dir_path, name)
            };
            let d_type: u8 = if self.directories.contains(&full_path) {
                4
            }
            // DT_DIR
            else if self.symlinks.contains_key(&full_path) {
                10
            }
            // DT_LNK
            else {
                8
            }; // DT_REG

            buf.extend_from_slice(&d_ino.to_le_bytes());
            buf.extend_from_slice(&d_off.to_le_bytes());
            buf.extend_from_slice(&(reclen as u16).to_le_bytes());
            buf.push(d_type);
            buf.extend_from_slice(name_bytes);
            buf.push(0); // NUL
                         // Pad to alignment
            while buf.len() % 8 != 0 {
                buf.push(0);
            }
            count += 1;
        }
        (buf, count)
    }
}

// ── Stat result ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum StatKind {
    File,
    Dir,
    Symlink,
}

#[derive(Debug, Clone)]
pub struct StatResult {
    pub kind: StatKind,
    pub size: usize,
    pub path: String,
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_file_operations() {
        let mut vfs = GpuVfs::new();
        vfs.write_file("/tmp/hello.txt", b"Hello, GPU!");
        assert!(vfs.exists("/tmp/hello.txt"));
        assert_eq!(
            vfs.read_file("/tmp/hello.txt"),
            Some(b"Hello, GPU!".as_slice())
        );
    }

    #[test]
    fn open_read_write_close() {
        let mut vfs = GpuVfs::new();
        let fd = vfs.open("/tmp/test.txt", O_CREAT | O_RDWR);
        assert!(fd >= 3);
        assert_eq!(vfs.write(fd, b"test data"), 9);
        assert_eq!(vfs.lseek(fd, 0, 0), 0); // SEEK_SET
        let data = vfs.read(fd, 100).unwrap();
        assert_eq!(data, b"test data");
        assert_eq!(vfs.close(fd), 0);
    }

    #[test]
    fn directory_operations() {
        let mut vfs = GpuVfs::new();
        assert_eq!(vfs.mkdir("/tmp/subdir"), 0);
        assert!(vfs.directories.contains("/tmp/subdir"));
        assert_eq!(vfs.mkdir("/tmp/subdir"), -1); // EEXIST
        assert_eq!(vfs.rmdir("/tmp/subdir"), 0);
        assert!(!vfs.directories.contains("/tmp/subdir"));
    }

    #[test]
    fn path_resolution() {
        let mut vfs = GpuVfs::new();
        assert_eq!(vfs.resolve_path("/foo/bar/../baz"), "/foo/baz");
        assert_eq!(vfs.resolve_path("/foo/./bar"), "/foo/bar");
        assert_eq!(vfs.resolve_path("/"), "/");
        vfs.cwd = "/home".to_string();
        vfs.directories.insert("/home".to_string());
        assert_eq!(vfs.resolve_path("test"), "/home/test");
    }

    #[test]
    fn symlink_operations() {
        let mut vfs = GpuVfs::new();
        vfs.write_file("/etc/real", b"real content");
        assert_eq!(vfs.symlink("/etc/real", "/etc/link"), 0);
        assert!(vfs.is_symlink("/etc/link"));
        assert_eq!(vfs.readlink("/etc/link"), Some("/etc/real"));
        let resolved = vfs.follow_symlink("/etc/link", 0);
        assert_eq!(resolved, "/etc/real");
    }

    #[test]
    fn pipe_operations() {
        let mut vfs = GpuVfs::new();
        let (rfd, wfd) = vfs.create_pipe();
        assert!(rfd >= 3);
        assert!(wfd > rfd);

        assert_eq!(vfs.write(wfd, b"pipe data"), 9);
        let data = vfs.read(rfd, 100).unwrap();
        assert_eq!(data, b"pipe data");

        // Close writer → reader gets EOF
        vfs.close(wfd);
        let eof = vfs.read(rfd, 100).unwrap();
        assert!(eof.is_empty());
    }

    #[test]
    fn stat_packing() {
        let info = StatResult {
            kind: StatKind::File,
            size: 42,
            path: "/test".to_string(),
        };
        let packed = GpuVfs::pack_stat64(&info);
        assert_eq!(packed.len(), 128);
        // st_mode at offset 16 should be 0o100755
        let mode = u32::from_le_bytes(packed[16..20].try_into().unwrap());
        assert_eq!(mode, 0o100755);
        // st_size at offset 48
        let size = i64::from_le_bytes(packed[48..56].try_into().unwrap());
        assert_eq!(size, 42);
    }

    #[test]
    fn getdents64_packing() {
        let mut vfs = GpuVfs::new();
        vfs.write_file("/tmp/a.txt", b"aaa");
        vfs.write_file("/tmp/b.txt", b"bbb");
        let (packed, count) = vfs.pack_getdents64("/tmp", 4096, 0);
        assert!(count >= 4); // . + .. + a.txt + b.txt
        assert!(!packed.is_empty());
        assert_eq!(packed.len() % 8, 0); // 8-byte aligned
    }

    #[test]
    fn rename_file() {
        let mut vfs = GpuVfs::new();
        vfs.write_file("/tmp/old.txt", b"content");
        assert_eq!(vfs.rename("/tmp/old.txt", "/tmp/new.txt"), 0);
        assert!(!vfs.files.contains_key("/tmp/old.txt"));
        assert_eq!(vfs.read_file("/tmp/new.txt"), Some(b"content".as_slice()));
    }

    #[test]
    fn dup2_basic() {
        let mut vfs = GpuVfs::new();
        vfs.write_file("/tmp/test", b"hello");
        let fd = vfs.open("/tmp/test", O_RDONLY);
        assert!(fd >= 3);
        let new_fd = vfs.dup2(fd, 10);
        assert_eq!(new_fd, 10);
        let data = vfs.read(10, 100).unwrap();
        assert_eq!(data, b"hello");
    }

    #[test]
    fn lseek_whence() {
        let mut vfs = GpuVfs::new();
        vfs.write_file("/tmp/seek", b"0123456789");
        let fd = vfs.open("/tmp/seek", O_RDONLY);
        assert_eq!(vfs.lseek(fd, 5, 0), 5); // SEEK_SET
        assert_eq!(vfs.lseek(fd, 2, 1), 7); // SEEK_CUR
        assert_eq!(vfs.lseek(fd, -3, 2), 7); // SEEK_END
    }
}
