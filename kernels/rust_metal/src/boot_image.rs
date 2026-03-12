//! Boot image format and memory map constants for the GPU boot runtime.
//!
//! This is the first execution slice of the GPU boot architecture plan:
//! the boot image becomes a concrete Rust artifact instead of a doc-only idea.
//!
//! The `BootImageBuilder` provides a builder-pattern API for constructing
//! complete boot images: `.add_region()` → `.set_task()` → `.set_rootfs()` → `.build()`.

use core::mem::size_of;
use std::io::{self, Write};

pub const BOOT_IMAGE_MAGIC: u32 = u32::from_le_bytes(*b"NCBT");
pub const BOOT_IMAGE_VERSION: u16 = 1;

pub const MEMORY_PROFILE_16M: u32 = 16;
pub const MEMORY_PROFILE_64M: u32 = 64;
pub const MEMORY_PROFILE_256M: u32 = 256;

pub const GPU_BOOT_STANDARD_MEMORY_SIZE: usize = 256 * 1024 * 1024;

pub const GPU_BOOT_GUARD_BASE: usize = 0x0000_0000;
pub const GPU_BOOT_STAGE0_BASE: usize = 0x0001_0000;
pub const GPU_BOOT_HEADER_BASE: usize = 0x0002_0000;
pub const GPU_BOOT_BRIDGE_DESC_BASE: usize = 0x0003_0000;
pub const GPU_BOOT_KERNEL_SCRATCH_BASE: usize = 0x0004_0000;
pub const GPU_BOOT_KERNEL_META_BASE: usize = 0x0008_0000;
pub const GPU_BOOT_KERNEL_HEAP_BASE: usize = 0x0010_0000;
pub const GPU_BOOT_ROOTFS_BASE: usize = 0x0040_0000;
pub const GPU_BOOT_OVERLAY_BASE: usize = 0x0140_0000;
pub const GPU_BOOT_RING_BASE: usize = 0x0180_0000;
pub const GPU_BOOT_TASK_BASE: usize = 0x01C0_0000;
pub const GPU_BOOT_BLOCK_CACHE_BASE: usize = 0x0200_0000;
pub const GPU_BOOT_IMAGE_SPACE_BASE: usize = 0x0300_0000;
pub const GPU_BOOT_USER_SPACE_BASE: usize = 0x0400_0000;
pub const GPU_BOOT_DEBUG_BASE: usize = 0x0C00_0000;
pub const GPU_BOOT_SNAPSHOT_BASE: usize = 0x0C20_0000;
pub const GPU_BOOT_EXPANSION_BASE: usize = 0x0C40_0000;

pub const IMAGE_FLAG_BRIDGED_MODE: u64 = 1 << 0;
pub const IMAGE_FLAG_TRACE_ON_BOOT: u64 = 1 << 1;
pub const IMAGE_FLAG_DEBUG_VISIBLE: u64 = 1 << 2;

pub const REGION_FLAG_READ: u32 = 1 << 0;
pub const REGION_FLAG_WRITE: u32 = 1 << 1;
pub const REGION_FLAG_EXEC: u32 = 1 << 2;
pub const REGION_FLAG_COMPRESSED: u32 = 1 << 3;
pub const REGION_FLAG_COPY_ON_WRITE: u32 = 1 << 4;
pub const REGION_FLAG_PRELOAD: u32 = 1 << 5;
pub const REGION_FLAG_DEBUG_VISIBLE: u32 = 1 << 6;

pub const TASK_FLAG_PRIVILEGED: u32 = 1 << 0;
pub const TASK_FLAG_BRIDGED: u32 = 1 << 1;
pub const TASK_FLAG_TRACE_ON_BOOT: u32 = 1 << 2;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BootRegionKind {
    KernelCode = 0,
    KernelData = 1,
    RootFs = 2,
    ElfText = 3,
    ElfData = 4,
    NativeModule = 5,
    Config = 6,
    DebugManifest = 7,
    Symbols = 8,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BootTaskPersona {
    Stage0 = 0,
    Linux = 1,
    Native = 2,
    Service = 3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BootImageHeader {
    pub magic: u32,
    pub version: u16,
    pub header_size: u16,
    pub flags: u64,
    pub image_size: u64,
    pub checksum64: u64,
    pub memory_profile: u32,
    pub region_count: u32,
    pub task_count: u32,
    pub root_task_index: u32,
    pub region_table_offset: u64,
    pub task_table_offset: u64,
    pub symbol_table_offset: u64,
    pub rootfs_offset: u64,
    pub rootfs_size: u64,
    pub boot_args_offset: u64,
    pub boot_args_size: u64,
    pub reserved: [u64; 3],
}

impl BootImageHeader {
    pub fn new(memory_profile: u32) -> Self {
        Self {
            magic: BOOT_IMAGE_MAGIC,
            version: BOOT_IMAGE_VERSION,
            header_size: size_of::<BootImageHeader>() as u16,
            memory_profile,
            ..Self::default()
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BootRegionDesc {
    pub kind: u32,
    pub flags: u32,
    pub guest_base: u64,
    pub mem_size: u64,
    pub file_offset: u64,
    pub file_size: u64,
    pub align: u64,
    pub checksum64: u64,
    pub reserved: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BootTaskDesc {
    pub persona: u32,
    pub flags: u32,
    pub entry_pc: u64,
    pub stack_top: u64,
    pub arg_ptr: u64,
    pub env_ptr: u64,
    pub addr_space_id: u32,
    pub fd_table_id: u32,
    pub cwd_node_id: u32,
    pub capability_mask: u32,
    pub reserved: u64,
}

// ── Serialization helpers ────────────────────────────────────────────────

/// XOR-fold checksum over 8-byte words (fast, deterministic, not cryptographic).
/// Pads the last partial word with zeros if `data.len()` is not a multiple of 8.
pub fn xor_fold_checksum(data: &[u8]) -> u64 {
    let mut checksum: u64 = 0;
    let chunks = data.chunks(8);
    for chunk in chunks {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        checksum ^= u64::from_le_bytes(word);
    }
    checksum
}

/// Serialize a `BootImageHeader` to 128 bytes (little-endian).
pub fn serialize_header(h: &BootImageHeader) -> [u8; 128] {
    let mut buf = [0u8; 128];
    let mut off = 0;

    macro_rules! put_u16 {
        ($v:expr) => {
            buf[off..off + 2].copy_from_slice(&($v).to_le_bytes());
            off += 2;
        };
    }
    macro_rules! put_u32 {
        ($v:expr) => {
            buf[off..off + 4].copy_from_slice(&($v).to_le_bytes());
            off += 4;
        };
    }
    macro_rules! put_u64 {
        ($v:expr) => {
            buf[off..off + 8].copy_from_slice(&($v).to_le_bytes());
            off += 8;
        };
    }

    put_u32!(h.magic);
    put_u16!(h.version);
    put_u16!(h.header_size);
    put_u64!(h.flags);
    put_u64!(h.image_size);
    put_u64!(h.checksum64);
    put_u32!(h.memory_profile);
    put_u32!(h.region_count);
    put_u32!(h.task_count);
    put_u32!(h.root_task_index);
    put_u64!(h.region_table_offset);
    put_u64!(h.task_table_offset);
    put_u64!(h.symbol_table_offset);
    put_u64!(h.rootfs_offset);
    put_u64!(h.rootfs_size);
    put_u64!(h.boot_args_offset);
    put_u64!(h.boot_args_size);
    for &r in &h.reserved {
        put_u64!(r);
    }
    debug_assert_eq!(off, 128);
    buf
}

/// Deserialize a `BootImageHeader` from 128 bytes (little-endian).
pub fn deserialize_header(buf: &[u8; 128]) -> BootImageHeader {
    let mut off = 0;

    macro_rules! get_u16 {
        () => {{
            let v = u16::from_le_bytes(buf[off..off + 2].try_into().unwrap());
            off += 2;
            v
        }};
    }
    macro_rules! get_u32 {
        () => {{
            let v = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
            off += 4;
            v
        }};
    }
    macro_rules! get_u64 {
        () => {{
            let v = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
            off += 8;
            v
        }};
    }

    let magic = get_u32!();
    let version = get_u16!();
    let header_size = get_u16!();
    let flags = get_u64!();
    let image_size = get_u64!();
    let checksum64 = get_u64!();
    let memory_profile = get_u32!();
    let region_count = get_u32!();
    let task_count = get_u32!();
    let root_task_index = get_u32!();
    let region_table_offset = get_u64!();
    let task_table_offset = get_u64!();
    let symbol_table_offset = get_u64!();
    let rootfs_offset = get_u64!();
    let rootfs_size = get_u64!();
    let boot_args_offset = get_u64!();
    let boot_args_size = get_u64!();
    let mut reserved = [0u64; 3];
    for r in reserved.iter_mut() {
        *r = get_u64!();
    }
    debug_assert_eq!(off, 128);

    BootImageHeader {
        magic,
        version,
        header_size,
        flags,
        image_size,
        checksum64,
        memory_profile,
        region_count,
        task_count,
        root_task_index,
        region_table_offset,
        task_table_offset,
        symbol_table_offset,
        rootfs_offset,
        rootfs_size,
        boot_args_offset,
        boot_args_size,
        reserved,
    }
}

/// Serialize a `BootRegionDesc` to 64 bytes (little-endian).
pub fn serialize_region(r: &BootRegionDesc) -> [u8; 64] {
    let mut buf = [0u8; 64];
    let mut off = 0;

    macro_rules! put_u32 {
        ($v:expr) => {
            buf[off..off + 4].copy_from_slice(&($v).to_le_bytes());
            off += 4;
        };
    }
    macro_rules! put_u64 {
        ($v:expr) => {
            buf[off..off + 8].copy_from_slice(&($v).to_le_bytes());
            off += 8;
        };
    }

    put_u32!(r.kind);
    put_u32!(r.flags);
    put_u64!(r.guest_base);
    put_u64!(r.mem_size);
    put_u64!(r.file_offset);
    put_u64!(r.file_size);
    put_u64!(r.align);
    put_u64!(r.checksum64);
    put_u64!(r.reserved);
    debug_assert_eq!(off, 64);
    buf
}

/// Deserialize a `BootRegionDesc` from 64 bytes (little-endian).
pub fn deserialize_region(buf: &[u8; 64]) -> BootRegionDesc {
    let mut off = 0;

    macro_rules! get_u32 {
        () => {{
            let v = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
            off += 4;
            v
        }};
    }
    macro_rules! get_u64 {
        () => {{
            let v = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
            off += 8;
            v
        }};
    }

    let kind = get_u32!();
    let flags = get_u32!();
    let guest_base = get_u64!();
    let mem_size = get_u64!();
    let file_offset = get_u64!();
    let file_size = get_u64!();
    let align = get_u64!();
    let checksum64 = get_u64!();
    let reserved = get_u64!();
    debug_assert_eq!(off, 64);

    BootRegionDesc {
        kind,
        flags,
        guest_base,
        mem_size,
        file_offset,
        file_size,
        align,
        checksum64,
        reserved,
    }
}

/// Serialize a `BootTaskDesc` to 64 bytes (little-endian).
pub fn serialize_task(t: &BootTaskDesc) -> [u8; 64] {
    let mut buf = [0u8; 64];
    let mut off = 0;

    macro_rules! put_u32 {
        ($v:expr) => {
            buf[off..off + 4].copy_from_slice(&($v).to_le_bytes());
            off += 4;
        };
    }
    macro_rules! put_u64 {
        ($v:expr) => {
            buf[off..off + 8].copy_from_slice(&($v).to_le_bytes());
            off += 8;
        };
    }

    put_u32!(t.persona);
    put_u32!(t.flags);
    put_u64!(t.entry_pc);
    put_u64!(t.stack_top);
    put_u64!(t.arg_ptr);
    put_u64!(t.env_ptr);
    put_u32!(t.addr_space_id);
    put_u32!(t.fd_table_id);
    put_u32!(t.cwd_node_id);
    put_u32!(t.capability_mask);
    put_u64!(t.reserved);
    debug_assert_eq!(off, 64);
    buf
}

/// Deserialize a `BootTaskDesc` from 64 bytes (little-endian).
pub fn deserialize_task(buf: &[u8; 64]) -> BootTaskDesc {
    let mut off = 0;

    macro_rules! get_u32 {
        () => {{
            let v = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
            off += 4;
            v
        }};
    }
    macro_rules! get_u64 {
        () => {{
            let v = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
            off += 8;
            v
        }};
    }

    let persona = get_u32!();
    let flags = get_u32!();
    let entry_pc = get_u64!();
    let stack_top = get_u64!();
    let arg_ptr = get_u64!();
    let env_ptr = get_u64!();
    let addr_space_id = get_u32!();
    let fd_table_id = get_u32!();
    let cwd_node_id = get_u32!();
    let capability_mask = get_u32!();
    let reserved = get_u64!();
    debug_assert_eq!(off, 64);

    BootTaskDesc {
        persona,
        flags,
        entry_pc,
        stack_top,
        arg_ptr,
        env_ptr,
        addr_space_id,
        fd_table_id,
        cwd_node_id,
        capability_mask,
        reserved,
    }
}

// ── Rootfs flat-pack format ─────────────────────────────────────────────
//
// Format: [u32 count][(u16 path_len)(path bytes)(u32 content_len)(content bytes)]...

/// A single rootfs entry (path → content) for packing into the boot image.
pub struct RootfsEntry {
    pub path: String,
    pub content: Vec<u8>,
}

/// Pack rootfs entries into the flat wire format.
pub fn pack_rootfs(entries: &[RootfsEntry]) -> Vec<u8> {
    let mut buf = Vec::new();
    // Entry count (u32 LE)
    let _ = buf.write_all(&(entries.len() as u32).to_le_bytes());
    for entry in entries {
        let path_bytes = entry.path.as_bytes();
        let _ = buf.write_all(&(path_bytes.len() as u16).to_le_bytes());
        let _ = buf.write_all(path_bytes);
        let _ = buf.write_all(&(entry.content.len() as u32).to_le_bytes());
        let _ = buf.write_all(&entry.content);
    }
    buf
}

/// Unpack rootfs entries from the flat wire format.
pub fn unpack_rootfs(data: &[u8]) -> io::Result<Vec<RootfsEntry>> {
    if data.len() < 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "rootfs too small",
        ));
    }
    let count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let mut off = 4;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        if off + 2 > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated rootfs path len",
            ));
        }
        let path_len = u16::from_le_bytes(data[off..off + 2].try_into().unwrap()) as usize;
        off += 2;
        if off + path_len > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated rootfs path",
            ));
        }
        let path = String::from_utf8_lossy(&data[off..off + path_len]).to_string();
        off += path_len;
        if off + 4 > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated rootfs content len",
            ));
        }
        let content_len = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        if off + content_len > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated rootfs content",
            ));
        }
        let content = data[off..off + content_len].to_vec();
        off += content_len;
        entries.push(RootfsEntry { path, content });
    }
    Ok(entries)
}

// ── Boot Image Builder ──────────────────────────────────────────────────

/// Errors from the boot image builder.
#[derive(Debug)]
pub enum BootImageError {
    NoTask,
    TooManyRegions,
    Io(io::Error),
}

impl std::fmt::Display for BootImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BootImageError::NoTask => write!(f, "no task descriptor set"),
            BootImageError::TooManyRegions => write!(f, "too many regions (max 255)"),
            BootImageError::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for BootImageError {}

impl From<io::Error> for BootImageError {
    fn from(e: io::Error) -> Self {
        BootImageError::Io(e)
    }
}

/// Builder for constructing a complete boot image blob.
///
/// Usage:
/// ```ignore
/// let image = BootImageBuilder::new(MEMORY_PROFILE_16M)
///     .add_region(region_desc, region_data)
///     .set_task(task_desc)
///     .set_rootfs(rootfs_entries)
///     .set_flags(IMAGE_FLAG_BRIDGED_MODE)
///     .build()?;
/// ```
pub struct BootImageBuilder {
    memory_profile: u32,
    flags: u64,
    regions: Vec<(BootRegionDesc, Vec<u8>)>,
    tasks: Vec<BootTaskDesc>,
    rootfs_data: Option<Vec<u8>>,
}

impl BootImageBuilder {
    pub fn new(memory_profile: u32) -> Self {
        BootImageBuilder {
            memory_profile,
            flags: 0,
            regions: Vec::new(),
            tasks: Vec::new(),
            rootfs_data: None,
        }
    }

    /// Add a region with its backing data. The `file_offset` and `file_size`
    /// fields in `desc` will be filled in by `build()`.
    pub fn add_region(mut self, desc: BootRegionDesc, data: Vec<u8>) -> Self {
        self.regions.push((desc, data));
        self
    }

    /// Convenience: add an ELF text or data region at a given guest address.
    pub fn add_elf_region(
        self,
        guest_base: u64,
        data: Vec<u8>,
        kind: BootRegionKind,
        flags: u32,
    ) -> Self {
        let desc = BootRegionDesc {
            kind: kind as u32,
            flags,
            guest_base,
            mem_size: data.len() as u64,
            align: 0x1000,
            ..BootRegionDesc::default()
        };
        self.add_region(desc, data)
    }

    /// Set a task descriptor (the root task).
    pub fn set_task(mut self, task: BootTaskDesc) -> Self {
        self.tasks.push(task);
        self
    }

    /// Set rootfs from pre-packed entries.
    pub fn set_rootfs(mut self, entries: &[RootfsEntry]) -> Self {
        self.rootfs_data = Some(pack_rootfs(entries));
        self
    }

    /// Set rootfs from raw pre-packed bytes.
    pub fn set_rootfs_raw(mut self, data: Vec<u8>) -> Self {
        self.rootfs_data = Some(data);
        self
    }

    /// Set image flags.
    pub fn set_flags(mut self, flags: u64) -> Self {
        self.flags = flags;
        self
    }

    /// Build the final boot image blob.
    ///
    /// Layout:
    /// ```text
    ///   [0..128)          BootImageHeader (128 bytes)
    ///   [128..128+R*64)   Region table (R × 64 bytes)
    ///   [after regions]   Task table (T × 64 bytes)
    ///   [after tasks]     Region data blobs (concatenated, 8-byte aligned)
    ///   [after data]      Rootfs blob (if any)
    /// ```
    pub fn build(mut self) -> Result<Vec<u8>, BootImageError> {
        if self.tasks.is_empty() {
            return Err(BootImageError::NoTask);
        }
        if self.regions.len() > 255 {
            return Err(BootImageError::TooManyRegions);
        }

        let header_size = size_of::<BootImageHeader>();
        let region_table_size = self.regions.len() * size_of::<BootRegionDesc>();
        let task_table_size = self.tasks.len() * size_of::<BootTaskDesc>();

        let region_table_offset = header_size as u64;
        let task_table_offset = region_table_offset + region_table_size as u64;
        let mut data_offset = task_table_offset + task_table_size as u64;
        // Align data start to 8 bytes
        data_offset = (data_offset + 7) & !7;

        // Assign file_offset/file_size to each region and collect data positions
        let mut data_positions = Vec::with_capacity(self.regions.len());
        let mut current_data_offset = data_offset;
        for (desc, data) in &mut self.regions {
            desc.file_offset = current_data_offset;
            desc.file_size = data.len() as u64;
            desc.checksum64 = xor_fold_checksum(data);
            data_positions.push(current_data_offset);
            current_data_offset += data.len() as u64;
            // Align next blob to 8 bytes
            current_data_offset = (current_data_offset + 7) & !7;
        }

        // Rootfs placement
        let rootfs_offset;
        let rootfs_size;
        if let Some(ref rootfs) = self.rootfs_data {
            rootfs_offset = current_data_offset;
            rootfs_size = rootfs.len() as u64;
            current_data_offset = rootfs_offset + rootfs_size;
            current_data_offset = (current_data_offset + 7) & !7;
        } else {
            rootfs_offset = 0;
            rootfs_size = 0;
        }

        let image_size = current_data_offset;

        // Build the header
        let mut header = BootImageHeader::new(self.memory_profile);
        header.flags = self.flags;
        header.image_size = image_size;
        header.region_count = self.regions.len() as u32;
        header.task_count = self.tasks.len() as u32;
        header.root_task_index = 0;
        header.region_table_offset = region_table_offset;
        header.task_table_offset = task_table_offset;
        header.rootfs_offset = rootfs_offset;
        header.rootfs_size = rootfs_size;

        // Assemble the image
        let mut image = Vec::with_capacity(image_size as usize);

        // Header (checksum placeholder = 0)
        image.extend_from_slice(&serialize_header(&header));

        // Region table
        for (desc, _) in &self.regions {
            image.extend_from_slice(&serialize_region(desc));
        }

        // Task table
        for task in &self.tasks {
            image.extend_from_slice(&serialize_task(task));
        }

        // Pad to data start
        while image.len() < data_offset as usize {
            image.push(0);
        }

        // Region data blobs
        for (_, data) in &self.regions {
            image.extend_from_slice(data);
            // Align to 8 bytes
            while image.len() % 8 != 0 {
                image.push(0);
            }
        }

        // Rootfs blob
        if let Some(ref rootfs) = self.rootfs_data {
            while image.len() < rootfs_offset as usize {
                image.push(0);
            }
            image.extend_from_slice(rootfs);
            while image.len() % 8 != 0 {
                image.push(0);
            }
        }

        // Pad to final size
        image.resize(image_size as usize, 0);

        // Compute checksum over everything except the checksum field itself.
        // The checksum field is at offset 24..32 in the header.
        // Zero it, compute, then write back.
        let checksum = {
            // Zero the checksum field
            image[24..32].copy_from_slice(&0u64.to_le_bytes());
            xor_fold_checksum(&image)
        };
        image[24..32].copy_from_slice(&checksum.to_le_bytes());

        Ok(image)
    }
}

/// Validate a boot image blob: check magic, version, and checksum.
pub fn validate_image(data: &[u8]) -> Result<BootImageHeader, String> {
    if data.len() < size_of::<BootImageHeader>() {
        return Err("image too small for header".to_string());
    }
    let mut header_bytes = [0u8; 128];
    header_bytes.copy_from_slice(&data[..128]);
    let header = deserialize_header(&header_bytes);

    if header.magic != BOOT_IMAGE_MAGIC {
        return Err(format!("bad magic: 0x{:08X}", header.magic));
    }
    if header.version != BOOT_IMAGE_VERSION {
        return Err(format!("unsupported version: {}", header.version));
    }

    // Verify checksum
    let stored_checksum = header.checksum64;
    let mut check_data = data.to_vec();
    check_data[24..32].copy_from_slice(&0u64.to_le_bytes());
    let computed = xor_fold_checksum(&check_data);
    if computed != stored_checksum {
        return Err(format!(
            "checksum mismatch: stored 0x{:016X}, computed 0x{:016X}",
            stored_checksum, computed
        ));
    }

    Ok(header)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boot_image_struct_layouts_are_stable() {
        assert_eq!(size_of::<BootImageHeader>(), 128);
        assert_eq!(size_of::<BootRegionDesc>(), 64);
        assert_eq!(size_of::<BootTaskDesc>(), 64);
    }

    #[test]
    fn boot_header_defaults_match_format() {
        let header = BootImageHeader::new(MEMORY_PROFILE_256M);
        assert_eq!(header.magic, BOOT_IMAGE_MAGIC);
        assert_eq!(header.version, BOOT_IMAGE_VERSION);
        assert_eq!(header.header_size as usize, size_of::<BootImageHeader>());
        assert_eq!(header.memory_profile, MEMORY_PROFILE_256M);
    }

    #[test]
    fn header_round_trip_serialize_deserialize() {
        let mut header = BootImageHeader::new(MEMORY_PROFILE_16M);
        header.flags = IMAGE_FLAG_BRIDGED_MODE | IMAGE_FLAG_TRACE_ON_BOOT;
        header.image_size = 0x1234_5678_9ABC_DEF0;
        header.region_count = 3;
        header.task_count = 1;
        header.rootfs_offset = 0xDEAD;
        header.rootfs_size = 0xBEEF;

        let bytes = serialize_header(&header);
        assert_eq!(bytes.len(), 128);

        let restored = deserialize_header(&bytes);
        assert_eq!(restored, header);
    }

    #[test]
    fn region_round_trip_serialize_deserialize() {
        let region = BootRegionDesc {
            kind: BootRegionKind::ElfData as u32,
            flags: REGION_FLAG_READ | REGION_FLAG_WRITE,
            guest_base: 0x0040_0000,
            mem_size: 0x2000,
            file_offset: 0x180,
            file_size: 0x1234,
            align: 0x1000,
            checksum64: 0xDEAD_BEEF_CAFE_BABE,
            reserved: 7,
        };

        let bytes = serialize_region(&region);
        let restored = deserialize_region(&bytes);
        assert_eq!(restored, region);
    }

    #[test]
    fn task_round_trip_serialize_deserialize() {
        let task = BootTaskDesc {
            persona: BootTaskPersona::Linux as u32,
            flags: TASK_FLAG_PRIVILEGED | TASK_FLAG_TRACE_ON_BOOT,
            entry_pc: 0x401000,
            stack_top: 0xFF000,
            arg_ptr: 0xFE000,
            env_ptr: 0xFD000,
            addr_space_id: 3,
            fd_table_id: 4,
            cwd_node_id: 5,
            capability_mask: 0xA5A5_5A5A,
            reserved: 9,
        };

        let bytes = serialize_task(&task);
        let restored = deserialize_task(&bytes);
        assert_eq!(restored, task);
    }

    #[test]
    fn xor_checksum_basic() {
        // All zeros → checksum is 0
        let zeros = vec![0u8; 64];
        assert_eq!(xor_fold_checksum(&zeros), 0);

        // Known pattern
        let data = 0xDEAD_BEEF_CAFE_BABEu64.to_le_bytes();
        assert_eq!(xor_fold_checksum(&data), 0xDEAD_BEEF_CAFE_BABE);

        // XOR of two words
        let mut two_words = Vec::new();
        two_words.extend_from_slice(&0xAAAA_AAAA_AAAA_AAAAu64.to_le_bytes());
        two_words.extend_from_slice(&0x5555_5555_5555_5555u64.to_le_bytes());
        assert_eq!(xor_fold_checksum(&two_words), 0xFFFF_FFFF_FFFF_FFFF);

        // Partial word (5 bytes) — padded with zeros
        let partial = vec![0xFF; 5];
        let expected = u64::from_le_bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0]);
        assert_eq!(xor_fold_checksum(&partial), expected);
    }

    #[test]
    fn rootfs_round_trip() {
        let entries = vec![
            RootfsEntry {
                path: "/etc/hostname".to_string(),
                content: b"ncpu-gpu\n".to_vec(),
            },
            RootfsEntry {
                path: "/bin/sh".to_string(),
                content: vec![0xEF, 0xBE, 0xAD, 0xDE],
            },
            RootfsEntry {
                path: "/empty".to_string(),
                content: vec![],
            },
        ];
        let packed = pack_rootfs(&entries);
        let unpacked = unpack_rootfs(&packed).unwrap();

        assert_eq!(unpacked.len(), 3);
        assert_eq!(unpacked[0].path, "/etc/hostname");
        assert_eq!(unpacked[0].content, b"ncpu-gpu\n");
        assert_eq!(unpacked[1].path, "/bin/sh");
        assert_eq!(unpacked[1].content, vec![0xEF, 0xBE, 0xAD, 0xDE]);
        assert_eq!(unpacked[2].path, "/empty");
        assert_eq!(unpacked[2].content, b"");
    }

    #[test]
    fn builder_minimal_image() {
        // Build a minimal boot image with one region and one task
        let region_data = vec![0x90; 256]; // NOP sled
        let task = BootTaskDesc {
            persona: BootTaskPersona::Linux as u32,
            entry_pc: 0x10000,
            stack_top: 0xFF000,
            ..BootTaskDesc::default()
        };

        let image = BootImageBuilder::new(MEMORY_PROFILE_16M)
            .add_elf_region(
                0x10000,
                region_data.clone(),
                BootRegionKind::ElfText,
                REGION_FLAG_READ | REGION_FLAG_EXEC,
            )
            .set_task(task)
            .build()
            .unwrap();

        // Validate
        let header = validate_image(&image).unwrap();
        assert_eq!(header.magic, BOOT_IMAGE_MAGIC);
        assert_eq!(header.version, BOOT_IMAGE_VERSION);
        assert_eq!(header.memory_profile, MEMORY_PROFILE_16M);
        assert_eq!(header.region_count, 1);
        assert_eq!(header.task_count, 1);

        // Verify region data is at the recorded offset
        let region_off = header.region_table_offset as usize;
        let mut rbuf = [0u8; 64];
        rbuf.copy_from_slice(&image[region_off..region_off + 64]);
        // kind=3 (ElfText), flags=5 (R|X)
        assert_eq!(
            u32::from_le_bytes(rbuf[0..4].try_into().unwrap()),
            BootRegionKind::ElfText as u32
        );
        assert_eq!(
            u32::from_le_bytes(rbuf[4..8].try_into().unwrap()),
            REGION_FLAG_READ | REGION_FLAG_EXEC
        );

        // Verify the data blob itself
        let file_offset = u64::from_le_bytes(rbuf[24..32].try_into().unwrap()) as usize;
        let file_size = u64::from_le_bytes(rbuf[32..40].try_into().unwrap()) as usize;
        assert_eq!(file_size, 256);
        assert_eq!(&image[file_offset..file_offset + file_size], &region_data);
    }

    #[test]
    fn builder_with_rootfs() {
        let entries = vec![RootfsEntry {
            path: "/etc/hostname".to_string(),
            content: b"ncpu\n".to_vec(),
        }];
        let task = BootTaskDesc {
            persona: BootTaskPersona::Linux as u32,
            entry_pc: 0x10000,
            ..BootTaskDesc::default()
        };

        let image = BootImageBuilder::new(MEMORY_PROFILE_16M)
            .set_task(task)
            .set_rootfs(&entries)
            .build()
            .unwrap();

        let header = validate_image(&image).unwrap();
        assert!(header.rootfs_offset > 0);
        assert!(header.rootfs_size > 0);

        // Verify rootfs can be unpacked from the image
        let rf_off = header.rootfs_offset as usize;
        let rf_size = header.rootfs_size as usize;
        let unpacked = unpack_rootfs(&image[rf_off..rf_off + rf_size]).unwrap();
        assert_eq!(unpacked.len(), 1);
        assert_eq!(unpacked[0].path, "/etc/hostname");
    }

    #[test]
    fn builder_no_task_fails() {
        let result = BootImageBuilder::new(MEMORY_PROFILE_16M).build();
        assert!(result.is_err());
    }

    #[test]
    fn checksum_detects_corruption() {
        let task = BootTaskDesc {
            persona: BootTaskPersona::Linux as u32,
            entry_pc: 0x10000,
            ..BootTaskDesc::default()
        };
        let mut image = BootImageBuilder::new(MEMORY_PROFILE_16M)
            .add_elf_region(
                0x10000,
                vec![0x42; 64],
                BootRegionKind::ElfText,
                REGION_FLAG_READ,
            )
            .set_task(task)
            .build()
            .unwrap();

        // Valid first
        assert!(validate_image(&image).is_ok());

        // Corrupt one byte in the data area
        let last = image.len() - 1;
        image[last] ^= 0xFF;
        assert!(validate_image(&image).is_err());
    }

    #[test]
    fn all_region_offsets_within_bounds() {
        let task = BootTaskDesc {
            persona: BootTaskPersona::Native as u32,
            entry_pc: 0x20000,
            stack_top: 0xF0000,
            ..BootTaskDesc::default()
        };
        let image = BootImageBuilder::new(MEMORY_PROFILE_64M)
            .add_elf_region(
                0x10000,
                vec![0xAA; 1024],
                BootRegionKind::ElfText,
                REGION_FLAG_READ | REGION_FLAG_EXEC,
            )
            .add_elf_region(
                0x50000,
                vec![0xBB; 512],
                BootRegionKind::ElfData,
                REGION_FLAG_READ | REGION_FLAG_WRITE,
            )
            .set_task(task)
            .set_rootfs(&[RootfsEntry {
                path: "/hello".to_string(),
                content: b"world".to_vec(),
            }])
            .build()
            .unwrap();

        let header = validate_image(&image).unwrap();
        let image_len = image.len() as u64;

        // Region table within bounds
        let rt_end = header.region_table_offset + (header.region_count as u64) * 64;
        assert!(rt_end <= image_len);

        // Task table within bounds
        let tt_end = header.task_table_offset + (header.task_count as u64) * 64;
        assert!(tt_end <= image_len);

        // Each region's data within bounds
        for i in 0..header.region_count as usize {
            let off = header.region_table_offset as usize + i * 64;
            let file_offset = u64::from_le_bytes(image[off + 24..off + 32].try_into().unwrap());
            let file_size = u64::from_le_bytes(image[off + 32..off + 40].try_into().unwrap());
            assert!(
                file_offset + file_size <= image_len,
                "region {} data overflows image",
                i
            );
        }

        // Rootfs within bounds
        if header.rootfs_size > 0 {
            assert!(header.rootfs_offset + header.rootfs_size <= image_len);
        }
    }
}
