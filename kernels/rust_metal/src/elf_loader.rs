//! ELF64 loader for aarch64 Linux binaries — Rust port of `ncpu/os/gpu/elf_loader.py`.
//!
//! Handles:
//! - ELF64 parsing (aarch64 / EM_AARCH64 = 183)
//! - PT_LOAD segment extraction
//! - Linux process stack setup (argc, argv, envp, auxv)
//! - Boot image integration via `prepare_memory_image()`

use std::fmt;

// ── ELF constants ────────────────────────────────────────────────────────

const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;
const ET_EXEC: u16 = 2;
const ET_DYN: u16 = 3;
const EM_AARCH64: u16 = 183;

const PT_LOAD: u32 = 1;
const PT_PHDR: u32 = 6;

pub const PF_X: u32 = 1;
pub const PF_W: u32 = 2;
pub const PF_R: u32 = 4;

// Auxiliary vector types (subset needed by musl)
const AT_NULL: u64 = 0;
const AT_PHDR: u64 = 3;
const AT_PHENT: u64 = 4;
const AT_PHNUM: u64 = 5;
const AT_PAGESZ: u64 = 6;
const AT_ENTRY: u64 = 9;
const AT_UID: u64 = 11;
const AT_EUID: u64 = 12;
const AT_GID: u64 = 13;
const AT_EGID: u64 = 14;
const AT_HWCAP: u64 = 16;
const AT_CLKTCK: u64 = 17;
const AT_SECURE: u64 = 23;
const AT_RANDOM: u64 = 25;

// ── Types ────────────────────────────────────────────────────────────────

/// Error from ELF loading.
#[derive(Debug)]
pub enum ElfError {
    TooSmall,
    BadMagic,
    NotElf64,
    NotLittleEndian,
    NotAarch64(u16),
    NotExecutable(u16),
    NoLoadSegments,
    SegmentOutOfBounds {
        index: usize,
        offset: u64,
        size: u64,
    },
}

impl fmt::Display for ElfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ElfError::TooSmall => write!(f, "file too small to be ELF"),
            ElfError::BadMagic => write!(f, "bad ELF magic"),
            ElfError::NotElf64 => write!(f, "not ELF64"),
            ElfError::NotLittleEndian => write!(f, "not little-endian"),
            ElfError::NotAarch64(m) => write!(f, "not aarch64 (machine={})", m),
            ElfError::NotExecutable(t) => write!(f, "not executable (type={})", t),
            ElfError::NoLoadSegments => write!(f, "no PT_LOAD segments found"),
            ElfError::SegmentOutOfBounds {
                index,
                offset,
                size,
            } => write!(
                f,
                "segment {} file data out of bounds (offset={}, size={})",
                index, offset, size
            ),
        }
    }
}

impl std::error::Error for ElfError {}

/// A loadable ELF segment.
#[derive(Debug, Clone)]
pub struct ElfSegment {
    pub vaddr: u64,
    pub memsz: u64,
    pub filesz: u64,
    pub offset: u64,
    pub flags: u32,
    pub align: u64,
}

/// Parsed ELF binary information.
#[derive(Debug, Clone)]
pub struct ElfInfo {
    pub entry: u64,
    pub segments: Vec<ElfSegment>,
    pub phdr_vaddr: u64,
    pub phdr_size: u16,
    pub phdr_count: u16,
    pub elf_type: u16,
    /// Offset of program headers in the file (e_phoff).
    pub phdr_file_offset: u64,
}

// ── Little-endian helpers ────────────────────────────────────────────────

fn le_u16(data: &[u8], off: usize) -> u16 {
    u16::from_le_bytes(data[off..off + 2].try_into().unwrap())
}

fn le_u32(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(data[off..off + 4].try_into().unwrap())
}

fn le_u64(data: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(data[off..off + 8].try_into().unwrap())
}

// ── Core parser ──────────────────────────────────────────────────────────

/// Parse an ELF64 binary for aarch64.
pub fn parse_elf(data: &[u8]) -> Result<ElfInfo, ElfError> {
    if data.len() < 64 {
        return Err(ElfError::TooSmall);
    }
    if data[0..4] != ELF_MAGIC {
        return Err(ElfError::BadMagic);
    }
    if data[4] != ELFCLASS64 {
        return Err(ElfError::NotElf64);
    }
    if data[5] != ELFDATA2LSB {
        return Err(ElfError::NotLittleEndian);
    }

    // ELF header fields at fixed offsets (ELF64)
    let e_type = le_u16(data, 16);
    let e_machine = le_u16(data, 18);
    let e_entry = le_u64(data, 24);
    let e_phoff = le_u64(data, 32);
    let e_phentsize = le_u16(data, 54);
    let e_phnum = le_u16(data, 56);

    if e_machine != EM_AARCH64 {
        return Err(ElfError::NotAarch64(e_machine));
    }
    if e_type != ET_EXEC && e_type != ET_DYN {
        return Err(ElfError::NotExecutable(e_type));
    }

    let mut info = ElfInfo {
        entry: e_entry,
        segments: Vec::new(),
        phdr_vaddr: 0,
        phdr_size: e_phentsize,
        phdr_count: e_phnum,
        elf_type: e_type,
        phdr_file_offset: e_phoff,
    };

    // Parse program headers
    for i in 0..e_phnum as usize {
        let off = e_phoff as usize + i * e_phentsize as usize;
        if off + e_phentsize as usize > data.len() {
            break;
        }

        let p_type = le_u32(data, off);
        let p_flags = le_u32(data, off + 4);
        let p_offset = le_u64(data, off + 8);
        let p_vaddr = le_u64(data, off + 16);
        // p_paddr at off+24, skip
        let p_filesz = le_u64(data, off + 32);
        let p_memsz = le_u64(data, off + 40);
        let p_align = le_u64(data, off + 48);

        if p_type == PT_LOAD {
            info.segments.push(ElfSegment {
                vaddr: p_vaddr,
                memsz: p_memsz,
                filesz: p_filesz,
                offset: p_offset,
                flags: p_flags,
                align: p_align,
            });
        } else if p_type == PT_PHDR {
            info.phdr_vaddr = p_vaddr;
        }
    }

    if info.segments.is_empty() {
        return Err(ElfError::NoLoadSegments);
    }

    Ok(info)
}

// ── Linux stack builder ──────────────────────────────────────────────────

/// Pre-computed memory image for loading an ELF into GPU memory.
///
/// Contains:
/// - Segment data ready to copy at their `vaddr` offsets
/// - Stack frame with argc/argv/envp/auxv at `stack_pointer`
/// - Entry point address
pub struct PreparedElf {
    /// Entry point address.
    pub entry_pc: u64,
    /// Stack pointer (SP) value to set before execution.
    pub stack_pointer: u64,
    /// Heap base address (first 64KB boundary after all segments).
    pub heap_base: u64,
    /// Memory writes: (guest_addr, data). Apply in order.
    pub writes: Vec<(u64, Vec<u8>)>,
    /// Parsed ELF metadata.
    pub elf_info: ElfInfo,
}

/// Prepare a complete memory image from an ELF binary.
///
/// This is the Rust equivalent of `load_elf_into_memory()` from elf_loader.py.
/// It computes all memory writes needed but does not touch GPU memory directly.
///
/// The caller applies the writes to GPU memory and sets PC and SP.
pub fn prepare_elf(
    elf_data: &[u8],
    argv: &[&str],
    envp: &[(&str, &str)],
    memory_limit: u64,
) -> Result<PreparedElf, ElfError> {
    let info = parse_elf(elf_data)?;
    let mut writes: Vec<(u64, Vec<u8>)> = Vec::new();

    // Compute memory layout
    let max_seg_end = info
        .segments
        .iter()
        .map(|s| s.vaddr + s.memsz)
        .max()
        .unwrap_or(0);
    let heap_base = (max_seg_end + 0xFFFF) & !0xFFFF;
    let stack_top = (memory_limit - 0x1000) & !0xF;

    // Zero the stack region (256KB below stack_top)
    let stack_clear = std::cmp::min(0x40000, stack_top);
    if stack_clear > 0 {
        writes.push((stack_top - stack_clear, vec![0u8; stack_clear as usize]));
    }

    // Zero the full image span (gaps between segments)
    if !info.segments.is_empty() {
        let image_lo = info.segments.iter().map(|s| s.vaddr).min().unwrap();
        let image_hi = info
            .segments
            .iter()
            .map(|s| std::cmp::min(s.vaddr + s.memsz, memory_limit))
            .max()
            .unwrap();
        if image_hi > image_lo {
            writes.push((image_lo, vec![0u8; (image_hi - image_lo) as usize]));
        }
    }

    // Load PT_LOAD segments
    for (idx, seg) in info.segments.iter().enumerate() {
        if seg.vaddr >= memory_limit {
            continue;
        }
        let avail = memory_limit - seg.vaddr;
        let filesz = std::cmp::min(seg.filesz, avail);
        let end = seg.offset as usize + filesz as usize;
        if end > elf_data.len() {
            return Err(ElfError::SegmentOutOfBounds {
                index: idx,
                offset: seg.offset,
                size: filesz,
            });
        }
        let seg_data = &elf_data[seg.offset as usize..end];
        writes.push((seg.vaddr, seg_data.to_vec()));
    }

    // Build Linux aarch64 process stack
    // String area: 4KB below stack_top
    let string_area = stack_top - 0x1000;
    let mut str_ptr = string_area;

    // Write argv strings
    let mut argv_addrs = Vec::with_capacity(argv.len());
    for arg in argv {
        let arg_bytes: Vec<u8> = arg
            .as_bytes()
            .iter()
            .copied()
            .chain(std::iter::once(0))
            .collect();
        writes.push((str_ptr, arg_bytes.clone()));
        argv_addrs.push(str_ptr);
        str_ptr += arg_bytes.len() as u64;
    }

    // Write envp strings
    let envp_strs: Vec<String> = envp.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
    let mut envp_addrs = Vec::with_capacity(envp_strs.len());
    for env_str in &envp_strs {
        let env_bytes: Vec<u8> = env_str
            .as_bytes()
            .iter()
            .copied()
            .chain(std::iter::once(0))
            .collect();
        writes.push((str_ptr, env_bytes.clone()));
        envp_addrs.push(str_ptr);
        str_ptr += env_bytes.len() as u64;
    }

    // 16 random bytes for AT_RANDOM (deterministic zeros for reproducibility in boot image)
    let random_addr = str_ptr;
    writes.push((random_addr, vec![0u8; 16]));
    str_ptr += 16;

    // Build auxiliary vector
    let phdr_addr = if info.phdr_vaddr != 0 {
        info.phdr_vaddr
    } else if !info.segments.is_empty() {
        info.segments[0].vaddr + info.phdr_file_offset
    } else {
        0
    };

    let auxv: Vec<(u64, u64)> = vec![
        (AT_PAGESZ, 4096),
        (AT_ENTRY, info.entry),
        (AT_UID, 0),
        (AT_EUID, 0),
        (AT_GID, 0),
        (AT_EGID, 0),
        (AT_HWCAP, 0),
        (AT_CLKTCK, 100),
        (AT_SECURE, 0),
        (AT_RANDOM, random_addr),
        (AT_PHDR, phdr_addr),
        (AT_PHENT, info.phdr_size as u64),
        (AT_PHNUM, info.phdr_count as u64),
        (AT_NULL, 0),
    ];

    // Calculate stack frame size
    let mut frame_size: u64 = 8; // argc
    frame_size += (argv.len() as u64) * 8 + 8; // argv pointers + NULL
    frame_size += (envp_strs.len() as u64) * 8 + 8; // envp pointers + NULL
    frame_size += (auxv.len() as u64) * 16; // auxv entries
    frame_size = (frame_size + 15) & !15; // 16-byte align

    let sp = (string_area - frame_size) & !0xF;

    // Write stack frame
    let mut frame = Vec::with_capacity(frame_size as usize);

    // argc
    frame.extend_from_slice(&(argv.len() as u64).to_le_bytes());

    // argv pointers
    for addr in &argv_addrs {
        frame.extend_from_slice(&addr.to_le_bytes());
    }
    frame.extend_from_slice(&0u64.to_le_bytes()); // NULL

    // envp pointers
    for addr in &envp_addrs {
        frame.extend_from_slice(&addr.to_le_bytes());
    }
    frame.extend_from_slice(&0u64.to_le_bytes()); // NULL

    // Auxiliary vector
    for (at_type, at_val) in &auxv {
        frame.extend_from_slice(&at_type.to_le_bytes());
        frame.extend_from_slice(&at_val.to_le_bytes());
    }

    writes.push((sp, frame));

    Ok(PreparedElf {
        entry_pc: info.entry,
        stack_pointer: sp,
        heap_base,
        writes,
        elf_info: info,
    })
}

/// Default environment variables matching the Python loader.
pub const DEFAULT_ENVP: [(&str, &str); 6] = [
    ("PATH", "/bin:/usr/bin:/sbin:/usr/sbin"),
    ("HOME", "/root"),
    ("TERM", "dumb"),
    ("USER", "root"),
    ("TZ", "UTC"),
    ("LOGNAME", "root"),
];

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid aarch64 ELF64 binary (static, single PT_LOAD).
    fn make_minimal_elf(entry: u64, vaddr: u64, code: &[u8]) -> Vec<u8> {
        let mut elf = Vec::new();

        // ELF header (64 bytes)
        elf.extend_from_slice(&ELF_MAGIC); // e_ident[0..4]: magic
        elf.push(ELFCLASS64); // e_ident[4]: class
        elf.push(ELFDATA2LSB); // e_ident[5]: data
        elf.push(1); // e_ident[6]: version
        elf.push(0); // e_ident[7]: OS/ABI
        elf.extend_from_slice(&[0u8; 8]); // e_ident[8..16]: padding

        elf.extend_from_slice(&ET_EXEC.to_le_bytes()); // e_type
        elf.extend_from_slice(&EM_AARCH64.to_le_bytes()); // e_machine
        elf.extend_from_slice(&1u32.to_le_bytes()); // e_version
        elf.extend_from_slice(&entry.to_le_bytes()); // e_entry
        elf.extend_from_slice(&64u64.to_le_bytes()); // e_phoff (right after header)
        elf.extend_from_slice(&0u64.to_le_bytes()); // e_shoff
        elf.extend_from_slice(&0u32.to_le_bytes()); // e_flags
        elf.extend_from_slice(&64u16.to_le_bytes()); // e_ehsize
        elf.extend_from_slice(&56u16.to_le_bytes()); // e_phentsize
        elf.extend_from_slice(&1u16.to_le_bytes()); // e_phnum
        elf.extend_from_slice(&0u16.to_le_bytes()); // e_shentsize
        elf.extend_from_slice(&0u16.to_le_bytes()); // e_shnum
        elf.extend_from_slice(&0u16.to_le_bytes()); // e_shstrndx
        assert_eq!(elf.len(), 64);

        // Program header (56 bytes) — single PT_LOAD
        let data_offset = 64 + 56; // after header + 1 phdr
        elf.extend_from_slice(&PT_LOAD.to_le_bytes()); // p_type
        elf.extend_from_slice(&(PF_R | PF_X).to_le_bytes()); // p_flags
        elf.extend_from_slice(&(data_offset as u64).to_le_bytes()); // p_offset
        elf.extend_from_slice(&vaddr.to_le_bytes()); // p_vaddr
        elf.extend_from_slice(&vaddr.to_le_bytes()); // p_paddr
        elf.extend_from_slice(&(code.len() as u64).to_le_bytes()); // p_filesz
        elf.extend_from_slice(&(code.len() as u64).to_le_bytes()); // p_memsz
        elf.extend_from_slice(&0x1000u64.to_le_bytes()); // p_align
        assert_eq!(elf.len(), 120);

        // Code/data
        elf.extend_from_slice(code);
        elf
    }

    #[test]
    fn parse_minimal_elf() {
        let code = vec![0xD4, 0x00, 0x00, 0x01]; // SVC #0
        let elf = make_minimal_elf(0x10000, 0x10000, &code);
        let info = parse_elf(&elf).unwrap();

        assert_eq!(info.entry, 0x10000);
        assert_eq!(info.segments.len(), 1);
        assert_eq!(info.segments[0].vaddr, 0x10000);
        assert_eq!(info.segments[0].filesz, 4);
        assert_eq!(info.segments[0].memsz, 4);
        assert_eq!(info.segments[0].flags, PF_R | PF_X);
        assert_eq!(info.phdr_count, 1);
        assert_eq!(info.phdr_size, 56);
        assert_eq!(info.elf_type, ET_EXEC);
    }

    #[test]
    fn parse_too_small() {
        assert!(parse_elf(&[0u8; 32]).is_err());
    }

    #[test]
    fn parse_bad_magic() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"\x7fFOO");
        assert!(matches!(parse_elf(&data), Err(ElfError::BadMagic)));
    }

    #[test]
    fn parse_not_aarch64() {
        let mut elf = make_minimal_elf(0x10000, 0x10000, &[0; 4]);
        // Change e_machine to x86_64 (62)
        elf[18..20].copy_from_slice(&62u16.to_le_bytes());
        assert!(matches!(parse_elf(&elf), Err(ElfError::NotAarch64(62))));
    }

    #[test]
    fn prepare_elf_basic() {
        let code = [0xD4u8, 0x00, 0x00, 0x01].repeat(4); // 4 instructions
        let elf = make_minimal_elf(0x10000, 0x10000, &code);

        let prepared = prepare_elf(
            &elf,
            &["busybox", "echo", "hello"],
            &DEFAULT_ENVP,
            16 * 1024 * 1024, // 16MB
        )
        .unwrap();

        assert_eq!(prepared.entry_pc, 0x10000);
        assert!(prepared.stack_pointer > 0);
        assert!(
            prepared.stack_pointer % 16 == 0,
            "SP must be 16-byte aligned"
        );
        assert!(prepared.heap_base > 0x10000);

        // Should have writes for: stack zero, image zero, segment data, strings, frame
        assert!(prepared.writes.len() >= 3);

        // Verify the segment data is in the writes
        let seg_write = prepared.writes.iter().find(|(addr, _)| *addr == 0x10000);
        assert!(seg_write.is_some());
    }

    #[test]
    fn prepare_elf_stack_layout() {
        let code = vec![0xD5, 0x03, 0x20, 0x1F]; // NOP
        let elf = make_minimal_elf(0x10000, 0x10000, &code);

        let prepared =
            prepare_elf(&elf, &["test_prog"], &[("HOME", "/root")], 4 * 1024 * 1024).unwrap();

        // Find the stack frame write (it's at SP)
        let frame_write = prepared
            .writes
            .iter()
            .find(|(addr, _)| *addr == prepared.stack_pointer);
        assert!(frame_write.is_some(), "should have a write at SP");

        let (_, frame_data) = frame_write.unwrap();

        // First 8 bytes = argc = 1
        let argc = u64::from_le_bytes(frame_data[0..8].try_into().unwrap());
        assert_eq!(argc, 1, "argc should be 1");

        // argv[0] pointer (non-null)
        let argv0_ptr = u64::from_le_bytes(frame_data[8..16].try_into().unwrap());
        assert!(argv0_ptr > 0, "argv[0] should be non-null");

        // argv[1] = NULL
        let argv_null = u64::from_le_bytes(frame_data[16..24].try_into().unwrap());
        assert_eq!(argv_null, 0, "argv terminator should be NULL");

        // envp[0] pointer (non-null)
        let envp0_ptr = u64::from_le_bytes(frame_data[24..32].try_into().unwrap());
        assert!(envp0_ptr > 0, "envp[0] should be non-null");

        // envp[1] = NULL
        let envp_null = u64::from_le_bytes(frame_data[32..40].try_into().unwrap());
        assert_eq!(envp_null, 0, "envp terminator should be NULL");

        // First auxv entry (AT_PAGESZ=6, value=4096)
        let at_type = u64::from_le_bytes(frame_data[40..48].try_into().unwrap());
        let at_val = u64::from_le_bytes(frame_data[48..56].try_into().unwrap());
        assert_eq!(at_type, AT_PAGESZ);
        assert_eq!(at_val, 4096);
    }

    #[test]
    fn parse_busybox_if_available() {
        // This test runs only when the BusyBox binary is present
        let path = std::path::Path::new("../../demos/busybox.elf");
        if !path.exists() {
            // Also try from the workspace root
            let alt = std::path::Path::new("demos/busybox.elf");
            if !alt.exists() {
                eprintln!("Skipping BusyBox test (binary not found)");
                return;
            }
            let data = std::fs::read(alt).unwrap();
            let info = parse_elf(&data).unwrap();
            check_busybox_info(&info);
            return;
        }
        let data = std::fs::read(path).unwrap();
        let info = parse_elf(&data).unwrap();
        check_busybox_info(&info);
    }

    fn check_busybox_info(info: &ElfInfo) {
        // BusyBox should be aarch64 static executable
        assert!(info.entry > 0, "entry point should be non-zero");
        assert!(!info.segments.is_empty(), "should have PT_LOAD segments");
        assert_eq!(info.elf_type, ET_EXEC, "BusyBox should be ET_EXEC");

        // Should have at least 2 segments (text + data)
        assert!(
            info.segments.len() >= 2,
            "BusyBox should have >=2 segments, got {}",
            info.segments.len()
        );

        // First segment should be executable
        assert!(
            info.segments[0].flags & PF_X != 0,
            "first segment should be executable"
        );

        // Verify phdr metadata
        assert!(info.phdr_size > 0);
        assert!(info.phdr_count > 0);
    }
}
