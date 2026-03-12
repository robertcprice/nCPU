//! Basic-block cache -- host-side analysis of ARM64 code to optimize GPU dispatch.
//!
//! A basic block is a straight-line sequence of instructions with a single entry point
//! and a single exit point (branch, SVC, or HLT). By caching block extents, the launcher
//! can dispatch exactly the right number of cycles for a block, avoiding over-dispatching.

use crate::micro_op::{MicroOp, OpKind};
use std::collections::HashMap;

/// A cached basic block.
#[derive(Clone, Debug)]
pub struct BasicBlock {
    /// Start address (PC) of the block
    pub start_pc: u64,
    /// Number of instructions in the block
    pub length: u32,
    /// How this block terminates
    pub terminator: BlockTerminator,
    /// Pre-decoded micro-ops (future use -- currently empty)
    pub ops: Vec<MicroOp>,
    /// Memory generation when this block was scanned
    pub mem_generation: u64,
}

/// How a basic block terminates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockTerminator {
    /// Unconditional branch (B, BL)
    Branch,
    /// Conditional branch (B.cond, CBZ, CBNZ, TBZ, TBNZ)
    ConditionalBranch,
    /// Register branch (BR, BLR, RET)
    IndirectBranch,
    /// System call (SVC)
    Syscall,
    /// Halt (HLT)
    Halt,
    /// Reached end of scanned memory
    EndOfMemory,
    /// Reached max block size limit
    MaxSize,
}

/// Host-side basic-block cache.
pub struct BBCache {
    /// Cached blocks keyed by (start_pc, mem_generation)
    blocks: HashMap<(u64, u64), BasicBlock>,
    /// Current memory generation counter
    mem_generation: u64,
    /// Maximum instructions per block
    max_block_size: u32,
    /// Cache statistics
    pub hits: u64,
    pub misses: u64,
}

const DEFAULT_MAX_BLOCK_SIZE: u32 = 4096;

impl BBCache {
    pub fn new() -> Self {
        BBCache {
            blocks: HashMap::new(),
            mem_generation: 0,
            max_block_size: DEFAULT_MAX_BLOCK_SIZE,
            hits: 0,
            misses: 0,
        }
    }

    /// Invalidate all cached blocks (e.g., after self-modifying code).
    pub fn invalidate_all(&mut self) {
        self.mem_generation += 1;
    }

    /// Invalidate blocks that overlap with a memory write to an executable region.
    pub fn invalidate_range(&mut self, addr: u64, len: u64) {
        let end = addr + len;
        self.blocks.retain(|&(start_pc, _), bb| {
            let bb_end = start_pc + (bb.length as u64) * 4;
            // Keep if no overlap
            bb_end <= addr || start_pc >= end
        });
    }

    /// Look up a basic block at the given PC. Returns None on cache miss.
    pub fn lookup(&mut self, pc: u64) -> Option<&BasicBlock> {
        let key = (pc, self.mem_generation);
        if self.blocks.contains_key(&key) {
            self.hits += 1;
            self.blocks.get(&key)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Scan memory starting at `pc` to find the next basic block boundary.
    /// `memory_reader` reads 4 bytes at an address and returns the instruction word.
    pub fn scan_block(
        &mut self,
        pc: u64,
        memory_reader: &dyn Fn(u64) -> Option<u32>,
    ) -> &BasicBlock {
        let key = (pc, self.mem_generation);

        // Already cached?
        if self.blocks.contains_key(&key) {
            self.hits += 1;
            return &self.blocks[&key];
        }

        self.misses += 1;

        let mut length = 0u32;
        let terminator;
        let mut current_pc = pc;

        loop {
            if length >= self.max_block_size {
                terminator = BlockTerminator::MaxSize;
                break;
            }

            let inst = match memory_reader(current_pc) {
                Some(i) => i,
                None => {
                    terminator = BlockTerminator::EndOfMemory;
                    break;
                }
            };

            length += 1;
            let kind = classify_instruction(inst);

            match kind {
                OpKind::Branch | OpKind::BranchLink => {
                    terminator = BlockTerminator::Branch;
                    break;
                }
                OpKind::BranchCond | OpKind::Cbz | OpKind::Cbnz | OpKind::Tbz | OpKind::Tbnz => {
                    terminator = BlockTerminator::ConditionalBranch;
                    break;
                }
                OpKind::BranchReg | OpKind::Ret => {
                    terminator = BlockTerminator::IndirectBranch;
                    break;
                }
                OpKind::Svc => {
                    terminator = BlockTerminator::Syscall;
                    break;
                }
                OpKind::Hlt => {
                    terminator = BlockTerminator::Halt;
                    break;
                }
                _ => {
                    current_pc += 4;
                }
            }
        }

        let bb = BasicBlock {
            start_pc: pc,
            length,
            terminator,
            ops: Vec::new(), // micro-ops populated later
            mem_generation: self.mem_generation,
        };

        self.blocks.insert(key, bb);
        &self.blocks[&key]
    }

    /// Get the recommended dispatch batch size for the block at `pc`.
    /// If the block ends at an SVC or branch, dispatch exactly that many cycles.
    pub fn recommended_batch_size(
        &mut self,
        pc: u64,
        memory_reader: &dyn Fn(u64) -> Option<u32>,
    ) -> u32 {
        let bb = self.scan_block(pc, memory_reader);
        bb.length
    }

    /// Number of cached blocks.
    pub fn cached_count(&self) -> usize {
        self.blocks.len()
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.hits = 0;
        self.misses = 0;
    }
}

/// Classify an ARM64 instruction word into its operation kind.
/// This is a lightweight decoder -- only needs to identify control flow.
fn classify_instruction(inst: u32) -> OpKind {
    let top8 = (inst >> 24) & 0xFF;

    match top8 {
        // Unconditional branch immediate: B
        0x14..=0x17 => OpKind::Branch,
        // Unconditional branch immediate: BL
        0x94..=0x97 => OpKind::BranchLink,

        // Conditional branch: B.cond
        0x54 => OpKind::BranchCond,

        // Compare and branch: CBZ (32-bit and 64-bit)
        0x34 | 0xB4 => OpKind::Cbz,
        // Compare and branch: CBNZ (32-bit and 64-bit)
        0x35 | 0xB5 => OpKind::Cbnz,

        // Test and branch: TBZ
        0x36 | 0xB6 => OpKind::Tbz,
        // Test and branch: TBNZ
        0x37 | 0xB7 => OpKind::Tbnz,

        // Exception generation and system
        0xD4 => {
            let opc = (inst >> 21) & 0x7;
            match opc {
                0 => OpKind::Svc, // SVC
                1 => OpKind::Hlt, // HLT
                _ => OpKind::Unknown,
            }
        }

        // Unconditional branch register: BR, BLR, RET
        0xD6 => {
            let opc = (inst >> 21) & 0xF;
            match opc {
                0 => OpKind::BranchReg,  // BR
                1 => OpKind::BranchLink, // BLR
                2 => OpKind::Ret,        // RET
                _ => OpKind::Unknown,
            }
        }

        _ => OpKind::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Helper: build a memory reader from a slice of instruction words ---
    fn make_reader(instructions: &[(u64, u32)]) -> impl Fn(u64) -> Option<u32> + '_ {
        move |addr| {
            instructions
                .iter()
                .find(|&&(a, _)| a == addr)
                .map(|&(_, inst)| inst)
        }
    }

    // -----------------------------------------------------------------------
    // classify_instruction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_classify_b_unconditional() {
        // B #offset -> top byte 0x14
        let inst: u32 = 0x14000010; // B #64
        assert_eq!(classify_instruction(inst), OpKind::Branch);
    }

    #[test]
    fn test_classify_bl() {
        // BL #offset -> top byte 0x94
        let inst: u32 = 0x94000010; // BL #64
        assert_eq!(classify_instruction(inst), OpKind::BranchLink);
    }

    #[test]
    fn test_classify_b_cond() {
        // B.EQ #offset -> top byte 0x54
        let inst: u32 = 0x54000040; // B.EQ
        assert_eq!(classify_instruction(inst), OpKind::BranchCond);
    }

    #[test]
    fn test_classify_cbz_32bit() {
        // CBZ W0, #offset -> top byte 0x34
        let inst: u32 = 0x34000060; // CBZ W0, #12
        assert_eq!(classify_instruction(inst), OpKind::Cbz);
    }

    #[test]
    fn test_classify_cbz_64bit() {
        // CBZ X0, #offset -> top byte 0xB4
        let inst: u32 = 0xB4000060; // CBZ X0, #12
        assert_eq!(classify_instruction(inst), OpKind::Cbz);
    }

    #[test]
    fn test_classify_cbnz_32bit() {
        // CBNZ W0, #offset -> top byte 0x35
        let inst: u32 = 0x35000060;
        assert_eq!(classify_instruction(inst), OpKind::Cbnz);
    }

    #[test]
    fn test_classify_cbnz_64bit() {
        // CBNZ X0, #offset -> top byte 0xB5
        let inst: u32 = 0xB5000060;
        assert_eq!(classify_instruction(inst), OpKind::Cbnz);
    }

    #[test]
    fn test_classify_tbz() {
        // TBZ X0, #bit, #offset -> top byte 0x36 or 0xB6
        let inst: u32 = 0x36000060;
        assert_eq!(classify_instruction(inst), OpKind::Tbz);
        let inst64: u32 = 0xB6000060;
        assert_eq!(classify_instruction(inst64), OpKind::Tbz);
    }

    #[test]
    fn test_classify_tbnz() {
        // TBNZ X0, #bit, #offset -> top byte 0x37 or 0xB7
        let inst: u32 = 0x37000060;
        assert_eq!(classify_instruction(inst), OpKind::Tbnz);
        let inst64: u32 = 0xB7000060;
        assert_eq!(classify_instruction(inst64), OpKind::Tbnz);
    }

    #[test]
    fn test_classify_svc() {
        // SVC #0 -> 0xD4000001
        let inst: u32 = 0xD4000001;
        assert_eq!(classify_instruction(inst), OpKind::Svc);
    }

    #[test]
    fn test_classify_hlt() {
        // HLT #0 -> 0xD4400000
        let inst: u32 = 0xD4400000;
        assert_eq!(classify_instruction(inst), OpKind::Hlt);
    }

    #[test]
    fn test_classify_br() {
        // BR X30 -> 0xD61F0000 + (30 << 5)
        let inst: u32 = 0xD61F0000 | (30 << 5);
        assert_eq!(classify_instruction(inst), OpKind::BranchReg);
    }

    #[test]
    fn test_classify_blr() {
        // BLR X0 -> 0xD63F0000
        let inst: u32 = 0xD63F0000;
        assert_eq!(classify_instruction(inst), OpKind::BranchLink);
    }

    #[test]
    fn test_classify_ret() {
        // RET -> 0xD65F03C0
        let inst: u32 = 0xD65F03C0;
        assert_eq!(classify_instruction(inst), OpKind::Ret);
    }

    #[test]
    fn test_classify_add_immediate_is_unknown() {
        // ADD X0, X1, #42 -> 0x910002A0 (not control flow)
        let inst: u32 = 0x91000AA0;
        assert_eq!(classify_instruction(inst), OpKind::Unknown);
    }

    #[test]
    fn test_classify_nop_is_unknown() {
        // NOP -> 0xD503201F (top byte 0xD5, not 0xD4 or 0xD6)
        let inst: u32 = 0xD503201F;
        assert_eq!(classify_instruction(inst), OpKind::Unknown);
    }

    // -----------------------------------------------------------------------
    // BBCache: empty memory
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_memory_yields_end_of_memory() {
        let mut cache = BBCache::new();
        let reader = |_addr: u64| -> Option<u32> { None };
        let bb = cache.scan_block(0x10000, &reader);
        assert_eq!(bb.start_pc, 0x10000);
        assert_eq!(bb.length, 0);
        assert_eq!(bb.terminator, BlockTerminator::EndOfMemory);
    }

    // -----------------------------------------------------------------------
    // BBCache: single HLT instruction
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_hlt_instruction() {
        let mut cache = BBCache::new();
        let instructions = [(0x10000u64, 0xD4400000u32)]; // HLT #0
        let reader = make_reader(&instructions);
        let bb = cache.scan_block(0x10000, &reader);
        assert_eq!(bb.start_pc, 0x10000);
        assert_eq!(bb.length, 1);
        assert_eq!(bb.terminator, BlockTerminator::Halt);
    }

    // -----------------------------------------------------------------------
    // BBCache: straight-line arithmetic terminated by branch
    // -----------------------------------------------------------------------

    #[test]
    fn test_straight_line_with_branch() {
        let mut cache = BBCache::new();
        // 3 ADD instructions + 1 B (unconditional branch)
        let instructions = [
            (0x10000u64, 0x91000400u32), // ADD X0, X0, #1
            (0x10004u64, 0x91000421u32), // ADD X1, X1, #1
            (0x10008u64, 0x91000442u32), // ADD X2, X2, #1
            (0x1000Cu64, 0x14000010u32), // B #offset
        ];
        let reader = make_reader(&instructions);
        let bb = cache.scan_block(0x10000, &reader);
        assert_eq!(bb.start_pc, 0x10000);
        assert_eq!(bb.length, 4);
        assert_eq!(bb.terminator, BlockTerminator::Branch);
    }

    // -----------------------------------------------------------------------
    // BBCache: SVC terminates a block
    // -----------------------------------------------------------------------

    #[test]
    fn test_svc_terminates_block() {
        let mut cache = BBCache::new();
        let instructions = [
            (0x10000u64, 0x91000400u32), // ADD X0, X0, #1
            (0x10004u64, 0xD4000001u32), // SVC #0
        ];
        let reader = make_reader(&instructions);
        let bb = cache.scan_block(0x10000, &reader);
        assert_eq!(bb.start_pc, 0x10000);
        assert_eq!(bb.length, 2);
        assert_eq!(bb.terminator, BlockTerminator::Syscall);
    }

    // -----------------------------------------------------------------------
    // BBCache: conditional branch terminates block
    // -----------------------------------------------------------------------

    #[test]
    fn test_conditional_branch_terminates_block() {
        let mut cache = BBCache::new();
        let instructions = [
            (0x10000u64, 0x91000400u32), // ADD X0, X0, #1
            (0x10004u64, 0x54000040u32), // B.EQ #offset
        ];
        let reader = make_reader(&instructions);
        let bb = cache.scan_block(0x10000, &reader);
        assert_eq!(bb.length, 2);
        assert_eq!(bb.terminator, BlockTerminator::ConditionalBranch);
    }

    // -----------------------------------------------------------------------
    // BBCache: RET terminates block as IndirectBranch
    // -----------------------------------------------------------------------

    #[test]
    fn test_ret_terminates_block() {
        let mut cache = BBCache::new();
        let instructions = [
            (0x10000u64, 0x91000400u32), // ADD X0, X0, #1
            (0x10004u64, 0xD65F03C0u32), // RET
        ];
        let reader = make_reader(&instructions);
        let bb = cache.scan_block(0x10000, &reader);
        assert_eq!(bb.length, 2);
        assert_eq!(bb.terminator, BlockTerminator::IndirectBranch);
    }

    // -----------------------------------------------------------------------
    // BBCache: cache hit/miss counting
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_hit_miss_counting() {
        let mut cache = BBCache::new();
        let instructions = [
            (0x10000u64, 0xD4400000u32), // HLT #0
        ];
        let reader = make_reader(&instructions);

        // First access: cache miss
        let _ = cache.scan_block(0x10000, &reader);
        assert_eq!(cache.hits, 0);
        assert_eq!(cache.misses, 1);

        // Second access: cache hit
        let _ = cache.scan_block(0x10000, &reader);
        assert_eq!(cache.hits, 1);
        assert_eq!(cache.misses, 1);

        // Third access: cache hit
        let _ = cache.scan_block(0x10000, &reader);
        assert_eq!(cache.hits, 2);
        assert_eq!(cache.misses, 1);
    }

    #[test]
    fn test_lookup_hit_miss() {
        let mut cache = BBCache::new();
        let instructions = [(0x10000u64, 0xD4400000u32)];
        let reader = make_reader(&instructions);

        // Lookup before scan: miss
        assert!(cache.lookup(0x10000).is_none());
        assert_eq!(cache.misses, 1);

        // Populate via scan
        let _ = cache.scan_block(0x10000, &reader);

        // Lookup after scan: hit
        let bb = cache.lookup(0x10000);
        assert!(bb.is_some());
        assert_eq!(bb.unwrap().terminator, BlockTerminator::Halt);
    }

    // -----------------------------------------------------------------------
    // BBCache: invalidate_all
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalidate_all() {
        let mut cache = BBCache::new();
        let instructions = [(0x10000u64, 0xD4400000u32)];
        let reader = make_reader(&instructions);

        // Populate
        let _ = cache.scan_block(0x10000, &reader);
        assert_eq!(cache.cached_count(), 1);

        // Invalidate: bumps generation, old entries still in map but unreachable
        cache.invalidate_all();

        // Lookup with new generation misses
        assert!(cache.lookup(0x10000).is_none());

        // Re-scan creates a new entry (new generation)
        let _ = cache.scan_block(0x10000, &reader);
        assert_eq!(cache.cached_count(), 2); // old gen + new gen both in map
    }

    // -----------------------------------------------------------------------
    // BBCache: invalidate_range
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalidate_range() {
        let mut cache = BBCache::new();
        // Two blocks: one at 0x10000 (length 2), one at 0x20000 (length 1)
        let instructions_a = [
            (0x10000u64, 0x91000400u32), // ADD
            (0x10004u64, 0xD4400000u32), // HLT
        ];
        let instructions_b = [
            (0x20000u64, 0xD4400000u32), // HLT
        ];

        let reader_a = make_reader(&instructions_a);
        let _ = cache.scan_block(0x10000, &reader_a);

        let reader_b = make_reader(&instructions_b);
        let _ = cache.scan_block(0x20000, &reader_b);

        assert_eq!(cache.cached_count(), 2);

        // Invalidate the range covering block A (0x10000..0x10008)
        cache.invalidate_range(0x10000, 8);

        // Block A should be gone, block B should remain
        assert_eq!(cache.cached_count(), 1);
        assert!(cache.lookup(0x20000).is_some());
    }

    #[test]
    fn test_invalidate_range_partial_overlap() {
        let mut cache = BBCache::new();
        // Block at 0x10000, length 4 instructions = 16 bytes (0x10000..0x10010)
        let instructions = [
            (0x10000u64, 0x91000400u32), // ADD
            (0x10004u64, 0x91000400u32), // ADD
            (0x10008u64, 0x91000400u32), // ADD
            (0x1000Cu64, 0x14000010u32), // B
        ];
        let reader = make_reader(&instructions);
        let _ = cache.scan_block(0x10000, &reader);
        assert_eq!(cache.cached_count(), 1);

        // Write overlapping the middle of the block
        cache.invalidate_range(0x10006, 4);
        assert_eq!(cache.cached_count(), 0);
    }

    #[test]
    fn test_invalidate_range_no_overlap() {
        let mut cache = BBCache::new();
        let instructions = [
            (0x10000u64, 0xD4400000u32), // HLT
        ];
        let reader = make_reader(&instructions);
        let _ = cache.scan_block(0x10000, &reader);
        assert_eq!(cache.cached_count(), 1);

        // Invalidate a range that does NOT overlap
        cache.invalidate_range(0x20000, 256);
        assert_eq!(cache.cached_count(), 1);
    }

    // -----------------------------------------------------------------------
    // BBCache: recommended_batch_size
    // -----------------------------------------------------------------------

    #[test]
    fn test_recommended_batch_size_returns_block_length() {
        let mut cache = BBCache::new();
        let instructions = [
            (0x10000u64, 0x91000400u32), // ADD
            (0x10004u64, 0x91000421u32), // ADD
            (0x10008u64, 0xD4000001u32), // SVC
        ];
        let reader = make_reader(&instructions);
        let size = cache.recommended_batch_size(0x10000, &reader);
        assert_eq!(size, 3);
    }

    #[test]
    fn test_recommended_batch_size_single_instruction() {
        let mut cache = BBCache::new();
        let instructions = [(0x10000u64, 0xD4400000u32)]; // HLT
        let reader = make_reader(&instructions);
        let size = cache.recommended_batch_size(0x10000, &reader);
        assert_eq!(size, 1);
    }

    // -----------------------------------------------------------------------
    // BBCache: clear
    // -----------------------------------------------------------------------

    #[test]
    fn test_clear() {
        let mut cache = BBCache::new();
        let instructions = [(0x10000u64, 0xD4400000u32)];
        let reader = make_reader(&instructions);

        let _ = cache.scan_block(0x10000, &reader);
        assert_eq!(cache.cached_count(), 1);
        assert_eq!(cache.misses, 1);

        cache.clear();
        assert_eq!(cache.cached_count(), 0);
        assert_eq!(cache.hits, 0);
        assert_eq!(cache.misses, 0);
    }

    // -----------------------------------------------------------------------
    // BBCache: CBZ/CBNZ terminate as ConditionalBranch
    // -----------------------------------------------------------------------

    #[test]
    fn test_cbz_terminates_block() {
        let mut cache = BBCache::new();
        let instructions = [
            (0x10000u64, 0x91000400u32), // ADD
            (0x10004u64, 0xB4000060u32), // CBZ X0, #12
        ];
        let reader = make_reader(&instructions);
        let bb = cache.scan_block(0x10000, &reader);
        assert_eq!(bb.length, 2);
        assert_eq!(bb.terminator, BlockTerminator::ConditionalBranch);
    }

    #[test]
    fn test_cbnz_terminates_block() {
        let mut cache = BBCache::new();
        let instructions = [
            (0x10000u64, 0x91000400u32), // ADD
            (0x10004u64, 0x35000060u32), // CBNZ W0, #12
        ];
        let reader = make_reader(&instructions);
        let bb = cache.scan_block(0x10000, &reader);
        assert_eq!(bb.length, 2);
        assert_eq!(bb.terminator, BlockTerminator::ConditionalBranch);
    }

    // -----------------------------------------------------------------------
    // BBCache: TBZ/TBNZ terminate as ConditionalBranch
    // -----------------------------------------------------------------------

    #[test]
    fn test_tbz_terminates_block() {
        let mut cache = BBCache::new();
        let instructions = [
            (0x10000u64, 0x36100060u32), // TBZ W0, #2, #12
        ];
        let reader = make_reader(&instructions);
        let bb = cache.scan_block(0x10000, &reader);
        assert_eq!(bb.length, 1);
        assert_eq!(bb.terminator, BlockTerminator::ConditionalBranch);
    }

    #[test]
    fn test_tbnz_terminates_block() {
        let mut cache = BBCache::new();
        let instructions = [
            (0x10000u64, 0x37100060u32), // TBNZ W0, #2, #12
        ];
        let reader = make_reader(&instructions);
        let bb = cache.scan_block(0x10000, &reader);
        assert_eq!(bb.length, 1);
        assert_eq!(bb.terminator, BlockTerminator::ConditionalBranch);
    }

    // -----------------------------------------------------------------------
    // BBCache: BL terminates as Branch (unconditional)
    // -----------------------------------------------------------------------

    #[test]
    fn test_bl_terminates_as_branch() {
        let mut cache = BBCache::new();
        let instructions = [
            (0x10000u64, 0x91000400u32), // ADD
            (0x10004u64, 0x94000010u32), // BL #offset
        ];
        let reader = make_reader(&instructions);
        let bb = cache.scan_block(0x10000, &reader);
        assert_eq!(bb.length, 2);
        assert_eq!(bb.terminator, BlockTerminator::Branch);
    }

    // -----------------------------------------------------------------------
    // BBCache: BLR terminates as BranchLink (via D6 handler)
    // -----------------------------------------------------------------------

    #[test]
    fn test_blr_terminates_block() {
        let mut cache = BBCache::new();
        // BLR X0 = 0xD63F0000
        let instructions = [
            (0x10000u64, 0x91000400u32), // ADD
            (0x10004u64, 0xD63F0000u32), // BLR X0
        ];
        let reader = make_reader(&instructions);
        let bb = cache.scan_block(0x10000, &reader);
        assert_eq!(bb.length, 2);
        // BLR goes through the D6 handler opc=1 -> BranchLink -> IndirectBranch?
        // Actually in classify_instruction, opc=1 maps to BranchLink, which matches Branch terminator
        assert_eq!(bb.terminator, BlockTerminator::Branch);
    }

    // -----------------------------------------------------------------------
    // BBCache: BR terminates as IndirectBranch
    // -----------------------------------------------------------------------

    #[test]
    fn test_br_terminates_as_indirect() {
        let mut cache = BBCache::new();
        // BR X30 = 0xD61F03C0
        let instructions = [
            (0x10000u64, 0xD61F03C0u32), // BR X30
        ];
        let reader = make_reader(&instructions);
        let bb = cache.scan_block(0x10000, &reader);
        assert_eq!(bb.length, 1);
        assert_eq!(bb.terminator, BlockTerminator::IndirectBranch);
    }

    // -----------------------------------------------------------------------
    // BBCache: ops vector starts empty
    // -----------------------------------------------------------------------

    #[test]
    fn test_ops_vector_is_empty() {
        let mut cache = BBCache::new();
        let instructions = [
            (0x10000u64, 0x91000400u32), // ADD
            (0x10004u64, 0xD4400000u32), // HLT
        ];
        let reader = make_reader(&instructions);
        let bb = cache.scan_block(0x10000, &reader);
        assert!(bb.ops.is_empty());
    }

    // -----------------------------------------------------------------------
    // BBCache: mem_generation is stored in the block
    // -----------------------------------------------------------------------

    #[test]
    fn test_mem_generation_stored_in_block() {
        let mut cache = BBCache::new();
        let instructions = [(0x10000u64, 0xD4400000u32)];
        let reader = make_reader(&instructions);

        let bb = cache.scan_block(0x10000, &reader);
        assert_eq!(bb.mem_generation, 0);

        cache.invalidate_all();

        let bb2 = cache.scan_block(0x10000, &reader);
        assert_eq!(bb2.mem_generation, 1);
    }
}
