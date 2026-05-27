//! Micro-op representation for pre-decoded ARM64 instructions.
//!
//! Each ARM64 instruction decodes to one or more micro-ops. Pre-decoding on the host
//! avoids redundant decode work on every GPU dispatch for the same basic block.

/// A decoded micro-op representing a single ARM64 operation.
#[derive(Clone, Debug)]
pub struct MicroOp {
    /// Original ARM64 instruction word
    pub raw: u32,
    /// Decoded operation type
    pub kind: OpKind,
    /// Destination register (0-31, or NONE=255)
    pub rd: u8,
    /// Source register 1
    pub rn: u8,
    /// Source register 2 (or immediate encoded)
    pub rm_or_imm: u8,
    /// Is this a 32-bit (W) operation?
    pub is_32bit: bool,
    /// Additional flags specific to the operation
    pub flags: u16,
}

/// Categories of ARM64 operations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OpKind {
    // Data processing
    Add,
    Sub,
    Mul,
    Div,
    And,
    Or,
    Xor,
    Bic,
    Lsl,
    Lsr,
    Asr,
    Ror,
    Mov,
    Mvn,
    // Comparison
    Cmp,
    Cmn,
    Tst,
    // Memory
    Load,
    Store,
    LoadPair,
    StorePair,
    // Branch
    Branch,
    BranchCond,
    BranchReg,
    BranchLink,
    Ret,
    Cbz,
    Cbnz,
    Tbz,
    Tbnz,
    // System
    Svc,
    Hlt,
    Nop,
    Mrs,
    Msr,
    // Extension
    Sxtb,
    Sxth,
    Sxtw,
    Uxtb,
    Uxth,
    // Other
    Adr,
    Adrp,
    Clz,
    Cls,
    Rbit,
    Rev,
    Madd,
    Msub,
    Csel,
    Csinc,
    Csinv,
    Csneg,
    // Unknown/unhandled
    Unknown,
}

impl MicroOp {
    pub const REG_NONE: u8 = 255;

    /// Create an unknown/pass-through micro-op from a raw instruction word.
    pub fn passthrough(raw: u32) -> Self {
        MicroOp {
            raw,
            kind: OpKind::Unknown,
            rd: Self::REG_NONE,
            rn: Self::REG_NONE,
            rm_or_imm: Self::REG_NONE,
            is_32bit: false,
            flags: 0,
        }
    }

    /// Returns true if this micro-op is a control flow instruction (branch, ret, svc, hlt).
    pub fn is_control_flow(&self) -> bool {
        matches!(
            self.kind,
            OpKind::Branch
                | OpKind::BranchCond
                | OpKind::BranchReg
                | OpKind::BranchLink
                | OpKind::Ret
                | OpKind::Cbz
                | OpKind::Cbnz
                | OpKind::Tbz
                | OpKind::Tbnz
                | OpKind::Svc
                | OpKind::Hlt
        )
    }

    /// Returns true if this is a memory-writing instruction.
    pub fn writes_memory(&self) -> bool {
        matches!(self.kind, OpKind::Store | OpKind::StorePair)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passthrough_creates_unknown_op() {
        let op = MicroOp::passthrough(0xDEADBEEF);
        assert_eq!(op.raw, 0xDEADBEEF);
        assert_eq!(op.kind, OpKind::Unknown);
        assert_eq!(op.rd, MicroOp::REG_NONE);
        assert_eq!(op.rn, MicroOp::REG_NONE);
        assert_eq!(op.rm_or_imm, MicroOp::REG_NONE);
        assert!(!op.is_32bit);
        assert_eq!(op.flags, 0);
    }

    #[test]
    fn test_is_control_flow_detects_branches() {
        let branch_kinds = [
            OpKind::Branch,
            OpKind::BranchCond,
            OpKind::BranchReg,
            OpKind::BranchLink,
            OpKind::Ret,
            OpKind::Cbz,
            OpKind::Cbnz,
            OpKind::Tbz,
            OpKind::Tbnz,
        ];
        for kind in &branch_kinds {
            let op = MicroOp {
                raw: 0,
                kind: *kind,
                rd: 0,
                rn: 0,
                rm_or_imm: 0,
                is_32bit: false,
                flags: 0,
            };
            assert!(
                op.is_control_flow(),
                "Expected {:?} to be control flow",
                kind
            );
        }
    }

    #[test]
    fn test_is_control_flow_detects_svc_and_hlt() {
        let svc = MicroOp {
            raw: 0,
            kind: OpKind::Svc,
            rd: 0,
            rn: 0,
            rm_or_imm: 0,
            is_32bit: false,
            flags: 0,
        };
        let hlt = MicroOp {
            raw: 0,
            kind: OpKind::Hlt,
            rd: 0,
            rn: 0,
            rm_or_imm: 0,
            is_32bit: false,
            flags: 0,
        };
        assert!(svc.is_control_flow());
        assert!(hlt.is_control_flow());
    }

    #[test]
    fn test_is_control_flow_rejects_non_branches() {
        let non_branch_kinds = [
            OpKind::Add,
            OpKind::Sub,
            OpKind::Mul,
            OpKind::Div,
            OpKind::Load,
            OpKind::Store,
            OpKind::Mov,
            OpKind::Nop,
            OpKind::Unknown,
        ];
        for kind in &non_branch_kinds {
            let op = MicroOp {
                raw: 0,
                kind: *kind,
                rd: 0,
                rn: 0,
                rm_or_imm: 0,
                is_32bit: false,
                flags: 0,
            };
            assert!(
                !op.is_control_flow(),
                "Expected {:?} to NOT be control flow",
                kind
            );
        }
    }

    #[test]
    fn test_writes_memory_detects_store() {
        let store = MicroOp {
            raw: 0,
            kind: OpKind::Store,
            rd: 0,
            rn: 0,
            rm_or_imm: 0,
            is_32bit: false,
            flags: 0,
        };
        let store_pair = MicroOp {
            raw: 0,
            kind: OpKind::StorePair,
            rd: 0,
            rn: 0,
            rm_or_imm: 0,
            is_32bit: false,
            flags: 0,
        };
        assert!(store.writes_memory());
        assert!(store_pair.writes_memory());
    }

    #[test]
    fn test_writes_memory_rejects_non_stores() {
        let non_store_kinds = [
            OpKind::Load,
            OpKind::LoadPair,
            OpKind::Add,
            OpKind::Branch,
            OpKind::Unknown,
        ];
        for kind in &non_store_kinds {
            let op = MicroOp {
                raw: 0,
                kind: *kind,
                rd: 0,
                rn: 0,
                rm_or_imm: 0,
                is_32bit: false,
                flags: 0,
            };
            assert!(
                !op.writes_memory(),
                "Expected {:?} to NOT write memory",
                kind
            );
        }
    }

    #[test]
    fn test_reg_none_constant() {
        assert_eq!(MicroOp::REG_NONE, 255);
    }
}
