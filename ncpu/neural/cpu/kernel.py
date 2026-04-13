"""
Kernel support mixin for NeuralCPU.

System registers (MRS/MSR), MMU/TLB translation, GIC interrupt controller,
PL011 UART emulation, and MMIO routing - all GPU-tensor-based.
"""

import logging
import torch
import time
from typing import Optional

from .constants import OpType

logger = logging.getLogger(__name__)


class KernelMixin:
    """System register, MMU, GIC, UART, and MMIO methods for NeuralCPU."""

    def _read_system_register(self, sysreg: int) -> int:
        """
        Read from ARM64 system register - ALL READS FROM GPU TENSORS.
        Sysreg encoding: op0:op1:CRn:CRm:op2 packed into 15 bits.
        """
        # Common userspace registers
        if sysreg == 0x5E10:  # TPIDR_EL0 (thread pointer)
            return self.sysreg_tpidr_el0
        elif sysreg == 0x5A10:  # FPCR
            return 0
        elif sysreg == 0x5A20:  # FPSR
            return 0

        # Counter-timer registers (ARM Generic Timer) - FROM GPU TENSORS
        elif sysreg == 0x5E00:  # CNTFRQ_EL0
            return int(self._sysregs[self._SR_CNTFRQ_EL0].item())
        elif sysreg == 0x5E02:  # CNTVCT_EL0 (virtual count)
            # Calculate from elapsed time using GPU tensors
            freq = self._sysregs[self._SR_CNTFRQ_EL0]
            elapsed_ns = torch.tensor(int(time.time() * 1e9), dtype=torch.int64, device=self.device) - self._timer_start
            count = (elapsed_ns * freq) // 1000000000
            self._sysregs[self._SR_CNTVCT_EL0] = count
            return int(count.item())
        elif sysreg == 0x5E08:  # CNTP_TVAL_EL0
            return int(self._sysregs[self._SR_CNTP_TVAL_EL0].item())
        elif sysreg == 0x5E09:  # CNTP_CTL_EL0
            return int(self._sysregs[self._SR_CNTP_CTL_EL0].item())
        elif sysreg == 0x5E18:  # CNTV_TVAL_EL0
            return int(self._sysregs[self._SR_CNTV_TVAL_EL0].item())
        elif sysreg == 0x5E19:  # CNTV_CTL_EL0
            return int(self._sysregs[self._SR_CNTV_CTL_EL0].item())

        # EL1 System Registers (Kernel) - FROM GPU TENSORS
        elif sysreg == 0x4000:  # SCTLR_EL1
            return int(self._sysregs[self._SR_SCTLR_EL1].item())
        elif sysreg == 0x4020:  # TTBR0_EL1
            return int(self._sysregs[self._SR_TTBR0_EL1].item())
        elif sysreg == 0x4021:  # TTBR1_EL1
            return int(self._sysregs[self._SR_TTBR1_EL1].item())
        elif sysreg == 0x4022:  # TCR_EL1
            return int(self._sysregs[self._SR_TCR_EL1].item())
        elif sysreg == 0x4A00:  # MAIR_EL1
            return int(self._sysregs[self._SR_MAIR_EL1].item())
        elif sysreg == 0x6000:  # VBAR_EL1
            return int(self._sysregs[self._SR_VBAR_EL1].item())
        elif sysreg == 0x6001:  # ELR_EL1
            return int(self._sysregs[self._SR_ELR_EL1].item())
        elif sysreg == 0x6002:  # SPSR_EL1
            return int(self._sysregs[self._SR_SPSR_EL1].item())
        elif sysreg == 0x6003:  # ESR_EL1
            return int(self._sysregs[self._SR_ESR_EL1].item())
        elif sysreg == 0x6004:  # FAR_EL1
            return int(self._sysregs[self._SR_FAR_EL1].item())
        elif sysreg == 0x6005:  # SP_EL0
            return int(self._sysregs[self._SR_SP_EL0].item())

        # CurrentEL
        elif sysreg == 0x4212:  # CurrentEL
            return int(self._sysregs[self._SR_CURRENT_EL].item()) << 2

        # ID registers (read-only, constant)
        elif sysreg == 0x4001:  # MIDR_EL1 - Main ID Register
            return 0x410FD0F0  # Cortex-A53 compatible
        elif sysreg == 0x4005:  # MPIDR_EL1 - Multiprocessor Affinity
            return 0x80000000  # Single core, Aff0=0
        elif sysreg == 0x4008:  # ID_AA64PFR0_EL1
            return 0x0000000000002222  # AArch64 EL0-3 support

        return 0  # Default

    def _write_system_register(self, sysreg: int, val: int):
        """
        Write to ARM64 system register - ALL WRITES TO GPU TENSORS.
        """
        # Userspace registers
        if sysreg == 0x5E10:  # TPIDR_EL0
            self.sysreg_tpidr_el0 = val

        # Counter-timer registers - TO GPU TENSORS
        elif sysreg == 0x5E08:  # CNTP_TVAL_EL0
            self._sysregs[self._SR_CNTP_TVAL_EL0] = val
        elif sysreg == 0x5E09:  # CNTP_CTL_EL0
            self._sysregs[self._SR_CNTP_CTL_EL0] = val
            self._check_timer_interrupt_gpu()
        elif sysreg == 0x5E18:  # CNTV_TVAL_EL0
            self._sysregs[self._SR_CNTV_TVAL_EL0] = val
        elif sysreg == 0x5E19:  # CNTV_CTL_EL0
            self._sysregs[self._SR_CNTV_CTL_EL0] = val
            self._check_timer_interrupt_gpu()

        # EL1 System Registers (Kernel) - TO GPU TENSORS
        elif sysreg == 0x4000:  # SCTLR_EL1
            old_mmu = self._sysregs[self._SR_SCTLR_EL1] & 1
            self._sysregs[self._SR_SCTLR_EL1] = val
            new_mmu = val & 1
            if int(old_mmu.item()) != new_mmu:
                if new_mmu:
                    self._tlb_valid.fill_(False)  # Flush TLB
        elif sysreg == 0x4020:  # TTBR0_EL1
            self._sysregs[self._SR_TTBR0_EL1] = val
            self._tlb_valid.fill_(False)  # Flush TLB
        elif sysreg == 0x4021:  # TTBR1_EL1
            self._sysregs[self._SR_TTBR1_EL1] = val
            self._tlb_valid.fill_(False)  # Flush TLB
        elif sysreg == 0x4022:  # TCR_EL1
            self._sysregs[self._SR_TCR_EL1] = val
        elif sysreg == 0x4A00:  # MAIR_EL1
            self._sysregs[self._SR_MAIR_EL1] = val
        elif sysreg == 0x6000:  # VBAR_EL1
            self._sysregs[self._SR_VBAR_EL1] = val
        elif sysreg == 0x6001:  # ELR_EL1
            self._sysregs[self._SR_ELR_EL1] = val
        elif sysreg == 0x6002:  # SPSR_EL1
            self._sysregs[self._SR_SPSR_EL1] = val
        elif sysreg == 0x6003:  # ESR_EL1
            self._sysregs[self._SR_ESR_EL1] = val
        elif sysreg == 0x6004:  # FAR_EL1
            self._sysregs[self._SR_FAR_EL1] = val
        elif sysreg == 0x6005:  # SP_EL0
            self._sysregs[self._SR_SP_EL0] = val

    # ═══════════════════════════════════════════════════════════════════════════
    # MMU PAGE TABLE TRANSLATION (GPU-ACCELERATED WITH TENSOR TLB)
    # ═══════════════════════════════════════════════════════════════════════════

    def _mmu_enabled(self) -> bool:
        """Check if MMU is enabled (from GPU tensor)."""
        return bool(self._sysregs[self._SR_SCTLR_EL1].item() & 1)

    def _translate_address_gpu(self, va_tensor: torch.Tensor) -> torch.Tensor:
        """
        GPU-accelerated address translation for batched addresses.
        Returns physical addresses (identity mapping when MMU off).
        """
        if not self._mmu_enabled():
            return va_tensor  # Identity mapping

        # GPU TLB lookup - all tensor ops
        page_va = va_tensor & self._page_mask
        page_offset = va_tensor & (self.PAGE_SIZE - 1)

        # Search TLB (vectorized)
        matches = (self._tlb_va == page_va.unsqueeze(-1)) & self._tlb_valid
        hit_mask = matches.any(dim=-1)
        hit_idx = matches.int().argmax(dim=-1)

        # For hits, use cached PA
        pa = torch.where(
            hit_mask,
            self._tlb_pa[hit_idx] | page_offset,
            va_tensor  # Miss: fallback to identity (or page walk)
        )
        return pa

    def _translate_address(self, va: int, is_write: bool = False) -> int:
        """
        Translate virtual address to physical address using page tables.
        Returns physical address or raises exception on fault.
        """
        if not self._mmu_enabled():
            return va  # Identity mapping when MMU disabled

        # Check GPU TLB cache first
        page_va = va & int(self._page_mask.item())
        va_tensor = torch.tensor(page_va, dtype=torch.int64, device=self.device)

        # Vectorized TLB lookup
        matches = (self._tlb_va == va_tensor) & self._tlb_valid
        if matches.any():
            hit_idx = matches.int().argmax()
            pa_base = int(self._tlb_pa[hit_idx].item())
            return pa_base | (va & (self.PAGE_SIZE - 1))

        # TLB miss - page table walk
        # Determine which TTBR to use based on address
        if va & (1 << 55):  # Upper half (kernel space)
            ttbr = int(self._sysregs[self._SR_TTBR1_EL1].item())
        else:  # Lower half (user space)
            ttbr = int(self._sysregs[self._SR_TTBR0_EL1].item())

        # 4-level page table walk (4KB granule)
        l0_idx = (va >> 39) & 0x1FF
        l1_idx = (va >> 30) & 0x1FF
        l2_idx = (va >> 21) & 0x1FF
        l3_idx = (va >> 12) & 0x1FF

        try:
            # Level 0
            l0_entry_addr = (ttbr & 0xFFFFFFFFF000) + (l0_idx * 8)
            l0_entry = self._read_physical_u64(l0_entry_addr)
            if not (l0_entry & 1):
                return self._handle_page_fault(va, is_write, "L0 invalid")

            # Level 1
            l1_table = l0_entry & 0xFFFFFFFFF000
            l1_entry_addr = l1_table + (l1_idx * 8)
            l1_entry = self._read_physical_u64(l1_entry_addr)
            if not (l1_entry & 1):
                return self._handle_page_fault(va, is_write, "L1 invalid")
            if (l1_entry & 3) == 1:  # 1GB block
                pa = (l1_entry & 0xFFFFC0000000) | (va & 0x3FFFFFFF)
                self._tlb_insert_gpu(page_va, pa & int(self._page_mask.item()), l1_entry)
                return pa

            # Level 2
            l2_table = l1_entry & 0xFFFFFFFFF000
            l2_entry_addr = l2_table + (l2_idx * 8)
            l2_entry = self._read_physical_u64(l2_entry_addr)
            if not (l2_entry & 1):
                return self._handle_page_fault(va, is_write, "L2 invalid")
            if (l2_entry & 3) == 1:  # 2MB block
                pa = (l2_entry & 0xFFFFFFE00000) | (va & 0x1FFFFF)
                self._tlb_insert_gpu(page_va, pa & int(self._page_mask.item()), l2_entry)
                return pa

            # Level 3
            l3_table = l2_entry & 0xFFFFFFFFF000
            l3_entry_addr = l3_table + (l3_idx * 8)
            l3_entry = self._read_physical_u64(l3_entry_addr)
            if not (l3_entry & 3) == 3:  # Page descriptor
                return self._handle_page_fault(va, is_write, "L3 invalid")

            pa = (l3_entry & 0xFFFFFFFFF000) | (va & 0xFFF)
            self._tlb_insert_gpu(page_va, pa & int(self._page_mask.item()), l3_entry)
            return pa

        except Exception as e:
            return self._handle_page_fault(va, is_write, str(e))

    def _tlb_insert_gpu(self, va_page: int, pa_page: int, perm: int):
        """Insert entry into GPU TLB (circular buffer)."""
        ptr = int(self._tlb_ptr.item())
        self._tlb_va[ptr] = va_page
        self._tlb_pa[ptr] = pa_page
        self._tlb_perm[ptr] = perm
        self._tlb_valid[ptr] = True
        self._tlb_ptr.fill_((ptr + 1) % self._tlb_max_entries)

    def _read_physical_u64(self, pa: int) -> int:
        """Read 64-bit value from physical memory (GPU tensor)."""
        if pa + 8 > self.mem_size:
            return 0
        # Use GPU tensor slicing, then single transfer
        data = self.memory[pa:pa+8]
        # Pack into 64-bit value using GPU
        val = data[0].to(torch.int64)
        for i in range(1, 8):
            val = val | (data[i].to(torch.int64) << (i * 8))
        return int(val.item())

    def _handle_page_fault(self, va: int, is_write: bool, reason: str) -> int:
        """Handle page fault - writes to GPU tensors."""
        self._sysregs[self._SR_FAR_EL1] = va
        self._sysregs[self._SR_ESR_EL1] = 0x92000000 if is_write else 0x82000000
        # Return identity mapping as fallback
        return va

    def tlb_flush(self):
        """Flush the GPU TLB cache."""
        self._tlb_valid.fill_(False)

    # ═══════════════════════════════════════════════════════════════════════════
    # GIC (GENERIC INTERRUPT CONTROLLER) - GPU TENSOR-BACKED
    # ═══════════════════════════════════════════════════════════════════════════

    def _gic_enabled(self) -> bool:
        """Check if GIC is enabled (from GPU tensor)."""
        return bool(self._sysregs[self._SR_GICD_CTLR].item() & 1)

    def _check_timer_interrupt_gpu(self):
        """Check if timer interrupt should be raised - GPU tensor version."""
        # Physical timer
        ctl_p = self._sysregs[self._SR_CNTP_CTL_EL0]
        tval_p = self._sysregs[self._SR_CNTP_TVAL_EL0]
        if (ctl_p & 1) and not (ctl_p & 2) and (tval_p <= 0):
            self._raise_irq_gpu(self.IRQ_TIMER)

        # Virtual timer
        ctl_v = self._sysregs[self._SR_CNTV_CTL_EL0]
        tval_v = self._sysregs[self._SR_CNTV_TVAL_EL0]
        if (ctl_v & 1) and not (ctl_v & 2) and (tval_v <= 0):
            self._raise_irq_gpu(self.IRQ_TIMER)

    def _raise_irq_gpu(self, irq_num: int):
        """Raise an interrupt - GPU tensor version."""
        if not self._gic_enabled():
            return
        # Check if already pending
        count = int(self._pending_irq_count.item())
        already_pending = (self._pending_irqs[:count] == irq_num).any()
        if not already_pending and count < 32:
            self._pending_irqs[count] = irq_num
            self._pending_irq_count.add_(1)
            # Set pending bit in distributor
            word = irq_num // 32
            bit = irq_num % 32
            self.gicd_ispendr[word] = self.gicd_ispendr[word] | (1 << bit)

    def _check_pending_irqs_gpu(self) -> bool:
        """Check for pending interrupts - GPU tensor version."""
        if not self._gic_enabled():
            return False
        count = int(self._pending_irq_count.item())
        if count == 0:
            return False
        if not (self._sysregs[self._SR_GICC_CTLR] & 1):
            return False

        # Find highest priority pending interrupt
        pmr = int(self._sysregs[self._SR_GICC_PMR].item())
        for i in range(count):
            irq = int(self._pending_irqs[i].item())
            if irq < 0:
                continue
            priority = int(self.gicd_ipriorityr[irq].item())
            if priority < pmr:
                self._sysregs[self._SR_GICC_IAR] = irq
                self._take_irq_exception_gpu(irq)
                return True
        return False

    def _take_irq_exception_gpu(self, irq_num: int):
        """Take an IRQ exception to EL1 - GPU tensor version."""
        # Save current state to exception stack
        ptr = int(self._exc_stack_ptr.item())
        if ptr < 8:
            self._exc_stack[ptr, 0] = self.pc.clone()
            self._exc_stack[ptr, 1] = self._sysregs[self._SR_CURRENT_EL]
            self._exc_stack[ptr, 2] = int(self.flags[0].item() > 0.5)
            self._exc_stack[ptr, 3] = int(self.flags[1].item() > 0.5)
            self._exc_stack[ptr, 4] = int(self.flags[2].item() > 0.5)
            self._exc_stack[ptr, 5] = int(self.flags[3].item() > 0.5)
            self._exc_stack_ptr.add_(1)

        # Set exception state
        self._sysregs[self._SR_ELR_EL1] = self.pc.clone()
        self._sysregs[self._SR_SPSR_EL1] = self._flags_to_spsr_gpu()
        self._sysregs[self._SR_CURRENT_EL] = 1

        # Jump to exception vector (IRQ from EL1 with SP_EL1)
        vector_offset = 0x280
        vbar = self._sysregs[self._SR_VBAR_EL1]
        self.pc = vbar + vector_offset

    def _flags_to_spsr_gpu(self) -> int:
        """Convert flags tensor to SPSR format - GPU tensor version."""
        n = int(self.flags[0].item() > 0.5)
        z = int(self.flags[1].item() > 0.5)
        c = int(self.flags[2].item() > 0.5)
        v = int(self.flags[3].item() > 0.5)
        el = int(self._sysregs[self._SR_CURRENT_EL].item())
        return (n << 31) | (z << 30) | (c << 29) | (v << 28) | (el << 2)

    def gic_read(self, offset: int) -> int:
        """Read from GIC registers - GPU tensor backed."""
        # Distributor
        if offset < 0x1000:
            if offset == 0x0:  # GICD_CTLR
                return int(self._sysregs[self._SR_GICD_CTLR].item())
            elif offset == 0x4:  # GICD_TYPER
                return 0x0000001F  # 32 interrupt lines
            elif 0x100 <= offset < 0x120:  # GICD_ISENABLER
                return int(self.gicd_isenabler[(offset - 0x100) // 4].item())
            elif 0x200 <= offset < 0x220:  # GICD_ISPENDR
                return int(self.gicd_ispendr[(offset - 0x200) // 4].item())
            elif 0x400 <= offset < 0x500:  # GICD_IPRIORITYR
                return int(self.gicd_ipriorityr[offset - 0x400].item())
        # CPU Interface
        elif offset >= 0x10000:
            cpu_off = offset - 0x10000
            if cpu_off == 0x0:  # GICC_CTLR
                return int(self._sysregs[self._SR_GICC_CTLR].item())
            elif cpu_off == 0x4:  # GICC_PMR
                return int(self._sysregs[self._SR_GICC_PMR].item())
            elif cpu_off == 0xC:  # GICC_IAR
                return int(self._sysregs[self._SR_GICC_IAR].item())
        return 0

    def gic_write(self, offset: int, val: int):
        """Write to GIC registers - GPU tensor backed."""
        # Distributor
        if offset < 0x1000:
            if offset == 0x0:  # GICD_CTLR
                self._sysregs[self._SR_GICD_CTLR] = val
            elif 0x100 <= offset < 0x120:  # GICD_ISENABLER
                idx = (offset - 0x100) // 4
                self.gicd_isenabler[idx] = self.gicd_isenabler[idx] | val
            elif 0x180 <= offset < 0x1A0:  # GICD_ICENABLER
                idx = (offset - 0x180) // 4
                self.gicd_isenabler[idx] = self.gicd_isenabler[idx] & ~val
            elif 0x400 <= offset < 0x500:  # GICD_IPRIORITYR
                self.gicd_ipriorityr[offset - 0x400] = val & 0xFF
        # CPU Interface
        elif offset >= 0x10000:
            cpu_off = offset - 0x10000
            if cpu_off == 0x0:  # GICC_CTLR
                self._sysregs[self._SR_GICC_CTLR] = val
            elif cpu_off == 0x4:  # GICC_PMR
                self._sysregs[self._SR_GICC_PMR] = val
            elif cpu_off == 0x10:  # GICC_EOIR
                # End of interrupt - remove from pending
                irq = val & 0x3FF
                count = int(self._pending_irq_count.item())
                # Find and remove the IRQ (GPU tensor ops)
                mask = self._pending_irqs[:count] != irq
                remaining = self._pending_irqs[:count][mask]
                new_count = remaining.shape[0]
                self._pending_irqs[:new_count] = remaining
                self._pending_irqs[new_count:count] = -1
                self._pending_irq_count.fill_(new_count)
                # Clear pending bit
                word = irq // 32
                bit = irq % 32
                self.gicd_ispendr[word] = self.gicd_ispendr[word] & ~(1 << bit)

    # ═══════════════════════════════════════════════════════════════════════════
    # UART (PL011) EMULATION - GPU TENSOR BACKED
    # ═══════════════════════════════════════════════════════════════════════════

    def uart_read(self, offset: int) -> int:
        """Read from UART registers - GPU tensor backed."""
        if offset == 0x00:  # UARTDR
            head = int(self._uart_rx_head.item())
            tail = int(self._uart_rx_tail.item())
            if head != tail:
                val = int(self._uart_rx_buf[tail].item())
                self._uart_rx_tail.fill_((tail + 1) % 256)
                return val
            return 0
        elif offset == 0x18:  # UARTFR (flags)
            # Bit 4: RX empty, Bit 5: TX full, Bit 7: TX empty
            flags = 0x90  # TX empty, RX empty
            if self._uart_rx_buffer:
                flags &= ~0x10  # RX not empty
            return flags
        elif offset == 0x24:  # UARTIBRD
            return self.uart_ibrd
        elif offset == 0x28:  # UARTFBRD
            return int(self._uart_regs[self._UART_FBRD].item())
        elif offset == 0x2C:  # UARTLCR_H
            return int(self._uart_regs[self._UART_LCR_H].item())
        elif offset == 0x30:  # UARTCR
            return int(self._uart_regs[self._UART_CR].item())
        elif offset == 0x38:  # UARTIMSC
            return int(self._uart_regs[self._UART_IMSC].item())
        elif offset == 0x3C:  # UARTRIS
            return int(self._uart_regs[self._UART_RIS].item())
        return 0

    def uart_write(self, offset: int, val: int):
        """Write to UART registers - GPU tensor backed."""
        if offset == 0x00:  # UARTDR
            # Output character to console and framebuffer
            char = chr(val & 0xFF)
            print(char, end='', flush=True)
            self.write_console_bytes(bytes([val & 0xFF]))
        elif offset == 0x24:  # UARTIBRD
            self._uart_regs[self._UART_IBRD] = val
        elif offset == 0x28:  # UARTFBRD
            self._uart_regs[self._UART_FBRD] = val
        elif offset == 0x2C:  # UARTLCR_H
            self._uart_regs[self._UART_LCR_H] = val
        elif offset == 0x30:  # UARTCR
            self._uart_regs[self._UART_CR] = val
        elif offset == 0x38:  # UARTIMSC
            self._uart_regs[self._UART_IMSC] = val

    def uart_input(self, data: bytes):
        """Feed input data to UART RX buffer (GPU tensor circular buffer)."""
        for b in data:
            head = int(self._uart_rx_head.item())
            next_head = (head + 1) % 256
            tail = int(self._uart_rx_tail.item())
            if next_head != tail:  # Buffer not full
                self._uart_rx_buf[head] = b
                self._uart_rx_head.fill_(next_head)

    # ═══════════════════════════════════════════════════════════════════════════
    # MMIO ACCESS ROUTER - GPU TENSOR BACKED
    # ═══════════════════════════════════════════════════════════════════════════

    def _mmio_read(self, addr: int, size: int) -> int:
        """Route MMIO reads to appropriate device (GPU tensor backed)."""
        # GIC
        if self.gic_base <= addr < self.gic_base + 0x20000:
            return self.gic_read(addr - self.gic_base)
        # UART
        if self.uart_base <= addr < self.uart_base + 0x1000:
            return self.uart_read(addr - self.uart_base)
        # VirtIO (GPU tensor backed)
        if self.virtio_base <= addr < self.virtio_base + 0x1000:
            offset = addr - self.virtio_base
            if offset < 64:
                return int(self._virtio_regs[offset // 4].item())
        return 0

    def _mmio_write(self, addr: int, val: int, size: int):
        """Route MMIO writes to appropriate device (GPU tensor backed)."""
        # GIC
        if self.gic_base <= addr < self.gic_base + 0x20000:
            self.gic_write(addr - self.gic_base, val)
        # UART
        elif self.uart_base <= addr < self.uart_base + 0x1000:
            self.uart_write(addr - self.uart_base, val)
        # VirtIO
        elif self.virtio_base <= addr < self.virtio_base + 0x1000:
            offset = addr - self.virtio_base
            if offset < 64:
                self._virtio_regs[offset // 4] = val

