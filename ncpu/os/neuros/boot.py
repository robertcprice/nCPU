"""neurOS Boot Sequence.

Initializes all OS components in the correct order:
    1. Device detection (MPS > CUDA > CPU)
    2. MMU initialization (page tables, physical frame pool)
    3. TLB initialization
    4. Cache hierarchy setup
    5. Interrupt controller (GIC)
    6. Process table
    7. Scheduler
    8. IPC subsystem
    9. Filesystem mount
    10. Shell process creation

After boot, the system is ready for interactive use or programmatic
control through the syscall interface.
"""

import torch
import time
import logging
from typing import Optional, Dict
from pathlib import Path

from .device import default_device
from .mmu import NeuralMMU
from .tlb import NeuralTLB
from .cache import NeuralCache
from .interrupts import NeuralGIC, IRQ_TIMER, IRQ_SYSCALL, IRQ_PAGE_FAULT
from .ipc import NeuralIPC
from .process import ProcessTable, ProcessState
from .scheduler import NeuralScheduler
from .filesystem import NeuralFilesystem
from .shell import NeuralShell
from .syscalls import SyscallInterface
from .assembler import NeuralAssembler
from .compiler import NeuralCompiler
from .watchdog import NeuralWatchdog
from .sync import SyncManager
from .protection import MemoryProtectionUnit

logger = logging.getLogger(__name__)


class NeurOS:
    """neurOS: GPU-Native Neural Operating System.

    Every component is a neural network running on GPU.
    All state lives as tensors — zero CPU-GPU sync during operation.

    Usage:
        os = NeurOS()
        os.boot()
        os.shell.execute("ls /")
        os.shell.run_interactive()  # Interactive mode
    """

    def __init__(self, device: Optional[torch.device] = None,
                 max_virtual_pages: int = 4096,
                 max_physical_frames: int = 4096,
                 tlb_size: int = 64,
                 cache_sets: int = 256,
                 cache_ways: int = 4,
                 fs_blocks: int = 4096,
                 max_processes: int = 256):
        self.device = device or default_device()
        self._booted = False
        self._boot_time = 0.0

        # Configuration
        self._config = {
            "max_virtual_pages": max_virtual_pages,
            "max_physical_frames": max_physical_frames,
            "tlb_size": tlb_size,
            "cache_sets": cache_sets,
            "cache_ways": cache_ways,
            "fs_blocks": fs_blocks,
            "max_processes": max_processes,
        }

        # Components (initialized during boot)
        self.mmu: Optional[NeuralMMU] = None
        self.tlb: Optional[NeuralTLB] = None
        self.cache: Optional[NeuralCache] = None
        self.gic: Optional[NeuralGIC] = None
        self.ipc: Optional[NeuralIPC] = None
        self.process_table: Optional[ProcessTable] = None
        self.scheduler: Optional[NeuralScheduler] = None
        self.fs: Optional[NeuralFilesystem] = None
        self.syscalls: Optional[SyscallInterface] = None
        self.shell: Optional[NeuralShell] = None
        self.assembler: Optional[NeuralAssembler] = None
        self.compiler: Optional[NeuralCompiler] = None
        self.watchdog: Optional[NeuralWatchdog] = None
        self.sync: Optional[SyncManager] = None
        self.mpu: Optional[MemoryProtectionUnit] = None

        # Online adaptation (initialized during boot if enabled)
        self.adaptation_manager = None

    def boot(self, load_models: bool = True, quiet: bool = False,
             online_adaptation: bool = False) -> Dict:
        """Boot neurOS.

        Initializes all components in dependency order.

        Args:
            load_models: Try to load trained neural models
            quiet: Suppress boot messages

        Returns:
            Boot statistics dict
        """
        t_start = time.perf_counter()

        if not quiet:
            self._boot_banner()

        stages = {}

        # ─── Stage 1: Memory Management ───────────────────────────────
        t = time.perf_counter()
        self.mmu = NeuralMMU(
            max_virtual_pages=self._config["max_virtual_pages"],
            max_physical_frames=self._config["max_physical_frames"],
            device=self.device,
        )
        self.tlb = NeuralTLB(
            size=self._config["tlb_size"],
            device=self.device,
        )
        self.cache = NeuralCache(
            num_sets=self._config["cache_sets"],
            ways=self._config["cache_ways"],
            device=self.device,
        )
        stages["memory"] = time.perf_counter() - t
        if not quiet:
            logger.info(f"  [BOOT] Memory subsystem: {stages['memory']*1000:.1f}ms")

        # ─── Stage 2: Interrupt Controller ────────────────────────────
        t = time.perf_counter()
        self.gic = NeuralGIC(device=self.device)
        # Register default interrupt handlers
        self.gic.register_handler(IRQ_TIMER, self._handle_timer_irq)
        self.gic.register_handler(IRQ_SYSCALL, self._handle_syscall_irq)
        self.gic.register_handler(IRQ_PAGE_FAULT, self._handle_page_fault_irq)
        stages["interrupts"] = time.perf_counter() - t
        if not quiet:
            logger.info(f"  [BOOT] Interrupt controller: {stages['interrupts']*1000:.1f}ms")

        # ─── Stage 3: Process Management ──────────────────────────────
        t = time.perf_counter()
        self.process_table = ProcessTable(
            max_processes=self._config["max_processes"],
            device=self.device,
        )
        self.scheduler = NeuralScheduler(
            process_table=self.process_table,
            device=self.device,
        )
        stages["processes"] = time.perf_counter() - t
        if not quiet:
            logger.info(f"  [BOOT] Process management: {stages['processes']*1000:.1f}ms")

        # ─── Stage 4: IPC ─────────────────────────────────────────────
        t = time.perf_counter()
        self.ipc = NeuralIPC(device=self.device)
        stages["ipc"] = time.perf_counter() - t
        if not quiet:
            logger.info(f"  [BOOT] IPC subsystem: {stages['ipc']*1000:.1f}ms")

        # ─── Stage 5: Filesystem ──────────────────────────────────────
        t = time.perf_counter()
        self.fs = NeuralFilesystem(
            num_blocks=self._config["fs_blocks"],
            device=self.device,
        )
        self._init_filesystem()
        stages["filesystem"] = time.perf_counter() - t
        if not quiet:
            logger.info(f"  [BOOT] Filesystem: {stages['filesystem']*1000:.1f}ms")

        # ─── Stage 6: Syscall Interface ───────────────────────────────
        t = time.perf_counter()
        self.syscalls = SyscallInterface(self)
        stages["syscalls"] = time.perf_counter() - t

        # ─── Stage 7: Shell ───────────────────────────────────────────
        t = time.perf_counter()
        self.shell = NeuralShell(self)

        # Create init process (PID 1)
        init_proc = self.process_table.create_process("init", priority=0)
        self.ipc.register_process(init_proc.pid)
        init_proc.state = ProcessState.RUNNING
        self.shell.pid = init_proc.pid
        stages["shell"] = time.perf_counter() - t
        if not quiet:
            logger.info(f"  [BOOT] Shell: {stages['shell']*1000:.1f}ms")

        # ─── Stage 8: Assembler & Compiler ───────────────────────────
        t = time.perf_counter()
        self.assembler = NeuralAssembler(device=self.device)
        self.compiler = NeuralCompiler(device=self.device)
        stages["toolchain"] = time.perf_counter() - t
        if not quiet:
            logger.info(f"  [BOOT] Toolchain: {stages['toolchain']*1000:.1f}ms")

        # ─── Stage 9: Watchdog, Sync, Protection ────────────────────────
        t = time.perf_counter()
        self.watchdog = NeuralWatchdog(device=self.device)
        self.sync = SyncManager(device=self.device)
        self.mpu = MemoryProtectionUnit(
            max_processes=self._config["max_processes"],
            device=self.device,
        )
        stages["security"] = time.perf_counter() - t
        if not quiet:
            logger.info(f"  [BOOT] Security subsystem: {stages['security']*1000:.1f}ms")

        # ─── Stage 10: Load Models (optional) ─────────────────────────
        if load_models:
            t = time.perf_counter()
            models_loaded = self._load_models()
            stages["models"] = time.perf_counter() - t
            if not quiet and models_loaded:
                logger.info(f"  [BOOT] Models: {stages['models']*1000:.1f}ms "
                           f"({models_loaded} loaded)")

        # ─── Stage 11: Online Adaptation (optional) ────────────────
        if online_adaptation:
            t = time.perf_counter()
            try:
                from ncpu.neural.online_adaptation import attach_to_neuros
                self.adaptation_manager = attach_to_neuros(self)
                stages["adaptation"] = time.perf_counter() - t
                if not quiet:
                    logger.info(f"  [BOOT] Online adaptation: {stages['adaptation']*1000:.1f}ms")
            except Exception as e:
                stages["adaptation"] = 0.0
                if not quiet:
                    logger.warning(f"  [BOOT] Online adaptation failed: {e}")

        self._boot_time = time.perf_counter() - t_start
        self._booted = True
        stages["total"] = self._boot_time

        if not quiet:
            logger.info(f"  [BOOT] neurOS ready in {self._boot_time*1000:.1f}ms")

        return stages

    def _init_filesystem(self):
        """Create standard directory structure."""
        for d in ["/bin", "/dev", "/etc", "/home", "/proc", "/tmp", "/var"]:
            self.fs.mkdir(d)

        # Create /etc/hostname
        hostname_data = torch.tensor(
            [ord(c) for c in "neuros\n"],
            dtype=torch.uint8, device=self.device,
        )
        self.fs.write_file("/etc/hostname", hostname_data)

        # Create /etc/motd
        motd = "Welcome to neurOS - the GPU-native neural operating system.\n"
        motd_data = torch.tensor(
            [ord(c) for c in motd],
            dtype=torch.uint8, device=self.device,
        )
        self.fs.write_file("/etc/motd", motd_data)

    def _load_models(self) -> int:
        """Attempt to load trained neural models for all components."""
        count = 0
        models_dir = Path("models/os")

        if self.mmu.load(str(models_dir / "mmu.pt")):
            count += 1
        if self.tlb.load(str(models_dir / "tlb.pt")):
            count += 1
        loaded = self.cache.load(
            str(models_dir / "cache_replace.pt"),
            str(models_dir / "prefetch.pt"),
        )
        count += len(loaded)
        if self.gic.load(str(models_dir / "gic.pt")):
            count += 1
        if self.scheduler.load(str(models_dir / "scheduler.pt")):
            count += 1
        if self.fs.load(str(models_dir / "block_alloc.pt")):
            count += 1
        loaded = self.assembler.load(
            str(models_dir / "assembler_tokenizer.pt"),
            str(models_dir / "assembler_codegen.pt"),
        )
        count += len(loaded)
        if self.compiler.load(str(models_dir / "compiler_optimizer.pt")):
            count += 1
        if self.watchdog.load(str(models_dir / "watchdog.pt")):
            count += 1

        return count

    # ─── Default IRQ Handlers ─────────────────────────────────────────────

    def _handle_timer_irq(self, irq: int):
        """Timer interrupt: advance scheduler tick."""
        current = self.process_table.running_process()
        if current:
            self.scheduler.tick_process(current)

    def _handle_syscall_irq(self, irq: int):
        """Syscall interrupt: dispatch through syscall interface."""
        pass  # Syscalls are dispatched directly, not through GIC

    def _handle_page_fault_irq(self, irq: int):
        """Page fault: allocate a new page (demand paging)."""
        pass  # Handled by MMU.translate()

    # ─── System Status ────────────────────────────────────────────────────

    def status(self) -> Dict:
        """Get complete system status."""
        return {
            "booted": self._booted,
            "boot_time_ms": self._boot_time * 1000,
            "device": str(self.device),
            "mmu": self.mmu.stats() if self.mmu else None,
            "tlb": self.tlb.stats() if self.tlb else None,
            "cache": self.cache.stats() if self.cache else None,
            "gic": self.gic.stats() if self.gic else None,
            "scheduler": self.scheduler.stats() if self.scheduler else None,
            "processes": self.process_table.stats() if self.process_table else None,
            "ipc": self.ipc.stats() if self.ipc else None,
            "filesystem": self.fs.stats() if self.fs else None,
            "syscalls": self.syscalls.stats() if self.syscalls else None,
            "assembler": self.assembler.stats() if self.assembler else None,
            "compiler": self.compiler.stats() if self.compiler else None,
            "watchdog": self.watchdog.stats() if self.watchdog else None,
        }

    def _boot_banner(self):
        """Print boot banner."""
        logger.info("╔══════════════════════════════════════════════════════════╗")
        logger.info("║  neurOS v0.1 — GPU-Native Neural Operating System       ║")
        logger.info("║  Every component is a trained neural network on GPU     ║")
        logger.info(f"║  Device: {str(self.device):<48}║")
        logger.info("╚══════════════════════════════════════════════════════════╝")

    def __repr__(self) -> str:
        if not self._booted:
            return "NeurOS(not booted)"
        return (f"NeurOS(device={self.device}, "
                f"procs={self.process_table.count}, "
                f"boot={self._boot_time*1000:.0f}ms)")
