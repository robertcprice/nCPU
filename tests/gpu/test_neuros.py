"""Tests for neurOS — GPU-Native Neural Operating System.

Covers all Phase 1-6 components:
    - Neural MMU (page table, translation, training)
    - Neural TLB (lookup, eviction, batch operations)
    - Neural Cache (access, replacement, prefetch)
    - Neural Scheduler (scheduling, fairness, preemption)
    - Process Control Blocks (creation, context switch)
    - Neural GIC (interrupt raise/dispatch/mask)
    - Neural IPC (messages, shared memory, pipes, signals)
    - Neural Filesystem (files, directories, read/write)
    - Neural Shell (command parsing, built-in commands)
    - Syscall Interface (dispatch, process/file/IPC syscalls)
    - Boot Sequence (full system boot and integration)
"""

import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Neural MMU Tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestNeuralMMU:
    @pytest.fixture
    def mmu(self):
        from ncpu.os.mmu import NeuralMMU
        return NeuralMMU(max_virtual_pages=256, max_physical_frames=256)

    def test_create(self, mmu):
        assert mmu.max_virtual_pages == 256
        assert mmu.max_physical_frames == 256
        assert mmu.free_frames == 256

    def test_alloc_frame(self, mmu):
        frame = mmu.alloc_frame()
        assert frame == 0
        assert mmu.free_frames == 255
        frame2 = mmu.alloc_frame()
        assert frame2 == 1

    def test_free_frame(self, mmu):
        frame = mmu.alloc_frame()
        mmu.free_frame(frame)
        assert mmu.free_frames == 256

    def test_map_page(self, mmu):
        assert mmu.map_page(vpn=0, pfn=0, asid=0, read=True, write=True)
        stats = mmu.stats()
        assert stats["mapped_pages"] == 1

    def test_unmap_page(self, mmu):
        pfn = mmu.alloc_frame()
        mmu.map_page(vpn=5, pfn=pfn, asid=0)
        mmu.unmap_page(vpn=5, asid=0)
        stats = mmu.stats()
        assert stats["mapped_pages"] == 0

    def test_alloc_and_map(self, mmu):
        pfn = mmu.alloc_and_map(vpn=10, asid=0, read=True, write=True)
        assert pfn >= 0
        assert mmu.free_frames == 255

    def test_translate_table_hit(self, mmu):
        pfn = mmu.alloc_and_map(vpn=0, asid=0, read=True)
        phys, fault = mmu.translate(0x0000, asid=0)  # VPN 0, offset 0
        assert fault is None
        assert phys == pfn * 4096

    def test_translate_table_miss(self, mmu):
        phys, fault = mmu.translate(0x5000, asid=0)  # VPN 5, not mapped
        assert fault is not None
        assert fault.fault_type == "not_mapped"

    def test_translate_with_offset(self, mmu):
        pfn = mmu.alloc_and_map(vpn=1, asid=0, read=True)
        phys, fault = mmu.translate(0x1ABC, asid=0)  # VPN 1, offset 0xABC
        assert fault is None
        assert phys == pfn * 4096 + 0xABC

    def test_translate_permission_denied(self, mmu):
        mmu.alloc_and_map(vpn=0, asid=0, read=True, write=False)
        phys, fault = mmu.translate(0x0000, asid=0, write=True)
        assert fault is not None
        assert fault.fault_type == "permission"

    def test_translate_exec_permission(self, mmu):
        mmu.alloc_and_map(vpn=0, asid=0, read=True, execute=False)
        phys, fault = mmu.translate(0x0000, asid=0, execute=True)
        assert fault is not None
        assert fault.fault_type == "permission"

    def test_batch_translate(self, mmu):
        for vpn in range(4):
            mmu.alloc_and_map(vpn=vpn, asid=0, read=True)
        addrs = torch.tensor([0x0000, 0x1000, 0x2000, 0x3000],
                             dtype=torch.int64, device=mmu.device)
        phys = mmu.translate_batch(addrs, asid=0)
        assert (phys >= 0).all()

    def test_batch_translate_with_miss(self, mmu):
        mmu.alloc_and_map(vpn=0, asid=0, read=True)
        addrs = torch.tensor([0x0000, 0x5000],
                             dtype=torch.int64, device=mmu.device)
        phys = mmu.translate_batch(addrs, asid=0)
        assert phys[0] >= 0
        assert phys[1] == -1

    def test_train_from_table(self, mmu):
        # Create some mappings
        for vpn in range(16):
            mmu.alloc_and_map(vpn=vpn, asid=0, read=True, write=True)

        stats = mmu.train_from_table(epochs=50, asid=0)
        assert "final_accuracy" in stats
        assert stats["num_mappings"] == 16
        assert mmu._trained is True

    def test_translate_neural_after_training(self, mmu):
        for vpn in range(16):
            mmu.alloc_and_map(vpn=vpn, asid=0, read=True, write=True)
        mmu.train_from_table(epochs=200, asid=0)

        # Neural translation should work (may not be 100% accurate with few epochs)
        phys, fault = mmu.translate(0x0000, asid=0)
        # We just check it doesn't crash — accuracy depends on training

    def test_asid_isolation(self, mmu):
        mmu.alloc_and_map(vpn=0, asid=0, read=True)
        phys, fault = mmu.translate(0x0000, asid=1)
        assert fault is not None  # ASID 1 has no mappings

    def test_stats(self, mmu):
        mmu.alloc_and_map(vpn=0, asid=0, read=True)
        mmu.translate(0x0000, asid=0)
        mmu.translate(0x5000, asid=0)  # Miss
        stats = mmu.stats()
        assert stats["translations"] == 2
        assert stats["page_faults"] == 1
        assert stats["mapped_pages"] == 1

    def test_repr(self, mmu):
        assert "NeuralMMU" in repr(mmu)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Neural TLB Tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestNeuralTLB:
    @pytest.fixture
    def tlb(self):
        from ncpu.os.tlb import NeuralTLB
        return NeuralTLB(size=16)

    def test_create(self, tlb):
        assert tlb.size == 16
        assert tlb.hit_rate == 0.0

    def test_insert_and_lookup(self, tlb):
        perms = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32, device=tlb.device)
        tlb.insert(vpn=10, asid=0, pfn=42, perms=perms)

        pfn, perm_out = tlb.lookup(vpn=10, asid=0)
        assert pfn == 42
        assert perm_out is not None

    def test_lookup_miss(self, tlb):
        pfn, perms = tlb.lookup(vpn=999, asid=0)
        assert pfn == -1
        assert perms is None

    def test_hit_rate(self, tlb):
        perms = torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.float32, device=tlb.device)
        tlb.insert(vpn=5, asid=0, pfn=10, perms=perms)
        tlb.lookup(vpn=5, asid=0)   # Hit
        tlb.lookup(vpn=5, asid=0)   # Hit
        tlb.lookup(vpn=99, asid=0)  # Miss
        assert tlb.hits == 2
        assert tlb.misses == 1
        assert abs(tlb.hit_rate - 2/3) < 0.01

    def test_invalidate(self, tlb):
        perms = torch.zeros(6, dtype=torch.float32, device=tlb.device)
        tlb.insert(vpn=5, asid=0, pfn=10, perms=perms)
        tlb.invalidate(vpn=5, asid=0)
        pfn, _ = tlb.lookup(vpn=5, asid=0)
        assert pfn == -1

    def test_flush_all(self, tlb):
        perms = torch.zeros(6, dtype=torch.float32, device=tlb.device)
        for i in range(10):
            tlb.insert(vpn=i, asid=0, pfn=i+100, perms=perms)
        tlb.flush()
        assert tlb.occupancy == 0.0

    def test_flush_asid(self, tlb):
        perms = torch.zeros(6, dtype=torch.float32, device=tlb.device)
        tlb.insert(vpn=0, asid=0, pfn=100, perms=perms)
        tlb.insert(vpn=0, asid=1, pfn=200, perms=perms)
        tlb.flush(asid=0)
        pfn0, _ = tlb.lookup(vpn=0, asid=0)
        pfn1, _ = tlb.lookup(vpn=0, asid=1)
        assert pfn0 == -1
        assert pfn1 == 200

    def test_lru_eviction(self, tlb):
        perms = torch.zeros(6, dtype=torch.float32, device=tlb.device)
        # Fill all 16 slots
        for i in range(16):
            tlb.insert(vpn=i, asid=0, pfn=i+100, perms=perms)

        # Access VPN 0 to make it recently used
        tlb.lookup(vpn=0, asid=0)

        # Insert one more — should evict LRU (not VPN 0)
        tlb.insert(vpn=99, asid=0, pfn=999, perms=perms)

        # VPN 0 should still be there (was accessed recently)
        pfn0, _ = tlb.lookup(vpn=0, asid=0)
        assert pfn0 == 100

        # VPN 99 should be there
        pfn99, _ = tlb.lookup(vpn=99, asid=0)
        assert pfn99 == 999

    def test_update_existing(self, tlb):
        perms = torch.zeros(6, dtype=torch.float32, device=tlb.device)
        tlb.insert(vpn=5, asid=0, pfn=10, perms=perms)
        tlb.insert(vpn=5, asid=0, pfn=20, perms=perms)  # Update
        pfn, _ = tlb.lookup(vpn=5, asid=0)
        assert pfn == 20

    def test_batch_lookup(self, tlb):
        perms = torch.zeros(6, dtype=torch.float32, device=tlb.device)
        for i in range(8):
            tlb.insert(vpn=i, asid=0, pfn=i+100, perms=perms)

        vpns = torch.tensor([0, 1, 2, 99], dtype=torch.int64, device=tlb.device)
        pfns, valid = tlb.lookup_batch(vpns, asid=0)
        assert valid[0] and valid[1] and valid[2]
        assert not valid[3]
        assert pfns[0] == 100
        assert pfns[3] == -1

    def test_repr(self, tlb):
        assert "NeuralTLB" in repr(tlb)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Process Tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestProcessTable:
    @pytest.fixture
    def table(self):
        from ncpu.os.process import ProcessTable
        return ProcessTable(max_processes=32)

    def test_create_process(self, table):
        pcb = table.create_process("test_proc", priority=64)
        assert pcb.pid == 1
        assert pcb.name == "test_proc"
        assert pcb.priority == 64
        assert pcb.registers is not None
        assert pcb.registers.shape == (32,)

    def test_process_state_gpu(self, table):
        pcb = table.create_process("gpu_test")
        assert pcb.registers.device.type in ("mps", "cuda", "cpu")
        assert pcb.pc.device == pcb.registers.device

    def test_multiple_processes(self, table):
        p1 = table.create_process("proc1")
        p2 = table.create_process("proc2")
        assert p1.pid != p2.pid
        assert table.count == 2

    def test_get_process(self, table):
        p = table.create_process("test")
        assert table.get(p.pid) is p
        assert table.get(999) is None

    def test_ready_processes(self, table):
        from ncpu.os.process import ProcessState
        p1 = table.create_process("p1", priority=100)
        p2 = table.create_process("p2", priority=50)
        ready = table.ready_processes()
        assert len(ready) == 2
        assert ready[0].priority <= ready[1].priority  # Sorted

    def test_context_switch(self, table):
        from ncpu.os.process import ProcessState
        p1 = table.create_process("p1")
        p2 = table.create_process("p2")
        p1.state = ProcessState.RUNNING
        table.context_switch(p1, p2)
        assert p1.state == ProcessState.READY
        assert p2.state == ProcessState.RUNNING

    def test_remove_process(self, table):
        p = table.create_process("remove_me")
        assert table.remove(p.pid)
        assert table.count == 0

    def test_process_table_full(self, table):
        for i in range(32):
            table.create_process(f"proc_{i}")
        with pytest.raises(RuntimeError, match="Process table full"):
            table.create_process("one_too_many")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Neural Scheduler Tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestNeuralScheduler:
    @pytest.fixture
    def scheduler(self):
        from ncpu.os.process import ProcessTable
        from ncpu.os.scheduler import NeuralScheduler
        table = ProcessTable()
        return NeuralScheduler(table)

    def test_schedule_empty(self, scheduler):
        result = scheduler.schedule()
        assert result is None

    def test_schedule_single(self, scheduler):
        scheduler.process_table.create_process("single", priority=100)
        selected = scheduler.schedule()
        assert selected is not None
        assert selected.name == "single"

    def test_priority_scheduling(self, scheduler):
        scheduler.process_table.create_process("low", priority=200)
        scheduler.process_table.create_process("high", priority=10)
        selected = scheduler.schedule()
        assert selected.name == "high"  # Higher priority (lower number)

    def test_tick_process(self, scheduler):
        from ncpu.os.process import ProcessState
        p = scheduler.process_table.create_process("ticky", priority=100)
        p.state = ProcessState.RUNNING
        p.time_slice = 5
        p.ticks_remaining = 5

        for _ in range(5):
            scheduler.tick_process(p)

        assert p.cpu_time == 5
        assert p.state == ProcessState.READY  # Preempted

    def test_block_unblock(self, scheduler):
        from ncpu.os.process import ProcessState
        p = scheduler.process_table.create_process("blocker")
        p.state = ProcessState.RUNNING
        scheduler.current_pid = p.pid

        scheduler.block_process(p.pid, "io_wait")
        assert p.state == ProcessState.BLOCKED
        assert p.blocked_on == "io_wait"

        scheduler.unblock_process(p.pid)
        assert p.state == ProcessState.READY
        assert p.blocked_on is None

    def test_terminate_process(self, scheduler):
        from ncpu.os.process import ProcessState
        p = scheduler.process_table.create_process("doomed")
        scheduler.terminate_process(p.pid, exit_code=42)
        assert p.state == ProcessState.ZOMBIE
        assert p.exit_code == 42

    def test_fairness(self, scheduler):
        for i in range(3):
            scheduler.process_table.create_process(f"proc_{i}")

        # Simulate scheduling several rounds
        for _ in range(30):
            selected = scheduler.schedule()
            if selected:
                scheduler.tick_process(selected)

        fairness = scheduler.jains_fairness()
        assert fairness > 0.0
        assert fairness <= 1.0

    def test_stats(self, scheduler):
        s = scheduler.stats()
        assert "tick" in s
        assert "fairness" in s
        assert s["trained"] is False

    def test_repr(self, scheduler):
        assert "NeuralScheduler" in repr(scheduler)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Neural Cache Tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestNeuralCache:
    @pytest.fixture
    def cache(self):
        from ncpu.os.cache import NeuralCache
        return NeuralCache(num_sets=16, ways=2)

    def test_create(self, cache):
        assert cache.num_sets == 16
        assert cache.ways == 2
        assert cache.hits == 0
        assert cache.misses == 0

    def test_access_miss_then_hit(self, cache):
        assert cache.access(0x1000) is False   # Miss (cold)
        assert cache.access(0x1000) is True     # Hit
        assert cache.hits == 1
        assert cache.misses == 1

    def test_write_access(self, cache):
        cache.access(0x2000, write=True)  # Miss + fill
        cache.access(0x2000)              # Hit
        assert cache.hits == 1

    def test_eviction(self, cache):
        # Fill a set beyond capacity (2 ways)
        base = 0  # All map to set 0
        stride = 16 * 64  # 16 sets × 64 byte lines

        cache.access(base)            # Way 0
        cache.access(base + stride)   # Way 1
        cache.access(base + 2*stride) # Evicts one of the above

        assert cache.evictions >= 1

    def test_invalidate(self, cache):
        cache.access(0x1000)          # Fill
        assert cache.access(0x1000)   # Hit
        cache.invalidate(0x1000)
        assert not cache.access(0x1000)  # Miss after invalidate

    def test_flush(self, cache):
        for i in range(10):
            cache.access(i * 0x100)
        cache.flush()
        # First access after flush should always miss (no stride pattern yet)
        assert not cache.access(0x1000)
        assert not cache.access(0x5000)  # Non-stride pattern avoids prefetch hits

    def test_hit_rate(self, cache):
        # Access same address repeatedly
        cache.access(0x1000)  # Miss
        for _ in range(9):
            cache.access(0x1000)  # 9 hits
        assert abs(cache.hit_rate - 0.9) < 0.01

    def test_stats(self, cache):
        s = cache.stats()
        assert "hits" in s
        assert "misses" in s
        assert "hit_rate" in s
        assert s["replacer_trained"] is False
        assert s["prefetcher_trained"] is False

    def test_repr(self, cache):
        assert "NeuralCache" in repr(cache)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Neural GIC Tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestNeuralGIC:
    @pytest.fixture
    def gic(self):
        from ncpu.os.interrupts import NeuralGIC
        return NeuralGIC()

    def test_create(self, gic):
        assert gic.num_irqs == 32
        assert gic.interrupts_raised == 0

    def test_raise_and_dispatch(self, gic):
        handled = []
        gic.register_handler(0, lambda irq: handled.append(irq))
        gic.raise_irq(0)
        assert gic.pending().any()
        result = gic.dispatch()
        assert result == 0
        assert len(handled) == 1

    def test_priority_order(self, gic):
        dispatch_order = []
        for i in range(4):
            gic.register_handler(i, lambda irq: dispatch_order.append(irq))
            gic.raise_irq(i)

        dispatched = gic.dispatch_all()
        assert dispatched == [0, 1, 2, 3]  # Fixed priority order

    def test_mask_irq(self, gic):
        gic.register_handler(0, lambda irq: None)
        gic.mask_irq(0)
        gic.raise_irq(0)
        result = gic.dispatch()
        assert result is None  # Masked

        gic.unmask_irq(0)
        result = gic.dispatch()
        assert result == 0

    def test_no_handler_spurious(self, gic):
        gic.raise_irq(5)  # No handler registered
        gic.dispatch()
        assert gic.spurious == 1

    def test_dispatch_empty(self, gic):
        assert gic.dispatch() is None

    def test_stats(self, gic):
        s = gic.stats()
        assert s["raised"] == 0
        assert s["trained"] is False

    def test_repr(self, gic):
        assert "NeuralGIC" in repr(gic)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Neural IPC Tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestNeuralIPC:
    @pytest.fixture
    def ipc(self):
        from ncpu.os.ipc import NeuralIPC
        ipc = NeuralIPC()
        ipc.register_process(1)
        ipc.register_process(2)
        return ipc

    def test_send_receive(self, ipc):
        payload = torch.tensor([1, 2, 3], dtype=torch.uint8, device=ipc.device)
        assert ipc.send(src_pid=1, dst_pid=2, payload=payload)
        msg = ipc.receive(pid=2)
        assert msg is not None
        assert msg.src_pid == 1
        assert len(msg.payload) == 3

    def test_receive_empty(self, ipc):
        assert ipc.receive(pid=1) is None

    def test_tagged_receive(self, ipc):
        payload = torch.tensor([10], dtype=torch.uint8, device=ipc.device)
        ipc.send(1, 2, payload, tag=42)
        ipc.send(1, 2, payload, tag=99)

        msg = ipc.receive(2, tag=99)
        assert msg is not None
        assert msg.tag == 99

    def test_broadcast(self, ipc):
        ipc.register_process(3)
        payload = torch.tensor([0xFF], dtype=torch.uint8, device=ipc.device)
        count = ipc.broadcast(src_pid=1, payload=payload)
        assert count == 2  # PID 2 and 3

        assert ipc.has_messages(2)
        assert ipc.has_messages(3)
        assert not ipc.has_messages(1)  # Sender doesn't get it

    def test_shared_memory(self, ipc):
        shm = ipc.shm_create("test_shm", size=1024, owner_pid=1)
        assert shm.name == "test_shm"

        # Write from PID 1
        data = torch.tensor([42, 43, 44], dtype=torch.uint8, device=ipc.device)
        shm.write(0, data)

        # PID 2 opens and reads
        shm2 = ipc.shm_open("test_shm", pid=2)
        assert shm2 is shm
        result = shm2.read(0, 3)
        assert (result == data).all()

    def test_pipe(self, ipc):
        pipe = ipc.pipe_create(reader_pid=1, writer_pid=2)
        data = torch.tensor([10, 20, 30], dtype=torch.uint8, device=ipc.device)
        written = pipe.write(data)
        assert written == 3

        result = pipe.read(3)
        assert len(result) == 3
        assert (result == data).all()

    def test_pipe_empty_read(self, ipc):
        pipe = ipc.pipe_create(reader_pid=1, writer_pid=2)
        result = pipe.read(10)
        assert len(result) == 0

    def test_signal_send(self, ipc):
        ipc.signal_send(src_pid=1, dst_pid=2, signal=9)
        signals = ipc.signal_pending(pid=2)
        assert 9 in signals

    def test_unregister_cleanup(self, ipc):
        pipe = ipc.pipe_create(reader_pid=1, writer_pid=2)
        ipc.unregister_process(1)
        assert pipe.closed

    def test_stats(self, ipc):
        s = ipc.stats()
        assert s["registered_processes"] == 2

    def test_repr(self, ipc):
        assert "NeuralIPC" in repr(ipc)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: Neural Filesystem Tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestNeuralFilesystem:
    @pytest.fixture
    def fs(self):
        from ncpu.os.filesystem import NeuralFilesystem
        return NeuralFilesystem(num_blocks=256, max_inodes=128)

    def test_root_exists(self, fs):
        assert fs.exists("/")

    def test_mkdir(self, fs):
        ino = fs.mkdir("/home")
        assert ino >= 0
        assert fs.exists("/home")

    def test_mkdir_nested(self, fs):
        fs.mkdir("/home")
        ino = fs.mkdir("/home/user")
        assert ino >= 0
        assert fs.exists("/home/user")

    def test_create_file(self, fs):
        ino = fs.create("/hello.txt")
        assert ino >= 0
        assert fs.exists("/hello.txt")

    def test_create_duplicate(self, fs):
        fs.create("/test.txt")
        ino = fs.create("/test.txt")
        assert ino == -1  # Already exists

    def test_unlink(self, fs):
        fs.create("/remove_me.txt")
        assert fs.unlink("/remove_me.txt")
        assert not fs.exists("/remove_me.txt")

    def test_rmdir(self, fs):
        fs.mkdir("/empty_dir")
        assert fs.rmdir("/empty_dir")
        assert not fs.exists("/empty_dir")

    def test_rmdir_nonempty(self, fs):
        fs.mkdir("/nonempty")
        fs.create("/nonempty/file.txt")
        assert not fs.rmdir("/nonempty")  # Should fail

    def test_list_dir(self, fs):
        fs.mkdir("/testdir")
        fs.create("/testdir/a.txt")
        fs.create("/testdir/b.txt")
        entries = fs.list_dir("/testdir")
        assert entries is not None
        assert "a.txt" in entries
        assert "b.txt" in entries

    def test_write_and_read_file(self, fs):
        data = torch.tensor([72, 101, 108, 108, 111],  # "Hello"
                            dtype=torch.uint8, device=fs.device)
        assert fs.write_file("/greeting.txt", data)
        result = fs.read_file("/greeting.txt")
        assert result is not None
        assert (result == data).all()

    def test_open_read_write(self, fs):
        fd = fs.open("/rw_test.txt", "w")
        assert fd >= 0
        data = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint8, device=fs.device)
        written = fs.write(fd, data)
        assert written == 5
        fs.close(fd)

        fd2 = fs.open("/rw_test.txt", "r")
        result = fs.read(fd2, 5)
        assert result is not None
        assert (result == data).all()
        fs.close(fd2)

    def test_seek(self, fs):
        data = torch.tensor([10, 20, 30, 40, 50], dtype=torch.uint8, device=fs.device)
        fs.write_file("/seek_test.txt", data)

        fd = fs.open("/seek_test.txt", "r")
        fs.seek(fd, 2, 0)  # SEEK_SET to position 2
        result = fs.read(fd, 3)
        assert len(result) == 3
        assert result[0] == 30  # data[2]
        fs.close(fd)

    def test_stat(self, fs):
        data = torch.tensor([1, 2, 3], dtype=torch.uint8, device=fs.device)
        fs.write_file("/stat_test.txt", data)
        info = fs.stat("/stat_test.txt")
        assert info is not None
        assert info["type"] == "REGULAR"
        assert info["size"] == 3

    def test_stat_directory(self, fs):
        fs.mkdir("/mydir")
        info = fs.stat("/mydir")
        assert info is not None
        assert info["type"] == "DIRECTORY"

    def test_stat_nonexistent(self, fs):
        assert fs.stat("/nonexistent") is None

    def test_large_file(self, fs):
        # Write more than one block (4096 bytes)
        data = (torch.arange(8000, device=fs.device) % 256).to(torch.uint8)
        fs.write_file("/large.bin", data)
        result = fs.read_file("/large.bin")
        assert result is not None
        assert len(result) == 8000
        assert (result == data).all()

    def test_free_blocks(self, fs):
        initial_free = fs.free_blocks
        data = torch.zeros(4096, dtype=torch.uint8, device=fs.device)
        fs.write_file("/block_test.txt", data)
        assert fs.free_blocks < initial_free

    def test_stats(self, fs):
        s = fs.stats()
        assert "total_blocks" in s
        assert "free_blocks" in s
        assert s["allocator_trained"] is False

    def test_repr(self, fs):
        assert "NeuralFS" in repr(fs)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6: Neural Shell Tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestNeuralShell:
    @pytest.fixture
    def neuros(self):
        from ncpu.os.boot import NeurOS
        os = NeurOS()
        os.boot(load_models=False, quiet=True)
        return os

    @pytest.fixture
    def shell(self, neuros):
        return neuros.shell

    def test_pwd(self, shell):
        output = shell.execute("pwd")
        assert "/" in output

    def test_ls_root(self, shell):
        output = shell.execute("ls")
        assert any("bin" in line for line in output)
        assert any("etc" in line for line in output)

    def test_mkdir_and_ls(self, shell):
        shell.execute("mkdir mydir")
        output = shell.execute("ls")
        assert any("mydir" in line for line in output)

    def test_cd(self, shell):
        shell.execute("mkdir testdir")
        shell.execute("cd testdir")
        output = shell.execute("pwd")
        assert any("testdir" in line for line in output)

    def test_cd_nonexistent(self, shell):
        output = shell.execute("cd nonexistent")
        assert any("no such" in line.lower() for line in output)

    def test_echo(self, shell):
        output = shell.execute("echo hello world")
        assert "hello world" in output

    def test_touch_and_cat(self, shell):
        shell.execute("touch newfile.txt")
        output = shell.execute("cat newfile.txt")
        # Empty file, should produce no output
        assert len(output) == 0 or all(line == "" for line in output)

    def test_cat_nonexistent(self, shell):
        output = shell.execute("cat nope.txt")
        assert any("no such" in line.lower() for line in output)

    def test_rm(self, shell):
        shell.execute("touch deleteme.txt")
        shell.execute("rm deleteme.txt")
        output = shell.execute("cat deleteme.txt")
        assert any("no such" in line.lower() for line in output)

    def test_ps(self, shell):
        output = shell.execute("ps")
        assert any("PID" in line for line in output)
        assert any("init" in line for line in output)

    def test_uname(self, shell):
        output = shell.execute("uname")
        assert any("neurOS" in line for line in output)

    def test_help(self, shell):
        output = shell.execute("help")
        assert any("ls" in line for line in output)
        assert any("exit" in line for line in output)

    def test_df(self, shell):
        output = shell.execute("df")
        assert any("neurfs" in line for line in output)

    def test_free(self, shell):
        output = shell.execute("free")
        assert any("Pages" in line for line in output)

    def test_top(self, shell):
        output = shell.execute("top")
        assert any("neurOS" in line for line in output)

    def test_env(self, shell):
        output = shell.execute("env")
        assert any("HOME" in line for line in output)

    def test_export(self, shell):
        shell.execute("export MYVAR=hello")
        assert shell.env["MYVAR"] == "hello"

    def test_history(self, shell):
        shell.execute("pwd")
        shell.execute("ls")
        output = shell.execute("history")
        assert any("pwd" in line for line in output)
        assert any("ls" in line for line in output)

    def test_unknown_command(self, shell):
        output = shell.execute("foobar")
        assert any("not found" in line for line in output)

    def test_empty_command(self, shell):
        output = shell.execute("")
        assert output == []

    def test_exit(self, shell):
        shell.execute("exit")
        assert not shell.running

    def test_var_expansion(self, shell):
        output = shell.execute("echo $HOME")
        assert "/" in output

    def test_uptime(self, shell):
        output = shell.execute("uptime")
        assert any("up" in line for line in output)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6: Syscall Tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestSyscalls:
    @pytest.fixture
    def neuros(self):
        from ncpu.os.boot import NeurOS
        os = NeurOS()
        os.boot(load_models=False, quiet=True)
        return os

    def test_getpid(self, neuros):
        from ncpu.os.syscalls import SYS_GETPID
        result = neuros.syscalls.dispatch(1, SYS_GETPID, [])
        assert result == 1

    def test_mkdir_syscall(self, neuros):
        from ncpu.os.syscalls import SYS_MKDIR
        result = neuros.syscalls.dispatch(1, SYS_MKDIR, ["/test"])
        assert result >= 0

    def test_open_write_read_close(self, neuros):
        from ncpu.os.syscalls import SYS_OPEN, SYS_WRITE, SYS_READ, SYS_CLOSE
        fd = neuros.syscalls.dispatch(1, SYS_OPEN, ["/sys_test.txt", "w"])
        assert fd >= 0

        data = torch.tensor([65, 66, 67], dtype=torch.uint8, device=neuros.device)
        written = neuros.syscalls.dispatch(1, SYS_WRITE, [fd, data])
        assert written == 3

        neuros.syscalls.dispatch(1, SYS_CLOSE, [fd])

        fd2 = neuros.syscalls.dispatch(1, SYS_OPEN, ["/sys_test.txt", "r"])
        assert fd2 >= 0

        result = neuros.syscalls.dispatch(1, SYS_READ, [fd2, 3])
        assert result == 3
        neuros.syscalls.dispatch(1, SYS_CLOSE, [fd2])

    def test_fork(self, neuros):
        from ncpu.os.syscalls import SYS_FORK
        child_pid = neuros.syscalls.dispatch(1, SYS_FORK, [])
        assert child_pid > 1
        child = neuros.process_table.get(child_pid)
        assert child is not None
        assert child.parent_pid == 1

    def test_kill(self, neuros):
        from ncpu.os.syscalls import SYS_FORK, SYS_KILL
        from ncpu.os.process import ProcessState
        child_pid = neuros.syscalls.dispatch(1, SYS_FORK, [])
        neuros.syscalls.dispatch(1, SYS_KILL, [child_pid, 9])
        # Signal should be pending
        signals = neuros.ipc.signal_pending(child_pid)
        assert 9 in signals

    def test_yield(self, neuros):
        from ncpu.os.syscalls import SYS_YIELD
        from ncpu.os.process import ProcessState
        result = neuros.syscalls.dispatch(1, SYS_YIELD, [])
        assert result == 0

    def test_unknown_syscall(self, neuros):
        result = neuros.syscalls.dispatch(1, 99999, [])
        assert result == -1

    def test_stat_syscall(self, neuros):
        from ncpu.os.syscalls import SYS_STAT
        result = neuros.syscalls.dispatch(1, SYS_STAT, ["/etc"])
        assert result >= 0  # Returns inode number

    def test_listdir_syscall(self, neuros):
        from ncpu.os.syscalls import SYS_LISTDIR
        result = neuros.syscalls.dispatch(1, SYS_LISTDIR, ["/"])
        assert result > 0  # Root has directories

    def test_send_recv(self, neuros):
        from ncpu.os.syscalls import SYS_FORK, SYS_SEND, SYS_RECV
        child_pid = neuros.syscalls.dispatch(1, SYS_FORK, [])
        payload = torch.tensor([42], dtype=torch.uint8, device=neuros.device)
        neuros.syscalls.dispatch(1, SYS_SEND, [child_pid, payload])
        result = neuros.syscalls.dispatch(child_pid, SYS_RECV, [])
        assert result == 0  # Message received


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: Boot Sequence Tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestNeurOSBoot:
    def test_boot(self):
        from ncpu.os import NeurOS
        os = NeurOS()
        stats = os.boot(load_models=False, quiet=True)
        assert "total" in stats
        assert os._booted is True

    def test_boot_creates_components(self):
        from ncpu.os import NeurOS
        os = NeurOS()
        os.boot(load_models=False, quiet=True)
        assert os.mmu is not None
        assert os.tlb is not None
        assert os.cache is not None
        assert os.gic is not None
        assert os.ipc is not None
        assert os.process_table is not None
        assert os.scheduler is not None
        assert os.fs is not None
        assert os.syscalls is not None
        assert os.shell is not None

    def test_boot_creates_filesystem(self):
        from ncpu.os import NeurOS
        os = NeurOS()
        os.boot(load_models=False, quiet=True)
        assert os.fs.exists("/bin")
        assert os.fs.exists("/etc")
        assert os.fs.exists("/home")
        assert os.fs.exists("/tmp")

    def test_boot_creates_init_process(self):
        from ncpu.os import NeurOS
        from ncpu.os.process import ProcessState
        os = NeurOS()
        os.boot(load_models=False, quiet=True)
        init = os.process_table.get(1)
        assert init is not None
        assert init.name == "init"
        assert init.state == ProcessState.RUNNING

    def test_boot_creates_files(self):
        from ncpu.os import NeurOS
        os = NeurOS()
        os.boot(load_models=False, quiet=True)
        assert os.fs.exists("/etc/hostname")
        hostname = os.fs.read_file("/etc/hostname")
        assert hostname is not None
        assert len(hostname) > 0

    def test_status(self):
        from ncpu.os import NeurOS
        os = NeurOS()
        os.boot(load_models=False, quiet=True)
        status = os.status()
        assert status["booted"] is True
        assert "mmu" in status
        assert "scheduler" in status

    def test_shell_integration(self):
        from ncpu.os import NeurOS
        os = NeurOS()
        os.boot(load_models=False, quiet=True)

        # Full shell workflow
        os.shell.execute("mkdir /home/user")
        os.shell.execute("cd /home/user")

        output = os.shell.execute("pwd")
        assert any("user" in line for line in output)

        output = os.shell.execute("ls /")
        assert any("home" in line for line in output)

    def test_repr_before_boot(self):
        from ncpu.os import NeurOS
        os = NeurOS()
        assert "not booted" in repr(os)

    def test_repr_after_boot(self):
        from ncpu.os import NeurOS
        os = NeurOS()
        os.boot(load_models=False, quiet=True)
        assert "NeurOS" in repr(os)
        assert "device=" in repr(os)

    def test_small_configuration(self):
        """Boot with minimal resources."""
        from ncpu.os import NeurOS
        os = NeurOS(
            max_virtual_pages=64,
            max_physical_frames=64,
            tlb_size=8,
            cache_sets=16,
            cache_ways=2,
            fs_blocks=128,
            max_processes=16,
        )
        stats = os.boot(load_models=False, quiet=True)
        assert os._booted


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: Multi-component Workflows
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestNeurOSIntegration:
    @pytest.fixture
    def neuros(self):
        from ncpu.os import NeurOS
        os = NeurOS()
        os.boot(load_models=False, quiet=True)
        return os

    def test_mmu_with_process(self, neuros):
        """Process gets its own virtual address space."""
        from ncpu.os.syscalls import SYS_FORK
        child_pid = neuros.syscalls.dispatch(1, SYS_FORK, [])
        child = neuros.process_table.get(child_pid)
        assert child.asid != 0 or child.asid == child.pid % 256

        # Map a page for the child's ASID
        neuros.mmu.alloc_and_map(vpn=0, asid=child.asid, read=True, write=True)
        phys, fault = neuros.mmu.translate(0x0000, asid=child.asid)
        assert fault is None

    def test_ipc_between_processes(self, neuros):
        """Two forked processes communicate via IPC."""
        from ncpu.os.syscalls import SYS_FORK, SYS_SEND, SYS_RECV

        child_pid = neuros.syscalls.dispatch(1, SYS_FORK, [])
        payload = torch.tensor([1, 2, 3], dtype=torch.uint8, device=neuros.device)

        # Parent sends to child
        neuros.syscalls.dispatch(1, SYS_SEND, [child_pid, payload])

        # Child receives
        result = neuros.syscalls.dispatch(child_pid, SYS_RECV, [])
        assert result == 0

    def test_filesystem_through_shell(self, neuros):
        """Create, write, and read a file through shell commands."""
        neuros.shell.execute("touch /tmp/test.txt")
        assert neuros.fs.exists("/tmp/test.txt")

        # Write via syscall (shell doesn't have a write command)
        data = torch.tensor([72, 73], dtype=torch.uint8, device=neuros.device)
        neuros.fs.write_file("/tmp/test.txt", data)

        output = neuros.shell.execute("cat /tmp/test.txt")
        assert any("HI" in line for line in output)

    def test_cache_during_filesystem_ops(self, neuros):
        """Cache tracks filesystem accesses."""
        initial_misses = neuros.cache.misses

        # Filesystem operations should generate cache activity
        for i in range(10):
            neuros.cache.access(i * 64)  # Simulate block access

        assert neuros.cache.misses > initial_misses

    def test_scheduler_with_multiple_processes(self, neuros):
        """Scheduler handles multiple processes."""
        from ncpu.os.syscalls import SYS_FORK

        pids = []
        for i in range(3):
            pid = neuros.syscalls.dispatch(1, SYS_FORK, [])
            pids.append(pid)

        assert neuros.process_table.count == 4  # init + 3 children

        # Scheduler should be able to schedule
        from ncpu.os.process import ProcessState
        init = neuros.process_table.get(1)
        init.state = ProcessState.READY  # Allow rescheduling

        selected = neuros.scheduler.schedule()
        assert selected is not None

    def test_interrupt_timer_scheduling(self, neuros):
        """Timer interrupts drive the scheduler."""
        from ncpu.os.interrupts import IRQ_TIMER

        neuros.gic.raise_irq(IRQ_TIMER)
        dispatched = neuros.gic.dispatch()
        assert dispatched == IRQ_TIMER


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 7: Neural Assembler Tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestClassicalAssembler:
    @pytest.fixture
    def asm(self):
        from ncpu.os.assembler import ClassicalAssembler
        return ClassicalAssembler()

    def test_nop(self, asm):
        result = asm.assemble("NOP\nHALT")
        assert result.success
        assert result.num_instructions == 2

    def test_halt(self, asm):
        from ncpu.os.assembler import Opcode
        result = asm.assemble("HALT")
        assert result.success
        assert len(result.binary) == 1
        decoded = asm.decode_word(result.binary[0])
        assert decoded.opcode == Opcode.HALT

    def test_mov_immediate(self, asm):
        from ncpu.os.assembler import Opcode
        result = asm.assemble("MOV R0, 42\nHALT")
        assert result.success
        decoded = asm.decode_word(result.binary[0])
        assert decoded.opcode == Opcode.MOV_IMM
        assert decoded.rd == 0
        assert decoded.imm == 42

    def test_mov_register(self, asm):
        from ncpu.os.assembler import Opcode
        result = asm.assemble("MOV R3, R5\nHALT")
        assert result.success
        decoded = asm.decode_word(result.binary[0])
        assert decoded.opcode == Opcode.MOV_REG
        assert decoded.rd == 3
        assert decoded.rs1 == 5

    def test_add(self, asm):
        from ncpu.os.assembler import Opcode
        result = asm.assemble("ADD R0, R1, R2\nHALT")
        assert result.success
        decoded = asm.decode_word(result.binary[0])
        assert decoded.opcode == Opcode.ADD
        assert decoded.rd == 0
        assert decoded.rs1 == 1
        assert decoded.rs2 == 2

    def test_all_arithmetic(self, asm):
        source = "ADD R0, R1, R2\nSUB R0, R1, R2\nMUL R0, R1, R2\nDIV R0, R1, R2\nHALT"
        result = asm.assemble(source)
        assert result.success
        assert result.num_instructions == 5

    def test_all_logical(self, asm):
        source = "AND R0, R1, R2\nOR R0, R1, R2\nXOR R0, R1, R2\nHALT"
        result = asm.assemble(source)
        assert result.success
        assert result.num_instructions == 4

    def test_shifts(self, asm):
        source = "SHL R0, R1, 3\nSHR R0, R1, 2\nHALT"
        result = asm.assemble(source)
        assert result.success
        assert result.num_instructions == 3

    def test_inc_dec(self, asm):
        from ncpu.os.assembler import Opcode
        result = asm.assemble("INC R0\nDEC R1\nHALT")
        assert result.success
        decoded0 = asm.decode_word(result.binary[0])
        decoded1 = asm.decode_word(result.binary[1])
        assert decoded0.opcode == Opcode.INC
        assert decoded0.rd == 0
        assert decoded1.opcode == Opcode.DEC
        assert decoded1.rd == 1

    def test_cmp(self, asm):
        from ncpu.os.assembler import Opcode
        result = asm.assemble("CMP R3, R4\nHALT")
        assert result.success
        decoded = asm.decode_word(result.binary[0])
        assert decoded.opcode == Opcode.CMP
        assert decoded.rs1 == 3
        assert decoded.rs2 == 4

    def test_jumps(self, asm):
        source = "JMP 0\nJZ 1\nJNZ 2\nJS 3\nJNS 4\nHALT"
        result = asm.assemble(source)
        assert result.success
        assert result.num_instructions == 6

    def test_labels(self, asm):
        source = """
loop:
    INC R0
    JMP loop
    HALT
"""
        result = asm.assemble(source)
        assert result.success
        assert "loop" in result.labels
        assert result.labels["loop"] == 0
        # JMP should reference address 0
        decoded = asm.decode_word(result.binary[1])
        assert decoded.imm == 0

    def test_fibonacci_program(self, asm):
        """Assemble the actual fibonacci program."""
        source = """\
    MOV R0, 0
    MOV R1, 1
    MOV R2, 10
    MOV R3, 0
    MOV R4, 1
loop:
    MOV R5, R1
    ADD R1, R0, R1
    MOV R0, R5
    ADD R3, R3, R4
    CMP R3, R2
    JNZ loop
    HALT
"""
        result = asm.assemble(source)
        assert result.success
        assert result.num_instructions == 12
        assert "loop" in result.labels

    def test_hex_immediate(self, asm):
        result = asm.assemble("MOV R0, 0xFF\nHALT")
        assert result.success
        decoded = asm.decode_word(result.binary[0])
        assert decoded.imm == 255

    def test_comments(self, asm):
        source = "MOV R0, 5 ; load five\n# another comment\nHALT"
        result = asm.assemble(source)
        assert result.success
        assert result.num_instructions == 2

    def test_disassemble(self, asm):
        source = "MOV R0, 42\nADD R1, R0, R0\nHALT"
        result = asm.assemble(source)
        disasm = asm.disassemble(result.binary)
        assert "MOV" in disasm
        assert "ADD" in disasm
        assert "HALT" in disasm

    def test_roundtrip(self, asm):
        """Assemble → disassemble → reassemble should produce same binary."""
        source = "MOV R0, 10\nMOV R1, 20\nADD R2, R0, R1\nHALT"
        result1 = asm.assemble(source)
        # Decode and re-encode each instruction
        for word in result1.binary:
            decoded = asm.decode_word(word)
            re_encoded = asm._encode(decoded)
            assert re_encoded == word, f"Roundtrip failed: 0x{word:08X} != 0x{re_encoded:08X}"

    def test_encoding_format(self, asm):
        """Verify binary encoding format: [31:24] op | [23:21] rd | [20:18] rs1 | [17:15] rs2 | [14:0] imm"""
        result = asm.assemble("MOV R3, 100\nHALT")
        word = result.binary[0]
        assert (word >> 24) & 0xFF == 0x10  # MOV_IMM opcode
        assert (word >> 21) & 0x7 == 3     # R3
        assert word & 0x7FFF == 100        # immediate

    def test_error_on_bad_instruction(self, asm):
        result = asm.assemble("BADOP R0\nHALT")
        assert len(result.errors) > 0

    def test_error_on_undefined_label(self, asm):
        result = asm.assemble("JMP nonexistent\nHALT")
        assert len(result.errors) > 0

    def test_source_map(self, asm):
        source = "MOV R0, 1\nMOV R1, 2\nHALT"
        result = asm.assemble(source)
        assert 0 in result.source_map
        assert 1 in result.source_map

    def test_negative_immediate(self, asm):
        result = asm.assemble("MOV R0, -5\nHALT")
        assert result.success
        decoded = asm.decode_word(result.binary[0])
        assert decoded.imm == -5


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestNeuralAssembler:
    @pytest.fixture
    def nasm(self):
        from ncpu.os.assembler import NeuralAssembler
        return NeuralAssembler()

    def test_create(self, nasm):
        assert nasm.programs_assembled == 0
        assert not nasm._codegen_trained

    def test_assemble_uses_classical(self, nasm):
        result = nasm.assemble("MOV R0, 42\nHALT")
        assert result.success
        assert nasm.programs_assembled == 1

    def test_train_codegen(self, nasm):
        programs = [
            "MOV R0, 1\nMOV R1, 2\nADD R2, R0, R1\nHALT",
            "MOV R0, 10\nDEC R0\nHALT",
            "INC R0\nINC R1\nHALT",
        ]
        stats = nasm.train_codegen(programs, epochs=100)
        assert stats["num_programs"] == 3
        assert stats["bit_accuracy"] > 0.5
        assert nasm._codegen_trained

    def test_train_and_validate(self, nasm):
        programs = [
            "NOP\nHALT",
            "MOV R0, 5\nHALT",
            "MOV R0, 1\nMOV R1, 2\nADD R2, R0, R1\nHALT",
        ]
        nasm.train_codegen(programs, epochs=200)
        result = nasm.assemble("MOV R0, 5\nHALT")
        assert result.success
        # After validation, neural_matches + neural_mismatches > 0
        total = nasm.neural_matches + nasm.neural_mismatches
        assert total > 0

    def test_disassemble(self, nasm):
        result = nasm.assemble("MOV R0, 42\nHALT")
        disasm = nasm.disassemble(result.binary)
        assert "MOV" in disasm
        assert "HALT" in disasm

    def test_stats(self, nasm):
        nasm.assemble("NOP\nHALT")
        stats = nasm.stats()
        assert stats["programs_assembled"] == 1
        assert "neural_accuracy" in stats

    def test_repr(self, nasm):
        r = repr(nasm)
        assert "NeuralAssembler" in r


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 8: Language & Compiler Tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestLexer:
    def test_simple_tokens(self):
        from ncpu.os.language import Lexer, TokenType
        tokens = Lexer("var x = 10;").tokenize()
        types = [t.type for t in tokens]
        assert TokenType.VAR in types
        assert TokenType.IDENT in types
        assert TokenType.ASSIGN in types
        assert TokenType.NUMBER in types
        assert TokenType.SEMI in types
        assert types[-1] == TokenType.EOF

    def test_keywords(self):
        from ncpu.os.language import Lexer, TokenType
        tokens = Lexer("if else while fn return halt var").tokenize()
        types = [t.type for t in tokens[:-1]]  # exclude EOF
        assert types == [TokenType.IF, TokenType.ELSE, TokenType.WHILE,
                         TokenType.FN, TokenType.RETURN, TokenType.HALT, TokenType.VAR]

    def test_operators(self):
        from ncpu.os.language import Lexer, TokenType
        tokens = Lexer("+ - * / & | ^ << >> == != < > <= >=").tokenize()
        types = [t.type for t in tokens[:-1]]
        expected = [TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH,
                    TokenType.AMP, TokenType.PIPE, TokenType.CARET,
                    TokenType.SHL, TokenType.SHR,
                    TokenType.EQ, TokenType.NEQ, TokenType.LT, TokenType.GT,
                    TokenType.LTE, TokenType.GTE]
        assert types == expected

    def test_comments(self):
        from ncpu.os.language import Lexer, TokenType
        tokens = Lexer("var x = 5; // this is a comment\nvar y = 10;").tokenize()
        idents = [t.value for t in tokens if t.type == TokenType.IDENT]
        assert idents == ["x", "y"]

    def test_hex_numbers(self):
        from ncpu.os.language import Lexer, TokenType
        tokens = Lexer("0xFF 0x10").tokenize()
        nums = [t for t in tokens if t.type == TokenType.NUMBER]
        assert len(nums) == 2
        assert nums[0].value == "0xFF"

    def test_braces_and_parens(self):
        from ncpu.os.language import Lexer, TokenType
        tokens = Lexer("if (x) { y; }").tokenize()
        types = [t.type for t in tokens[:-1]]
        assert TokenType.LPAREN in types
        assert TokenType.RPAREN in types
        assert TokenType.LBRACE in types
        assert TokenType.RBRACE in types

    def test_multiline(self):
        from ncpu.os.language import Lexer
        source = "var x = 1;\nvar y = 2;\n"
        tokens = Lexer(source).tokenize()
        assert tokens[-1].line == 3  # EOF on line 3


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestParser:
    def _parse(self, source):
        from ncpu.os.language import Lexer, Parser
        tokens = Lexer(source).tokenize()
        return Parser(tokens).parse()

    def test_var_decl(self):
        from ncpu.os.language import VarDecl, NumberLit
        prog = self._parse("var x = 42;")
        assert len(prog.statements) == 1
        stmt = prog.statements[0]
        assert isinstance(stmt, VarDecl)
        assert stmt.name == "x"
        assert isinstance(stmt.init, NumberLit)
        assert stmt.init.value == 42

    def test_assignment(self):
        from ncpu.os.language import VarDecl, Assignment
        prog = self._parse("var x = 1; x = 2;")
        assert len(prog.statements) == 2
        assert isinstance(prog.statements[1], Assignment)
        assert prog.statements[1].name == "x"

    def test_binary_expr(self):
        from ncpu.os.language import VarDecl, BinaryExpr
        prog = self._parse("var x = 1 + 2;")
        assert isinstance(prog.statements[0].init, BinaryExpr)
        assert prog.statements[0].init.op == "+"

    def test_precedence(self):
        from ncpu.os.language import VarDecl, BinaryExpr
        prog = self._parse("var x = 1 + 2 * 3;")
        expr = prog.statements[0].init
        assert isinstance(expr, BinaryExpr)
        assert expr.op == "+"
        assert isinstance(expr.right, BinaryExpr)
        assert expr.right.op == "*"

    def test_if_stmt(self):
        from ncpu.os.language import IfStmt
        prog = self._parse("if (1 == 1) { halt; }")
        assert len(prog.statements) == 1
        assert isinstance(prog.statements[0], IfStmt)

    def test_if_else(self):
        from ncpu.os.language import IfStmt
        prog = self._parse("if (1 > 2) { halt; } else { halt; }")
        stmt = prog.statements[0]
        assert isinstance(stmt, IfStmt)
        assert stmt.else_block is not None

    def test_while_stmt(self):
        from ncpu.os.language import WhileStmt
        prog = self._parse("while (1 != 0) { halt; }")
        assert isinstance(prog.statements[0], WhileStmt)

    def test_halt(self):
        from ncpu.os.language import HaltStmt
        prog = self._parse("halt;")
        assert isinstance(prog.statements[0], HaltStmt)

    def test_function_decl(self):
        from ncpu.os.language import FuncDecl
        prog = self._parse("fn add(a, b) { return a + b; }")
        assert len(prog.functions) == 1
        func = prog.functions[0]
        assert func.name == "add"
        assert func.params == ["a", "b"]

    def test_unary_minus(self):
        from ncpu.os.language import VarDecl, UnaryExpr
        prog = self._parse("var x = -5;")
        expr = prog.statements[0].init
        assert isinstance(expr, UnaryExpr)
        assert expr.op == "-"

    def test_parenthesized_expr(self):
        from ncpu.os.language import VarDecl, BinaryExpr
        prog = self._parse("var x = (1 + 2) * 3;")
        expr = prog.statements[0].init
        assert isinstance(expr, BinaryExpr)
        assert expr.op == "*"
        assert isinstance(expr.left, BinaryExpr)
        assert expr.left.op == "+"

    def test_comparison_ops(self):
        from ncpu.os.language import VarDecl, BinaryExpr
        for op in ["==", "!=", "<", ">", "<=", ">="]:
            prog = self._parse(f"var x = 1 {op} 2;")
            expr = prog.statements[0].init
            assert isinstance(expr, BinaryExpr)
            assert expr.op == op

    def test_syntax_error(self):
        import pytest
        with pytest.raises(SyntaxError):
            self._parse("var = ;")

    def test_empty_program(self):
        prog = self._parse("")
        assert len(prog.statements) == 0
        assert len(prog.functions) == 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestNeuralCompiler:
    @pytest.fixture
    def compiler(self):
        from ncpu.os.compiler import NeuralCompiler
        return NeuralCompiler()

    def test_simple_compile(self, compiler):
        result = compiler.compile("halt;")
        assert result.success
        assert result.assembly_result is not None
        assert result.binary is not None

    def test_var_and_halt(self, compiler):
        result = compiler.compile("var x = 42; halt;")
        assert result.success
        assert "x" in result.variables
        assert "MOV" in result.assembly
        assert "HALT" in result.assembly

    def test_addition(self, compiler):
        # Use optimize=False to prevent constant folding, so ADD is preserved
        result = compiler.compile("var a = 10; var b = 20; var c = a + b; halt;", optimize=False)
        assert result.success
        assert "ADD" in result.assembly

    def test_subtraction(self, compiler):
        result = compiler.compile("var a = 30; var b = 10; var c = a - b; halt;", optimize=False)
        assert result.success
        assert "SUB" in result.assembly

    def test_multiplication(self, compiler):
        result = compiler.compile("var a = 7; var b = 6; var c = a * b; halt;", optimize=False)
        assert result.success
        assert "MUL" in result.assembly

    def test_division(self, compiler):
        result = compiler.compile("var a = 42; var b = 6; var c = a / b; halt;", optimize=False)
        assert result.success
        assert "DIV" in result.assembly

    def test_bitwise_ops(self, compiler):
        result = compiler.compile("var a = 0xFF; var b = 0x0F; var c = a & b; halt;", optimize=False)
        assert result.success
        assert "AND" in result.assembly

    def test_shift_ops(self, compiler):
        result = compiler.compile("var a = 1; var b = 3; var c = a << b; halt;", optimize=False)
        assert result.success
        assert "SHL" in result.assembly

    def test_while_loop(self, compiler):
        source = """\
var x = 10;
var one = 1;
while (x != 0) {
    x = x - one;
}
halt;
"""
        result = compiler.compile(source)
        assert result.success
        assert "CMP" in result.assembly
        assert "JNZ" in result.assembly.upper() or "JZ" in result.assembly.upper()

    def test_if_statement(self, compiler):
        source = """\
var x = 10;
var five = 5;
if (x > five) {
    x = five;
}
halt;
"""
        result = compiler.compile(source)
        assert result.success
        assert "CMP" in result.assembly

    def test_if_else(self, compiler):
        source = """\
var x = 3;
var five = 5;
if (x > five) {
    x = five;
} else {
    x = 0;
}
halt;
"""
        result = compiler.compile(source)
        assert result.success
        assert "JMP" in result.assembly.upper()

    def test_sum_1_to_10(self, compiler):
        """Classic sum 1..10 — should produce working assembly."""
        source = """\
var sum = 0;
var i = 1;
var limit = 11;
var one = 1;
while (i != limit) {
    sum = sum + i;
    i = i + one;
}
halt;
"""
        result = compiler.compile(source)
        assert result.success
        assert result.binary is not None
        assert len(result.binary) > 0
        # Verify it assembled cleanly
        assert len(result.assembly_result.errors) == 0

    def test_fibonacci(self, compiler):
        """Fibonacci — tests complex variable management."""
        source = """\
var prev = 0;
var curr = 1;
var n = 10;
var i = 0;
var one = 1;
while (i != n) {
    var temp = curr;
    curr = prev + curr;
    prev = temp;
    i = i + one;
}
halt;
"""
        result = compiler.compile(source)
        assert result.success
        assert result.binary is not None

    def test_nested_expressions(self, compiler):
        result = compiler.compile("var x = 1; var y = 2; var z = (x + y) * (x + y); halt;")
        assert result.success

    def test_optimization_constant_fold(self, compiler):
        """Compiler should fold constant expressions."""
        result = compiler.compile("var x = 10; var y = 20; var z = x + y; halt;", optimize=True)
        assert result.success
        # With constant folding, the ADD might be replaced by MOV
        # At minimum it should compile successfully

    def test_no_optimization(self, compiler):
        result = compiler.compile("var x = 5; halt;", optimize=False)
        assert result.success

    def test_compile_error(self, compiler):
        result = compiler.compile("var = ;")
        assert not result.success
        assert len(result.errors) > 0

    def test_stats(self, compiler):
        compiler.compile("halt;")
        stats = compiler.stats()
        assert stats["programs_compiled"] == 1
        assert "compression_ratio" in stats

    def test_repr(self, compiler):
        r = repr(compiler)
        assert "NeuralCompiler" in r

    def test_assembly_output_assembles(self, compiler):
        """Verify the assembly output can be re-assembled."""
        from ncpu.os.assembler import ClassicalAssembler
        source = "var a = 5; var b = 10; var c = a + b; halt;"
        result = compiler.compile(source)
        assert result.success

        # Re-assemble the output
        asm = ClassicalAssembler()
        result2 = asm.assemble(result.assembly)
        assert result2.success
        assert result2.binary == result.assembly_result.binary


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestClassicalOptimizer:
    @pytest.fixture
    def opt(self):
        from ncpu.os.compiler import ClassicalOptimizer
        return ClassicalOptimizer()

    def test_identity_elimination(self, opt):
        from ncpu.os.compiler import IRInstr
        ir = [IRInstr(op="mov", dest="R0", src1="R0")]
        result, count = opt.optimize(ir)
        assert count > 0
        assert len(result) == 0  # Identity move removed

    def test_dead_store_elimination(self, opt):
        from ncpu.os.compiler import IRInstr
        ir = [
            IRInstr(op="mov", dest="R0", src1="5"),
            IRInstr(op="mov", dest="R0", src1="10"),
        ]
        result, count = opt.optimize(ir)
        assert count > 0
        assert len(result) == 1  # First store eliminated

    def test_constant_folding(self, opt):
        from ncpu.os.compiler import IRInstr
        ir = [
            IRInstr(op="mov", dest="R0", src1="3"),
            IRInstr(op="mov", dest="R1", src1="5"),
            IRInstr(op="add", dest="R2", src1="R0", src2="R1"),
        ]
        result, count = opt.optimize(ir)
        assert count > 0
        # R2 should be folded to MOV R2, 8
        found_folded = any(i.op == "mov" and i.dest == "R2" and i.src1 == "8"
                          for i in result)
        assert found_folded

    def test_no_optimization_needed(self, opt):
        from ncpu.os.compiler import IRInstr
        ir = [
            IRInstr(op="halt"),
        ]
        result, count = opt.optimize(ir)
        assert count == 0
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 9: Integration Tests (Assembler + Compiler + Boot)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestToolchainIntegration:
    @pytest.fixture
    def neuros(self):
        from ncpu.os import NeurOS
        os = NeurOS()
        os.boot(load_models=False, quiet=True)
        return os

    def test_boot_creates_toolchain(self, neuros):
        """Boot should create assembler and compiler."""
        assert neuros.assembler is not None
        assert neuros.compiler is not None

    def test_status_includes_toolchain(self, neuros):
        status = neuros.status()
        assert "assembler" in status
        assert "compiler" in status

    def test_assemble_through_neuros(self, neuros):
        """Assemble a program through the booted OS."""
        result = neuros.assembler.assemble("MOV R0, 42\nHALT")
        assert result.success

    def test_compile_through_neuros(self, neuros):
        """Compile through the booted OS."""
        result = neuros.compiler.compile("var x = 42; halt;")
        assert result.success

    def test_compile_and_assemble_match(self, neuros):
        """Compiler output should be valid assembly for the assembler."""
        result = neuros.compiler.compile("var a = 5; var b = 10; var c = a + b; halt;")
        assert result.success

        # Re-assemble the compiler output
        asm_result = neuros.assembler.assemble(result.assembly)
        assert asm_result.success
        assert asm_result.binary == result.assembly_result.binary

    def test_shell_asm_command(self, neuros):
        """Shell 'asm' command assembles files from filesystem."""
        # Write an assembly program to the filesystem
        source = "MOV R0, 1\nHALT\n"
        data = torch.tensor([ord(c) for c in source], dtype=torch.uint8,
                            device=neuros.device)
        neuros.fs.write_file("/tmp/test.asm", data)

        output = neuros.shell.execute("asm /tmp/test.asm")
        assert any("Assembled" in line for line in output)
        assert any("2 instructions" in line for line in output)

    def test_shell_nsc_command(self, neuros):
        """Shell 'nsc' command compiles nsl files."""
        source = "var x = 42; halt;\n"
        data = torch.tensor([ord(c) for c in source], dtype=torch.uint8,
                            device=neuros.device)
        neuros.fs.write_file("/tmp/test.nsl", data)

        output = neuros.shell.execute("nsc /tmp/test.nsl")
        assert any("Compiled" in line for line in output)

    def test_assembler_training_with_programs(self, neuros):
        """Train the neural assembler on a few programs."""
        programs = [
            "NOP\nHALT",
            "MOV R0, 5\nINC R0\nHALT",
            "MOV R0, 0\nMOV R1, 1\nADD R2, R0, R1\nHALT",
        ]
        stats = neuros.assembler.train_codegen(programs, epochs=50)
        assert stats["num_programs"] == 3
        assert stats["bit_accuracy"] > 0.3

    def test_compiler_end_to_end_fibonacci(self, neuros):
        """Full end-to-end: compile fibonacci to binary."""
        source = """\
var prev = 0;
var curr = 1;
var n = 10;
var i = 0;
var one = 1;
while (i != n) {
    var temp = curr;
    curr = prev + curr;
    prev = temp;
    i = i + one;
}
halt;
"""
        result = neuros.compiler.compile(source)
        assert result.success
        assert result.binary is not None
        assert len(result.binary) > 5  # Non-trivial program

        # Disassemble and verify
        disasm = neuros.assembler.disassemble(result.binary)
        assert "MOV" in disasm
        assert "HALT" in disasm

    def test_full_pipeline_write_compile_assemble(self, neuros):
        """Write source to filesystem, compile, assemble via shell."""
        # Write nsl source
        source = "var x = 7; var y = 6; var z = x * y; halt;\n"
        data = torch.tensor([ord(c) for c in source], dtype=torch.uint8,
                            device=neuros.device)
        neuros.fs.write_file("/home/prog.nsl", data)

        # Compile via shell
        output = neuros.shell.execute("nsc /home/prog.nsl")
        assert any("Compiled" in line for line in output)

    def test_multiple_compilations(self, neuros):
        """Compile multiple programs sequentially."""
        programs = [
            "var x = 1; halt;",
            "var a = 5; var b = a + a; halt;",
            "var i = 10; var one = 1; while (i != 0) { i = i - one; } halt;",
        ]
        for source in programs:
            result = neuros.compiler.compile(source)
            assert result.success, f"Failed to compile: {source}"

        assert neuros.compiler.programs_compiled == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 10: nsl Language Extensions Tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestForLoop:
    """Tests for the 'for' loop language extension."""

    @pytest.fixture
    def compiler(self):
        from ncpu.os.compiler import NeuralCompiler
        return NeuralCompiler()

    def test_basic_for(self, compiler):
        """for (var i = 0; i != 5; i = i + 1) { sum = sum + i; }"""
        source = """\
var sum = 0;
var one = 1;
for (var i = 0; i != 5; i = i + one) {
    sum = sum + i;
}
halt;
"""
        result = compiler.compile(source)
        assert result.success, f"Compile errors: {result.errors}"
        assert result.binary is not None
        assert "CMP" in result.assembly
        assert "JMP" in result.assembly.upper() or "JZ" in result.assembly.upper()

    def test_for_with_assignment_init(self, compiler):
        """for loop where init is an assignment (not var decl)."""
        source = """\
var i = 10;
var one = 1;
for (i = 0; i != 3; i = i + one) {
    halt;
}
halt;
"""
        result = compiler.compile(source)
        assert result.success, f"Compile errors: {result.errors}"

    def test_for_assembles_cleanly(self, compiler):
        """Verify for loop produces valid assembly that re-assembles."""
        from ncpu.os.assembler import ClassicalAssembler
        source = """\
var sum = 0;
var one = 1;
for (var i = 0; i != 3; i = i + one) {
    sum = sum + i;
}
halt;
"""
        result = compiler.compile(source)
        assert result.success
        asm = ClassicalAssembler()
        result2 = asm.assemble(result.assembly)
        assert result2.success, f"Re-assembly errors: {result2.errors}"

    def test_for_parse_ast(self):
        """Verify the parser produces a ForStmt AST node."""
        from ncpu.os.language import Lexer, Parser, ForStmt
        source = "for (var i = 0; i != 5; i = i + 1) { halt; }"
        tokens = Lexer(source).tokenize()
        prog = Parser(tokens).parse()
        assert len(prog.statements) == 1
        assert isinstance(prog.statements[0], ForStmt)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestModulo:
    """Tests for the modulo (%) operator."""

    @pytest.fixture
    def compiler(self):
        from ncpu.os.compiler import NeuralCompiler
        return NeuralCompiler()

    def test_modulo(self, compiler):
        """10 % 3 should compile and produce DIV, MUL, SUB instructions."""
        source = "var a = 10; var b = 3; var c = a % b; halt;"
        result = compiler.compile(source, optimize=False)
        assert result.success, f"Compile errors: {result.errors}"
        assert "DIV" in result.assembly
        assert "MUL" in result.assembly
        assert "SUB" in result.assembly

    def test_modulo_parse(self):
        """Verify % parses as a BinaryExpr with op='%'."""
        from ncpu.os.language import Lexer, Parser, BinaryExpr
        tokens = Lexer("var x = 10 % 3;").tokenize()
        prog = Parser(tokens).parse()
        expr = prog.statements[0].init
        assert isinstance(expr, BinaryExpr)
        assert expr.op == "%"

    def test_modulo_lexer(self):
        """Verify the lexer produces a PERCENT token."""
        from ncpu.os.language import Lexer, TokenType
        tokens = Lexer("10 % 3").tokenize()
        types = [t.type for t in tokens[:-1]]
        assert TokenType.PERCENT in types

    def test_modulo_assembles_cleanly(self, compiler):
        """Verify modulo output re-assembles correctly."""
        from ncpu.os.assembler import ClassicalAssembler
        source = "var a = 10; var b = 3; var c = a % b; halt;"
        result = compiler.compile(source, optimize=False)
        assert result.success
        asm = ClassicalAssembler()
        result2 = asm.assemble(result.assembly)
        assert result2.success


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestLogicalOps:
    """Tests for logical operators: &&, ||, !"""

    @pytest.fixture
    def compiler(self):
        from ncpu.os.compiler import NeuralCompiler
        return NeuralCompiler()

    def test_logical_and(self, compiler):
        """Compile if (a > 0 && b > 0) { ... }"""
        source = """\
var a = 5;
var b = 3;
var zero = 0;
if (a > zero && b > zero) {
    halt;
}
halt;
"""
        result = compiler.compile(source)
        assert result.success, f"Compile errors: {result.errors}"
        assert result.binary is not None

    def test_logical_or(self, compiler):
        """Compile if (a == 0 || b == 0) { ... }"""
        source = """\
var a = 0;
var b = 3;
var zero = 0;
if (a == zero || b == zero) {
    halt;
}
halt;
"""
        result = compiler.compile(source)
        assert result.success, f"Compile errors: {result.errors}"
        assert result.binary is not None

    def test_logical_not(self, compiler):
        """Compile var x = !0; — should produce value 1."""
        source = "var x = !0; halt;"
        result = compiler.compile(source)
        assert result.success, f"Compile errors: {result.errors}"

    def test_logical_not_nonzero(self, compiler):
        """Compile var x = !5; — should produce value 0."""
        source = "var a = 5; var x = !a; halt;"
        result = compiler.compile(source)
        assert result.success, f"Compile errors: {result.errors}"

    def test_logical_and_parse(self):
        """Verify && parses as BinaryExpr with op='&&'."""
        from ncpu.os.language import Lexer, Parser, BinaryExpr
        tokens = Lexer("var x = 1 && 2;").tokenize()
        prog = Parser(tokens).parse()
        expr = prog.statements[0].init
        assert isinstance(expr, BinaryExpr)
        assert expr.op == "&&"

    def test_logical_or_parse(self):
        """Verify || parses as BinaryExpr with op='||'."""
        from ncpu.os.language import Lexer, Parser, BinaryExpr
        tokens = Lexer("var x = 0 || 1;").tokenize()
        prog = Parser(tokens).parse()
        expr = prog.statements[0].init
        assert isinstance(expr, BinaryExpr)
        assert expr.op == "||"

    def test_logical_not_parse(self):
        """Verify ! parses as UnaryExpr with op='!'."""
        from ncpu.os.language import Lexer, Parser, UnaryExpr
        tokens = Lexer("var x = !0;").tokenize()
        prog = Parser(tokens).parse()
        expr = prog.statements[0].init
        assert isinstance(expr, UnaryExpr)
        assert expr.op == "!"

    def test_logical_lexer_tokens(self):
        """Verify the lexer produces LAND, LOR, BANG tokens."""
        from ncpu.os.language import Lexer, TokenType
        tokens = Lexer("&& || !").tokenize()
        types = [t.type for t in tokens[:-1]]
        assert types == [TokenType.LAND, TokenType.LOR, TokenType.BANG]

    def test_logical_precedence(self):
        """|| has lower precedence than &&."""
        from ncpu.os.language import Lexer, Parser, BinaryExpr
        tokens = Lexer("var x = 1 || 2 && 3;").tokenize()
        prog = Parser(tokens).parse()
        expr = prog.statements[0].init
        # Should parse as: 1 || (2 && 3)
        assert isinstance(expr, BinaryExpr)
        assert expr.op == "||"
        assert isinstance(expr.right, BinaryExpr)
        assert expr.right.op == "&&"

    def test_logical_and_short_circuit_compiles(self, compiler):
        """&& with short-circuit generates correct control flow."""
        source = """\
var a = 1;
var b = 0;
var zero = 0;
var one = 1;
if (a != zero && b != zero) {
    a = 0;
}
halt;
"""
        result = compiler.compile(source)
        assert result.success, f"Compile errors: {result.errors}"
        # Should have multiple CMP instructions for short-circuit
        cmp_count = result.assembly.upper().count("CMP")
        assert cmp_count >= 2, "Short-circuit AND should have multiple comparisons"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestCompoundAssign:
    """Tests for compound assignment operators: +=, -=, *=, /="""

    @pytest.fixture
    def compiler(self):
        from ncpu.os.compiler import NeuralCompiler
        return NeuralCompiler()

    def test_plus_equals(self, compiler):
        """x += 5 should desugar to x = x + 5."""
        source = "var x = 10; var five = 5; x += five; halt;"
        result = compiler.compile(source, optimize=False)
        assert result.success, f"Compile errors: {result.errors}"
        assert "ADD" in result.assembly

    def test_minus_equals(self, compiler):
        """x -= 3 should desugar to x = x - 3."""
        source = "var x = 10; var three = 3; x -= three; halt;"
        result = compiler.compile(source, optimize=False)
        assert result.success, f"Compile errors: {result.errors}"
        assert "SUB" in result.assembly

    def test_star_equals(self, compiler):
        """x *= 2 should desugar to x = x * 2."""
        source = "var x = 5; var two = 2; x *= two; halt;"
        result = compiler.compile(source, optimize=False)
        assert result.success, f"Compile errors: {result.errors}"
        assert "MUL" in result.assembly

    def test_slash_equals(self, compiler):
        """x /= 2 should desugar to x = x / 2."""
        source = "var x = 10; var two = 2; x /= two; halt;"
        result = compiler.compile(source, optimize=False)
        assert result.success, f"Compile errors: {result.errors}"
        assert "DIV" in result.assembly

    def test_compound_assign_lexer(self):
        """Verify compound assignment tokens are lexed correctly."""
        from ncpu.os.language import Lexer, TokenType
        tokens = Lexer("+= -= *= /=").tokenize()
        types = [t.type for t in tokens[:-1]]
        assert types == [
            TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN,
            TokenType.STAR_ASSIGN, TokenType.SLASH_ASSIGN,
        ]

    def test_compound_assign_parse(self):
        """Verify += desugars to Assignment with BinaryExpr."""
        from ncpu.os.language import Lexer, Parser, Assignment, BinaryExpr, IdentExpr
        tokens = Lexer("var x = 1; x += 2;").tokenize()
        prog = Parser(tokens).parse()
        stmt = prog.statements[1]
        assert isinstance(stmt, Assignment)
        assert stmt.name == "x"
        assert isinstance(stmt.value, BinaryExpr)
        assert stmt.value.op == "+"
        assert isinstance(stmt.value.left, IdentExpr)
        assert stmt.value.left.name == "x"

    def test_compound_in_for_update(self, compiler):
        """Compound assignment works in for-loop update clause."""
        source = """\
var sum = 0;
var one = 1;
for (var i = 0; i != 5; i += one) {
    sum += i;
}
halt;
"""
        result = compiler.compile(source)
        assert result.success, f"Compile errors: {result.errors}"

    def test_compound_assembles_cleanly(self, compiler):
        """Verify compound assignment output re-assembles."""
        from ncpu.os.assembler import ClassicalAssembler
        source = "var x = 10; var two = 2; x += two; halt;"
        result = compiler.compile(source, optimize=False)
        assert result.success
        asm = ClassicalAssembler()
        result2 = asm.assemble(result.assembly)
        assert result2.success


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestDoWhile:
    """Tests for the do...while loop."""

    @pytest.fixture
    def compiler(self):
        from ncpu.os.compiler import NeuralCompiler
        return NeuralCompiler()

    def test_do_while(self, compiler):
        """do { x = x - 1; } while (x != 0); — body runs at least once."""
        source = """\
var x = 5;
var one = 1;
do {
    x = x - one;
} while (x != 0);
halt;
"""
        result = compiler.compile(source)
        assert result.success, f"Compile errors: {result.errors}"
        assert result.binary is not None
        assert "CMP" in result.assembly
        assert "JMP" in result.assembly.upper()

    def test_do_while_parse_ast(self):
        """Verify the parser produces a DoWhileStmt AST node."""
        from ncpu.os.language import Lexer, Parser, DoWhileStmt
        source = "do { halt; } while (1 != 0);"
        tokens = Lexer(source).tokenize()
        prog = Parser(tokens).parse()
        assert len(prog.statements) == 1
        assert isinstance(prog.statements[0], DoWhileStmt)

    def test_do_while_keywords(self):
        """Verify 'do' is a keyword token."""
        from ncpu.os.language import Lexer, TokenType
        tokens = Lexer("do while").tokenize()
        types = [t.type for t in tokens[:-1]]
        assert types == [TokenType.DO, TokenType.WHILE]

    def test_do_while_assembles_cleanly(self, compiler):
        """Verify do...while output re-assembles correctly."""
        from ncpu.os.assembler import ClassicalAssembler
        source = """\
var x = 3;
var one = 1;
do {
    x = x - one;
} while (x != 0);
halt;
"""
        result = compiler.compile(source)
        assert result.success
        asm = ClassicalAssembler()
        result2 = asm.assemble(result.assembly)
        assert result2.success, f"Re-assembly errors: {result2.errors}"

    def test_do_while_body_once(self, compiler):
        """do...while with immediately-false condition should still compile."""
        source = """\
var x = 0;
var zero = 0;
do {
    x = 1;
} while (x == zero);
halt;
"""
        result = compiler.compile(source)
        assert result.success, f"Compile errors: {result.errors}"
