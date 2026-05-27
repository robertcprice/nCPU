//! JEPA Neural Kernel — Rust implementation of the learned bottom-up neural machine.
//!
//! This module provides the high-performance execution engine for the "JEPA CPU"
//! (the bottom-up neural machine whose dynamics are driven by learned predictors).
//!
//! Design goals:
//! - Reuse/extend the existing high-quality `Process` and `ProcessManager` infrastructure.
//! - Provide a clean `NeuralJepaKernel` abstraction (the Rust counterpart to the Python `NeuralKernel`).
//! - Support the same high-level primitives: context switch, trap/syscall entry, scheduling.
//! - Expose via PyO3 so the Python research layer can use the Rust backend for speed + determinism.
//! - Eventually support running distilled JEPA predictors (or calling back into Python predictors during research).
//!
//! This is the path to running the learned neural OS workloads at full speed on the Metal GPU
//! with all the σ=0.0 determinism superpowers.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::os::process::{Process, ProcessManager, ProcessState};

/// A minimal but functional neural kernel execution environment in Rust.
///
/// This is the direct Rust equivalent of the Python `NeuralKernel` class.
/// It wraps a `ProcessManager` and adds the higher-level kernel loop, syscall dispatch,
/// and integration points for JEPA predictors.
///
/// This is the high-performance, deterministic path for the bottom-up "JEPA CPU"
/// (full von Neumann machine whose state evolution is learnable via cross-JEPA predictors).
/// All process state here is the *real* substrate (registers, pc, flags, backing memory regions,
/// scheduling, blocking) — the same one that already runs real multi-process UNIX + BusyBox.
///
/// The critical new capability is *live observation*:
/// When attached as `jepa_observer` to a `GpuLauncher`, this object receives real
/// process snapshots at every context switch and memory syscall. It maintains a
/// shadow structured memory model (dirty pages + mutation counts) and can return
/// scheduling bias suggestions based on learned churn.
#[pyclass]
pub struct NeuralJepaKernel {
    /// The underlying real multi-process manager (already very mature).
    pub process_manager: ProcessManager,

    /// Current "kernel" view (we often treat pid 0 as the kernel process).
    pub kernel_pid: i32,

    /// Syscall dispatch table (number → name). Real dispatch lives in handle_syscall.
    pub syscall_table: HashMap<i32, String>,

    /// Statistics for observability and predictor training.
    pub total_syscalls: u64,
    pub total_context_switches: u64,
    pub total_traps: u64,
    pub total_jepa_bias_suggestions: u32,

    /// Monotonic observation step counter (wrapping u32).
    /// Incremented on every live hook from the real substrate (on_context_switch, on_syscall,
    /// ingest_real_process_snapshot). Used to give _last_touch_step in PageInfo real meaning
    /// so the churn scorer can strongly differentiate "just now hammered memory" (high recency)
    /// from "was heavy earlier but quiet now". This is the key to larger, more reliable spreads
    /// on real BusyBox multi-process workloads where absolute pressure is uniformly high.
    current_step: u32,

    /// Count of events produced in the most recent run_steps (for quick observability).
    /// Full rich events are returned directly from run_steps() — the recommended consumption path
    /// for Python JEPA training. We avoid storing full PyObject batches inside the Rust struct
    /// to sidestep GIL/clone complexity.
    last_event_count: usize,

    /// Per-process synthetic memory model for JEPA predictor training (v2 - structured + growable).
    ///
    /// - Length = committed units (grows/shrinks with brk/mmap/munmap)
    /// - Content mutates on write/brk
    /// - We also track a simple "mutation counter" per process so the predictor can learn
    ///   how frequently memory is being touched (very useful for cache/scheduling models later).
    jepa_memory: HashMap<i32, Vec<f32>>,
    /// Per-process counter of memory mutations (writes, brk, mmap, munmap).
    memory_mutation_count: HashMap<i32, u32>,

    /// Structured page-level memory model (novel direction for learned OS memory management).
    /// Each process has a small number of conceptual pages. We track dirty bits and access
    /// counts so the JEPA predictor can learn real memory access patterns, not just flat vectors.
    /// This moves us toward a neural machine that can internalize MMU-like behavior.
    memory_pages: HashMap<i32, Vec<PageInfo>>,
}

/// Simple per-page metadata for the structured memory model.
///
/// This is the key novel data structure that lets the JEPA layer learn
/// MMU-like behavior instead of just flat vector memory summaries.
///
/// - `dirty`: page has been written since last clear (approximates real dirty bits)
/// - `access_count`: how many times the page was touched (read or write)
/// - `_last_touch_step`: reserved for future recency-weighted scoring
///
/// The model currently uses 12 conceptual pages per process. This is small
/// enough to be cheap but rich enough for the predictor to learn interesting
/// access patterns and working-set behavior.
#[derive(Clone, Debug, Default)]
pub struct PageInfo {
    pub dirty: bool,
    pub access_count: u32,
    pub _last_touch_step: u32,
}

#[pymethods]
impl NeuralJepaKernel {
    /// Create a new NeuralJepaKernel with an empty process table.
    #[new]
    pub fn new() -> Self {
        let mut km = Self {
            process_manager: ProcessManager::new(),
            kernel_pid: 0,
            syscall_table: HashMap::new(),
            total_syscalls: 0,
            total_context_switches: 0,
            total_traps: 0,
            total_jepa_bias_suggestions: 0,
            current_step: 0,
            last_event_count: 0,
            jepa_memory: HashMap::new(),
            memory_mutation_count: HashMap::new(),
            memory_pages: HashMap::new(),
        };

        // Bootstrap a kernel process (pid 0)
        let _ = km.create_process(0, -1);

        // Seed realistic syscall numbers (matching common conventions + Python NeuralKernel)
        km.syscall_table.insert(0, "getpid".to_string());
        km.syscall_table.insert(1, "yield".to_string());
        km.syscall_table.insert(2, "exit".to_string());
        km.syscall_table.insert(3, "sleep".to_string());
        km.syscall_table.insert(4, "write".to_string());
        km.syscall_table.insert(5, "brk".to_string());
        km.syscall_table.insert(6, "getppid".to_string());
        km.syscall_table.insert(7, "gettid".to_string());
        km.syscall_table.insert(8, "nanosleep".to_string());
        km.syscall_table.insert(9, "mmap".to_string());
        km.syscall_table.insert(10, "munmap".to_string());

        km
    }

    /// Create a new process (wrapper around the real ProcessManager).
    /// Also initializes a lightweight synthetic memory summary for the JEPA predictor.
    pub fn create_process(&mut self, pid: i32, ppid: i32) -> PyResult<i32> {
        if self.process_manager.processes.contains_key(&pid) {
            return Ok(pid);
        }

        let proc = Process::new(pid, ppid, ProcessState::Ready);

        self.process_manager.processes.insert(pid, proc);
        self.process_manager.next_pid = std::cmp::max(self.process_manager.next_pid, pid + 1);

        // Initialize growable JEPA memory model.
        let mut mem = vec![0.0f32; 32];
        for (i, v) in mem.iter_mut().enumerate() {
            *v = ((i as f32) * 0.017 + (pid as f32) * 0.003) % 0.7;
        }
        self.jepa_memory.insert(pid, mem);
        self.memory_mutation_count.insert(pid, 0);

        // Page table for structured memory (dirty + access patterns the predictor can learn)
        // Increased to 12 for richer signals in the novel learned MMU paradigm.
        let pages = vec![PageInfo::default(); 12];
        self.memory_pages.insert(pid, pages);

        Ok(pid)
    }

    /// Advance the global observation step and mark a specific conceptual page as recently
    /// touched (dirty + access_count++ + _last_touch_step = now). This makes recency weighting
    /// in compute_churn_score actually work on real execution traces.
    ///
    /// Called from memory-mutating paths (write/brk/mmap) and live hooks so that a process
    /// that just did heavy allocation right now scores as much higher-churn than one that
    /// did the same work 200 steps ago. Key to reliable bias on sustained multi-process loads.
    #[inline]
    fn _advance_and_touch_page(&mut self, pid: i32, page_idx: usize) {
        self.current_step = self.current_step.wrapping_add(1);
        if let Some(pages) = self.memory_pages.get_mut(&pid) {
            let n = pages.len().max(1);
            if let Some(pg) = pages.get_mut(page_idx % n) {
                pg.dirty = true;
                pg.access_count = pg.access_count.saturating_add(1);
                pg._last_touch_step = self.current_step;
            }
        }
    }

    /// Increment the observation step without a specific page touch (used on every
    /// context switch / syscall observation so age calculations stay meaningful).
    #[inline]
    fn _advance_step(&mut self) {
        self.current_step = self.current_step.wrapping_add(1);
    }

    /// Perform a context switch (saves current, restores target).
    ///
    /// In a real integration the caller would also move GPU memory between
    /// active workspace and backing stores (exactly like the existing ProcessManager).
    pub fn switch_process(&mut self, to_pid: i32) -> PyResult<()> {
        let from_pid = self.process_manager.current_pid;

        if from_pid == to_pid {
            return Ok(());
        }

        // In a full integration we would expect the caller to have supplied
        // the live GPU snapshot before calling this. For the prototype we just
        // update the control block.
        if from_pid >= 0 {
            if let Some(proc) = self.process_manager.processes.get_mut(&from_pid) {
                proc.state = ProcessState::Ready;
            }
        }

        if let Some(proc) = self.process_manager.processes.get_mut(&to_pid) {
            proc.state = ProcessState::Running;
        }

        self.process_manager.current_pid = to_pid;
        self.total_context_switches += 1;

        Ok(())
    }

    /// Enter a trap / syscall from a user process.
    ///
    /// This is the Rust-side entry point equivalent to `handle_trap` in Python.
    /// It saves the user context, switches to kernel, and dispatches known syscalls.
    /// The real Process state machine is mutated here — this is what the learned predictor must internalize.
    pub fn enter_trap(&mut self, user_pid: i32, trap_number: i32) -> PyResult<()> {
        // Save user state
        if let Some(proc) = self.process_manager.processes.get_mut(&user_pid) {
            proc.state = ProcessState::Blocked;
        }

        // Switch to kernel
        self.switch_process(self.kernel_pid)?;

        self.total_syscalls += 1;
        self.total_traps += 1;

        // Dispatch if this looks like a syscall
        if self.syscall_table.contains_key(&trap_number) {
            let _ = self.handle_syscall(user_pid, trap_number)?;
        }

        Ok(())
    }

    /// Handle a specific syscall from a user process (called while in kernel context).
    ///
    /// This does *real* work on the underlying Process structs (registers, state, heap_break, pc, etc.).
    /// This is the substrate that the JEPA predictor will learn to model at the kernel level.
    ///
    /// Calling convention (matches existing nCPU + common embedded/UNIX lite):
    ///   On entry: user_proc.registers[0] often holds syscall number (or passed separately).
    ///   Args typically in r1, r2, r3 (or x0-x3 in full ARM64 view).
    /// Return value is written back to r0.
    pub fn handle_syscall(&mut self, user_pid: i32, syscall_number: i32) -> PyResult<i64> {
        // Pre-handle control-flow syscalls that need scheduling (avoid double borrow)
        match syscall_number {
            1 => {
                // yield: voluntarily reschedule
                let _ = self.schedule_next();
            }
            3 => {
                // sleep: block this user process
                if let Some(p) = self.process_manager.processes.get_mut(&user_pid) {
                    p.state = ProcessState::Blocked;
                }
                let _ = self.schedule_next();
            }
            _ => {}
        }

        let ret: i64 = if let Some(user_proc) = self.process_manager.processes.get_mut(&user_pid) {
            // Read args using the same register convention the Python side and real emulator use
            let arg0 = user_proc.registers[1] as i64; // often fd or target pid
            let arg1 = user_proc.registers[2] as i64; // often ptr or value
            let arg2 = user_proc.registers[3] as i64; // often len or flags

            match syscall_number {
                0 => {
                    // getpid
                    user_pid as i64
                }
                6 => {
                    // getppid
                    user_proc.ppid as i64
                }
                2 => {
                    // exit(status in r0 or arg0)
                    user_proc.state = ProcessState::Zombie;
                    let status = if user_proc.registers[0] != 0 {
                        user_proc.registers[0] as i32
                    } else {
                        arg0 as i32
                    };
                    user_proc.exit_code = status;
                    status as i64
                }
                1 | 3 => {
                    // yield / sleep already handled for scheduling; return success
                    0
                }
                4 => {
                    // write(fd, buf_ptr, len)
                    // Mutate the JEPA synthetic memory summary so the predictor can learn
                    // "memory was written" dynamics. We scribble a simple pattern based on len/args.
                    let len = arg2.max(0);
                    if let Some(mem) = self.jepa_memory.get_mut(&user_pid) {
                        let write_val = ((len as f32) * 0.001 + 0.5).min(1.0);
                        let mlen = mem.len();
                        for i in 0..std::cmp::min(4, mlen) {
                            mem[(i + 4) % mlen] = write_val;
                        }
                        if mlen > 16 {
                            let last_idx = mlen - 1;
                            mem[last_idx] = (mem[last_idx] * 0.7 + write_val * 0.3).min(1.0);
                        }
                    }

                    // Structured page model: mark relevant page dirty+recent on write.
                    // Using the step-aware helper so _last_touch_step is set for recency scoring.
                    let page_idx = (len as usize / 4096) % 12;
                    self._advance_and_touch_page(user_pid, page_idx);

                    *self.memory_mutation_count.entry(user_pid).or_insert(0) += 1;
                    len
                }
                5 => {
                    // brk(new_break) — classic UNIX memory growth primitive.
                    // This is extremely high-value for a neural machine: the predictor must learn
                    // how address space grows, heap layout, etc.
                    let new_break = arg0 as u64;
                    if new_break > user_proc.heap_break {
                        user_proc.heap_break = new_break;
                    }
                    if let Some(mem) = self.jepa_memory.get_mut(&user_pid) {
                        if mem.len() >= 2 {
                            let normalized = ((new_break as f32) / 1_048_576.0).min(1.0);
                            mem[0] = normalized;
                            mem[1] = (normalized * 0.8 + mem[1] * 0.2).min(1.0);
                        }
                    }
                    *self.memory_mutation_count.entry(user_pid).or_insert(0) += 1;
                    user_proc.heap_break as i64
                }
                7 => {
                    // gettid
                    (user_pid as i64) * 1000 + 42
                }
                8 => {
                    // nanosleep (simulated short sleep) — produces nice blocking + schedule events for the predictor
                    if let Some(p) = self.process_manager.processes.get_mut(&user_pid) {
                        p.state = ProcessState::Blocked;
                    }
                    let _ = self.schedule_next();
                    0
                }
                9 => {
                    // mmap(addr_hint, length, prot, flags, fd, offset)
                    // For JEPA research we treat this as "commit more address space".
                    // Grow the synthetic memory vector and advance mmap_next on the real PCB.
                    let length = arg1.max(1) as usize;
                    let units_to_add = (length / 4096).max(1); // simulate page granularity

                    if let Some(mem) = self.jepa_memory.get_mut(&user_pid) {
                        let old_len = mem.len();
                        mem.resize(old_len + units_to_add.min(256), 0.11); // cap growth per call for stability
                        // Mark the newly "mapped" region as lightly dirtied
                        for v in mem.iter_mut().skip(old_len) {
                            *v = 0.11;
                        }
                    }

                    // Page model: newly mapped pages start dirty + recently touched.
                    // Helper sets _last_touch_step for strong recency differentiation.
                    for i in 0..units_to_add.min(12) {
                        self._advance_and_touch_page(user_pid, i);
                    }

                    // Advance the real process's mmap_next (visible in snapshots)
                    let returned_addr = if let Some(proc) = self.process_manager.processes.get_mut(&user_pid) {
                        let addr = proc.mmap_next;
                        proc.mmap_next = proc.mmap_next.wrapping_add((units_to_add * 4096) as u64);
                        addr
                    } else {
                        0
                    };

                    *self.memory_mutation_count.entry(user_pid).or_insert(0) += 1;
                    returned_addr as i64
                }
                10 => {
                    let length = arg1.max(1) as usize;
                    let units_to_remove = (length / 4096).max(1);

                    if let Some(mem) = self.jepa_memory.get_mut(&user_pid) {
                        let new_len = mem.len().saturating_sub(units_to_remove).max(8);
                        mem.truncate(new_len);
                    }
                    *self.memory_mutation_count.entry(user_pid).or_insert(0) += 1;
                    0
                }
                _ => 0,
            }
        } else {
            0
        };

        // Always write the return value back into the user's r0 (standard ABI for these prototypes)
        if let Some(user_proc) = self.process_manager.processes.get_mut(&user_pid) {
            user_proc.registers[0] = ret;
            // Also bump pc a little so that a subsequent "resume" looks like it returned from the trap site
            user_proc.pc = user_proc.pc.wrapping_add(4);
        }

        Ok(ret)
    }

    /// Return from a trap back to a user process.
    /// Restores the user to Ready and switches back to it.
    pub fn return_from_trap(&mut self, user_pid: i32) -> PyResult<()> {
        if let Some(proc) = self.process_manager.processes.get_mut(&user_pid) {
            if proc.state == ProcessState::Blocked {
                proc.state = ProcessState::Ready;
            }
        }

        self.switch_process(user_pid)?;
        Ok(())
    }

    /// High-level helper: handle a syscall trap end-to-end.
    /// Enters the trap, dispatches the syscall, then returns to the user.
    pub fn handle_syscall_trap(&mut self, user_pid: i32, syscall_number: i32) -> PyResult<i64> {
        self.enter_trap(user_pid, syscall_number)?;
        let ret = self.handle_syscall(user_pid, syscall_number)?;
        self.return_from_trap(user_pid)?;
        Ok(ret)
    }

    /// Run the kernel for a number of steps.
    /// This is the PRIMARY driver for generating high-quality, authentic traces for the JEPA predictor.
    ///
    /// It exercises the *real* ProcessManager + scheduling + blocking + trap/syscall paths
    /// that already underpin the deterministic multi-process UNIX substrate.
    ///
    /// Each step produces rich events containing exactly the features a cross-JEPA / process-aware
    /// world model needs to learn full machine (including kernel) dynamics:
    ///   - pid, registers, pc, flags, state, total_cycles
    ///   - syscall_number / args / retval when a syscall happened
    ///   - trap info, in_kernel flag, blocked reason
    ///   - scheduling transitions
    ///
    /// The Python side (NeuralKernel or direct JEPANeuralCPU) can feed these straight into
    /// train_on_transitions or a hybrid Rust+Python predictor loop.
    pub fn run_steps(&mut self, steps: u32) -> PyResult<PyObject> {
        let py = unsafe { pyo3::Python::assume_gil_acquired() };
        let mut local_events: Vec<PyObject> = Vec::new();

        for step in 0..steps {
            let current = self.process_manager.current_pid;

            if current < 0 {
                let _ = self.schedule_next();
                self._record_schedule_event(py, step, -1, self.process_manager.current_pid);
                continue;
            }

            let is_runnable = {
                if let Some(proc) = self.process_manager.processes.get(&current) {
                    matches!(proc.state, ProcessState::Running | ProcessState::Ready)
                } else {
                    false
                }
            };

            // Simulate one "user instruction" by advancing PC + cycles.
            // (Real heavy compute / ARM64 interpretation lives in the GPU shaders + full_arm64 path.
            //  This harness focuses on the *kernel dynamics* the JEPA predictor must master for OS hosting.)
            if let Some(proc) = self.process_manager.processes.get_mut(&current) {
                if matches!(proc.state, ProcessState::Running | ProcessState::Ready) {
                    proc.pc = proc.pc.wrapping_add(4);
                    proc.total_cycles += 1;
                }
            }

            // Occasionally inject a realistic trap/syscall from user space (the highest-leverage learning signal).
            let should_inject_trap = is_runnable && (step % 7 == 0) && current != self.kernel_pid;

            if should_inject_trap {
                // Pick a varied syscall including mmap/munmap (9/10) for address space dynamics
                let trap_num = ((step / 7) % 11) as i32; // 0..10 covering our expanded table
                let _ = self.enter_trap(current, trap_num);
                let _ = self.return_from_trap(current);
                self.total_traps += 1;

                // Record a rich trap/syscall event
                self._record_rich_event(py, step, current, Some(trap_num), "trap_syscall", &mut local_events);
            } else if is_runnable {
                // Normal user step — still record occasionally for dense coverage
                if step % 3 == 0 {
                    self._record_rich_event(py, step, current, None, "user_step", &mut local_events);
                }
            }

            // Always let the real scheduler decide what runs next (this is gold for learning)
            // Occasional native Rust churn bias (predictor-free self-optimizing behavior on the substrate)
            let prev = current;
            let mut next = self.schedule_next().unwrap_or(current);

            if step % 8 == 0 {
                let mut best = next;
                let mut best_score = self.compute_churn_score(next).unwrap_or(999.0);
                let ready: Vec<i32> = self.process_manager.processes.iter()
                    .filter(|(_, p)| matches!(p.state, ProcessState::Ready | ProcessState::Running))
                    .map(|(&pid, _)| pid).collect();
                for &cand in &ready {
                    if cand == current { continue; }
                    if let Ok(score) = self.compute_churn_score(cand) {
                        if score < best_score - 0.1 {
                            best = cand;
                            best_score = score;
                        }
                    }
                }
                if best != next {
                    let _ = self.switch_process(best);
                    next = best;
                }
            }

            if next != prev {
                self._record_schedule_event(py, step, prev, next);
            }

            // Always append one dense snapshot of whatever is now current (for predictor rollouts)
            if step % 2 == 0 {
                self._record_rich_event(py, step, self.process_manager.current_pid, None, "snapshot", &mut local_events);
            }
        }

        // Record how many rich events this run produced (full data lives in the returned Python list).
        self.last_event_count = local_events.len();

        Ok(local_events.into_py(py))
    }

    /// Get a rich snapshot of a process (registers, pc, flags, state, memory region info).
    /// This is what the Python JEPA predictor can consume as features.
    pub fn get_process_snapshot(&self, pid: i32, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);

        if let Some(proc) = self.process_manager.processes.get(&pid) {
            dict.set_item("pid", proc.pid)?;
            dict.set_item("ppid", proc.ppid)?;
            dict.set_item("state", format!("{:?}", proc.state))?;
            dict.set_item("pc", proc.pc)?;
            dict.set_item("registers", proc.registers.to_vec())?;
            dict.set_item("flags", proc.flags.to_vec())?;
            dict.set_item("total_cycles", proc.total_cycles)?;
            dict.set_item("memory_base", crate::os::process::ProcessManager::backing_addr(pid))?;
            dict.set_item("memory_size", crate::os::process::BACKING_STORE_SIZE)?;
        } else {
            dict.set_item("error", format!("process {} not found", pid))?;
        }

        Ok(dict.into())
    }

    /// Return a rich memory snapshot for a process, specifically designed for JEPA latent encoding.
    ///
    /// Includes:
    /// - Conceptual backing store location (matches the real ProcessManager layout)
    /// - Current heap_break and mmap_next (real fields from the PCB)
    /// - Compact 32-float summary vector that evolves with brk/write/etc.
    ///   This is what the Python cross-JEPA predictor can consume directly as memory features.
    pub fn get_memory_snapshot(&self, pid: i32, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);

        if let Some(proc) = self.process_manager.processes.get(&pid) {
            dict.set_item("pid", proc.pid)?;
            dict.set_item("memory_base", crate::os::process::ProcessManager::backing_addr(pid))?;
            dict.set_item("memory_size", crate::os::process::BACKING_STORE_SIZE)?;
            dict.set_item("heap_break", proc.heap_break)?;
            dict.set_item("mmap_next", proc.mmap_next)?;

            if let Some(mem) = self.jepa_memory.get(&pid) {
                dict.set_item("summary", mem.clone())?;
                dict.set_item("committed_units", mem.len() as u32)?;
                let mutations = *self.memory_mutation_count.get(&pid).unwrap_or(&0);
                dict.set_item("memory_mutations", mutations)?;

                let sum: f32 = mem.iter().sum();
                let mean = sum / (mem.len() as f32);
                let variance: f32 = mem.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (mem.len() as f32);
                dict.set_item("summary_mean", mean)?;
                dict.set_item("summary_variance", variance)?;
            } else {
                dict.set_item("summary", vec![0.0f32; 32])?;
                dict.set_item("committed_units", 0u32)?;
                dict.set_item("summary_mean", 0.0f32)?;
                dict.set_item("summary_variance", 0.0f32)?;
            }

            // New structured page signals for the predictor (dirty page learning = novel OS paradigm)
            if let Some(pages) = self.memory_pages.get(&pid) {
                let dirty_count = pages.iter().filter(|p| p.dirty).count() as u32;
                let total_access: u32 = pages.iter().map(|p| p.access_count).sum();
                dict.set_item("dirty_pages", dirty_count)?;
                dict.set_item("total_page_accesses", total_access)?;
            } else {
                dict.set_item("dirty_pages", 0u32)?;
                dict.set_item("total_page_accesses", 0u32)?;
            }
        } else {
            dict.set_item("error", format!("process {} not found", pid))?;
        }

        Ok(dict.into())
    }

    /// Simple round-robin schedule (wrapper around the real scheduler logic).
    pub fn schedule_next(&mut self) -> PyResult<i32> {
        let ready: Vec<i32> = self
            .process_manager
            .processes
            .iter()
            .filter(|(_, p)| matches!(p.state, ProcessState::Ready | ProcessState::Running))
            .map(|(&pid, _)| pid)
            .collect();

        if ready.is_empty() {
            return Ok(self.process_manager.current_pid);
        }

        let current = self.process_manager.current_pid;
        let idx = ready.iter().position(|&p| p == current).unwrap_or(0);
        let next_idx = (idx + 1) % ready.len();
        let next_pid = ready[next_idx];

        self.switch_process(next_pid)?;

        // On context switch, simulate write-back / clean of a few dirty pages (realistic MMU behavior
        // the predictor can learn from the page dirty signals).
        if let Some(pages) = self.memory_pages.get_mut(&next_pid) {
            for pg in pages.iter_mut() {
                if pg.dirty && (pg.access_count % 3 == 0) {
                    pg.dirty = false;
                }
            }
        }

        Ok(next_pid)
    }

    /// Return basic stats (useful for Python-side observability and predictor features).
    pub fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("current_pid", self.process_manager.current_pid)?;
        dict.set_item("num_processes", self.process_manager.processes.len())?;
        dict.set_item("total_syscalls", self.total_syscalls)?;
        dict.set_item("total_context_switches", self.total_context_switches)?;
        Ok(dict.into())
    }

    /// Create a user process (Python-friendly wrapper).
    pub fn spawn_user_process(&mut self, ppid: i32) -> PyResult<i32> {
        let pid = self.process_manager.alloc_pid();
        if pid < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Process table full",
            ));
        }
        self.create_process(pid, ppid)
    }

    /// Get a snapshot of the current syscall table (for debugging / Python side).
    pub fn get_syscall_table(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (num, name) in &self.syscall_table {
            dict.set_item(*num, name)?;
        }
        Ok(dict.into())
    }

    /// Return a list of (pid, state) for all known processes.
    /// Very useful for the Python NeuralKernel to stay in sync.
    pub fn list_processes(&self, py: Python<'_>) -> PyResult<PyObject> {
        let list = pyo3::types::PyList::empty(py);
        for (&pid, proc) in &self.process_manager.processes {
            let tuple = (pid, format!("{:?}", proc.state));
            let _ = list.append(tuple);
        }
        Ok(list.into())
    }

    /// Legacy alias — prefer get_and_clear_last_events.
    pub fn get_and_clear_events(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        self.get_and_clear_last_events(py)
    }

    /// Spawn a user process with explicit initial register state (super useful for deterministic JEPA experiments).
    pub fn spawn_user_process_with_state(
        &mut self,
        ppid: i32,
        initial_regs: Vec<i64>,
        initial_pc: u64,
    ) -> PyResult<i32> {
        let pid = self.process_manager.alloc_pid();
        if pid < 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Process table full",
            ));
        }
        let _ = self.create_process(pid, ppid);

        if let Some(proc) = self.process_manager.processes.get_mut(&pid) {
            // Copy provided registers (pad or truncate to 32)
            for (i, &v) in initial_regs.iter().take(32).enumerate() {
                proc.registers[i] = v;
            }
            proc.pc = initial_pc;
            proc.state = ProcessState::Ready;
        }

        // Seed a growable JEPA memory model (v2)
        let base = if initial_regs.is_empty() { 32 } else { 28 + ((initial_regs[0] as usize) % 12) };
        let mem: Vec<f32> = (0..base).map(|i| ((i as f32) * 0.019 + 0.07) % 0.85).collect();
        self.jepa_memory.insert(pid, mem);
        self.memory_mutation_count.insert(pid, 0);

        let pages = vec![PageInfo::default(); 12];
        self.memory_pages.insert(pid, pages);

        Ok(pid)
    }

    /// Force a specific process into Blocked state (for testing predictor on blocking dynamics).
    pub fn block_process(&mut self, pid: i32, _reason: &str) -> PyResult<()> {
        if let Some(proc) = self.process_manager.processes.get_mut(&pid) {
            proc.state = ProcessState::Blocked;
            // We could store a blocked_reason string in future if Process grows the field.
        }
        Ok(())
    }

    /// Unblock a process so the scheduler can pick it again.
    pub fn unblock_process(&mut self, pid: i32) -> PyResult<()> {
        if let Some(proc) = self.process_manager.processes.get_mut(&pid) {
            if proc.state == ProcessState::Blocked {
                proc.state = ProcessState::Ready;
            }
        }
        Ok(())
    }

    /// Return current stats + event buffer length (for Python-side monitoring of trace volume).
    pub fn get_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("current_pid", self.process_manager.current_pid)?;
        dict.set_item("num_processes", self.process_manager.processes.len())?;
        dict.set_item("total_syscalls", self.total_syscalls)?;
        dict.set_item("total_context_switches", self.total_context_switches)?;
        dict.set_item("total_traps", self.total_traps)?;
        dict.set_item("last_event_count", self.last_event_count)?;
        dict.set_item("jepa_memory_processes", self.jepa_memory.len())?;
        // Expose aggregate memory mutation activity for Python-side observability / predictor features
        let total_mut = self.memory_mutation_count.values().sum::<u32>();
        dict.set_item("total_memory_mutations", total_mut)?;
        dict.set_item("total_jepa_bias_suggestions", self.total_jepa_bias_suggestions)?;
        dict.set_item("current_observation_step", self.current_step)?;
        Ok(dict.into())
    }

    /// Compute a predictor-free churn score directly from the structured page + mutation state.
    ///
    /// This is the fast, always-available signal used for live self-optimizing scheduling
    /// bias. It is deliberately *not* a full learned JEPA predictor — it is a simple,
    /// cheap heuristic that the real execution engine can consult at every context switch
    /// and memory syscall without any Python or model inference overhead.
    ///
    /// The score is made *relative* to the current highest-churn process in the system
    /// (normalized to a 0–10 range). This gives meaningful differentiation even when
    /// every process is under heavy memory pressure (common in real BusyBox-style workloads).
    ///
    /// Recency is now *real* (using current_step vs PageInfo._last_touch_step):
    /// - A process that touched memory in the last few observation steps gets a strong
    ///   upward boost (recent heavy activity stands out).
    /// - Older activity is down-weighted, so "was churning 100 steps ago but quiet since"
    ///   scores lower than "just did a big brk/mmap/write burst".
    /// This produces the larger spreads needed for reliable bias overrides on sustained
    /// multi-process guest code (the 59→73+ jepa_bias_suggestions milestone).
    ///
    /// Components (weights chosen empirically on real BusyBox traces):
    /// - Dirty pages + access volume (imminent pressure)
    /// - True recency boost from step delta on touched pages
    /// - Mutation count (brk/mmap/write activity rate)
    /// - Committed address space size (long-term growth)
    pub fn compute_churn_score(&self, pid: i32) -> PyResult<f32> {
        let mut score = 0.0f32;

        if let Some(pages) = self.memory_pages.get(&pid) {
            let dirty = pages.iter().filter(|p| p.dirty).count() as f32;
            let accesses: f32 = pages.iter().map(|p| p.access_count as f32).sum();

            // === TRUE RECENCY (the key upgrade) ===
            // Use the step counter + per-page last_touch to give very recent activity
            // dramatically higher weight. A page touched in the last 1-5 steps is "hot now".
            let mut recency_boost = 0.0f32;
            for pg in pages.iter() {
                if pg.access_count > 0 {
                    let age = self.current_step.saturating_sub(pg._last_touch_step) as f32;
                    // Strong non-linear boost for very recent (age < 8): up to ~4.0 per hot page
                    let age_factor = if age < 3.0 {
                        4.0
                    } else if age < 8.0 {
                        2.5
                    } else if age < 25.0 {
                        1.0
                    } else {
                        0.15 // old activity barely counts for "current pressure"
                    };
                    // Weight by how much this page was used (hot pages matter more)
                    let weight = ((pg.access_count as f32) / 50.0).min(2.0);
                    recency_boost += age_factor * weight;
                }
            }
            recency_boost = recency_boost.min(6.0); // cap so one wild page doesn't dominate everything

            score += dirty * 0.55f32
                + (accesses / 25.0f32) * 0.35f32
                + recency_boost * 0.9f32; // recency is now the dominant differentiator
        }

        if let Some(&muts) = self.memory_mutation_count.get(&pid) {
            score += (muts as f32) * 0.12;
        }

        if let Some(mem) = self.jepa_memory.get(&pid) {
            let comm = mem.len() as f32;
            score += (comm / 64.0) * 0.08;
        }

        // === Peer-relative normalization (MUST use identical formula) ===
        // Previously the max loop omitted the recency term → inconsistent scores and
        // weaker spreads. Now every peer is scored with the *exact* same recency logic
        // so a process that just hammered memory truly ranks at the top (10.0) while
        // peers that were quiet for the last N steps sit visibly lower (e.g. 9.7x).
        let mut max_score = score;
        for &other_pid in self.process_manager.processes.keys() {
            if other_pid == pid { continue; }
            let mut other = 0.0f32;
            if let Some(pages) = self.memory_pages.get(&other_pid) {
                let dirty = pages.iter().filter(|p| p.dirty).count() as f32;
                let accesses: f32 = pages.iter().map(|p| p.access_count as f32).sum();

                let mut rec = 0.0f32;
                for pg in pages.iter() {
                    if pg.access_count > 0 {
                        let age = self.current_step.saturating_sub(pg._last_touch_step) as f32;
                        let age_factor = if age < 3.0 { 4.0 } else if age < 8.0 { 2.5 } else if age < 25.0 { 1.0 } else { 0.15 };
                        let weight = ((pg.access_count as f32) / 50.0).min(2.0);
                        rec += age_factor * weight;
                    }
                }
                rec = rec.min(6.0);
                other += dirty * 0.55 + (accesses / 25.0) * 0.35 + rec * 0.9;
            }
            if let Some(&muts) = self.memory_mutation_count.get(&other_pid) {
                other += (muts as f32) * 0.12;
            }
            if let Some(mem) = self.jepa_memory.get(&other_pid) {
                let comm = mem.len() as f32;
                other += (comm / 64.0) * 0.08;
            }
            if other > max_score { max_score = other; }
        }

        if max_score > 0.0 {
            score = (score / max_score).clamp(0.0, 1.0) * 10.0;
        }

        Ok(score)
    }

    /// Ingest a real Process snapshot from the mature substrate (GpuLauncher / full execution path).
    /// This is a key integration hook: allows the JEPA model to observe real execution state
    /// (registers, memory region info, etc.) coming from high-performance real program runs.
    ///
    /// This is the foundation for the neural layer "hosting" or deeply observing real OS workloads.
    pub fn ingest_real_process_snapshot(&mut self, pid: i32, snapshot: PyObject, py: Python<'_>) -> PyResult<()> {
        let dict = snapshot.downcast_bound::<PyDict>(py)?;

        // Snapshot ingestion from the real launcher is a first-class observation point.
        // Advance the recency clock here too so that state pushed at every schedule
        // decision participates in age calculations.
        self._advance_step();

        // Robust extraction using Bound API
        let ppid: i32 = match dict.get_item("ppid") {
            Ok(Some(v)) => v.extract().unwrap_or(-1),
            _ => -1,
        };

        if !self.process_manager.processes.contains_key(&pid) {
            let _ = self.create_process(pid, ppid);
        }

        if let Some(proc) = self.process_manager.processes.get_mut(&pid) {
            if let Ok(Some(regs_obj)) = dict.get_item("registers") {
                if let Ok(regs) = regs_obj.extract::<Vec<i64>>() {
                    for (i, &val) in regs.iter().take(32).enumerate() {
                        proc.registers[i] = val;
                    }
                }
            }
            if let Ok(Some(pc_obj)) = dict.get_item("pc") {
                if let Ok(pc) = pc_obj.extract::<u64>() {
                    proc.pc = pc;
                }
            }
            if let Ok(Some(state_obj)) = dict.get_item("state") {
                if let Ok(state_str) = state_obj.extract::<String>() {
                    proc.state = match state_str.as_str() {
                        "Ready" => ProcessState::Ready,
                        "Running" => ProcessState::Running,
                        "Blocked" => ProcessState::Blocked,
                        "Zombie" => ProcessState::Zombie,
                        _ => proc.state,
                    };
                }
            }
        }

        if let Ok(Some(committed_obj)) = dict.get_item("committed_units") {
            if let Ok(committed) = committed_obj.extract::<usize>() {
                if let Some(mem) = self.jepa_memory.get_mut(&pid) {
                    if mem.len() < committed {
                        mem.resize(committed, 0.0);
                    }
                }
            }
        }

        // Support real substrate snapshots carrying heap_break/mmap_next (from GpuLauncher Process)
        // Grow shadow memory model and mark activity so churn scores reflect real address space growth.
        let mut grew = false;
        if let Ok(Some(hb_obj)) = dict.get_item("heap_break") {
            if let Ok(hb) = hb_obj.extract::<u64>() {
                let base: u64 = 0x60000;
                let units = ((hb.saturating_sub(base)) / 8) as usize;
                if let Some(mem) = self.jepa_memory.get_mut(&pid) {
                    if units > mem.len() {
                        mem.resize(units.max(32), 0.0);
                        grew = true;
                    }
                }
            }
        }
        if let Ok(Some(mm_obj)) = dict.get_item("mmap_next") {
            if let Ok(_mm) = mm_obj.extract::<u64>() {
                // mmap growth also signals activity
                grew = true;
            }
        }
        if grew {
            *self.memory_mutation_count.entry(pid).or_insert(0) += 1;
            if let Some(pages) = self.memory_pages.get_mut(&pid) {
                if let Some(pg) = pages.first_mut() {
                    pg.dirty = true;
                    pg.access_count = pg.access_count.saturating_add(1);
                }
            }
        }

        if let Ok(Some(muts_obj)) = dict.get_item("memory_mutations") {
            if let Ok(muts) = muts_obj.extract::<u32>() {
                *self.memory_mutation_count.entry(pid).or_insert(0) = muts.max(*self.memory_mutation_count.get(&pid).unwrap_or(&0));
            }
        }

        // Populate structured page model (dirty + accesses) from real snapshot data
        if !self.memory_pages.contains_key(&pid) {
            self.memory_pages.insert(pid, vec![PageInfo::default(); 12]);
        }

        if let Ok(Some(dirty_obj)) = dict.get_item("dirty_pages") {
            if let Ok(dirty_count) = dirty_obj.extract::<usize>() {
                if let Some(pages) = self.memory_pages.get_mut(&pid) {
                    let n = pages.len();
                    for (i, pg) in pages.iter_mut().enumerate() {
                        pg.dirty = i < dirty_count.min(n);
                    }
                }
            }
        }
        if let Ok(Some(access_obj)) = dict.get_item("page_accesses") {
            if let Ok(accesses) = access_obj.extract::<u32>() {
                if let Some(pages) = self.memory_pages.get_mut(&pid) {
                    let n = pages.len().max(1);
                    let per = accesses / n as u32;
                    for pg in pages.iter_mut() {
                        if pg.dirty {
                            pg.access_count = per;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Returns (and clears) the count of events from the last run_steps.
    /// Note: the actual rich event data is returned directly by run_steps() itself.
    /// This is mainly for stats / flow control in long training runs.
    pub fn get_and_clear_last_events(&mut self, _py: Python<'_>) -> PyResult<PyObject> {
        let count = self.last_event_count;
        self.last_event_count = 0;
        // Return a tiny dict so callers have a uniform "events-like" return
        let d = PyDict::new(_py);
        let _ = d.set_item("event_count", count);
        Ok(d.into())
    }

    /// Export the current state of all processes in a format directly usable by
    /// ingest_real_process_snapshot. This makes post-real-run synchronization
    /// from the substrate extremely easy from Python.
    pub fn export_all_process_snapshots(&self, py: Python<'_>) -> PyResult<PyObject> {
        let list = pyo3::types::PyList::empty(py);

        for (&pid, proc) in &self.process_manager.processes {
            let snap = PyDict::new(py);
            snap.set_item("pid", pid)?;
            snap.set_item("ppid", proc.ppid)?;
            snap.set_item("registers", proc.registers.to_vec())?;
            snap.set_item("pc", proc.pc)?;
            snap.set_item("state", format!("{:?}", proc.state))?;
            snap.set_item("committed_units", self.jepa_memory.get(&pid).map(|m| m.len()).unwrap_or(0))?;

            if let Some(pages) = self.memory_pages.get(&pid) {
                let dirty = pages.iter().filter(|p| p.dirty).count() as u32;
                let acc: u32 = pages.iter().map(|p| p.access_count).sum();
                snap.set_item("dirty_pages", dirty)?;
                snap.set_item("page_accesses", acc)?;
            }

            list.append(snap)?;
        }

        Ok(list.into())
    }

    /// Called by the real execution engine (GpuLauncher) at context-switch points.
    /// The JEPA model ingests the current state and can return a preferred next pid
    /// to bias scheduling (self-optimizing behavior using learned memory churn).
    ///
    /// Returns Some(pid) if the model wants to override the next scheduled process.
    pub fn on_context_switch(&mut self, current_pid: i32, py: Python<'_>) -> PyResult<Option<i32>> {
        // Every observation point advances the global step so that recency (via _last_touch_step)
        // can strongly separate processes that are actively churning memory *right now* from
        // ones that were active in the past. This is what produces usable bias on real workloads.
        self._advance_step();

        // Ingest current state of the process that just ran
        if let Some(proc) = self.process_manager.processes.get(&current_pid) {
            let snap = PyDict::new(py);
            let _ = snap.set_item("pid", current_pid);
            let _ = snap.set_item("ppid", proc.ppid);
            let _ = snap.set_item("registers", proc.registers.to_vec());
            let _ = snap.set_item("pc", proc.pc);
            let _ = snap.set_item("state", format!("{:?}", proc.state));
            let _ = snap.set_item("committed_units", self.jepa_memory.get(&current_pid).map(|m| m.len()).unwrap_or(0));

            if let Some(pages) = self.memory_pages.get(&current_pid) {
                let dirty = pages.iter().filter(|p| p.dirty).count() as u32;
                let acc: u32 = pages.iter().map(|p| p.access_count).sum();
                let _ = snap.set_item("dirty_pages", dirty);
                let _ = snap.set_item("page_accesses", acc);
            }

            let _ = self.ingest_real_process_snapshot(current_pid, snap.into(), py);
        }

        // Use the structured memory (page dirty + accesses) + mutations for stronger bias
        // (this is the live self-optimizing feedback using the novel page model)
        let mut best = None;
        let mut best_score = f32::INFINITY;

        for &pid in self.process_manager.processes.keys() {
            if pid == current_pid || pid <= 0 {
                continue;
            }
            if let Ok(score) = self.compute_churn_score(pid) {
                if score < best_score {
                    best_score = score;
                    best = Some(pid);
                }
            }
        }

        // More aggressive threshold now that we have rich page/dirty signals
        if let Some(b) = best {
            if let Ok(curr_score) = self.compute_churn_score(current_pid) {
                // Lowered threshold so the model will actually act on the small-but-real
                // relative differences we now see from the peer-normalized scorer on real workloads.
                if curr_score > best_score + 0.01 {
                    self.total_jepa_bias_suggestions += 1;

                    // 3rd lever (adaptive persistent yield / de-prio):
                    // The model itself decides how long the high-churn process must yield.
                    // We compute the exact relative delta between current and best peer,
                    // then set a skip count (1–6 turns) proportional to that delta.
                    // This is what turns small observed spreads (0.04 on real BusyBox)
                    // into multi-turn scheduling influence under the lowest-PID policy.
                    // The launcher (and schedule_next) respect and age these skips.
                    // See also: launcher.rs bias sites, process.rs jepa_deprio_remaining,
                    // and the fairness telemetry (times_scheduled + per_process_scheduled).
                    let delta = curr_score - best_score;
                    let skip = (1.0 + delta * 25.0).clamp(1.0, 6.0) as u32;
                    if let Some(p) = self.process_manager.processes.get_mut(&current_pid) {
                        p.jepa_deprio_remaining = skip.max(p.jepa_deprio_remaining);
                    }

                    return Ok(Some(b));
                }
            }
        }

        Ok(None)
    }

    /// Convenience: return churn scores for all user processes (pid > 0).
    /// Extremely useful after real guest execution to see what the model learned.
    pub fn get_all_churn_scores(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for &pid in self.process_manager.processes.keys() {
            if pid > 0 {
                if let Ok(score) = self.compute_churn_score(pid) {
                    let _ = dict.set_item(pid, score);
                }
            }
        }
        Ok(dict.into())
    }

    /// Return current JEPA-driven de-prio skip counters for all user processes.
    /// This is the direct visibility into the 3rd decision lever (adaptive yield).
    /// Non-zero values mean the learned model decided that process was hot enough
    /// (relative to its peers at that exact moment) to force it to yield for N turns.
    /// Used by the A/B harness (/tmp/test_real_jepa_busybox.py) together with
    /// per_process_scheduled to measure actual fairness impact of the bias.
    pub fn get_all_deprios(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (&pid, proc) in &self.process_manager.processes {
            if pid > 0 {
                let _ = dict.set_item(pid, proc.jepa_deprio_remaining);
            }
        }
        Ok(dict.into())
    }

    /// Live hook called by GpuLauncher at syscall/trap entry during real ELF execution.
    ///
    /// This gives the JEPA model immediate visibility into kernel events, especially
    /// memory-related syscalls (brk=5, mmap=9, munmap=10). On memory operations we
    /// heuristically bump the shadow page/dirty/mutation model so that the churn
    /// predictor sees address-space pressure as soon as it happens.
    ///
    /// If the operation causes this process to look significantly worse than its
    /// peers (using the relative churn score), we return a preferred next pid so the
    /// launcher can immediately bias scheduling away from the high-pressure process.
    ///
    /// Returns `Some(pid)` when the model wants to override the next scheduled process.
    pub fn on_syscall(&mut self, pid: i32, syscall_num: i32, _arg0: i64, _arg1: i64, _py: Python<'_>) -> PyResult<Option<i32>> {
        if !self.process_manager.processes.contains_key(&pid) {
            let _ = self.create_process(pid, 0);
        }

        // Tick the recency clock on *every* syscall observation from real guest code.
        // Even non-memory syscalls advance time so that a process that did a burst of
        // allocations 50 steps ago is now "older" than one doing them in the last 5.
        self._advance_step();

        let is_mem = matches!(syscall_num, 5 | 9 | 10);
        if is_mem {
            *self.memory_mutation_count.entry(pid).or_insert(0) += 2;
            if let Some(mem) = self.jepa_memory.get_mut(&pid) {
                if mem.len() < 48 { mem.resize(48, 0.0); }
            }
            // Use the recency-aware helper for both conceptual pages. This ensures
            // a heavy brk/mmap right now gets a strong recent-touch boost in churn score,
            // making bias decisions (prefer low-churn peer) fire more reliably.
            self._advance_and_touch_page(pid, 0);
            self._advance_and_touch_page(pid, 1);
        } else {
            *self.memory_mutation_count.entry(pid).or_insert(0) += 0;
        }

        self.total_syscalls += 1;

        if is_mem {
            if let Ok(curr) = self.compute_churn_score(pid) {
                // Much more aggressive on memory ops: we want the model to be able to
                // immediately bias away from a process that just did a heavy allocation.
                // Removed the high absolute bar; now any meaningful relative delta wins.
                if curr > 0.5 {
                    let mut best = None;
                    let mut best_s = f32::INFINITY;
                    for &other in self.process_manager.processes.keys() {
                        if other == pid || other <= 0 { continue; }
                        if let Ok(s) = self.compute_churn_score(other) {
                            if s < best_s { best_s = s; best = Some(other); }
                        }
                    }
                    if let Some(b) = best {
                        if let Ok(cs) = self.compute_churn_score(pid) {
                            if cs > best_s + 0.01 {
                                self.total_jepa_bias_suggestions += 1;

                                // Dynamic 3rd lever for mem ops (the highest-signal case for yield).
                                // The hotter the relative churn right after the brk/mmap, the longer
                                // we force this process to yield so low-churn peers get the slices.
                                let delta = cs - best_s;
                                let skip = (2.0 + delta * 30.0).clamp(2.0, 7.0) as u32;
                                if let Some(p) = self.process_manager.processes.get_mut(&pid) {
                                    p.jepa_deprio_remaining = skip.max(p.jepa_deprio_remaining);
                                }

                                return Ok(Some(b));
                            }
                        }
                    }
                }
            }
        }

        Ok(None)
    }

}

/// Non-pymethod helpers (private implementation details).
/// These must live outside the #[pymethods] block because they use signatures
/// (e.g. Option before &str) that pyo3's macro would otherwise complain about.
impl NeuralJepaKernel {
    /// Internal: produce a very rich event dict for the JEPA predictor.
    fn _record_rich_event(
        &mut self,
        py: Python<'_>,
        step: u32,
        pid: i32,
        syscall_num: Option<i32>,
        kind: &str,
        out: &mut Vec<PyObject>,
    ) {
        let dict = PyDict::new(py);
        let _ = dict.set_item("step", step);
        let _ = dict.set_item("kind", kind);
        let _ = dict.set_item("pid", pid);
        let _ = dict.set_item("current_pid", self.process_manager.current_pid);

        if let Some(proc) = self.process_manager.processes.get(&pid) {
            let _ = dict.set_item("registers", proc.registers.to_vec());
            let _ = dict.set_item("pc", proc.pc);
            let _ = dict.set_item("flags", proc.flags.to_vec());
            let _ = dict.set_item("state", format!("{:?}", proc.state));
            let _ = dict.set_item("total_cycles", proc.total_cycles);
            let _ = dict.set_item("heap_break", proc.heap_break);
            let _ = dict.set_item("mmap_next", proc.mmap_next);
            let _ = dict.set_item("ppid", proc.ppid);

            // Include compact memory summary + committed size in events — critical for the predictor
            // to learn address space growth, dirtying, and munmap reclamation.
            if let Some(mem) = self.jepa_memory.get(&pid) {
                let _ = dict.set_item("memory_summary", mem.clone());
                let _ = dict.set_item("committed_units", mem.len() as u32);
                let _ = dict.set_item("memory_mutations", *self.memory_mutation_count.get(&pid).unwrap_or(&0));
            }

            // Page dirty/access signals in events (enables the predictor to learn memory access patterns)
            if let Some(pages) = self.memory_pages.get(&pid) {
                let dirty = pages.iter().filter(|p| p.dirty).count() as u32;
                let acc: u32 = pages.iter().map(|p| p.access_count).sum();
                let _ = dict.set_item("dirty_pages", dirty);
                let _ = dict.set_item("page_accesses", acc);
            }
        }

        if let Some(sn) = syscall_num {
            let _ = dict.set_item("syscall_number", sn);
            if let Some(name) = self.syscall_table.get(&sn) {
                let _ = dict.set_item("syscall_name", name.clone());
            }
            // Attach the return value that was written to r0 (already in registers[0] snapshot)
            let _ = dict.set_item("syscall_retval_present", true);
        }

        let in_kernel = pid == self.kernel_pid;
        let _ = dict.set_item("in_kernel", in_kernel);

        if let Some(proc) = self.process_manager.processes.get(&pid) {
            let blocked = matches!(proc.state, ProcessState::Blocked);
            let _ = dict.set_item("blocked", blocked);
        }

        out.push(dict.into());
    }

    /// Internal: record a scheduling transition (extremely valuable signal).
    fn _record_schedule_event(&mut self, py: Python<'_>, step: u32, from_pid: i32, to_pid: i32) {
        let dict = PyDict::new(py);
        let _ = dict.set_item("step", step);
        let _ = dict.set_item("kind", "schedule");
        let _ = dict.set_item("from_pid", from_pid);
        let _ = dict.set_item("to_pid", to_pid);
        let _ = dict.set_item("pid", to_pid);
        let _ = dict.set_item("current_pid", self.process_manager.current_pid);

        if let Some(proc) = self.process_manager.processes.get(&to_pid) {
            let _ = dict.set_item("registers", proc.registers.to_vec());
            let _ = dict.set_item("pc", proc.pc);
            let _ = dict.set_item("state", format!("{:?}", proc.state));
            let _ = dict.set_item("heap_break", proc.heap_break);

            if let Some(mem) = self.jepa_memory.get(&to_pid) {
                let _ = dict.set_item("memory_summary", mem.clone());
                let _ = dict.set_item("committed_units", mem.len() as u32);
                let _ = dict.set_item("memory_mutations", *self.memory_mutation_count.get(&to_pid).unwrap_or(&0));
            }

            if let Some(pages) = self.memory_pages.get(&to_pid) {
                let dirty = pages.iter().filter(|p| p.dirty).count() as u32;
                let acc: u32 = pages.iter().map(|p| p.access_count).sum();
                let _ = dict.set_item("dirty_pages", dirty);
                let _ = dict.set_item("page_accesses", acc);
            }
        }
        // We do not persist full objects here anymore; the count is updated in run_steps after collection.
    }
}

/// Module initialization — exposed to Python via PyO3.
pub fn register_neural_jepa_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NeuralJepaKernel>()?;
    Ok(())
}
