//! Multi-process manager for ARM64 programs running on a Metal GPU.
//!
//! Rust port of the Python `ProcessManager` from `ncpu/os/gpu/runner.py`.
//!
//! Manages process fork/wait/exit/kill, round-robin scheduling, and
//! memory-swapping between an active GPU workspace and per-process backing
//! stores. The manager itself never touches GPU buffers directly -- the caller
//! (GpuLauncher) is responsible for reading/writing GPU memory using the
//! addresses returned by [`ProcessManager::backing_addr`].

use std::collections::HashMap;

use crate::vfs::{FdEntry, FdKind, GpuVfs};

// ── Constants ────────────────────────────────────────────────────────────

pub const MAX_PROCESSES: i32 = 15;
pub const MAX_FORKS_PER_PROCESS: u32 = 32;
pub const MAX_CYCLE_LIMIT: u64 = 100_000_000;
pub const MAX_FDS_PER_PROCESS: usize = 64;

pub const ACTIVE_BASE: u64 = 0x10000;
pub const ACTIVE_END: u64 = 0xFF000;
pub const ACTIVE_SIZE: u64 = ACTIVE_END - ACTIVE_BASE;

pub const BACKING_STORE_BASE: u64 = 0x100000;
pub const BACKING_STORE_SIZE: u64 = 0x100000; // 1 MB

pub const HEAP_BASE: u64 = 0x60000;
pub const MMAP_BASE: u64 = HEAP_BASE + 0x400000;

pub const SIGTERM: i32 = 15;
pub const SIGKILL: i32 = 9;

// ── ProcessState ─────────────────────────────────────────────────────────

/// Lifecycle state of a process.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum ProcessState {
    Free = 0,
    Ready = 1,
    Running = 2,
    Blocked = 3,
    Zombie = 4,
}

// ── Process ──────────────────────────────────────────────────────────────

/// Per-process control block.
///
/// Mirrors the Python `Process` dataclass. Registers, PC, and flags are
/// snapshots taken at context-switch time -- they are **not** live GPU state.
#[derive(Clone, Debug)]
pub struct Process {
    pub pid: i32,
    pub ppid: i32,
    pub state: ProcessState,
    pub exit_code: i32,
    /// General-purpose registers X0..X30 + XZR (index 31).
    pub registers: [i64; 32],
    pub pc: u64,
    /// NZCV flags stored as `[N, Z, C, V]` (1.0 = set, 0.0 = clear).
    pub flags: [f32; 4],
    pub heap_break: u64,
    pub mmap_next: u64,
    pub fd_table: HashMap<i32, FdEntry>,
    pub cwd: String,
    /// PID this process is blocked waiting on (-1 = any child).
    pub wait_target: i32,
    /// Fork bomb protection counter.
    pub fork_count: u32,
    /// Cumulative cycles consumed by this process.
    pub total_cycles: u64,
    /// Pending signal to deliver on next schedule (None = no signal).
    pub pending_signal: Option<i32>,
    /// Per-process environment variables.
    pub env: HashMap<String, String>,
}

impl Process {
    /// Create a new process with default (zeroed) register state.
    pub fn new(pid: i32, ppid: i32, state: ProcessState) -> Self {
        Self {
            pid,
            ppid,
            state,
            exit_code: 0,
            registers: [0i64; 32],
            pc: 0,
            flags: [0.0f32; 4],
            heap_break: HEAP_BASE,
            mmap_next: MMAP_BASE,
            fd_table: HashMap::new(),
            cwd: "/".to_string(),
            wait_target: -1,
            fork_count: 0,
            total_cycles: 0,
            pending_signal: None,
            env: HashMap::new(),
        }
    }
}

// ── Saved GPU snapshot (returned by save_context / consumed by restore) ──

/// Snapshot of the GPU-side execution state at context-switch time.
///
/// The caller captures these values from the GPU and hands them to
/// [`ProcessManager::save_context`]. Conversely, [`ProcessManager::restore_context`]
/// returns a snapshot that the caller writes back to the GPU.
#[derive(Clone, Debug)]
pub struct GpuSnapshot {
    pub registers: [i64; 32],
    pub pc: u64,
    pub flags: [f32; 4],
}

// ── ProcessManager ───────────────────────────────────────────────────────

/// Manages multiple processes on a single GPU via memory swapping.
///
/// The manager maintains per-process control blocks and orchestrates
/// context switches, fork, exit, kill, and round-robin scheduling. Actual
/// GPU memory transfers are delegated to the caller through the snapshot
/// and backing-address APIs.
pub struct ProcessManager {
    pub processes: HashMap<i32, Process>,
    pub next_pid: i32,
    pub current_pid: i32,
    pub total_forks: u32,
    pub total_context_switches: u32,
}

impl ProcessManager {
    // ── Construction ─────────────────────────────────────────────────

    /// Create an empty process manager with no processes.
    pub fn new() -> Self {
        Self {
            processes: HashMap::new(),
            next_pid: 1,
            current_pid: -1,
            total_forks: 0,
            total_context_switches: 0,
        }
    }

    // ── PID allocation ───────────────────────────────────────────────

    /// Allocate the next available PID.
    ///
    /// Returns a PID in `[1, MAX_PROCESSES]`, or `-1` if the process table
    /// is full. After the linear counter exceeds `MAX_PROCESSES` the
    /// allocator scans for reusable slots.
    pub fn alloc_pid(&mut self) -> i32 {
        if self.next_pid <= MAX_PROCESSES {
            let pid = self.next_pid;
            self.next_pid += 1;
            return pid;
        }
        // Linear scan for a free slot.
        for i in 1..=MAX_PROCESSES {
            if !self.processes.contains_key(&i) {
                return i;
            }
        }
        -1 // Table full
    }

    // ── Backing-store addressing ─────────────────────────────────────

    /// Return the base address of the backing store for *pid*.
    ///
    /// Layout: `BACKING_STORE_BASE + (pid - 1) * BACKING_STORE_SIZE`.
    /// The caller uses this address with GPU `read_memory`/`write_memory`
    /// to move data between the active workspace and the backing store.
    #[inline]
    pub fn backing_addr(pid: i32) -> u64 {
        BACKING_STORE_BASE + ((pid - 1) as u64) * BACKING_STORE_SIZE
    }

    // ── Context switch ───────────────────────────────────────────────

    /// Save live GPU state into the process control block.
    ///
    /// The caller must supply the current GPU registers, PC, and flags.
    /// It must also copy `ACTIVE_BASE..ACTIVE_END` to the backing store
    /// at `backing_addr(pid)` **after** calling this method.
    ///
    /// `fd_table` is cloned from the VFS so the process keeps its own
    /// snapshot of open file descriptors.
    pub fn save_context(
        &mut self,
        pid: i32,
        snapshot: &GpuSnapshot,
        fd_table: &HashMap<i32, FdEntry>,
    ) {
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.registers = snapshot.registers;
            proc.pc = snapshot.pc;
            proc.flags = snapshot.flags;
            proc.fd_table = fd_table.clone();
        }
    }

    /// Prepare to restore a process onto the GPU.
    ///
    /// Returns the process snapshot (registers, PC, flags) and a clone of
    /// the process fd_table. The caller is responsible for:
    ///
    /// 1. Copying the backing store at `backing_addr(pid)` into
    ///    `ACTIVE_BASE..ACTIVE_END`.
    /// 2. Loading the returned registers/PC/flags into the GPU.
    /// 3. Installing the returned fd_table into the VFS.
    ///
    /// The process is marked `Running` and `current_pid` is updated.
    pub fn restore_context(
        &mut self,
        pid: i32,
    ) -> Option<(GpuSnapshot, HashMap<i32, FdEntry>, String)> {
        let proc = self.processes.get_mut(&pid)?;
        let snapshot = GpuSnapshot {
            registers: proc.registers,
            pc: proc.pc,
            flags: proc.flags,
        };
        let fd_table = proc.fd_table.clone();
        let cwd = proc.cwd.clone();
        proc.state = ProcessState::Running;
        self.current_pid = pid;
        self.total_context_switches += 1;
        Some((snapshot, fd_table, cwd))
    }

    // ── Scheduling ───────────────────────────────────────────────────

    /// Round-robin scheduler: pick the next `Ready` process.
    ///
    /// Selects the lowest-PID ready process whose PID is greater than
    /// `current_pid`, wrapping around to the beginning if necessary.
    /// Returns `None` when no process is ready.
    pub fn schedule_next(&self) -> Option<i32> {
        let mut ready: Vec<i32> = self
            .processes
            .values()
            .filter(|p| p.state == ProcessState::Ready)
            .map(|p| p.pid)
            .collect();
        if ready.is_empty() {
            return None;
        }
        ready.sort_unstable();
        // Pick the first ready PID after current_pid.
        for &pid in &ready {
            if pid > self.current_pid {
                return Some(pid);
            }
        }
        // Wrap around to the lowest.
        Some(ready[0])
    }

    // ── Fork ─────────────────────────────────────────────────────────

    /// Fork the process identified by `parent_pid`.
    ///
    /// Creates a child process with a copy of the parent's registers, PC,
    /// flags, fd_table, cwd, and environment. The caller is responsible for
    /// memory copying between backing stores (parent -> child).
    ///
    /// On success the child's X0 is set to 0 and the parent's X0 is set
    /// to the child PID. Both processes are left in `Ready` state.
    ///
    /// Returns the child PID, or `-1` on failure (process table full or
    /// fork-bomb limit reached).
    ///
    /// **Important**: The caller should advance the parent PC past the SVC
    /// instruction *before* calling this method and should save the parent
    /// context so that `parent.pc` already points past the syscall.
    pub fn fork(
        &mut self,
        parent_pid: i32,
        parent_snapshot: &GpuSnapshot,
        parent_fd_table: &HashMap<i32, FdEntry>,
        vfs: &mut GpuVfs,
    ) -> i32 {
        let parent = match self.processes.get(&parent_pid) {
            Some(p) => p,
            None => return -1,
        };

        // Fork bomb protection.
        if parent.fork_count >= MAX_FORKS_PER_PROCESS {
            return -1; // EAGAIN
        }

        let child_pid = self.alloc_pid();
        if child_pid < 0 {
            return -1; // EAGAIN -- table full
        }

        // Save parent snapshot into its PCB.
        self.save_context(parent_pid, parent_snapshot, parent_fd_table);

        // Clone the fd_table with proper pipe refcounting.
        let child_fd_table = vfs.clone_fd_table();

        // Read parent state *after* saving context.
        let parent = self.processes.get(&parent_pid).unwrap();

        let mut child = Process {
            pid: child_pid,
            ppid: parent_pid,
            state: ProcessState::Ready,
            exit_code: 0,
            registers: parent.registers,
            pc: parent.pc, // already advanced past SVC by caller
            flags: parent.flags,
            heap_break: parent.heap_break,
            mmap_next: parent.mmap_next,
            fd_table: child_fd_table,
            cwd: parent.cwd.clone(),
            wait_target: -1,
            fork_count: 0,
            total_cycles: 0,
            pending_signal: None,
            env: parent.env.clone(),
        };

        // Return values: child gets 0, parent gets child_pid.
        child.registers[0] = 0;

        // Bump parent fork count and set parent return value + state.
        let parent_mut = self.processes.get_mut(&parent_pid).unwrap();
        parent_mut.fork_count += 1;
        parent_mut.registers[0] = child_pid as i64;
        parent_mut.state = ProcessState::Ready;

        self.processes.insert(child_pid, child);
        self.total_forks += 1;

        child_pid
    }

    // ── Process exit ─────────────────────────────────────────────────

    /// Mark a process as `Zombie`, close its fds, reparent orphans, and
    /// wake any waiting parent.
    ///
    /// Pipe endpoints are closed through the VFS so that blocked readers
    /// or writers see EOF.
    pub fn process_exit(&mut self, pid: i32, exit_code: i32, vfs: &mut GpuVfs) {
        // Close all pipe fds through the VFS.
        if let Some(proc) = self.processes.get(&pid) {
            let pipe_fds: Vec<(i32, FdKind)> = proc
                .fd_table
                .iter()
                .map(|(&fd, entry)| (fd, entry.kind.clone()))
                .collect();
            for (_fd, kind) in &pipe_fds {
                match kind {
                    FdKind::PipeRead(idx) => {
                        if let Some(pb) = vfs.pipes.get_mut(*idx) {
                            pb.close_reader();
                        }
                    }
                    FdKind::PipeWrite(idx) => {
                        if let Some(pb) = vfs.pipes.get_mut(*idx) {
                            pb.close_writer();
                        }
                    }
                    _ => {}
                }
            }
        }

        // Mark zombie and set exit code.
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.state = ProcessState::Zombie;
            proc.exit_code = exit_code;
            proc.fd_table.clear();
        }

        // Reparent orphaned children to init (PID 1).
        let child_pids: Vec<i32> = self
            .processes
            .values()
            .filter(|p| p.ppid == pid && p.pid != pid)
            .map(|p| p.pid)
            .collect();

        for cpid in child_pids {
            if let Some(child) = self.processes.get_mut(&cpid) {
                child.ppid = 1;
                // If child is a zombie, wake init if it is waiting.
                if child.state == ProcessState::Zombie {
                    if let Some(init) = self.processes.get_mut(&1) {
                        if init.state == ProcessState::Blocked
                            && (init.wait_target == cpid || init.wait_target == -1)
                        {
                            init.state = ProcessState::Ready;
                        }
                    }
                }
            }
        }

        // Wake parent if it is blocked waiting for this child.
        let ppid = self.processes.get(&pid).map(|p| p.ppid).unwrap_or(-1);
        if ppid > 0 {
            if let Some(parent) = self.processes.get_mut(&ppid) {
                if parent.state == ProcessState::Blocked
                    && (parent.wait_target == pid || parent.wait_target == -1)
                {
                    parent.state = ProcessState::Ready;
                }
            }
        }
    }

    // ── Kill ─────────────────────────────────────────────────────────

    /// Send a signal to a process. Returns `0` on success, `-1` on error.
    ///
    /// - `SIGKILL` (9): immediate termination, cannot be caught.
    /// - `SIGTERM` (15): marks the process for termination on next schedule.
    /// - Signal 0: existence check only.
    pub fn kill_process(
        &mut self,
        target_pid: i32,
        signal: i32,
        _sender_pid: i32,
        vfs: &mut GpuVfs,
    ) -> i32 {
        let target_state = match self.processes.get(&target_pid) {
            Some(t) => t.state,
            None => return -1, // ESRCH
        };

        if target_state == ProcessState::Zombie {
            return -1; // Already dead
        }

        if signal == SIGKILL {
            self.process_exit(target_pid, 128 + SIGKILL, vfs);
            return 0;
        } else if signal == SIGTERM {
            if let Some(target) = self.processes.get_mut(&target_pid) {
                target.pending_signal = Some(SIGTERM);
                if target.state == ProcessState::Blocked {
                    target.state = ProcessState::Ready;
                }
            }
            return 0;
        } else if signal == 0 {
            return 0; // Existence check
        }

        -1 // EINVAL -- unsupported signal
    }

    // ── Zombie reaping ───────────────────────────────────────────────

    /// Reap a zombie child of `parent_pid`.
    ///
    /// If `child_pid` is `-1`, reaps any zombie child. Otherwise reaps
    /// the specific child. Returns `Some(process)` on success, or `None`
    /// if no matching zombie was found.
    pub fn reap_zombie(&mut self, parent_pid: i32, child_pid: i32) -> Option<Process> {
        if child_pid == -1 {
            // Find any zombie child of parent.
            let zombie_pid = self
                .processes
                .values()
                .find(|p| p.ppid == parent_pid && p.state == ProcessState::Zombie)
                .map(|p| p.pid)?;
            self.processes.remove(&zombie_pid)
        } else {
            let matches = self
                .processes
                .get(&child_pid)
                .map(|p| p.ppid == parent_pid && p.state == ProcessState::Zombie)
                .unwrap_or(false);
            if matches {
                self.processes.remove(&child_pid)
            } else {
                None
            }
        }
    }

    // ── Init process creation ────────────────────────────────────────

    /// Create the initial process (PID 1) with default state.
    ///
    /// The caller should load the binary into GPU memory at `0x10000`
    /// and save active workspace to backing store afterwards.
    pub fn create_init_process(
        &mut self,
        fd_table: Option<HashMap<i32, FdEntry>>,
        cwd: &str,
        env: Option<HashMap<String, String>>,
    ) -> i32 {
        let pid = self.alloc_pid();
        if pid < 0 {
            return -1;
        }

        let mut proc = Process::new(pid, 0, ProcessState::Ready);
        proc.pc = ACTIVE_BASE;
        proc.cwd = cwd.to_string();
        if let Some(fdt) = fd_table {
            proc.fd_table = fdt;
        }
        if let Some(e) = env {
            proc.env = e;
        }
        // X31 = 0 (XZR); SP is set by _start code.
        proc.registers[31] = 0;

        self.processes.insert(pid, proc);
        pid
    }

    // ── Query helpers ────────────────────────────────────────────────

    /// Get an immutable reference to a process by PID.
    #[inline]
    pub fn get(&self, pid: i32) -> Option<&Process> {
        self.processes.get(&pid)
    }

    /// Get a mutable reference to a process by PID.
    #[inline]
    pub fn get_mut(&mut self, pid: i32) -> Option<&mut Process> {
        self.processes.get_mut(&pid)
    }

    /// Check whether any non-zombie children exist for `parent_pid`.
    pub fn has_live_children(&self, parent_pid: i32) -> bool {
        self.processes
            .values()
            .any(|p| p.ppid == parent_pid && p.state != ProcessState::Zombie)
    }

    /// Return a sorted list of `(pid, ppid, state_name)` for ps-like output.
    pub fn ps_list(&self) -> Vec<(i32, i32, &'static str)> {
        let mut list: Vec<(i32, i32, &'static str)> = self
            .processes
            .values()
            .map(|p| {
                let sname = match p.state {
                    ProcessState::Free => "FREE",
                    ProcessState::Ready => "READY",
                    ProcessState::Running => "RUN",
                    ProcessState::Blocked => "BLOCK",
                    ProcessState::Zombie => "ZOMBIE",
                };
                (p.pid, p.ppid, sname)
            })
            .collect();
        list.sort_by_key(|&(pid, _, _)| pid);
        list
    }
}

impl Default for ProcessManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vfs::GpuVfs;

    /// Helper: build a minimal VFS for testing.
    fn test_vfs() -> GpuVfs {
        GpuVfs::new()
    }

    /// Helper: create a manager with an init process (PID 1).
    fn manager_with_init() -> (ProcessManager, GpuVfs) {
        let mut pm = ProcessManager::new();
        let vfs = test_vfs();
        let pid = pm.create_init_process(None, "/", None);
        assert_eq!(pid, 1);
        (pm, vfs)
    }

    /// Helper: build a GpuSnapshot with deterministic values.
    fn dummy_snapshot(pc: u64) -> GpuSnapshot {
        let mut regs = [0i64; 32];
        regs[0] = 42;
        regs[1] = 100;
        regs[29] = 0xFF000; // FP
        GpuSnapshot {
            registers: regs,
            pc,
            flags: [0.0, 1.0, 0.0, 0.0], // Z set
        }
    }

    // ── PID allocation ───────────────────────────────────────────────

    #[test]
    fn test_pid_allocation_sequential() {
        let mut pm = ProcessManager::new();
        for expected in 1..=MAX_PROCESSES {
            let pid = pm.alloc_pid();
            assert_eq!(pid, expected);
            // Insert a dummy process so the slot is occupied.
            pm.processes
                .insert(pid, Process::new(pid, 0, ProcessState::Ready));
        }
        // Table full.
        assert_eq!(pm.alloc_pid(), -1);
    }

    #[test]
    fn test_pid_reuse_after_removal() {
        let mut pm = ProcessManager::new();
        // Fill the table.
        for _ in 1..=MAX_PROCESSES {
            let pid = pm.alloc_pid();
            pm.processes
                .insert(pid, Process::new(pid, 0, ProcessState::Ready));
        }
        assert_eq!(pm.alloc_pid(), -1);

        // Free PID 5.
        pm.processes.remove(&5);
        let reused = pm.alloc_pid();
        assert_eq!(reused, 5);
    }

    // ── Init process creation ────────────────────────────────────────

    #[test]
    fn test_create_init_process() {
        let mut pm = ProcessManager::new();
        let pid = pm.create_init_process(None, "/home", None);
        assert_eq!(pid, 1);

        let proc = pm.get(1).unwrap();
        assert_eq!(proc.ppid, 0);
        assert_eq!(proc.state, ProcessState::Ready);
        assert_eq!(proc.pc, ACTIVE_BASE);
        assert_eq!(proc.cwd, "/home");
        assert_eq!(proc.heap_break, HEAP_BASE);
        assert_eq!(proc.mmap_next, MMAP_BASE);
    }

    // ── Context switch ───────────────────────────────────────────────

    #[test]
    fn test_save_and_restore_context() {
        let (mut pm, _vfs) = manager_with_init();

        let snap = dummy_snapshot(0x10100);
        let fd_table = HashMap::new();
        pm.save_context(1, &snap, &fd_table);

        let proc = pm.get(1).unwrap();
        assert_eq!(proc.pc, 0x10100);
        assert_eq!(proc.registers[0], 42);
        assert_eq!(proc.flags[1], 1.0); // Z

        // Restore.
        let (restored, _fdt, cwd) = pm.restore_context(1).unwrap();
        assert_eq!(restored.pc, 0x10100);
        assert_eq!(restored.registers[0], 42);
        assert_eq!(cwd, "/");
        assert_eq!(pm.current_pid, 1);
        assert_eq!(pm.total_context_switches, 1);

        // Process should now be Running.
        assert_eq!(pm.get(1).unwrap().state, ProcessState::Running);
    }

    // ── Scheduling ───────────────────────────────────────────────────

    #[test]
    fn test_round_robin_scheduling() {
        let mut pm = ProcessManager::new();
        for _i in 1..=3 {
            let pid = pm.alloc_pid();
            pm.processes
                .insert(pid, Process::new(pid, 0, ProcessState::Ready));
        }

        // current_pid = -1, should pick PID 1.
        assert_eq!(pm.schedule_next(), Some(1));

        pm.current_pid = 1;
        assert_eq!(pm.schedule_next(), Some(2));

        pm.current_pid = 2;
        assert_eq!(pm.schedule_next(), Some(3));

        // Wrap around.
        pm.current_pid = 3;
        assert_eq!(pm.schedule_next(), Some(1));
    }

    #[test]
    fn test_schedule_skips_non_ready() {
        let mut pm = ProcessManager::new();
        for _i in 1..=3 {
            let pid = pm.alloc_pid();
            pm.processes
                .insert(pid, Process::new(pid, 0, ProcessState::Ready));
        }
        // Block PID 2.
        pm.get_mut(2).unwrap().state = ProcessState::Blocked;

        pm.current_pid = 1;
        assert_eq!(pm.schedule_next(), Some(3));
    }

    #[test]
    fn test_schedule_none_when_all_blocked() {
        let mut pm = ProcessManager::new();
        let pid = pm.alloc_pid();
        pm.processes
            .insert(pid, Process::new(pid, 0, ProcessState::Blocked));

        assert_eq!(pm.schedule_next(), None);
    }

    // ── Fork ─────────────────────────────────────────────────────────

    #[test]
    fn test_fork_creates_child() {
        let (mut pm, mut vfs) = manager_with_init();

        let snap = dummy_snapshot(0x10104); // PC already past SVC
        let fd_table = HashMap::new();
        let child_pid = pm.fork(1, &snap, &fd_table, &mut vfs);
        assert_eq!(child_pid, 2);

        // Child exists and is Ready.
        let child = pm.get(child_pid).unwrap();
        assert_eq!(child.ppid, 1);
        assert_eq!(child.state, ProcessState::Ready);
        assert_eq!(child.registers[0], 0); // child return value

        // Parent gets child PID.
        let parent = pm.get(1).unwrap();
        assert_eq!(parent.registers[0], child_pid as i64);
        assert_eq!(parent.state, ProcessState::Ready);

        // Stats.
        assert_eq!(pm.total_forks, 1);
    }

    #[test]
    fn test_fork_copies_parent_state() {
        let (mut pm, mut vfs) = manager_with_init();

        // Give parent some distinctive state.
        {
            let p = pm.get_mut(1).unwrap();
            p.cwd = "/usr/bin".to_string();
            p.heap_break = 0x70000;
            p.mmap_next = 0x90000;
            p.env.insert("PATH".to_string(), "/bin".to_string());
        }

        let snap = dummy_snapshot(0x10200);
        let fd_table = HashMap::new();
        let child_pid = pm.fork(1, &snap, &fd_table, &mut vfs);
        assert!(child_pid > 0);

        let child = pm.get(child_pid).unwrap();
        assert_eq!(child.cwd, "/usr/bin");
        assert_eq!(child.heap_break, 0x70000);
        assert_eq!(child.mmap_next, 0x90000);
        assert_eq!(child.env.get("PATH").map(|s| s.as_str()), Some("/bin"));
        assert_eq!(child.pc, 0x10200);
        // Register 1 should be copied from parent (before return-value overwrite).
        assert_eq!(child.registers[1], 100);
    }

    #[test]
    fn test_fork_bomb_protection() {
        let (mut pm, mut vfs) = manager_with_init();

        // Exhaust the fork limit.
        for _i in 0..MAX_FORKS_PER_PROCESS {
            let snap = dummy_snapshot(0x10104);
            let fd_table = HashMap::new();
            let result = pm.fork(1, &snap, &fd_table, &mut vfs);
            if result < 0 {
                // Process table may fill before fork limit.
                break;
            }
        }

        // Force fork_count to the limit.
        pm.get_mut(1).unwrap().fork_count = MAX_FORKS_PER_PROCESS;

        let snap = dummy_snapshot(0x10104);
        let fd_table = HashMap::new();
        let result = pm.fork(1, &snap, &fd_table, &mut vfs);
        assert_eq!(result, -1);
    }

    #[test]
    fn test_fork_table_full() {
        let mut pm = ProcessManager::new();
        let mut vfs = test_vfs();
        // Fill all slots manually.
        for i in 1..=MAX_PROCESSES {
            pm.processes
                .insert(i, Process::new(i, 0, ProcessState::Ready));
        }
        pm.next_pid = MAX_PROCESSES + 1;

        let snap = dummy_snapshot(0x10104);
        let fd_table = HashMap::new();
        let result = pm.fork(1, &snap, &fd_table, &mut vfs);
        assert_eq!(result, -1);
    }

    // ── Process exit ─────────────────────────────────────────────────

    #[test]
    fn test_process_exit_zombie() {
        let (mut pm, mut vfs) = manager_with_init();

        // Fork a child.
        let snap = dummy_snapshot(0x10104);
        let fd_table = HashMap::new();
        let child_pid = pm.fork(1, &snap, &fd_table, &mut vfs);
        assert!(child_pid > 0);

        pm.process_exit(child_pid, 42, &mut vfs);

        let child = pm.get(child_pid).unwrap();
        assert_eq!(child.state, ProcessState::Zombie);
        assert_eq!(child.exit_code, 42);
        assert!(child.fd_table.is_empty());
    }

    #[test]
    fn test_exit_wakes_waiting_parent() {
        let (mut pm, mut vfs) = manager_with_init();

        let snap = dummy_snapshot(0x10104);
        let fd_table = HashMap::new();
        let child_pid = pm.fork(1, &snap, &fd_table, &mut vfs);

        // Parent waits for child.
        pm.get_mut(1).unwrap().state = ProcessState::Blocked;
        pm.get_mut(1).unwrap().wait_target = child_pid;

        pm.process_exit(child_pid, 0, &mut vfs);

        // Parent should now be Ready.
        assert_eq!(pm.get(1).unwrap().state, ProcessState::Ready);
    }

    #[test]
    fn test_exit_wakes_parent_waiting_any() {
        let (mut pm, mut vfs) = manager_with_init();

        let snap = dummy_snapshot(0x10104);
        let fd_table = HashMap::new();
        let child_pid = pm.fork(1, &snap, &fd_table, &mut vfs);

        // Parent waits for any child (-1).
        pm.get_mut(1).unwrap().state = ProcessState::Blocked;
        pm.get_mut(1).unwrap().wait_target = -1;

        pm.process_exit(child_pid, 0, &mut vfs);

        assert_eq!(pm.get(1).unwrap().state, ProcessState::Ready);
    }

    #[test]
    fn test_orphan_reparenting() {
        let (mut pm, mut vfs) = manager_with_init();

        // Create parent (PID 2) and grandchild (PID 3).
        let snap = dummy_snapshot(0x10104);
        let fd_table = HashMap::new();
        let pid2 = pm.fork(1, &snap, &fd_table, &mut vfs);
        assert_eq!(pid2, 2);

        let snap2 = dummy_snapshot(0x10108);
        let fd_table2 = HashMap::new();
        let pid3 = pm.fork(2, &snap2, &fd_table2, &mut vfs);
        assert_eq!(pid3, 3);

        // PID 2 exits -> PID 3 should be reparented to PID 1.
        pm.process_exit(2, 0, &mut vfs);

        assert_eq!(pm.get(3).unwrap().ppid, 1);
    }

    #[test]
    fn test_orphan_zombie_wakes_init() {
        let (mut pm, mut vfs) = manager_with_init();

        // PID 2 forks PID 3, PID 3 exits (zombie), then PID 2 exits.
        let snap = dummy_snapshot(0x10104);
        let fd_table = HashMap::new();
        let pid2 = pm.fork(1, &snap, &fd_table, &mut vfs);

        let snap2 = dummy_snapshot(0x10108);
        let fd_table2 = HashMap::new();
        let pid3 = pm.fork(pid2, &snap2, &fd_table2, &mut vfs);

        // PID 3 becomes zombie.
        pm.process_exit(pid3, 0, &mut vfs);
        assert_eq!(pm.get(pid3).unwrap().ppid, pid2);

        // Init blocks waiting for any child.
        pm.get_mut(1).unwrap().state = ProcessState::Blocked;
        pm.get_mut(1).unwrap().wait_target = -1;

        // PID 2 exits, orphan PID 3 (zombie) reparented to init.
        pm.process_exit(pid2, 0, &mut vfs);

        assert_eq!(pm.get(pid3).unwrap().ppid, 1);
        // Init should be woken because a zombie was reparented to it.
        assert_eq!(pm.get(1).unwrap().state, ProcessState::Ready);
    }

    // ── Kill ─────────────────────────────────────────────────────────

    #[test]
    fn test_kill_sigkill() {
        let (mut pm, mut vfs) = manager_with_init();

        let snap = dummy_snapshot(0x10104);
        let fd_table = HashMap::new();
        let child_pid = pm.fork(1, &snap, &fd_table, &mut vfs);

        let result = pm.kill_process(child_pid, SIGKILL, 1, &mut vfs);
        assert_eq!(result, 0);
        assert_eq!(pm.get(child_pid).unwrap().state, ProcessState::Zombie);
        assert_eq!(pm.get(child_pid).unwrap().exit_code, 128 + SIGKILL);
    }

    #[test]
    fn test_kill_sigterm() {
        let (mut pm, mut vfs) = manager_with_init();

        let snap = dummy_snapshot(0x10104);
        let fd_table = HashMap::new();
        let child_pid = pm.fork(1, &snap, &fd_table, &mut vfs);

        let result = pm.kill_process(child_pid, SIGTERM, 1, &mut vfs);
        assert_eq!(result, 0);
        assert_eq!(pm.get(child_pid).unwrap().pending_signal, Some(SIGTERM));
        // Process should still be alive.
        assert_ne!(pm.get(child_pid).unwrap().state, ProcessState::Zombie);
    }

    #[test]
    fn test_kill_sigterm_unblocks() {
        let (mut pm, mut vfs) = manager_with_init();

        let snap = dummy_snapshot(0x10104);
        let fd_table = HashMap::new();
        let child_pid = pm.fork(1, &snap, &fd_table, &mut vfs);

        pm.get_mut(child_pid).unwrap().state = ProcessState::Blocked;

        pm.kill_process(child_pid, SIGTERM, 1, &mut vfs);
        assert_eq!(pm.get(child_pid).unwrap().state, ProcessState::Ready);
    }

    #[test]
    fn test_kill_signal_zero_existence_check() {
        let (mut pm, mut vfs) = manager_with_init();
        assert_eq!(pm.kill_process(1, 0, 1, &mut vfs), 0);
        assert_eq!(pm.kill_process(99, 0, 1, &mut vfs), -1);
    }

    #[test]
    fn test_kill_nonexistent_process() {
        let (mut pm, mut vfs) = manager_with_init();
        assert_eq!(pm.kill_process(99, SIGKILL, 1, &mut vfs), -1);
    }

    #[test]
    fn test_kill_zombie_fails() {
        let (mut pm, mut vfs) = manager_with_init();

        let snap = dummy_snapshot(0x10104);
        let fd_table = HashMap::new();
        let child_pid = pm.fork(1, &snap, &fd_table, &mut vfs);

        pm.process_exit(child_pid, 0, &mut vfs);
        assert_eq!(pm.kill_process(child_pid, SIGKILL, 1, &mut vfs), -1);
    }

    // ── Zombie reaping ───────────────────────────────────────────────

    #[test]
    fn test_reap_specific_zombie() {
        let (mut pm, mut vfs) = manager_with_init();

        let snap = dummy_snapshot(0x10104);
        let fd_table = HashMap::new();
        let child_pid = pm.fork(1, &snap, &fd_table, &mut vfs);

        pm.process_exit(child_pid, 77, &mut vfs);

        let zombie = pm.reap_zombie(1, child_pid);
        assert!(zombie.is_some());
        let z = zombie.unwrap();
        assert_eq!(z.pid, child_pid);
        assert_eq!(z.exit_code, 77);

        // Should be removed from the table.
        assert!(pm.get(child_pid).is_none());
    }

    #[test]
    fn test_reap_any_zombie() {
        let (mut pm, mut vfs) = manager_with_init();

        let snap = dummy_snapshot(0x10104);
        let fd_table = HashMap::new();
        let child_pid = pm.fork(1, &snap, &fd_table, &mut vfs);

        pm.process_exit(child_pid, 0, &mut vfs);

        let zombie = pm.reap_zombie(1, -1);
        assert!(zombie.is_some());
        assert_eq!(zombie.unwrap().pid, child_pid);
    }

    #[test]
    fn test_reap_no_zombie_returns_none() {
        let (mut pm, _vfs) = manager_with_init();
        assert!(pm.reap_zombie(1, -1).is_none());
        assert!(pm.reap_zombie(1, 99).is_none());
    }

    #[test]
    fn test_reap_wrong_parent_returns_none() {
        let (mut pm, mut vfs) = manager_with_init();

        let snap = dummy_snapshot(0x10104);
        let fd_table = HashMap::new();
        let child_pid = pm.fork(1, &snap, &fd_table, &mut vfs);

        pm.process_exit(child_pid, 0, &mut vfs);

        // PID 99 is not the parent.
        assert!(pm.reap_zombie(99, child_pid).is_none());
    }

    // ── Backing address ──────────────────────────────────────────────

    #[test]
    fn test_backing_addr_layout() {
        assert_eq!(ProcessManager::backing_addr(1), BACKING_STORE_BASE);
        assert_eq!(
            ProcessManager::backing_addr(2),
            BACKING_STORE_BASE + BACKING_STORE_SIZE
        );
        assert_eq!(
            ProcessManager::backing_addr(15),
            BACKING_STORE_BASE + 14 * BACKING_STORE_SIZE
        );
    }

    // ── ps helper ────────────────────────────────────────────────────

    #[test]
    fn test_ps_list() {
        let (mut pm, mut vfs) = manager_with_init();

        let snap = dummy_snapshot(0x10104);
        let fd_table = HashMap::new();
        pm.fork(1, &snap, &fd_table, &mut vfs);

        let list = pm.ps_list();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].0, 1); // PID 1
        assert_eq!(list[1].0, 2); // PID 2
    }

    // ── has_live_children ────────────────────────────────────────────

    #[test]
    fn test_has_live_children() {
        let (mut pm, mut vfs) = manager_with_init();

        assert!(!pm.has_live_children(1));

        let snap = dummy_snapshot(0x10104);
        let fd_table = HashMap::new();
        let child_pid = pm.fork(1, &snap, &fd_table, &mut vfs);

        assert!(pm.has_live_children(1));

        pm.process_exit(child_pid, 0, &mut vfs);
        assert!(!pm.has_live_children(1));
    }

    // ── Comprehensive multi-process scenario ─────────────────────────

    #[test]
    fn test_full_lifecycle() {
        let mut pm = ProcessManager::new();
        let mut vfs = test_vfs();

        // 1. Create init.
        let init_pid = pm.create_init_process(None, "/", None);
        assert_eq!(init_pid, 1);

        // 2. Fork child from init.
        let snap = dummy_snapshot(0x10104);
        let fd_table = HashMap::new();
        let child_pid = pm.fork(init_pid, &snap, &fd_table, &mut vfs);
        assert!(child_pid > 0);

        // 3. Schedule should pick one of the ready processes.
        pm.current_pid = -1;
        let next = pm.schedule_next();
        assert!(next.is_some());

        // 4. Run child, then exit.
        pm.get_mut(child_pid).unwrap().state = ProcessState::Running;
        pm.process_exit(child_pid, 0, &mut vfs);

        // 5. Reap.
        let z = pm.reap_zombie(init_pid, child_pid);
        assert!(z.is_some());
        assert_eq!(z.unwrap().exit_code, 0);

        // 6. Child should be gone.
        assert!(pm.get(child_pid).is_none());
        assert_eq!(pm.processes.len(), 1); // only init
    }
}
