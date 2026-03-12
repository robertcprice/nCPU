//! GPU Microkernel - A minimal OS that runs entirely on the Metal GPU
//!
//! This module implements a tiny GPU-native microkernel that doesn't require
//! Linux. It provides:
//! - Process scheduling and management
//! - Memory management (simple page allocator)
//! - VFS with files, directories, pipes
//! - Device emulation (timer, console, random)
//! - Syscall interface
//! - ARM64 execution context

use std::collections::HashMap;

/// Maximum number of processes
pub const MAX_PROCESSES: usize = 64;

/// Maximum number of open file descriptors per process
pub const MAX_FDS: usize = 128;

/// Page size (4KB)
pub const PAGE_SIZE: u64 = 4096;

/// Process states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessState {
    Running,
    Ready,
    Blocked,
    Zombie,
    Dead,
}

/// Process priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Priority(pub u8);

impl Priority {
    pub const IDLE: Priority = Priority(0);
    pub const LOW: Priority = Priority(32);
    pub const NORMAL: Priority = Priority(64);
    pub const HIGH: Priority = Priority(96);
    pub const REAL_TIME: Priority = Priority(128);
}

/// Memory region flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryFlags(u32);

impl MemoryFlags {
    pub const READ: MemoryFlags = MemoryFlags(1);
    pub const WRITE: MemoryFlags = MemoryFlags(2);
    pub const EXEC: MemoryFlags = MemoryFlags(4);
    pub const SHARED: MemoryFlags = MemoryFlags(8);
    pub const PRIVATE: MemoryFlags = MemoryFlags(16);
    pub const ANONYMOUS: MemoryFlags = MemoryFlags(32);

    pub fn bits(&self) -> u32 { self.0 }
}

/// Memory region description
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    pub base: u64,
    pub size: u64,
    pub flags: MemoryFlags,
    pub file_backed: Option<(String, u64)>, // (path, offset)
}

/// File descriptor kind
#[derive(Debug, Clone)]
pub enum FdKind {
    File(String),      // path
    PipeRead(usize),  // pipe index
    PipeWrite(usize),
    Console,
    Socket,
    Device(String),
}

/// A file descriptor entry
#[derive(Debug, Clone)]
pub struct FdEntry {
    pub kind: FdKind,
    pub flags: u32,
    pub offset: u64,
}

/// Process control block
#[derive(Debug, Clone)]
pub struct Process {
    pub pid: u32,
    pub ppid: u32,
    pub state: ProcessState,
    pub priority: Priority,
    pub pc: u64,
    pub sp: u64,
    pub regs: [u64; 32],  // ARM64 registers x0-x30, plus SP in regs[31]
    pub flags: u64,       // NZCV flags
    pub heap_base: u64,
    pub heap_brk: u64,
    pub mm: MemoryManager,
    pub fd_table: Vec<Option<FdEntry>>,
    pub cwd: String,
    pub exit_code: Option<i32>,
    pub parent_wait: Option<u32>, // PID of parent waiting
}

impl Process {
    /// Create a new process
    pub fn new(pid: u32, ppid: u32, entry: u64, sp: u64) -> Self {
        let mut regs = [0u64; 32];
        regs[0] = 0;  // x0 = first arg (argc)
        regs[1] = 0;  // x1 = argv pointer (will be set up)
        regs[31] = sp; // SP

        Process {
            pid,
            ppid,
            state: ProcessState::Ready,
            priority: Priority::NORMAL,
            pc: entry,
            sp,
            regs,
            flags: 0,
            heap_base: 0,
            heap_brk: 0,
            mm: MemoryManager::new(),
            fd_table: vec![None; MAX_FDS],
            cwd: "/".to_string(),
            exit_code: None,
            parent_wait: None,
        }
    }

    /// Allocate a file descriptor
    pub fn alloc_fd(&mut self, kind: FdKind, flags: u32) -> Option<i32> {
        for (i, entry) in self.fd_table.iter_mut().enumerate() {
            if entry.is_none() {
                *entry = Some(FdEntry {
                    kind,
                    flags,
                    offset: 0,
                });
                return Some(i as i32);
            }
        }
        None
    }

    /// Get file descriptor entry
    pub fn get_fd(&self, fd: i32) -> Option<&FdEntry> {
        if fd >= 0 && (fd as usize) < self.fd_table.len() {
            self.fd_table[fd as usize].as_ref()
        } else {
            None
        }
    }

    /// Get mutable file descriptor entry
    pub fn get_fd_mut(&mut self, fd: i32) -> Option<&mut FdEntry> {
        if fd >= 0 && (fd as usize) < self.fd_table.len() {
            self.fd_table[fd as usize].as_mut()
        } else {
            None
        }
    }

    /// Close a file descriptor
    pub fn close_fd(&mut self, fd: i32) -> bool {
        if fd >= 0 && (fd as usize) < self.fd_table.len() {
            self.fd_table[fd as usize].is_some()
        } else {
            false
        }
    }
}

/// Simple memory manager
#[derive(Debug, Clone)]
pub struct MemoryManager {
    pub regions: Vec<MemoryRegion>,
    next_addr: u64,
}

impl MemoryManager {
    pub fn new() -> Self {
        // Start user space at 0x10000 (64KB)
        MemoryManager {
            regions: Vec::new(),
            next_addr: 0x10000,
        }
    }

    /// Allocate memory region
    pub fn mmap(&mut self, addr: Option<u64>, size: u64, flags: MemoryFlags) -> u64 {
        let base = addr.unwrap_or(self.next_addr);
        let aligned = (base + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
        let aligned_size = (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);

        self.regions.push(MemoryRegion {
            base: aligned,
            size: aligned_size,
            flags,
            file_backed: None,
        });

        self.next_addr = aligned + aligned_size;
        aligned
    }

    /// Free memory region
    pub fn munmap(&mut self, addr: u64, _size: u64) {
        self.regions.retain(|r| r.base != addr);
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Pipe buffer for inter-process communication
#[derive(Debug, Clone)]
pub struct PipeBuffer {
    pub data: Vec<u8>,
    pub read_pos: usize,
    pub write_pos: usize,
    pub capacity: usize,
    pub readers: usize,
    pub writers: usize,
}

impl PipeBuffer {
    pub fn new(capacity: usize) -> Self {
        PipeBuffer {
            data: vec![0u8; capacity],
            read_pos: 0,
            write_pos: 0,
            capacity,
            readers: 1,
            writers: 1,
        }
    }

    /// Read from pipe
    pub fn read(&mut self, max_len: usize) -> Option<Vec<u8>> {
        if self.read_pos == self.write_pos && self.writers == 0 {
            return None; // EOF
        }

        let available = if self.write_pos >= self.read_pos {
            self.write_pos - self.read_pos
        } else {
            self.capacity - self.read_pos + self.write_pos
        };

        let to_read = available.min(max_len);
        let mut result = Vec::with_capacity(to_read);

        for _ in 0..to_read {
            result.push(self.data[self.read_pos]);
            self.read_pos = (self.read_pos + 1) % self.capacity;
        }

        Some(result)
    }

    /// Write to pipe
    pub fn write(&mut self, data: &[u8]) -> usize {
        let available = self.capacity - if self.write_pos >= self.read_pos {
            self.write_pos - self.read_pos
        } else {
            self.capacity - self.read_pos + self.write_pos
        };

        let to_write = available.min(data.len());

        for &byte in data.iter().take(to_write) {
            self.data[self.write_pos] = byte;
            self.write_pos = (self.write_pos + 1) % self.capacity;
        }

        to_write
    }
}

/// Timer device - provides time and scheduling
#[derive(Debug, Clone)]
pub struct TimerDevice {
    pub ticks: u64,
    pub freq_hz: u64,  // 100 Hz default
}

impl TimerDevice {
    pub fn new() -> Self {
        TimerDevice {
            ticks: 0,
            freq_hz: 100,
        }
    }

    /// Advance time (called on each scheduler tick)
    pub fn tick(&mut self) {
        self.ticks += 1;
    }

    /// Get current time in nanoseconds
    pub fn get_time_ns(&self) -> u64 {
        // Convert ticks to nanoseconds
        let secs = self.ticks / self.freq_hz;
        let nsecs = ((self.ticks % self.freq_hz) * 1_000_000_000) / self.freq_hz;
        secs * 1_000_000_000 + nsecs
    }

    /// Get uptime in seconds
    pub fn uptime(&self) -> u64 {
        self.ticks / self.freq_hz
    }
}

impl Default for TimerDevice {
    fn default() -> Self {
        Self::new()
    }
}

/// Random device - provides deterministic randomness
#[derive(Debug, Clone)]
pub struct RandomDevice {
    seed: u64,
}

impl RandomDevice {
    pub fn new() -> Self {
        RandomDevice {
            seed: 0x12345678, // Fixed seed for determinism
        }
    }

    /// Generate next random u32 (LCG)
    pub fn next_u32(&mut self) -> u32 {
        self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
        (self.seed >> 16) as u32
    }

    /// Fill buffer with random bytes
    pub fn fill(&mut self, buf: &mut [u8]) {
        for chunk in buf.chunks_mut(4) {
            let val = self.next_u32();
            for (i, byte) in chunk.iter_mut().enumerate() {
                *byte = (val >> (i * 8)) as u8;
            }
        }
    }
}

impl Default for RandomDevice {
    fn default() -> Self {
        Self::new()
    }
}

/// Console device - for I/O
#[derive(Debug, Clone)]
pub struct ConsoleDevice {
    pub input_buffer: Vec<u8>,
    pub output_buffer: Vec<u8>,
}

impl ConsoleDevice {
    pub fn new() -> Self {
        ConsoleDevice {
            input_buffer: Vec::new(),
            output_buffer: Vec::new(),
        }
    }

    /// Write to console
    pub fn write(&mut self, data: &[u8]) {
        self.output_buffer.extend_from_slice(data);
    }

    /// Read from console (non-blocking)
    pub fn read(&mut self, max: usize) -> Option<Vec<u8>> {
        if self.input_buffer.is_empty() {
            None
        } else {
            let len = max.min(self.input_buffer.len());
            Some(self.input_buffer.drain(..len).collect())
        }
    }

    /// Check if input available
    pub fn has_input(&self) -> bool {
        !self.input_buffer.is_empty()
    }
}

impl Default for ConsoleDevice {
    fn default() -> Self {
        Self::new()
    }
}

/// The GPU Microkernel
pub struct GpuMicrokernel {
    pub processes: HashMap<u32, Process>,
    pub ready_queue: Vec<u32>,       // PIDs of ready processes
    pub current_pid: Option<u32>,
    pub next_pid: u32,
    pub ticks: u64,
    pub vfs: MicroVfs,
    pub timer: TimerDevice,
    pub random: RandomDevice,
    pub console: ConsoleDevice,
}

impl GpuMicrokernel {
    /// Create a new microkernel
    pub fn new() -> Self {
        GpuMicrokernel {
            processes: HashMap::new(),
            ready_queue: Vec::new(),
            current_pid: None,
            next_pid: 1,
            ticks: 0,
            vfs: MicroVfs::new(),
            timer: TimerDevice::new(),
            random: RandomDevice::new(),
            console: ConsoleDevice::new(),
        }
    }

    /// Fork current process
    pub fn fork(&mut self) -> Option<(u32, u32)> {
        let parent_pid = self.current_pid?;
        let parent = self.processes.get(&parent_pid)?.clone();

        let child_pid = self.next_pid;
        self.next_pid += 1;

        let mut child = parent;
        child.pid = child_pid;
        child.ppid = parent_pid;
        child.state = ProcessState::Ready;
        child.parent_wait = None;

        self.processes.insert(child_pid, child);
        self.ready_queue.push(child_pid);

        Some((parent_pid, child_pid))
    }

    /// Execute new program in current process
    pub fn exec(&mut self, entry: u64, argc: u64, argv_ptr: u64) {
        if let Some(pid) = self.current_pid {
            if let Some(proc) = self.processes.get_mut(&pid) {
                proc.pc = entry;
                proc.regs[0] = argc;
                proc.regs[1] = argv_ptr;
                proc.regs[31] = proc.sp; // Reset stack
                proc.heap_brk = proc.heap_base;
            }
        }
    }

    /// Schedule next process (round-robin)
    pub fn schedule(&mut self) -> Option<u32> {
        self.schedule_rr(false)
    }

    /// Neural-guided scheduling - uses ML heuristics to pick best process
    pub fn schedule_neural(&mut self) -> Option<u32> {
        // First do regular round-robin housekeeping
        self.timer.tick();
        self.ticks += 1;

        if self.ready_queue.is_empty() {
            return None;
        }

        // Update current process state
        if let Some(current) = self.current_pid {
            if let Some(proc) = self.processes.get_mut(&current) {
                if proc.state == ProcessState::Running {
                    proc.state = ProcessState::Ready;
                }
            }
        }

        // Neural scoring: prioritize based on multiple heuristics
        // Score = w1*priority + w2*cpu_burst + w3*io_wait - w4*age
        let mut scores: Vec<(u32, f32)> = self.ready_queue.iter().map(|&pid| {
            let score = self.compute_neural_score(pid);
            (pid, score)
        }).collect();

        // Higher score = higher priority
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select best process
        if let Some(&(pid, _)) = scores.first() {
            // Remove from current position in queue
            self.ready_queue.retain(|&p| p != pid);
            // Add to front (will be selected next)
            self.ready_queue.insert(0, pid);

            if let Some(proc) = self.processes.get_mut(&pid) {
                proc.state = ProcessState::Running;
            }

            self.current_pid = Some(pid);
            return Some(pid);
        }

        None
    }

    /// Compute neural score for a process based on heuristics
    fn compute_neural_score(&self, pid: u32) -> f32 {
        let proc = match self.processes.get(&pid) {
            Some(p) => p,
            None => return 0.0,
        };

        // Weight factors (these could be learned)
        let w_priority = 1.0;    // Priority weight
        let w_cpu_burst = 0.8;  // Recent CPU usage
        let w_io_wait = 0.5;     // I/O waiting bonus
        let w_age = 0.3;         // Age bonus (older = higher)

        // Priority score (0-255 normalized to 0-1)
        let priority_score = proc.priority.0 as f32 / 255.0;

        // CPU burst estimation (would be tracked in real implementation)
        // For now, use a simple heuristic based on state
        let cpu_burst_score = match proc.state {
            ProcessState::Running => 0.8,   // Currently running = high CPU
            ProcessState::Ready => 0.5,     // Ready = medium
            ProcessState::Blocked => 0.1,   // Blocked = low
            _ => 0.0,
        };

        // I/O wait bonus - processes with open file descriptors
        let io_wait_score = if proc.fd_table.iter().any(|f| f.is_some()) {
            0.6  // Has open files = might be waiting for I/O
        } else {
            0.2
        };

        // Age score - how long since last scheduled
        let age_score = (self.ticks % 100) as f32 / 100.0;

        // Compute weighted score
        let score = w_priority * priority_score
            + w_cpu_burst * cpu_burst_score
            + w_io_wait * io_wait_score
            + w_age * age_score;

        score
    }

    /// Round-robin scheduler with optional neural enhancement
    fn schedule_rr(&mut self, _neural: bool) -> Option<u32> {
        // Tick the timer
        self.timer.tick();
        self.ticks += 1;

        // Find next ready process
        if self.ready_queue.is_empty() {
            return None;
        }

        // Simple round-robin: move current to back, get next
        if let Some(current) = self.current_pid {
            if let Some(proc) = self.processes.get_mut(&current) {
                if proc.state == ProcessState::Running {
                    proc.state = ProcessState::Ready;
                }
            }
        }

        // Get next from queue
        let next_pid = self.ready_queue.remove(0);
        self.ready_queue.push(next_pid);

        if let Some(proc) = self.processes.get_mut(&next_pid) {
            proc.state = ProcessState::Running;
        }

        self.current_pid = Some(next_pid);
        Some(next_pid)
    }

    /// Exit current process
    pub fn exit(&mut self, code: i32) {
        if let Some(pid) = self.current_pid {
            if let Some(proc) = self.processes.get_mut(&pid) {
                proc.state = ProcessState::Zombie;
                proc.exit_code = Some(code);
            }

            // Remove from ready queue
            self.ready_queue.retain(|&p| p != pid);

            // Wake up parent if waiting
            if let Some(proc) = self.processes.get(&pid) {
                let ppid = proc.ppid;
                if let Some(parent) = self.processes.get_mut(&ppid) {
                    parent.parent_wait = None;
                    parent.state = ProcessState::Ready;
                    if !self.ready_queue.contains(&ppid) {
                        self.ready_queue.push(ppid);
                    }
                }
            }

            self.current_pid = None;
        }
    }

    /// Wait for child process
    pub fn wait(&mut self) -> Option<i32> {
        let pid = self.current_pid?;

        // Find a zombie child - collect keys first to avoid borrow issues
        let zombie_pids: Vec<u32> = self.processes
            .iter()
            .filter(|(_, p)| p.ppid == pid && p.state == ProcessState::Zombie)
            .map(|(&k, _)| k)
            .collect();

        // Reap first zombie child
        if let Some(child_pid) = zombie_pids.first() {
            if let Some(child) = self.processes.get(child_pid) {
                let code = child.exit_code.unwrap_or(0);
                self.processes.remove(child_pid);
                return Some(code as i32);
            }
        }

        // Block if no zombie children
        if let Some(proc) = self.processes.get_mut(&pid) {
            proc.state = ProcessState::Blocked;
            proc.parent_wait = Some(pid);
        }

        None
    }

    /// Add process to ready queue
    pub fn wake(&mut self, pid: u32) {
        if let Some(proc) = self.processes.get_mut(&pid) {
            if proc.state == ProcessState::Blocked || proc.state == ProcessState::Ready {
                proc.state = ProcessState::Ready;
                if !self.ready_queue.contains(&pid) {
                    self.ready_queue.push(pid);
                }
            }
        }
    }

    /// Block current process
    pub fn block(&mut self) {
        if let Some(pid) = self.current_pid {
            if let Some(proc) = self.processes.get_mut(&pid) {
                proc.state = ProcessState::Blocked;
            }
            self.ready_queue.retain(|&p| p != pid);
            self.current_pid = None;
        }
    }

    /// Spawn a new process with the given entry point and arguments
    pub fn spawn(&mut self, entry: u64, _argv: Vec<String>, _envp: Vec<String>) -> u32 {
        let pid = self.next_pid;
        self.next_pid += 1;

        // Set up stack - start high and grow down
        let stack_base: u64 = 0x7fff_ffff_0000;

        // Use Process::new which sets up the correct structure
        let mut process = Process::new(pid, 0, entry, stack_base);

        // Set up heap
        process.heap_base = 0x0000_1000;
        process.heap_brk = process.heap_base + (16 * 1024 * 1024); // 16MB heap

        self.processes.insert(pid, process);
        self.ready_queue.push(pid);

        pid
    }

    /// Run the microkernel scheduler (returns when no more processes)
    pub fn run(&mut self, max_ticks: u64) -> (u32, i32) {
        // Ensure we have a process to run
        if self.ready_queue.is_empty() {
            return (0, -1);
        }

        // Initialize first process
        if self.current_pid.is_none() {
            if let Some(pid) = self.ready_queue.first() {
                self.current_pid = Some(*pid);
                if let Some(proc) = self.processes.get_mut(pid) {
                    proc.state = ProcessState::Running;
                }
            }
        }

        // Run scheduler loop
        while self.ticks < max_ticks {
            // Check if we have processes
            let has_runnable = self.processes.values()
                .any(|p| p.state == ProcessState::Running || p.state == ProcessState::Ready);

            if !has_runnable {
                // No runnable processes - find exit code of last process
                let exit_code = self.processes.values()
                    .find(|p| p.state == ProcessState::Zombie)
                    .and_then(|p| p.exit_code)
                    .unwrap_or(0);
                break;
            }

            // Schedule next tick
            self.schedule();
            self.ticks += 1;
        }

        let final_pid = self.current_pid.unwrap_or(0);
        let exit_code = self.processes.get(&final_pid)
            .and_then(|p| p.exit_code)
            .unwrap_or(-1);

        (final_pid, exit_code)
    }
}

impl Default for GpuMicrokernel {
    fn default() -> Self {
        Self::new()
    }
}

/// Minimal VFS for the microkernel
#[derive(Debug, Clone)]
pub struct MicroVfs {
    pub files: HashMap<String, Vec<u8>>,
    pub directories: Vec<String>,
    pub symlinks: HashMap<String, String>,
    pub pipes: Vec<PipeBuffer>,
}

impl MicroVfs {
    pub fn new() -> Self {
        MicroVfs {
            files: HashMap::new(),
            directories: vec!["/".to_string()],
            symlinks: HashMap::new(),
            pipes: Vec::new(),
        }
    }

    /// Create a pipe
    pub fn pipe(&mut self) -> (i32, i32) {
        let idx = self.pipes.len();
        self.pipes.push(PipeBuffer::new(4096));
        // Return fake file descriptors
        (-(idx as i32 + 1), idx as i32)
    }

    /// Read from pipe
    pub fn read_pipe(&mut self, idx: usize, buf: &mut [u8]) -> Option<usize> {
        if let Some(pipe) = self.pipes.get_mut(idx) {
            let data = pipe.read(buf.len())?;
            let len = data.len();
            buf[..len].copy_from_slice(&data);
            Some(len)
        } else {
            None
        }
    }

    /// Write to pipe
    pub fn write_pipe(&mut self, idx: usize, data: &[u8]) -> Option<usize> {
        if let Some(pipe) = self.pipes.get_mut(idx) {
            Some(pipe.write(data))
        } else {
            None
        }
    }

    /// Create directory
    pub fn mkdir(&mut self, path: &str) -> i32 {
        if self.directories.contains(&path.to_string()) {
            return -1; // Already exists
        }
        // Create parent directories as needed
        let mut current = "/".to_string();
        for part in path.trim_start_matches('/').split('/') {
            current = format!("{}/{}", current.trim_end_matches('/'), part);
            if !self.directories.contains(&current) {
                self.directories.push(current.clone());
            }
        }
        0
    }

    /// Create file
    pub fn create(&mut self, path: &str, data: &[u8]) -> i32 {
        if self.files.contains_key(path) {
            self.files.insert(path.to_string(), data.to_vec());
        } else {
            self.files.insert(path.to_string(), data.to_vec());
        }
        0
    }

    /// Read file
    pub fn read(&self, path: &str) -> Option<Vec<u8>> {
        self.files.get(path).cloned()
    }

    /// Write file
    pub fn write(&mut self, path: &str, data: &[u8]) -> Option<usize> {
        if let Some(existing) = self.files.get_mut(path) {
            let len = data.len();
            *existing = data.to_vec();
            Some(len)
        } else {
            self.files.insert(path.to_string(), data.to_vec());
            Some(data.len())
        }
    }

    /// Delete file
    pub fn unlink(&mut self, path: &str) -> i32 {
        if self.files.remove(path).is_some() {
            0
        } else {
            -1
        }
    }

    /// Remove directory
    pub fn rmdir(&mut self, path: &str) -> i32 {
        if self.directories.contains(&path.to_string()) {
            self.directories.retain(|d| d != path);
            0
        } else {
            -1
        }
    }

    /// Create symlink
    pub fn symlink(&mut self, target: &str, link: &str) -> i32 {
        self.symlinks.insert(link.to_string(), target.to_string());
        0
    }

    /// Read symlink
    pub fn readlink(&self, path: &str) -> Option<String> {
        self.symlinks.get(path).cloned()
    }
}

impl Default for MicroVfs {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_creation() {
        let proc = Process::new(1, 0, 0x400000, 0x800000);
        assert_eq!(proc.pid, 1);
        assert_eq!(proc.ppid, 0);
        assert_eq!(proc.state, ProcessState::Ready);
    }

    #[test]
    fn test_fd_allocation() {
        let mut proc = Process::new(1, 0, 0, 0);
        let fd = proc.alloc_fd(FdKind::Console, 0);
        assert!(fd.is_some());
        assert_eq!(fd.unwrap(), 0);
    }

    #[test]
    fn test_microkernel_fork() {
        let mut kernel = GpuMicrokernel::new();

        // Create initial process
        let mut proc = Process::new(1, 0, 0x400000, 0x800000);
        proc.heap_base = 0x60000;
        kernel.processes.insert(1, proc);
        kernel.ready_queue.push(1);
        kernel.current_pid = Some(1);

        // Fork
        let (parent, child) = kernel.fork().unwrap();
        assert_eq!(parent, 1);
        assert_eq!(child, 2);
    }

    #[test]
    fn test_pipe() {
        let mut vfs = MicroVfs::new();
        let (rfd, wfd) = vfs.pipe();

        // Write then read
        vfs.write_pipe(wfd as usize, b"hello").unwrap();
        let mut buf = [0u8; 64];
        let len = vfs.read_pipe(rfd.unsigned_abs() as usize, &mut buf).unwrap();
        assert_eq!(&buf[..len], b"hello");
    }
}
