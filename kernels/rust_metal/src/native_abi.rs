//! Native nCPU Hypercall ABI -- SVC #0x4E43
//!
//! This module implements the nCPU-native hypercall interface.  Programs using
//! this ABI issue `SVC #0x4E43` (the ASCII codes for "NC") instead of the
//! Linux-compatible `SVC #0`.  The native ABI is cleaner, faster, and
//! purpose-built for the nCPU execution environment.
//!
//! # Register convention
//!
//! | Register | Purpose                          |
//! |----------|----------------------------------|
//! | x16      | Service group                    |
//! | x8       | Operation ID within the group    |
//! | x0..x5   | Arguments                        |
//! | x6       | Flags                            |
//! | x7       | Capability token (future use)    |
//! | x0       | Return value                     |
//!
//! # Service groups
//!
//! | ID | Name    | Description                  |
//! |----|---------|------------------------------|
//! | 0  | TASK    | Process lifecycle            |
//! | 1  | MEMORY  | Memory management            |
//! | 2  | VFS     | File operations              |
//! | 3  | IPC     | Inter-process communication  |
//! | 4  | TIMER   | Timing services              |
//! | 5  | CONSOLE | Terminal I/O                 |
//! | 6  | BRIDGE  | Host bridge (Python interop) |
//! | 7  | DEBUG   | Debugging services           |

use std::time::SystemTime;

use crate::vfs::GpuVfs;

// ── SVC discriminator ────────────────────────────────────────────────────

/// The SVC immediate used by the native nCPU ABI: 0x4E43 == "NC" in ASCII.
pub const SVC_NATIVE: u16 = 0x4E43;

// ── Service group IDs (carried in x16) ───────────────────────────────────

pub const SERVICE_TASK: u64 = 0;
pub const SERVICE_MEMORY: u64 = 1;
pub const SERVICE_VFS: u64 = 2;
pub const SERVICE_IPC: u64 = 3;
pub const SERVICE_TIMER: u64 = 4;
pub const SERVICE_CONSOLE: u64 = 5;
pub const SERVICE_BRIDGE: u64 = 6;
pub const SERVICE_DEBUG: u64 = 7;

// ── TASK operations (x8 values) ──────────────────────────────────────────

pub const TASK_EXIT: u64 = 0;
pub const TASK_YIELD: u64 = 1;
pub const TASK_SPAWN: u64 = 2;
pub const TASK_GETID: u64 = 3;

// ── MEMORY operations ────────────────────────────────────────────────────

pub const MEM_ALLOC: u64 = 0;
pub const MEM_FREE: u64 = 1;
pub const MEM_MAP: u64 = 2;
pub const MEM_QUERY: u64 = 3;

// ── VFS operations ───────────────────────────────────────────────────────

pub const VFS_OPEN: u64 = 0;
pub const VFS_CLOSE: u64 = 1;
pub const VFS_READ: u64 = 2;
pub const VFS_WRITE: u64 = 3;
pub const VFS_STAT: u64 = 4;

// ── IPC operations ───────────────────────────────────────────────────────

pub const IPC_PIPE: u64 = 0;
pub const IPC_SEND: u64 = 1;
pub const IPC_RECV: u64 = 2;

// ── TIMER operations ────────────────────────────────────────────────────

pub const TIMER_GET_NS: u64 = 0;
pub const TIMER_GET_CYCLES: u64 = 1;

// ── CONSOLE operations ──────────────────────────────────────────────────

pub const CONSOLE_WRITE: u64 = 0;
pub const CONSOLE_READ: u64 = 1;
pub const CONSOLE_LOG: u64 = 2;

// ── BRIDGE operations ───────────────────────────────────────────────────

pub const BRIDGE_CALL: u64 = 0;
pub const BRIDGE_EVAL: u64 = 1;

// ── DEBUG operations ─────────────────────────────────────────────────────

pub const DEBUG_TRACE_ON: u64 = 0;
pub const DEBUG_TRACE_OFF: u64 = 1;
pub const DEBUG_BREAKPOINT: u64 = 2;
pub const DEBUG_DUMP_REGS: u64 = 3;
pub const DEBUG_PERF_COUNTER: u64 = 4;

// ── Standard POSIX-ish error codes (negative) ────────────────────────────

const ENOSYS: i64 = -38;
const EBADF: i64 = -9;

// ── Alignment helper for MEM_ALLOC ───────────────────────────────────────

fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

// ── Dispatch result ──────────────────────────────────────────────────────

/// The result of dispatching a native hypercall.
///
/// Most operations return `Continue(retval)` which places the return value in
/// x0 and resumes execution.  Special operations can halt the program, yield
/// control, or toggle instruction tracing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeResult {
    /// Resume execution with the given return value in x0.
    Continue(i64),
    /// Program exited with the given status code.
    Exit(i32),
    /// Program yielded its time-slice (cooperative scheduling).
    Yield,
    /// Toggle instruction tracing.  `true` = enable, `false` = disable.
    Trace(bool),
}

// ── Dispatch ─────────────────────────────────────────────────────────────

/// Dispatch a native nCPU hypercall.
///
/// This is the main entry point called by the execution loop when an
/// `SVC #0x4E43` is decoded.  The service group (x16) and operation (x8)
/// determine which handler runs.
///
/// # Arguments
///
/// * `service`       - Service group from x16
/// * `op`            - Operation ID from x8
/// * `x0`..`x5`     - Argument registers
/// * `flags`         - Flags register (x6)
/// * `_capability`   - Capability token (x7), reserved for future use
/// * `vfs`           - Optional reference to the GPU virtual filesystem
/// * `stdout`        - Accumulated stdout bytes for this execution
/// * `stderr`        - Accumulated stderr bytes for this execution
/// * `memory_reader` - Closure to read `count` bytes starting at `addr`
/// * `memory_writer` - Closure to write `data` bytes at `addr`
/// * `total_cycles`  - Current cycle counter
/// * `heap_break`    - Current program break (bump allocator state)
#[allow(clippy::too_many_arguments)]
pub fn dispatch_native_hypercall(
    service: u64,
    op: u64,
    x0: i64,
    x1: i64,
    x2: i64,
    x3: i64,
    _x4: i64,
    _x5: i64,
    _flags: i64,
    _capability: i64,
    vfs: &mut Option<GpuVfs>,
    stdout: &mut Vec<u8>,
    stderr: &mut Vec<u8>,
    memory_reader: &dyn Fn(usize, usize) -> Vec<u8>,
    memory_writer: &dyn Fn(usize, &[u8]),
    total_cycles: u64,
    heap_break: &mut u64,
) -> NativeResult {
    match service {
        SERVICE_TASK => dispatch_task(op, x0),
        SERVICE_MEMORY => dispatch_memory(op, x0, x1, heap_break),
        SERVICE_VFS => dispatch_vfs(op, x0, x1, x2, x3, vfs, memory_reader, memory_writer),
        SERVICE_IPC => dispatch_ipc(op),
        SERVICE_TIMER => dispatch_timer(op, total_cycles),
        SERVICE_CONSOLE => dispatch_console(op, x0, x1, x2, stdout, stderr, memory_reader),
        SERVICE_BRIDGE => dispatch_bridge(op),
        SERVICE_DEBUG => dispatch_debug(op, total_cycles),
        _ => NativeResult::Continue(ENOSYS),
    }
}

/// Check whether an SVC immediate value belongs to the native nCPU ABI.
#[inline]
pub fn is_native_svc(svc_immediate: u16) -> bool {
    svc_immediate == SVC_NATIVE
}

// ── SERVICE_TASK ─────────────────────────────────────────────────────────

fn dispatch_task(op: u64, x0: i64) -> NativeResult {
    match op {
        TASK_EXIT => NativeResult::Exit(x0 as i32),
        TASK_YIELD => NativeResult::Yield,
        TASK_SPAWN => {
            // Spawn is not yet implemented; return -ENOSYS so the caller
            // knows to fall back or report an error.
            NativeResult::Continue(ENOSYS)
        }
        TASK_GETID => {
            // Single-process environment: always PID 1.
            NativeResult::Continue(1)
        }
        _ => NativeResult::Continue(ENOSYS),
    }
}

// ── SERVICE_MEMORY ───────────────────────────────────────────────────────

fn dispatch_memory(op: u64, x0: i64, x1: i64, heap_break: &mut u64) -> NativeResult {
    match op {
        MEM_ALLOC => {
            // Bump allocator.
            //   x0 = requested size in bytes
            //   x1 = alignment (0 or power-of-2; defaults to 16)
            let size = x0 as u64;
            if size == 0 {
                return NativeResult::Continue(0);
            }
            let alignment = if x1 <= 0 { 16u64 } else { x1 as u64 };
            let base = align_up(*heap_break, alignment);
            *heap_break = base + size;
            NativeResult::Continue(base as i64)
        }
        MEM_FREE => {
            // Bump allocator does not reclaim memory; acknowledge silently.
            NativeResult::Continue(0)
        }
        MEM_MAP => {
            // Memory mapping is not yet implemented.
            NativeResult::Continue(ENOSYS)
        }
        MEM_QUERY => {
            // Return current heap break so the program can inspect usage.
            NativeResult::Continue(*heap_break as i64)
        }
        _ => NativeResult::Continue(ENOSYS),
    }
}

// ── SERVICE_VFS ──────────────────────────────────────────────────────────

fn dispatch_vfs(
    op: u64,
    x0: i64,
    x1: i64,
    x2: i64,
    _x3: i64,
    vfs: &mut Option<GpuVfs>,
    memory_reader: &dyn Fn(usize, usize) -> Vec<u8>,
    memory_writer: &dyn Fn(usize, &[u8]),
) -> NativeResult {
    let fs = match vfs.as_mut() {
        Some(fs) => fs,
        None => return NativeResult::Continue(ENOSYS),
    };

    match op {
        VFS_OPEN => {
            // x0 = pointer to path string (NUL-terminated)
            // x1 = open flags
            let path = read_cstring(memory_reader, x0 as usize);
            let fd = fs.open(&path, x1 as i32);
            NativeResult::Continue(fd as i64)
        }
        VFS_CLOSE => {
            // x0 = fd
            let rc = fs.close(x0 as i32);
            NativeResult::Continue(rc as i64)
        }
        VFS_READ => {
            // x0 = fd
            // x1 = pointer to destination buffer
            // x2 = count
            let count = x2 as usize;
            match fs.read(x0 as i32, count) {
                Some(data) => {
                    let n = data.len();
                    if n > 0 {
                        memory_writer(x1 as usize, &data);
                    }
                    NativeResult::Continue(n as i64)
                }
                None => NativeResult::Continue(EBADF),
            }
        }
        VFS_WRITE => {
            // x0 = fd
            // x1 = pointer to source buffer
            // x2 = count
            let count = x2 as usize;
            let data = memory_reader(x1 as usize, count);
            let written = fs.write(x0 as i32, &data);
            NativeResult::Continue(written as i64)
        }
        VFS_STAT => {
            // x0 = pointer to path string (NUL-terminated)
            // x1 = pointer to stat buffer (128 bytes)
            let path = read_cstring(memory_reader, x0 as usize);
            match fs.stat(&path) {
                Some(info) => {
                    let buf = GpuVfs::pack_stat64(&info);
                    memory_writer(x1 as usize, &buf);
                    NativeResult::Continue(0)
                }
                None => NativeResult::Continue(-2), // -ENOENT
            }
        }
        _ => NativeResult::Continue(ENOSYS),
    }
}

// ── SERVICE_IPC ──────────────────────────────────────────────────────────

fn dispatch_ipc(op: u64) -> NativeResult {
    match op {
        IPC_PIPE | IPC_SEND | IPC_RECV => {
            // IPC primitives are not yet implemented in the native ABI.
            NativeResult::Continue(ENOSYS)
        }
        _ => NativeResult::Continue(ENOSYS),
    }
}

// ── SERVICE_TIMER ────────────────────────────────────────────────────────

fn dispatch_timer(op: u64, total_cycles: u64) -> NativeResult {
    match op {
        TIMER_GET_NS => {
            let ns = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map_or(0i64, |d| d.as_nanos() as i64);
            NativeResult::Continue(ns)
        }
        TIMER_GET_CYCLES => NativeResult::Continue(total_cycles as i64),
        _ => NativeResult::Continue(ENOSYS),
    }
}

// ── SERVICE_CONSOLE ──────────────────────────────────────────────────────

fn dispatch_console(
    op: u64,
    x0: i64,
    x1: i64,
    x2: i64,
    stdout: &mut Vec<u8>,
    stderr: &mut Vec<u8>,
    memory_reader: &dyn Fn(usize, usize) -> Vec<u8>,
) -> NativeResult {
    match op {
        CONSOLE_WRITE => {
            // x0 = fd (1 = stdout, 2 = stderr)
            // x1 = pointer to data
            // x2 = length
            let count = x2 as usize;
            let data = memory_reader(x1 as usize, count);
            match x0 {
                2 => stderr.extend_from_slice(&data),
                _ => stdout.extend_from_slice(&data),
            }
            NativeResult::Continue(count as i64)
        }
        CONSOLE_READ => {
            // Console read from stdin -- return 0 (EOF) since there is no
            // interactive input channel in the GPU execution environment.
            NativeResult::Continue(0)
        }
        CONSOLE_LOG => {
            // x0 = pointer to message (NUL-terminated)
            // Writes to stderr with an "[ncpu] " prefix.
            let msg = read_cstring(memory_reader, x0 as usize);
            let prefixed = format!("[ncpu] {}\n", msg);
            stderr.extend_from_slice(prefixed.as_bytes());
            NativeResult::Continue(0)
        }
        _ => NativeResult::Continue(ENOSYS),
    }
}

// ── SERVICE_BRIDGE ───────────────────────────────────────────────────────

fn dispatch_bridge(op: u64) -> NativeResult {
    match op {
        BRIDGE_CALL | BRIDGE_EVAL => {
            // Host bridge operations require Python interop and are not yet
            // implemented at the Rust level.
            NativeResult::Continue(ENOSYS)
        }
        _ => NativeResult::Continue(ENOSYS),
    }
}

// ── SERVICE_DEBUG ────────────────────────────────────────────────────────

fn dispatch_debug(op: u64, total_cycles: u64) -> NativeResult {
    match op {
        DEBUG_TRACE_ON => NativeResult::Trace(true),
        DEBUG_TRACE_OFF => NativeResult::Trace(false),
        DEBUG_BREAKPOINT => {
            // Yield back to the host so a debugger can inspect state.
            NativeResult::Yield
        }
        DEBUG_DUMP_REGS => {
            // Register dumping is handled by the host after we yield.
            NativeResult::Yield
        }
        DEBUG_PERF_COUNTER => NativeResult::Continue(total_cycles as i64),
        _ => NativeResult::Continue(ENOSYS),
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Read a NUL-terminated C string from guest memory.
///
/// Reads up to 4096 bytes to prevent runaway reads on unterminated strings.
fn read_cstring(memory_reader: &dyn Fn(usize, usize) -> Vec<u8>, addr: usize) -> String {
    const MAX_READ: usize = 4096;
    let raw = memory_reader(addr, MAX_READ);
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    String::from_utf8_lossy(&raw[..end]).into_owned()
}

// ══════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vfs::GpuVfs;

    // ── Helpers for tests ────────────────────────────────────────────────

    use std::cell::UnsafeCell;

    /// A simple flat memory buffer for test use.
    ///
    /// Uses `UnsafeCell` internally so that both `reader()` and `writer()`
    /// can be obtained simultaneously as `&dyn Fn(...)` closures (the
    /// dispatch API requires `Fn`, not `FnMut`).
    struct TestMemory {
        buf: UnsafeCell<Vec<u8>>,
    }

    impl TestMemory {
        fn new(size: usize) -> Self {
            TestMemory {
                buf: UnsafeCell::new(vec![0u8; size]),
            }
        }

        fn reader(&self) -> impl Fn(usize, usize) -> Vec<u8> + '_ {
            move |addr, count| {
                let buf = unsafe { &*self.buf.get() };
                let end = (addr + count).min(buf.len());
                buf[addr..end].to_vec()
            }
        }

        fn writer(&self) -> impl Fn(usize, &[u8]) + '_ {
            move |addr, data| {
                let buf = unsafe { &mut *self.buf.get() };
                let end = (addr + data.len()).min(buf.len());
                let len = end - addr;
                buf[addr..end].copy_from_slice(&data[..len]);
            }
        }

        fn write_cstring(&self, addr: usize, s: &str) {
            let buf = unsafe { &mut *self.buf.get() };
            let bytes = s.as_bytes();
            buf[addr..addr + bytes.len()].copy_from_slice(bytes);
            buf[addr + bytes.len()] = 0;
        }

        fn write_bytes(&self, addr: usize, data: &[u8]) {
            let buf = unsafe { &mut *self.buf.get() };
            buf[addr..addr + data.len()].copy_from_slice(data);
        }

        fn read_bytes(&self, addr: usize, len: usize) -> Vec<u8> {
            let buf = unsafe { &*self.buf.get() };
            buf[addr..addr + len].to_vec()
        }
    }

    // ── is_native_svc ────────────────────────────────────────────────────

    #[test]
    fn test_is_native_svc_accepts_nc() {
        assert!(is_native_svc(0x4E43));
    }

    #[test]
    fn test_is_native_svc_rejects_linux() {
        assert!(!is_native_svc(0x0000)); // Linux SVC #0
    }

    #[test]
    fn test_is_native_svc_rejects_arbitrary() {
        assert!(!is_native_svc(0x1234));
        assert!(!is_native_svc(0x4E42)); // off-by-one
        assert!(!is_native_svc(0x4E44)); // off-by-one
        assert!(!is_native_svc(0xFFFF));
    }

    // ── TASK_EXIT ────────────────────────────────────────────────────────

    #[test]
    fn test_task_exit_returns_exit_with_code() {
        let mem = TestMemory::new(1024);
        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        let result = dispatch_native_hypercall(
            SERVICE_TASK,
            TASK_EXIT,
            42,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            1000,
            &mut heap,
        );
        assert_eq!(result, NativeResult::Exit(42));
    }

    #[test]
    fn test_task_exit_negative_code() {
        let mem = TestMemory::new(1024);
        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        let result = dispatch_native_hypercall(
            SERVICE_TASK,
            TASK_EXIT,
            -1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        assert_eq!(result, NativeResult::Exit(-1));
    }

    // ── TASK_YIELD ───────────────────────────────────────────────────────

    #[test]
    fn test_task_yield() {
        let mem = TestMemory::new(1024);
        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        let result = dispatch_native_hypercall(
            SERVICE_TASK,
            TASK_YIELD,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        assert_eq!(result, NativeResult::Yield);
    }

    // ── TASK_GETID ───────────────────────────────────────────────────────

    #[test]
    fn test_task_getid_returns_one() {
        let mem = TestMemory::new(1024);
        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        let result = dispatch_native_hypercall(
            SERVICE_TASK,
            TASK_GETID,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        assert_eq!(result, NativeResult::Continue(1));
    }

    // ── TASK_SPAWN (stub) ────────────────────────────────────────────────

    #[test]
    fn test_task_spawn_returns_enosys() {
        let mem = TestMemory::new(1024);
        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        let result = dispatch_native_hypercall(
            SERVICE_TASK,
            TASK_SPAWN,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        assert_eq!(result, NativeResult::Continue(ENOSYS));
    }

    // ── CONSOLE_WRITE to stdout ──────────────────────────────────────────

    #[test]
    fn test_console_write_stdout() {
        let mem = TestMemory::new(4096);
        // Place "Hello" at address 0x100
        mem.write_bytes(0x100, b"Hello");

        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        let result = dispatch_native_hypercall(
            SERVICE_CONSOLE,
            CONSOLE_WRITE,
            1,     // fd = stdout
            0x100, // buf ptr
            5,     // length
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        assert_eq!(result, NativeResult::Continue(5));
        assert_eq!(&out, b"Hello");
        assert!(err.is_empty());
    }

    // ── CONSOLE_WRITE to stderr ──────────────────────────────────────────

    #[test]
    fn test_console_write_stderr() {
        let mem = TestMemory::new(4096);
        mem.write_bytes(0x200, b"Err");

        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        let result = dispatch_native_hypercall(
            SERVICE_CONSOLE,
            CONSOLE_WRITE,
            2,     // fd = stderr
            0x200, // buf ptr
            3,     // length
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        assert_eq!(result, NativeResult::Continue(3));
        assert!(out.is_empty());
        assert_eq!(&err, b"Err");
    }

    // ── CONSOLE_LOG ──────────────────────────────────────────────────────

    #[test]
    fn test_console_log_prefixes_message() {
        let mem = TestMemory::new(4096);
        mem.write_cstring(0x300, "boot ok");

        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        let result = dispatch_native_hypercall(
            SERVICE_CONSOLE,
            CONSOLE_LOG,
            0x300, // message ptr
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        assert_eq!(result, NativeResult::Continue(0));
        assert_eq!(String::from_utf8_lossy(&err), "[ncpu] boot ok\n");
    }

    // ── CONSOLE_READ (returns EOF) ───────────────────────────────────────

    #[test]
    fn test_console_read_returns_eof() {
        let mem = TestMemory::new(1024);
        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        let result = dispatch_native_hypercall(
            SERVICE_CONSOLE,
            CONSOLE_READ,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        assert_eq!(result, NativeResult::Continue(0));
    }

    // ── TIMER_GET_CYCLES ─────────────────────────────────────────────────

    #[test]
    fn test_timer_get_cycles_returns_counter() {
        let mem = TestMemory::new(1024);
        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        let result = dispatch_native_hypercall(
            SERVICE_TIMER,
            TIMER_GET_CYCLES,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            999_999,
            &mut heap,
        );
        assert_eq!(result, NativeResult::Continue(999_999));
    }

    // ── TIMER_GET_NS ─────────────────────────────────────────────────────

    #[test]
    fn test_timer_get_ns_returns_positive() {
        let mem = TestMemory::new(1024);
        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        let result = dispatch_native_hypercall(
            SERVICE_TIMER,
            TIMER_GET_NS,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        match result {
            NativeResult::Continue(ns) => assert!(ns > 0, "wall-clock ns should be positive"),
            other => panic!("expected Continue, got {:?}", other),
        }
    }

    // ── DEBUG_TRACE_ON / DEBUG_TRACE_OFF ─────────────────────────────────

    #[test]
    fn test_debug_trace_on() {
        let mem = TestMemory::new(1024);
        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        let result = dispatch_native_hypercall(
            SERVICE_DEBUG,
            DEBUG_TRACE_ON,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        assert_eq!(result, NativeResult::Trace(true));
    }

    #[test]
    fn test_debug_trace_off() {
        let mem = TestMemory::new(1024);
        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        let result = dispatch_native_hypercall(
            SERVICE_DEBUG,
            DEBUG_TRACE_OFF,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        assert_eq!(result, NativeResult::Trace(false));
    }

    // ── DEBUG_BREAKPOINT ─────────────────────────────────────────────────

    #[test]
    fn test_debug_breakpoint_yields() {
        let mem = TestMemory::new(1024);
        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        let result = dispatch_native_hypercall(
            SERVICE_DEBUG,
            DEBUG_BREAKPOINT,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        assert_eq!(result, NativeResult::Yield);
    }

    // ── DEBUG_PERF_COUNTER ───────────────────────────────────────────────

    #[test]
    fn test_debug_perf_counter() {
        let mem = TestMemory::new(1024);
        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        let result = dispatch_native_hypercall(
            SERVICE_DEBUG,
            DEBUG_PERF_COUNTER,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            12345,
            &mut heap,
        );
        assert_eq!(result, NativeResult::Continue(12345));
    }

    // ── MEM_ALLOC bumps heap_break ───────────────────────────────────────

    #[test]
    fn test_mem_alloc_bumps_heap() {
        let mem = TestMemory::new(1024);
        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        // Allocate 256 bytes with default alignment (16)
        let result = dispatch_native_hypercall(
            SERVICE_MEMORY,
            MEM_ALLOC,
            256,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );

        match result {
            NativeResult::Continue(addr) => {
                assert_eq!(addr, 0x60000, "first alloc should return heap base");
                assert_eq!(heap, 0x60000 + 256, "heap_break should advance by size");
            }
            other => panic!("expected Continue, got {:?}", other),
        }
    }

    #[test]
    fn test_mem_alloc_respects_alignment() {
        let mem = TestMemory::new(1024);
        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60001u64; // deliberately unaligned

        let result = dispatch_native_hypercall(
            SERVICE_MEMORY,
            MEM_ALLOC,
            100,
            64,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );

        match result {
            NativeResult::Continue(addr) => {
                assert_eq!(addr % 64, 0, "returned address must be 64-aligned");
                assert_eq!(addr, 0x60040, "should round up to next 64-byte boundary");
                assert_eq!(heap, 0x60040 + 100);
            }
            other => panic!("expected Continue, got {:?}", other),
        }
    }

    #[test]
    fn test_mem_alloc_zero_returns_null() {
        let mem = TestMemory::new(1024);
        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        let result = dispatch_native_hypercall(
            SERVICE_MEMORY,
            MEM_ALLOC,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        assert_eq!(result, NativeResult::Continue(0));
        assert_eq!(heap, 0x60000, "heap should not move for zero-size alloc");
    }

    #[test]
    fn test_mem_alloc_sequential() {
        let mem = TestMemory::new(1024);
        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        // First allocation: 32 bytes
        let r1 = dispatch_native_hypercall(
            SERVICE_MEMORY,
            MEM_ALLOC,
            32,
            16,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        let addr1 = match r1 {
            NativeResult::Continue(a) => a,
            other => panic!("expected Continue, got {:?}", other),
        };

        // Second allocation: 64 bytes
        let r2 = dispatch_native_hypercall(
            SERVICE_MEMORY,
            MEM_ALLOC,
            64,
            16,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        let addr2 = match r2 {
            NativeResult::Continue(a) => a,
            other => panic!("expected Continue, got {:?}", other),
        };

        assert!(
            addr2 >= addr1 + 32,
            "second allocation must not overlap first"
        );
        assert_eq!(heap, (addr2 + 64) as u64);
    }

    // ── MEM_FREE (no-op) ─────────────────────────────────────────────────

    #[test]
    fn test_mem_free_succeeds() {
        let mem = TestMemory::new(1024);
        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        let result = dispatch_native_hypercall(
            SERVICE_MEMORY,
            MEM_FREE,
            0x60000,
            256,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        assert_eq!(result, NativeResult::Continue(0));
    }

    // ── MEM_QUERY ────────────────────────────────────────────────────────

    #[test]
    fn test_mem_query_returns_heap_break() {
        let mem = TestMemory::new(1024);
        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x70000u64;

        let result = dispatch_native_hypercall(
            SERVICE_MEMORY,
            MEM_QUERY,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        assert_eq!(result, NativeResult::Continue(0x70000));
    }

    // ── VFS round-trip: OPEN + WRITE + READ ──────────────────────────────

    #[test]
    fn test_vfs_open_write_read_roundtrip() {
        let mem = TestMemory::new(0x10000);
        // Write the path "/tmp/test.txt\0" at address 0x1000
        mem.write_cstring(0x1000, "/tmp/test.txt");

        let mut vfs = Some(GpuVfs::new());
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        // --- VFS_OPEN (O_CREAT | O_RDWR) ---
        let open_result = dispatch_native_hypercall(
            SERVICE_VFS,
            VFS_OPEN,
            0x1000, // path ptr
            (crate::vfs::O_CREAT | crate::vfs::O_RDWR) as i64,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        let fd = match open_result {
            NativeResult::Continue(fd) => {
                assert!(fd >= 3, "fd should be >= 3 (after stdin/out/err)");
                fd
            }
            other => panic!("expected Continue with fd, got {:?}", other),
        };

        // --- VFS_WRITE "hello" at memory address 0x2000 ---
        let payload = b"hello";
        mem.write_bytes(0x2000, payload);

        let write_result = dispatch_native_hypercall(
            SERVICE_VFS,
            VFS_WRITE,
            fd,                   // fd
            0x2000,               // buf ptr
            payload.len() as i64, // count
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        assert_eq!(write_result, NativeResult::Continue(5));

        // --- VFS_CLOSE ---
        let close_result = dispatch_native_hypercall(
            SERVICE_VFS,
            VFS_CLOSE,
            fd,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        assert_eq!(close_result, NativeResult::Continue(0));

        // --- Re-open for reading ---
        let open2 = dispatch_native_hypercall(
            SERVICE_VFS,
            VFS_OPEN,
            0x1000, // same path
            crate::vfs::O_RDONLY as i64,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        let fd2 = match open2 {
            NativeResult::Continue(fd) => fd,
            other => panic!("expected Continue, got {:?}", other),
        };

        // --- VFS_READ into buffer at 0x3000 ---
        let read_result = dispatch_native_hypercall(
            SERVICE_VFS,
            VFS_READ,
            fd2,    // fd
            0x3000, // dest buf
            256,    // count
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        assert_eq!(read_result, NativeResult::Continue(5));
        assert_eq!(mem.read_bytes(0x3000, 5), b"hello".to_vec());
    }

    // ── VFS_STAT ─────────────────────────────────────────────────────────

    #[test]
    fn test_vfs_stat_existing_file() {
        let mem = TestMemory::new(0x10000);
        mem.write_cstring(0x1000, "/etc/hostname");

        let mut vfs = Some(GpuVfs::new());
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        let result = dispatch_native_hypercall(
            SERVICE_VFS,
            VFS_STAT,
            0x1000, // path ptr
            0x2000, // stat buf
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        assert_eq!(result, NativeResult::Continue(0));

        // Verify that something was written to the stat buffer (128 bytes).
        // The first 8 bytes are st_dev; just check the buffer is non-zero.
        let stat_buf = mem.read_bytes(0x2000, 128);
        assert!(
            stat_buf.iter().any(|&b| b != 0),
            "stat buffer should be populated"
        );
    }

    #[test]
    fn test_vfs_stat_nonexistent() {
        let mem = TestMemory::new(0x10000);
        mem.write_cstring(0x1000, "/no/such/file");

        let mut vfs = Some(GpuVfs::new());
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        let result = dispatch_native_hypercall(
            SERVICE_VFS,
            VFS_STAT,
            0x1000,
            0x2000,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        assert_eq!(result, NativeResult::Continue(-2)); // -ENOENT
    }

    // ── VFS without filesystem ───────────────────────────────────────────

    #[test]
    fn test_vfs_returns_enosys_without_fs() {
        let mem = TestMemory::new(0x10000);
        mem.write_cstring(0x1000, "/tmp/x");

        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        let result = dispatch_native_hypercall(
            SERVICE_VFS,
            VFS_OPEN,
            0x1000,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        assert_eq!(result, NativeResult::Continue(ENOSYS));
    }

    // ── Unknown service group ────────────────────────────────────────────

    #[test]
    fn test_unknown_service_returns_enosys() {
        let mem = TestMemory::new(1024);
        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        let result = dispatch_native_hypercall(
            255,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        assert_eq!(result, NativeResult::Continue(ENOSYS));
    }

    // ── Unknown operation within service ─────────────────────────────────

    #[test]
    fn test_unknown_op_returns_enosys() {
        let mem = TestMemory::new(1024);
        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        let result = dispatch_native_hypercall(
            SERVICE_TASK,
            99,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            &mut vfs,
            &mut out,
            &mut err,
            &mem.reader(),
            &mem.writer(),
            0,
            &mut heap,
        );
        assert_eq!(result, NativeResult::Continue(ENOSYS));
    }

    // ── IPC stubs ────────────────────────────────────────────────────────

    #[test]
    fn test_ipc_stubs_return_enosys() {
        let mem = TestMemory::new(1024);
        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        for op in [IPC_PIPE, IPC_SEND, IPC_RECV] {
            let result = dispatch_native_hypercall(
                SERVICE_IPC,
                op,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                &mut vfs,
                &mut out,
                &mut err,
                &mem.reader(),
                &mem.writer(),
                0,
                &mut heap,
            );
            assert_eq!(
                result,
                NativeResult::Continue(ENOSYS),
                "IPC op {} should return ENOSYS",
                op
            );
        }
    }

    // ── BRIDGE stubs ─────────────────────────────────────────────────────

    #[test]
    fn test_bridge_stubs_return_enosys() {
        let mem = TestMemory::new(1024);
        let mut vfs: Option<GpuVfs> = None;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut heap = 0x60000u64;

        for op in [BRIDGE_CALL, BRIDGE_EVAL] {
            let result = dispatch_native_hypercall(
                SERVICE_BRIDGE,
                op,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                &mut vfs,
                &mut out,
                &mut err,
                &mem.reader(),
                &mem.writer(),
                0,
                &mut heap,
            );
            assert_eq!(
                result,
                NativeResult::Continue(ENOSYS),
                "BRIDGE op {} should return ENOSYS",
                op
            );
        }
    }

    // ── Alignment helper ─────────────────────────────────────────────────

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 16), 0);
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
        assert_eq!(align_up(0x60001, 64), 0x60040);
        assert_eq!(align_up(0x60040, 64), 0x60040);
    }

    // ── read_cstring helper ──────────────────────────────────────────────

    #[test]
    fn test_read_cstring_basic() {
        let mem = TestMemory::new(4096);
        mem.write_cstring(0x100, "hello world");
        let s = read_cstring(&mem.reader(), 0x100);
        assert_eq!(s, "hello world");
    }

    #[test]
    fn test_read_cstring_empty() {
        let mem = TestMemory::new(4096); // all zeros
        let s = read_cstring(&mem.reader(), 0x100);
        assert_eq!(s, "");
    }
}
