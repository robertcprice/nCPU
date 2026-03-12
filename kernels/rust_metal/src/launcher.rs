//! GPU Launcher — Rust-only Metal execution engine for ARM64 boot images.
//!
//! Provides `GpuLauncher` which allocates StorageModeShared Metal buffers,
//! compiles the full ARM64 shader, and runs an execution loop with syscall
//! dispatch — all without Python involvement.

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, UdpSocket};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::elf_loader::PreparedElf;
use crate::full_arm64::FULL_ARM64_SHADER;
use crate::get_default_device;
use crate::process::{
    GpuSnapshot, ProcessManager, ProcessState, HEAP_BASE, MAX_CYCLE_LIMIT, MMAP_BASE,
};
use crate::vfs::{FdEntry, FdKind, GpuVfs, HostSocketState};

// ── Constants ────────────────────────────────────────────────────────────

const SVC_BUF_BASE: usize = 0x3F0000;
const SVC_BUF_HDR: usize = 16;
const SVC_BUF_DATA: usize = SVC_BUF_BASE + SVC_BUF_HDR;

const SIGNAL_RUNNING: u32 = 0;
const SIGNAL_HALT: u32 = 1;
const SIGNAL_SYSCALL: u32 = 2;
const SIGNAL_CHECKPOINT: u32 = 3;

const SYS_COMPILE: i64 = 300;
const SYS_EXEC: i64 = 301;
const SYS_GETCHAR: i64 = 302;
const SYS_CLOCK: i64 = 303;
const SYS_SLEEP: i64 = 304;
const SYS_SOCKET: i64 = 305;
const SYS_BIND: i64 = 306;
const SYS_LISTEN: i64 = 307;
const SYS_ACCEPT: i64 = 308;
const SYS_CONNECT: i64 = 309;
const SYS_SEND: i64 = 310;
const SYS_RECV: i64 = 311;
const SYS_PS: i64 = 312;
const SYS_KILL: i64 = 314;
const SYS_GETENV: i64 = 315;
const SYS_SETENV: i64 = 316;
const SYS_EXECVE: i64 = 221;

const SIGTERM: i32 = 15;
const SIGKILL: i32 = 9;
const WAIT_TARGET_PIPE: i32 = -2;
const RAW_EXEC_ADDR: usize = 0x10000;

// ── Execution result ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct VfsSnapshot {
    pub files: Vec<(String, Vec<u8>)>,   // (path, content) for modified/new files
    pub directories: Vec<String>,         // new directories created
    pub symlinks: Vec<(String, String)>, // (path, target) for symlinks
}

#[derive(Debug)]
pub struct LaunchResult {
    pub total_cycles: u64,
    pub elapsed_secs: f64,
    pub stop_reason: String,
    pub exit_code: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub total_forks: u32,
    pub total_context_switches: u32,
    pub processes_created: i32,
    pub vfs_snapshot: Option<VfsSnapshot>,  // Filesystem changes after execution
}

enum RuntimeAction {
    Continue { ret: i64, advance_pc: bool },
    Exit(i32),
    Forked,
    Blocked { advance_pc: bool },
    Execed,
}

// ── GpuLauncher ──────────────────────────────────────────────────────────

pub struct GpuLauncher {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,

    memory_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    registers_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pc_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    flags_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    max_cycles_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    mem_size_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    vreg_lo_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    vreg_hi_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    signal_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    total_cycles_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    batch_count_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,

    memory_size: usize,
    cycles_per_batch: u32,
}

impl GpuLauncher {
    /// Create a new GPU launcher with the given memory size.
    pub fn new(memory_size: usize, cycles_per_batch: u32) -> Result<Self, String> {
        let device = get_default_device().ok_or("no Metal device found")?;
        let command_queue = device
            .newCommandQueue()
            .ok_or("failed to create command queue")?;

        let source = NSString::from_str(FULL_ARM64_SHADER);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|e| format!("shader compilation failed: {:?}", e))?;

        let function_name = NSString::from_str("arm64_execute_full");
        let function = library
            .newFunctionWithName(&function_name)
            .ok_or("kernel function 'arm64_execute_full' not found")?;

        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| format!("pipeline creation failed: {:?}", e))?;

        let shared = MTLResourceOptions::StorageModeShared;

        macro_rules! buf {
            ($size:expr) => {
                device
                    .newBufferWithLength_options($size, shared)
                    .ok_or("buffer creation failed")?
            };
        }

        let memory_buffer = buf!(memory_size);
        let registers_buffer = buf!(32 * 8);
        let pc_buffer = buf!(8);
        let flags_buffer = buf!(4 * 4);
        let max_cycles_buffer = buf!(4);
        let mem_size_buffer = buf!(4);
        let vreg_lo_buffer = buf!(32 * 8);
        let vreg_hi_buffer = buf!(32 * 8);
        let signal_buffer = buf!(4);
        let total_cycles_buffer = buf!(4);
        let batch_count_buffer = buf!(4);

        // Initialize config
        unsafe {
            *(mem_size_buffer.contents().as_ptr() as *mut u32) = memory_size as u32;
            *(max_cycles_buffer.contents().as_ptr() as *mut u32) = cycles_per_batch;
        }

        Ok(GpuLauncher {
            device,
            command_queue,
            pipeline,
            memory_buffer,
            registers_buffer,
            pc_buffer,
            flags_buffer,
            max_cycles_buffer,
            mem_size_buffer,
            vreg_lo_buffer,
            vreg_hi_buffer,
            signal_buffer,
            total_cycles_buffer,
            batch_count_buffer,
            memory_size,
            cycles_per_batch,
        })
    }

    // ── Memory access ────────────────────────────────────────────────────

    pub fn write_memory(&self, addr: usize, data: &[u8]) {
        if addr + data.len() > self.memory_size {
            return;
        }
        unsafe {
            let ptr = self.memory_buffer.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(addr), data.len());
        }
    }

    pub fn read_memory(&self, addr: usize, len: usize) -> Vec<u8> {
        if addr + len > self.memory_size {
            return vec![];
        }
        let mut result = vec![0u8; len];
        unsafe {
            let ptr = self.memory_buffer.contents().as_ptr() as *const u8;
            std::ptr::copy_nonoverlapping(ptr.add(addr), result.as_mut_ptr(), len);
        }
        result
    }

    fn read_u64(&self, addr: usize) -> u64 {
        let bytes = self.read_memory(addr, 8);
        if bytes.len() == 8 {
            u64::from_le_bytes(bytes.try_into().unwrap())
        } else {
            0
        }
    }

    fn read_u32_mem(&self, addr: usize) -> u32 {
        let bytes = self.read_memory(addr, 4);
        if bytes.len() == 4 {
            u32::from_le_bytes(bytes.try_into().unwrap())
        } else {
            0
        }
    }

    fn write_u64(&self, addr: usize, val: u64) {
        self.write_memory(addr, &val.to_le_bytes());
    }

    pub fn set_pc(&self, pc: u64) {
        unsafe {
            *(self.pc_buffer.contents().as_ptr() as *mut u64) = pc;
        }
    }

    pub fn get_pc(&self) -> u64 {
        unsafe { *(self.pc_buffer.contents().as_ptr() as *const u64) }
    }

    pub fn set_register(&self, reg: usize, val: i64) {
        if reg >= 32 {
            return;
        }
        unsafe {
            let ptr = self.registers_buffer.contents().as_ptr() as *mut i64;
            *ptr.add(reg) = val;
        }
    }

    pub fn get_register(&self, reg: usize) -> i64 {
        if reg >= 32 {
            return 0;
        }
        unsafe {
            let ptr = self.registers_buffer.contents().as_ptr() as *const i64;
            *ptr.add(reg)
        }
    }

    fn get_registers(&self) -> [i64; 32] {
        let mut regs = [0i64; 32];
        unsafe {
            let ptr = self.registers_buffer.contents().as_ptr() as *const i64;
            std::ptr::copy_nonoverlapping(ptr, regs.as_mut_ptr(), regs.len());
        }
        regs
    }

    fn set_registers(&self, regs: &[i64; 32]) {
        unsafe {
            let ptr = self.registers_buffer.contents().as_ptr() as *mut i64;
            std::ptr::copy_nonoverlapping(regs.as_ptr(), ptr, regs.len());
        }
    }

    fn get_flags(&self) -> [f32; 4] {
        let mut flags = [0.0f32; 4];
        unsafe {
            let ptr = self.flags_buffer.contents().as_ptr() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, flags.as_mut_ptr(), flags.len());
        }
        flags
    }

    fn set_flags(&self, flags: &[f32; 4]) {
        unsafe {
            let ptr = self.flags_buffer.contents().as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(flags.as_ptr(), ptr, flags.len());
        }
    }

    fn snapshot_memory_image(&self) -> Vec<u8> {
        self.read_memory(0, self.memory_size)
    }

    fn restore_memory_image(&self, image: &[u8]) {
        if image.len() != self.memory_size {
            return;
        }
        self.write_memory(0, image);
    }

    fn init_svc_buffer(&self, heap_base: u64) {
        if self.memory_size > SVC_BUF_BASE + SVC_BUF_HDR {
            // Zero header (write_pos + entry_count)
            self.write_memory(SVC_BUF_BASE, &[0u8; 8]);
            // Set BRK base at SVC_BUF_BASE + 8
            self.write_u64(SVC_BUF_BASE + 8, heap_base);
        }
    }

    // ── GPU dispatch ─────────────────────────────────────────────────────

    fn dispatch_batch(&self) -> Result<(u32, u32), String> {
        unsafe {
            *(self.signal_buffer.contents().as_ptr() as *mut u32) = SIGNAL_RUNNING;
            *(self.total_cycles_buffer.contents().as_ptr() as *mut u32) = 0;
            *(self.batch_count_buffer.contents().as_ptr() as *mut u32) = 0;
        }

        let command_buffer = self
            .command_queue
            .commandBuffer()
            .ok_or("failed to create command buffer")?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or("failed to create compute encoder")?;

        encoder.setComputePipelineState(&self.pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&self.memory_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&self.registers_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&self.pc_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&self.flags_buffer), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&self.max_cycles_buffer), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&self.mem_size_buffer), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&self.vreg_lo_buffer), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(&self.vreg_hi_buffer), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(&self.signal_buffer), 0, 8);
            encoder.setBuffer_offset_atIndex(Some(&self.total_cycles_buffer), 0, 9);
            encoder.setBuffer_offset_atIndex(Some(&self.batch_count_buffer), 0, 10);

            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            );
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        let (cycles, signal) = unsafe {
            let c = *(self.total_cycles_buffer.contents().as_ptr() as *const u32);
            let s = *(self.signal_buffer.contents().as_ptr() as *const u32);
            (c, s)
        };
        Ok((cycles, signal))
    }

    /// Drain the GPU SVC write buffer — returns list of (fd, data) tuples.
    fn drain_svc_buffer(&self) -> Vec<(u8, Vec<u8>)> {
        let mut entries = Vec::new();
        let write_pos = self.read_u32_mem(SVC_BUF_BASE);
        let entry_count = self.read_u32_mem(SVC_BUF_BASE + 4);

        if entry_count > 0 {
            let mut offset = 0u32;
            for _ in 0..entry_count {
                if offset + 3 > write_pos {
                    break;
                }
                let base = SVC_BUF_DATA + offset as usize;
                let mem = self.read_memory(base, 3);
                if mem.len() < 3 {
                    break;
                }
                let fd = mem[0];
                let len = u16::from_le_bytes([mem[1], mem[2]]) as usize;
                let data = self.read_memory(base + 3, len);
                entries.push((fd, data));
                offset += 3 + len as u32;
            }
            // Clear buffer
            self.write_memory(SVC_BUF_BASE, &[0u8; 8]);
        }
        entries
    }

    // ── String reading ───────────────────────────────────────────────────

    fn read_string(&self, addr: u64, max_len: usize) -> String {
        let data = self.read_memory(addr as usize, max_len);
        let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
        String::from_utf8_lossy(&data[..end]).to_string()
    }

    fn read_string_array(&self, addr: u64, max_entries: usize, max_len: usize) -> Vec<String> {
        if addr == 0 {
            return Vec::new();
        }

        let mut values = Vec::new();
        for idx in 0..max_entries {
            let ptr = self.read_u64(addr as usize + idx * 8);
            if ptr == 0 {
                break;
            }
            values.push(self.read_string(ptr, max_len));
        }
        values
    }

    fn default_env() -> HashMap<String, String> {
        [
            ("PATH", "/bin:/usr/bin:/sbin:/usr/sbin"),
            ("HOME", "/root"),
            ("TERM", "dumb"),
            ("USER", "root"),
            ("TZ", "UTC"),
            ("LOGNAME", "root"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    fn env_map_from_strings(values: &[String]) -> HashMap<String, String> {
        values
            .iter()
            .filter_map(|entry| {
                entry
                    .split_once('=')
                    .map(|(key, value)| (key.to_string(), value.to_string()))
            })
            .collect()
    }

    fn runtime_src_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../ncpu/os/gpu/src")
            .to_path_buf()
    }

    fn unique_temp_path(stem: &str, ext: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "ncpu_{}_{}_{}.{}",
            stem,
            std::process::id(),
            nonce,
            ext
        ))
    }

    fn compile_guest_c(
        &self,
        vfs: &mut GpuVfs,
        src_path: &str,
        bin_path: &str,
    ) -> Result<usize, String> {
        let resolved_src = vfs.resolve_path(src_path);
        let resolved_bin = vfs.resolve_path(bin_path);
        let source = vfs
            .read_file(&resolved_src)
            .ok_or_else(|| format!("source not found: {}", resolved_src))?
            .to_vec();

        let src_dir = Self::runtime_src_dir();
        let linker_script = src_dir.join("arm64.ld");
        let startup_asm = src_dir.join("arm64_start.S");
        if !linker_script.exists() || !startup_asm.exists() {
            return Err(format!(
                "missing runtime sources under {}",
                src_dir.display()
            ));
        }

        let c_path = Self::unique_temp_path("guest_compile", "c");
        let elf_path = c_path.with_extension("elf");
        let out_path = c_path.with_extension("bin");
        fs::write(&c_path, source).map_err(|e| format!("failed to write temp source: {}", e))?;

        let compile_result = (|| -> Result<usize, String> {
            let gcc = Command::new("aarch64-elf-gcc")
                .arg("-nostdlib")
                .arg("-ffreestanding")
                .arg("-static")
                .arg("-O2")
                .arg("-march=armv8-a")
                .arg("-mgeneral-regs-only")
                .arg(format!("-T{}", linker_script.display()))
                .arg(format!("-I{}", src_dir.display()))
                .arg("-e")
                .arg("_start")
                .arg(&startup_asm)
                .arg(&c_path)
                .arg("-o")
                .arg(&elf_path)
                .output()
                .map_err(|e| format!("failed to launch aarch64-elf-gcc: {}", e))?;
            if !gcc.status.success() {
                let stderr = String::from_utf8_lossy(&gcc.stderr).trim().to_string();
                return Err(if stderr.is_empty() {
                    "aarch64-elf-gcc failed".to_string()
                } else {
                    stderr
                });
            }

            let objcopy = Command::new("aarch64-elf-objcopy")
                .arg("-O")
                .arg("binary")
                .arg(&elf_path)
                .arg(&out_path)
                .output()
                .map_err(|e| format!("failed to launch aarch64-elf-objcopy: {}", e))?;
            if !objcopy.status.success() {
                let stderr = String::from_utf8_lossy(&objcopy.stderr).trim().to_string();
                return Err(if stderr.is_empty() {
                    "aarch64-elf-objcopy failed".to_string()
                } else {
                    stderr
                });
            }

            let binary = fs::read(&out_path)
                .map_err(|e| format!("failed to read compiled binary: {}", e))?;
            let binary_len = binary.len();
            vfs.write_file(&resolved_bin, &binary);
            Ok(binary_len)
        })();

        let _ = fs::remove_file(&c_path);
        let _ = fs::remove_file(&elf_path);
        let _ = fs::remove_file(&out_path);
        compile_result
    }

    fn exec_guest_binary(
        &self,
        pid: i32,
        path: &str,
        argv: Vec<String>,
        env_map: HashMap<String, String>,
        proc_mgr: &mut ProcessManager,
        vfs: &mut Option<GpuVfs>,
        heap_break: &mut u64,
        mmap_next: &mut u64,
    ) -> RuntimeAction {
        let (resolved_path, binary) = match vfs.as_ref().and_then(|fs| {
            let resolved = fs.resolve_path(path);
            fs.read_file(&resolved)
                .map(|data| (resolved, data.to_vec()))
        }) {
            Some(found) => found,
            None => {
                return RuntimeAction::Continue {
                    ret: -2,
                    advance_pc: true,
                };
            }
        };

        let zeroed = vec![0u8; self.memory_size];
        self.restore_memory_image(&zeroed);
        self.set_registers(&[0i64; 32]);
        self.set_flags(&[0.0f32; 4]);

        if binary.starts_with(b"\x7FELF") {
            let argv = if argv.is_empty() {
                vec![resolved_path.clone()]
            } else {
                argv
            };
            let env_map = if env_map.is_empty() {
                Self::default_env()
            } else {
                env_map
            };
            let argv_refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let mut env_pairs = env_map
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect::<Vec<_>>();
            env_pairs.sort_by(|a, b| a.0.cmp(&b.0));
            let env_refs = env_pairs
                .iter()
                .map(|(key, value)| (key.as_str(), value.as_str()))
                .collect::<Vec<_>>();
            let prepared = match crate::elf_loader::prepare_elf(
                &binary,
                &argv_refs,
                &env_refs,
                self.memory_size as u64,
            ) {
                Ok(prepared) => prepared,
                Err(_) => {
                    return RuntimeAction::Continue {
                        ret: -8,
                        advance_pc: true,
                    };
                }
            };

            self.load_prepared_elf(&prepared);
            *heap_break = prepared.heap_base;
            *mmap_next = prepared.heap_base + 0x400000;
            if let Some(proc) = proc_mgr.get_mut(pid) {
                proc.heap_break = *heap_break;
                proc.mmap_next = *mmap_next;
                proc.wait_target = -1;
                proc.env = env_map;
            }
            return RuntimeAction::Execed;
        }

        self.write_memory(RAW_EXEC_ADDR, &binary);
        self.set_pc(RAW_EXEC_ADDR as u64);
        *heap_break = HEAP_BASE;
        *mmap_next = MMAP_BASE;
        self.init_svc_buffer(*heap_break);
        if let Some(proc) = proc_mgr.get_mut(pid) {
            proc.heap_break = *heap_break;
            proc.mmap_next = *mmap_next;
            proc.wait_target = -1;
            if !env_map.is_empty() {
                proc.env = env_map;
            }
        }
        RuntimeAction::Execed
    }

    fn init_runtime(
        &self,
        proc_mgr: &mut ProcessManager,
        memory_images: &mut HashMap<i32, Vec<u8>>,
        vfs: &mut Option<GpuVfs>,
        heap_break: u64,
        mmap_next: u64,
    ) -> Result<(), String> {
        let cwd = vfs
            .as_ref()
            .map(|fs| fs.getcwd().to_string())
            .unwrap_or_else(|| "/".to_string());
        let init_pid = proc_mgr.create_init_process(None, &cwd, Some(Self::default_env()));
        if init_pid < 0 {
            return Err("failed to create initial process".to_string());
        }

        let fd_table = vfs
            .as_ref()
            .map(|fs| fs.fd_table.clone())
            .unwrap_or_default();
        let proc = proc_mgr
            .get_mut(init_pid)
            .ok_or_else(|| "initial process missing after creation".to_string())?;
        proc.state = ProcessState::Ready;
        proc.pc = self.get_pc();
        proc.registers = self.get_registers();
        proc.flags = self.get_flags();
        proc.heap_break = heap_break;
        proc.mmap_next = mmap_next;
        proc.fd_table = fd_table;
        proc.cwd = cwd;

        proc_mgr.current_pid = init_pid;
        memory_images.insert(init_pid, self.snapshot_memory_image());
        Ok(())
    }

    fn save_runtime_process(
        &self,
        proc_mgr: &mut ProcessManager,
        memory_images: &mut HashMap<i32, Vec<u8>>,
        vfs: &mut Option<GpuVfs>,
        pid: i32,
        heap_break: u64,
        mmap_next: u64,
    ) {
        let snapshot = GpuSnapshot {
            registers: self.get_registers(),
            pc: self.get_pc(),
            flags: self.get_flags(),
        };
        let fd_table: HashMap<i32, FdEntry> = vfs
            .as_ref()
            .map(|fs| fs.fd_table.clone())
            .unwrap_or_default();
        proc_mgr.save_context(pid, &snapshot, &fd_table);
        if let Some(proc) = proc_mgr.get_mut(pid) {
            proc.heap_break = heap_break;
            proc.mmap_next = mmap_next;
            if let Some(fs) = vfs.as_ref() {
                proc.cwd = fs.getcwd().to_string();
            }
        }
        memory_images.insert(pid, self.snapshot_memory_image());
    }

    fn restore_runtime_process(
        &self,
        proc_mgr: &mut ProcessManager,
        memory_images: &HashMap<i32, Vec<u8>>,
        vfs: &mut Option<GpuVfs>,
        pid: i32,
    ) -> Result<(u64, u64), String> {
        let image = memory_images
            .get(&pid)
            .ok_or_else(|| format!("missing memory image for pid {}", pid))?;
        self.restore_memory_image(image);

        let (snapshot, fd_table, cwd) = proc_mgr
            .restore_context(pid)
            .ok_or_else(|| format!("missing process context for pid {}", pid))?;
        self.set_registers(&snapshot.registers);
        self.set_pc(snapshot.pc);
        self.set_flags(&snapshot.flags);

        if let Some(fs) = vfs.as_mut() {
            fs.fd_table = fd_table;
            fs.cwd = cwd;
        }

        let proc = proc_mgr
            .get(pid)
            .ok_or_else(|| format!("missing process state for pid {}", pid))?;
        self.init_svc_buffer(proc.heap_break);
        Ok((proc.heap_break, proc.mmap_next))
    }

    fn wake_pipe_readers(proc_mgr: &mut ProcessManager) {
        for proc in proc_mgr.processes.values_mut() {
            if proc.state == ProcessState::Blocked && proc.wait_target == WAIT_TARGET_PIPE {
                proc.state = ProcessState::Ready;
            }
        }
    }

    // ── Load an ELF ──────────────────────────────────────────────────────

    /// Load a prepared ELF into GPU memory and set PC/SP.
    pub fn load_prepared_elf(&self, prepared: &PreparedElf) {
        for (addr, data) in &prepared.writes {
            self.write_memory(*addr as usize, data);
        }
        self.load_boot_task(
            prepared.entry_pc,
            prepared.stack_pointer,
            prepared.heap_base,
        );
    }

    /// Initialize a Linux-style task that has already been loaded into memory.
    pub fn load_boot_task(&self, entry_pc: u64, stack_pointer: u64, heap_base: u64) {
        self.set_pc(entry_pc);
        self.set_register(31, stack_pointer as i64);
        self.init_svc_buffer(heap_base);
    }

    // ── Main execution loop ──────────────────────────────────────────────

    /// Run a loaded ELF binary with syscall handling.
    /// Returns stdout output and execution metadata.
    pub fn run(
        &self,
        vfs: &mut Option<GpuVfs>,
        max_total_cycles: u64,
        timeout_secs: f64,
        quiet: bool,
    ) -> Result<LaunchResult, String> {
        let start = Instant::now();
        let timeout = Duration::from_secs_f64(timeout_secs);
        let mut total_cycles: u64 = 0;
        let mut stdout_buf: Vec<u8> = Vec::new();
        let mut stderr_buf: Vec<u8> = Vec::new();
        let mut exit_code: i32 = 0;
        let mut stop_reason = "COMPLETE".to_string();

        let mut fallback_vfs = if vfs.is_none() {
            Some(GpuVfs::new())
        } else {
            None
        };
        let vfs = if vfs.is_some() {
            vfs
        } else {
            &mut fallback_vfs
        };

        let mut heap_break = self.read_u64(SVC_BUF_BASE + 8);
        if heap_break == 0 {
            heap_break = 0x60000;
            self.init_svc_buffer(heap_break);
        }
        let mmap_next = heap_break + 0x400000;

        let mut proc_mgr = ProcessManager::new();
        let mut memory_images: HashMap<i32, Vec<u8>> = HashMap::new();
        self.init_runtime(
            &mut proc_mgr,
            &mut memory_images,
            vfs,
            heap_break,
            mmap_next,
        )?;

        loop {
            if total_cycles >= max_total_cycles {
                stop_reason = "MAX_CYCLES".to_string();
                break;
            }
            if start.elapsed() > timeout {
                stop_reason = "TIMEOUT".to_string();
                break;
            }

            let Some(pid) = proc_mgr.schedule_next() else {
                if proc_mgr
                    .processes
                    .values()
                    .any(|proc| proc.state == ProcessState::Blocked)
                {
                    stop_reason = "DEADLOCK".to_string();
                } else if stop_reason == "COMPLETE" {
                    stop_reason = "HALT".to_string();
                }
                break;
            };

            let (mut heap_break, mut mmap_next) =
                self.restore_runtime_process(&mut proc_mgr, &memory_images, vfs, pid)?;

            let pending_signal = proc_mgr
                .get_mut(pid)
                .and_then(|proc| proc.pending_signal.take());
            if let Some(sig) = pending_signal {
                if sig == SIGTERM || sig == SIGKILL {
                    exit_code = 128 + sig;
                    proc_mgr.process_exit(pid, exit_code, vfs.as_mut().unwrap());
                    memory_images.remove(&pid);
                    Self::wake_pipe_readers(&mut proc_mgr);
                    stop_reason = "EXIT".to_string();
                    continue;
                }
            }

            if proc_mgr
                .get(pid)
                .map(|proc| proc.total_cycles >= MAX_CYCLE_LIMIT)
                .unwrap_or(false)
            {
                exit_code = 137;
                proc_mgr.process_exit(pid, exit_code, vfs.as_mut().unwrap());
                memory_images.remove(&pid);
                Self::wake_pipe_readers(&mut proc_mgr);
                stop_reason = "EXIT".to_string();
                continue;
            }

            let (cycles, signal) = self.dispatch_batch()?;
            total_cycles += cycles as u64;
            if let Some(proc) = proc_mgr.get_mut(pid) {
                proc.total_cycles += cycles as u64;
            }

            // Drain SVC buffer (GPU-side writes)
            for (fd, data) in self.drain_svc_buffer() {
                match fd {
                    1 => stdout_buf.extend_from_slice(&data),
                    2 => stderr_buf.extend_from_slice(&data),
                    _ => {}
                }
            }

            match signal {
                SIGNAL_HALT => {
                    let x8 = self.get_register(8);
                    exit_code = if x8 == 93 || x8 == 94 {
                        self.get_register(0) as i32
                    } else {
                        0
                    };
                    stop_reason = if x8 == 93 || x8 == 94 {
                        "EXIT".to_string()
                    } else {
                        "HALT".to_string()
                    };
                    proc_mgr.process_exit(pid, exit_code, vfs.as_mut().unwrap());
                    memory_images.remove(&pid);
                    Self::wake_pipe_readers(&mut proc_mgr);
                }
                SIGNAL_SYSCALL => {
                    let action = self.handle_runtime_syscall(
                        pid,
                        &mut proc_mgr,
                        &mut memory_images,
                        vfs,
                        &mut heap_break,
                        &mut mmap_next,
                        &mut stdout_buf,
                        &mut stderr_buf,
                        &start,
                        quiet,
                    );

                    match action {
                        RuntimeAction::Continue { ret, advance_pc } => {
                            self.set_register(0, ret);
                            if advance_pc {
                                let pc = self.get_pc();
                                self.set_pc(pc + 4);
                            }
                            self.save_runtime_process(
                                &mut proc_mgr,
                                &mut memory_images,
                                vfs,
                                pid,
                                heap_break,
                                mmap_next,
                            );
                            if let Some(proc) = proc_mgr.get_mut(pid) {
                                if proc.state == ProcessState::Running {
                                    proc.state = ProcessState::Ready;
                                }
                            }
                        }
                        RuntimeAction::Exit(code) => {
                            exit_code = code;
                            stop_reason = "EXIT".to_string();
                            proc_mgr.process_exit(pid, code, vfs.as_mut().unwrap());
                            memory_images.remove(&pid);
                            Self::wake_pipe_readers(&mut proc_mgr);
                        }
                        RuntimeAction::Forked => {}
                        RuntimeAction::Blocked { advance_pc } => {
                            if advance_pc {
                                let pc = self.get_pc();
                                self.set_pc(pc + 4);
                            }
                            self.save_runtime_process(
                                &mut proc_mgr,
                                &mut memory_images,
                                vfs,
                                pid,
                                heap_break,
                                mmap_next,
                            );
                        }
                        RuntimeAction::Execed => {
                            self.save_runtime_process(
                                &mut proc_mgr,
                                &mut memory_images,
                                vfs,
                                pid,
                                heap_break,
                                mmap_next,
                            );
                            if let Some(proc) = proc_mgr.get_mut(pid) {
                                proc.state = ProcessState::Ready;
                            }
                        }
                    }
                }
                SIGNAL_CHECKPOINT => {
                    self.save_runtime_process(
                        &mut proc_mgr,
                        &mut memory_images,
                        vfs,
                        pid,
                        heap_break,
                        mmap_next,
                    );
                    if let Some(proc) = proc_mgr.get_mut(pid) {
                        if proc.state == ProcessState::Running {
                            proc.state = ProcessState::Ready;
                        }
                    }
                }
                _ => {
                    stop_reason = format!("SIGNAL_{}", signal);
                    break;
                }
            }
        }

        // Capture VFS snapshot at end of execution
        let vfs_snapshot = vfs.as_ref().map(|vfs| {
            VfsSnapshot {
                files: vfs.files.iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
                directories: vfs.directories.iter().cloned().collect(),
                symlinks: vfs.symlinks.iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            }
        });

        Ok(LaunchResult {
            total_cycles,
            elapsed_secs: start.elapsed().as_secs_f64(),
            stop_reason,
            exit_code,
            stdout: stdout_buf,
            stderr: stderr_buf,
            total_forks: proc_mgr.total_forks,
            total_context_switches: proc_mgr.total_context_switches,
            processes_created: proc_mgr.next_pid - 1,
            vfs_snapshot,
        })
    }

    // ── Syscall dispatch ─────────────────────────────────────────────────

    fn handle_runtime_syscall(
        &self,
        pid: i32,
        proc_mgr: &mut ProcessManager,
        memory_images: &mut HashMap<i32, Vec<u8>>,
        vfs: &mut Option<GpuVfs>,
        heap_break: &mut u64,
        mmap_next: &mut u64,
        stdout_buf: &mut Vec<u8>,
        stderr_buf: &mut Vec<u8>,
        start: &Instant,
        quiet: bool,
    ) -> RuntimeAction {
        let num = self.get_register(8);
        let x0 = self.get_register(0);
        let x1 = self.get_register(1);
        let x2 = self.get_register(2);
        let x3 = self.get_register(3);

        match num {
            SYS_COMPILE => {
                let src_path = self.read_string(x0 as u64, 256);
                let bin_path = self.read_string(x1 as u64, 256);
                let result = vfs
                    .as_mut()
                    .ok_or_else(|| "missing VFS".to_string())
                    .and_then(|fs| self.compile_guest_c(fs, &src_path, &bin_path));

                match result {
                    Ok(binary_len) => {
                        if !quiet {
                            eprintln!(
                                "[launcher] compiled {} -> {} ({} bytes)",
                                src_path, bin_path, binary_len
                            );
                        }
                        RuntimeAction::Continue {
                            ret: 0,
                            advance_pc: true,
                        }
                    }
                    Err(err) => {
                        if !err.is_empty() {
                            stderr_buf.extend_from_slice(err.as_bytes());
                            if !stderr_buf.ends_with(b"\n") {
                                stderr_buf.push(b'\n');
                            }
                        }
                        if !quiet {
                            eprintln!("[launcher] compile failed: {}", err);
                        }
                        RuntimeAction::Continue {
                            ret: -1,
                            advance_pc: true,
                        }
                    }
                }
            }

            220 => {
                let snapshot = GpuSnapshot {
                    registers: self.get_registers(),
                    pc: self.get_pc() + 4,
                    flags: self.get_flags(),
                };
                if let Some(parent) = proc_mgr.get_mut(pid) {
                    parent.heap_break = *heap_break;
                    parent.mmap_next = *mmap_next;
                    if let Some(fs) = vfs.as_ref() {
                        parent.cwd = fs.getcwd().to_string();
                    }
                }

                let image = self.snapshot_memory_image();
                let fs = vfs.as_mut().unwrap();
                let parent_fd_table = fs.fd_table.clone();
                let child_pid = proc_mgr.fork(pid, &snapshot, &parent_fd_table, fs);
                if child_pid < 0 {
                    return RuntimeAction::Continue {
                        ret: -1,
                        advance_pc: true,
                    };
                }

                memory_images.insert(pid, image.clone());
                memory_images.insert(child_pid, image);
                RuntimeAction::Forked
            }

            SYS_EXECVE => {
                let path = self.read_string(x0 as u64, 256);
                let mut argv = self.read_string_array(x1 as u64, 64, 512);
                if argv.is_empty() {
                    argv.push(path.clone());
                }

                let env_map = if x2 != 0 {
                    let env_entries = self.read_string_array(x2 as u64, 128, 512);
                    let parsed = Self::env_map_from_strings(&env_entries);
                    if parsed.is_empty() {
                        proc_mgr
                            .get(pid)
                            .map(|proc| proc.env.clone())
                            .unwrap_or_else(Self::default_env)
                    } else {
                        parsed
                    }
                } else {
                    proc_mgr
                        .get(pid)
                        .map(|proc| proc.env.clone())
                        .unwrap_or_else(Self::default_env)
                };

                self.exec_guest_binary(
                    pid, &path, argv, env_map, proc_mgr, vfs, heap_break, mmap_next,
                )
            }

            260 => {
                let target_pid = x0 as i32;
                if let Some(zombie) = proc_mgr.reap_zombie(pid, target_pid) {
                    memory_images.remove(&zombie.pid);
                    if x1 != 0 {
                        let status = ((zombie.exit_code & 0xFF) << 8) as i32;
                        self.write_memory(x1 as usize, &status.to_le_bytes());
                    }
                    return RuntimeAction::Continue {
                        ret: zombie.pid as i64,
                        advance_pc: true,
                    };
                }

                if !proc_mgr.has_live_children(pid) {
                    return RuntimeAction::Continue {
                        ret: -10,
                        advance_pc: true,
                    };
                }

                if let Some(proc) = proc_mgr.get_mut(pid) {
                    proc.state = ProcessState::Blocked;
                    proc.wait_target = target_pid;
                }
                RuntimeAction::Blocked { advance_pc: false }
            }

            59 => {
                if proc_mgr
                    .get(pid)
                    .map(|proc| proc.fd_table.len() >= 64)
                    .unwrap_or(false)
                {
                    return RuntimeAction::Continue {
                        ret: -24,
                        advance_pc: true,
                    };
                }

                let fs = vfs.as_mut().unwrap();
                let (read_fd, write_fd) = fs.create_pipe();
                if read_fd < 0 {
                    RuntimeAction::Continue {
                        ret: -24,
                        advance_pc: true,
                    }
                } else {
                    self.write_memory(x0 as usize, &(read_fd as i32).to_le_bytes());
                    self.write_memory(x0 as usize + 4, &(write_fd as i32).to_le_bytes());
                    RuntimeAction::Continue {
                        ret: 0,
                        advance_pc: true,
                    }
                }
            }

            24 => {
                let fs = vfs.as_mut().unwrap();
                let result = fs.dup2(x0 as i32, x1 as i32);
                RuntimeAction::Continue {
                    ret: result as i64,
                    advance_pc: true,
                }
            }

            172 => RuntimeAction::Continue {
                ret: pid as i64,
                advance_pc: true,
            },

            173 => RuntimeAction::Continue {
                ret: proc_mgr.get(pid).map(|proc| proc.ppid as i64).unwrap_or(0),
                advance_pc: true,
            },

            SYS_KILL => {
                let target_pid = x0 as i32;
                let signal = x1 as i32;
                let result = {
                    let fs = vfs.as_mut().unwrap();
                    proc_mgr.kill_process(target_pid, signal, pid, fs)
                };
                if result == 0 && signal == SIGKILL {
                    memory_images.remove(&target_pid);
                    Self::wake_pipe_readers(proc_mgr);
                }
                RuntimeAction::Continue {
                    ret: result as i64,
                    advance_pc: true,
                }
            }

            SYS_GETENV => {
                let key = self.read_string(x0 as u64, 256);
                let buf_size = x2 as usize;
                let value = proc_mgr
                    .get(pid)
                    .and_then(|proc| proc.env.get(&key))
                    .cloned()
                    .unwrap_or_default();
                if value.is_empty() {
                    return RuntimeAction::Continue {
                        ret: 0,
                        advance_pc: true,
                    };
                }

                let value_bytes = value.into_bytes();
                if value_bytes.len() + 1 > buf_size {
                    return RuntimeAction::Continue {
                        ret: -1,
                        advance_pc: true,
                    };
                }

                self.write_memory(x1 as usize, &value_bytes);
                self.write_memory(x1 as usize + value_bytes.len(), &[0]);
                RuntimeAction::Continue {
                    ret: value_bytes.len() as i64,
                    advance_pc: true,
                }
            }

            SYS_SETENV => {
                let key = self.read_string(x0 as u64, 256);
                let value = self.read_string(x1 as u64, 512);
                if let Some(proc) = proc_mgr.get_mut(pid) {
                    proc.env.insert(key, value);
                }
                RuntimeAction::Continue {
                    ret: 0,
                    advance_pc: true,
                }
            }

            SYS_PS => {
                let mut info = String::from("PID  PPID  STATE\n");
                for proc in proc_mgr.processes.values() {
                    let state = match proc.state {
                        ProcessState::Free => "FREE",
                        ProcessState::Ready => "READY",
                        ProcessState::Running => "RUN",
                        ProcessState::Blocked => "BLOCK",
                        ProcessState::Zombie => "ZOMBIE",
                    };
                    info.push_str(&format!("{:3}  {:4}  {}\n", proc.pid, proc.ppid, state));
                }
                if x0 != 0 {
                    self.write_memory(x0 as usize, info.as_bytes());
                    self.write_memory(x0 as usize + info.len(), &[0]);
                } else {
                    stdout_buf.extend_from_slice(info.as_bytes());
                }
                RuntimeAction::Continue {
                    ret: info.len() as i64,
                    advance_pc: true,
                }
            }

            SYS_EXEC => {
                let path = self.read_string(x0 as u64, 256);
                let env_map = proc_mgr
                    .get(pid)
                    .map(|proc| proc.env.clone())
                    .unwrap_or_else(Self::default_env);
                self.exec_guest_binary(
                    pid,
                    &path,
                    vec![path.clone()],
                    env_map,
                    proc_mgr,
                    vfs,
                    heap_break,
                    mmap_next,
                )
            }

            SYS_GETCHAR => RuntimeAction::Continue {
                ret: -1,
                advance_pc: true,
            },

            SYS_CLOCK => RuntimeAction::Continue {
                ret: start.elapsed().as_millis() as i64,
                advance_pc: true,
            },

            SYS_SLEEP => {
                let sleep_ms = x0.max(0) as u64;
                if sleep_ms > 0 {
                    std::thread::sleep(Duration::from_millis(sleep_ms));
                }
                RuntimeAction::Continue {
                    ret: 0,
                    advance_pc: true,
                }
            }

            SYS_SOCKET => {
                if x0 != 2 || x1 != 1 {
                    return RuntimeAction::Continue {
                        ret: -97,
                        advance_pc: true,
                    };
                }

                let fs = vfs.as_mut().unwrap();
                let fd = fs.insert_host_socket(
                    HostSocketState::PendingTcp {
                        bind_host: None,
                        bind_port: None,
                    },
                    x1 as i32,
                    "<socket>",
                );
                RuntimeAction::Continue {
                    ret: if fd >= 0 { fd as i64 } else { -24 },
                    advance_pc: true,
                }
            }

            SYS_BIND => {
                let fd = x0 as i32;
                let host = if x1 > 255 {
                    self.read_string(x1 as u64, 128)
                } else {
                    "0.0.0.0".to_string()
                };
                let port = x2.clamp(0, u16::MAX as i64) as u16;
                let fs = vfs.as_mut().unwrap();
                let result = match fs.fd_table.get(&fd).cloned() {
                    Some(entry) => match entry.kind {
                        FdKind::HostSocket(socket) => {
                            let mut state = socket.lock().unwrap();
                            match &mut *state {
                                HostSocketState::PendingTcp {
                                    bind_host,
                                    bind_port,
                                } => {
                                    *bind_host = Some(host);
                                    *bind_port = Some(port);
                                    0
                                }
                                _ => -1,
                            }
                        }
                        _ => -9,
                    },
                    None => -9,
                };
                RuntimeAction::Continue {
                    ret: result as i64,
                    advance_pc: true,
                }
            }

            SYS_LISTEN => {
                let fd = x0 as i32;
                let fs = vfs.as_mut().unwrap();
                let result = match fs.fd_table.get(&fd).cloned() {
                    Some(entry) => match entry.kind {
                        FdKind::HostSocket(socket) => {
                            let mut state = socket.lock().unwrap();
                            match &mut *state {
                                HostSocketState::PendingTcp { bind_host, bind_port }
                                | HostSocketState::PendingUdp { bind_host, bind_port } => {
                                    let host =
                                        bind_host.clone().unwrap_or_else(|| "0.0.0.0".to_string());
                                    let port = bind_port.unwrap_or(0);
                                    // For UDP, bind creates a UdpSocket
                                    if let HostSocketState::PendingUdp { .. } = &*state {
                                        match UdpSocket::bind((host.as_str(), port)) {
                                            Ok(socket) => {
                                                *state = HostSocketState::UdpSocket(socket);
                                                0
                                            }
                                            Err(_) => -1,
                                        }
                                    } else {
                                        match TcpListener::bind((host.as_str(), port)) {
                                            Ok(listener) => {
                                                *state = HostSocketState::Listener(listener);
                                                0
                                            }
                                            Err(_) => -1,
                                        }
                                    }
                                }
                                HostSocketState::Listener(_) => 0,
                                HostSocketState::Stream(_) => 0,
                                HostSocketState::UdpSocket(_) => 0,
                            }
                        }
                        _ => -9,
                    },
                    None => -9,
                };
                RuntimeAction::Continue {
                    ret: result as i64,
                    advance_pc: true,
                }
            }

            SYS_ACCEPT => {
                let fd = x0 as i32;
                let fs = vfs.as_mut().unwrap();
                let accepted = match fs.fd_table.get(&fd).cloned() {
                    Some(entry) => match entry.kind {
                        FdKind::HostSocket(socket) => {
                            let mut state = socket.lock().unwrap();
                            match &mut *state {
                                HostSocketState::Listener(listener) => {
                                    listener.accept().ok().map(|(stream, _)| stream)
                                }
                                _ => None,
                            }
                        }
                        _ => None,
                    },
                    None => None,
                };
                let ret = match accepted {
                    Some(stream) => fs.insert_host_socket(
                        HostSocketState::Stream(stream),
                        0,
                        "<socket:accepted>",
                    ) as i64,
                    None => -1,
                };
                RuntimeAction::Continue {
                    ret,
                    advance_pc: true,
                }
            }

            SYS_CONNECT => {
                let fd = x0 as i32;
                let host = if x1 > 255 {
                    self.read_string(x1 as u64, 128)
                } else {
                    "127.0.0.1".to_string()
                };
                let port = x2.clamp(0, u16::MAX as i64) as u16;
                let fs = vfs.as_mut().unwrap();
                let result = match fs.fd_table.get(&fd).cloned() {
                    Some(entry) => match entry.kind {
                        FdKind::HostSocket(socket) => {
                            let mut state = socket.lock().unwrap();
                            match &mut *state {
                                HostSocketState::PendingTcp { .. } | HostSocketState::PendingUdp { .. } => {
                                    match TcpStream::connect((host.as_str(), port)) {
                                        Ok(stream) => {
                                            *state = HostSocketState::Stream(stream);
                                            0
                                        }
                                        Err(_) => -1,
                                    }
                                }
                                _ => -1,
                            }
                        }
                        _ => -9,
                    },
                    None => -9,
                };
                RuntimeAction::Continue {
                    ret: result as i64,
                    advance_pc: true,
                }
            }

            SYS_SEND => {
                let fd = x0 as i32;
                let data = self.read_memory(x1 as usize, x2 as usize);
                let fs = vfs.as_mut().unwrap();
                let result = match fs.fd_table.get(&fd).cloned() {
                    Some(entry) => match entry.kind {
                        FdKind::HostSocket(socket) => {
                            let mut state = socket.lock().unwrap();
                            match &mut *state {
                                HostSocketState::Stream(stream) => {
                                    stream.write(&data).map(|n| n as i64).unwrap_or(-1)
                                }
                                _ => -1,
                            }
                        }
                        _ => -9,
                    },
                    None => -9,
                };
                RuntimeAction::Continue {
                    ret: result,
                    advance_pc: true,
                }
            }

            SYS_RECV => {
                let fd = x0 as i32;
                let buf_addr = x1 as usize;
                let max_len = (x2 as usize).min(64 * 1024);
                let fs = vfs.as_mut().unwrap();
                let mut recv_buf = vec![0u8; max_len];
                let result = match fs.fd_table.get(&fd).cloned() {
                    Some(entry) => match entry.kind {
                        FdKind::HostSocket(socket) => {
                            let mut state = socket.lock().unwrap();
                            match &mut *state {
                                HostSocketState::Stream(stream) => {
                                    match stream.read(&mut recv_buf) {
                                        Ok(n) => {
                                            if n > 0 {
                                                self.write_memory(buf_addr, &recv_buf[..n]);
                                            }
                                            n as i64
                                        }
                                        Err(_) => -1,
                                    }
                                }
                                _ => -1,
                            }
                        }
                        _ => -9,
                    },
                    None => -9,
                };
                RuntimeAction::Continue {
                    ret: result,
                    advance_pc: true,
                }
            }

            63 => {
                let fd = x0 as i32;
                let fs = vfs.as_mut().unwrap();
                if let Some(entry) = fs.fd_table.get(&fd).cloned() {
                    if let FdKind::PipeRead(idx) = entry.kind {
                        let data = fs
                            .pipes
                            .get_mut(idx)
                            .and_then(|pipe| pipe.read(x2 as usize));
                        return match data {
                            Some(data) => {
                                if !data.is_empty() {
                                    self.write_memory(x1 as usize, &data);
                                }
                                RuntimeAction::Continue {
                                    ret: data.len() as i64,
                                    advance_pc: true,
                                }
                            }
                            None => {
                                if let Some(proc) = proc_mgr.get_mut(pid) {
                                    proc.state = ProcessState::Blocked;
                                    proc.wait_target = WAIT_TARGET_PIPE;
                                }
                                RuntimeAction::Blocked { advance_pc: false }
                            }
                        };
                    }
                }
                match self.handle_syscall(
                    num, x0, x1, x2, x3, vfs, heap_break, mmap_next, stdout_buf, stderr_buf, quiet,
                ) {
                    SyscallResult::Continue(ret) => RuntimeAction::Continue {
                        ret,
                        advance_pc: true,
                    },
                    SyscallResult::Exit(code) => RuntimeAction::Exit(code),
                }
            }

            64 => {
                let fd = x0 as i32;
                let fs = vfs.as_mut().unwrap();
                if let Some(entry) = fs.fd_table.get(&fd).cloned() {
                    if let FdKind::PipeWrite(idx) = entry.kind {
                        let data = self.read_memory(x1 as usize, x2 as usize);
                        let written = fs
                            .pipes
                            .get_mut(idx)
                            .map(|pipe| pipe.write(&data))
                            .unwrap_or(-1);
                        if written >= 0 {
                            Self::wake_pipe_readers(proc_mgr);
                        }
                        return RuntimeAction::Continue {
                            ret: written as i64,
                            advance_pc: true,
                        };
                    }
                }
                match self.handle_syscall(
                    num, x0, x1, x2, x3, vfs, heap_break, mmap_next, stdout_buf, stderr_buf, quiet,
                ) {
                    SyscallResult::Continue(ret) => RuntimeAction::Continue {
                        ret,
                        advance_pc: true,
                    },
                    SyscallResult::Exit(code) => RuntimeAction::Exit(code),
                }
            }

            57 => {
                let fs = vfs.as_mut().unwrap();
                let result = fs.close(x0 as i32);
                if result == 0 {
                    Self::wake_pipe_readers(proc_mgr);
                }
                RuntimeAction::Continue {
                    ret: result as i64,
                    advance_pc: true,
                }
            }

            _ => match self.handle_syscall(
                    num, x0, x1, x2, x3, vfs, heap_break, mmap_next, stdout_buf, stderr_buf, quiet,
                ) {
                    SyscallResult::Continue(ret) => RuntimeAction::Continue {
                        ret,
                        advance_pc: true,
                    },
                    SyscallResult::Exit(code) => RuntimeAction::Exit(code),
                }
            }
    }

    fn handle_syscall(
        &self,
        num: i64,
        x0: i64,
        x1: i64,
        x2: i64,
        x3: i64,
        vfs: &mut Option<GpuVfs>,
        heap_break: &mut u64,
        mmap_next: &mut u64,
        stdout_buf: &mut Vec<u8>,
        stderr_buf: &mut Vec<u8>,
        _quiet: bool,
    ) -> SyscallResult {
        // Import syscall numbers (Linux aarch64)
        match num {
            // ── Exit ─────────────────────────────────────────
            93 | 94 => {
                // exit / exit_group
                SyscallResult::Exit(x0 as i32)
            }

            // ── Write ────────────────────────────────────────
            64 => {
                // write(fd, buf, count)
                let fd = x0;
                let buf_addr = x1 as u64;
                let count = x2 as usize;
                if count == 0 {
                    return SyscallResult::Continue(0);
                }
                let data = self.read_memory(buf_addr as usize, count);
                match fd {
                    1 => {
                        stdout_buf.extend_from_slice(&data);
                        SyscallResult::Continue(data.len() as i64)
                    }
                    2 => {
                        stderr_buf.extend_from_slice(&data);
                        SyscallResult::Continue(data.len() as i64)
                    }
                    _ => {
                        if let Some(ref mut vfs) = vfs {
                            let written = vfs.write(fd as i32, &data);
                            SyscallResult::Continue(written as i64)
                        } else {
                            SyscallResult::Continue(data.len() as i64)
                        }
                    }
                }
            }

            // ── Read ─────────────────────────────────────────
            63 => {
                // read(fd, buf, count)
                let fd = x0 as i32;
                let buf_addr = x1 as usize;
                let count = x2 as usize;
                if let Some(ref mut vfs) = vfs {
                    match vfs.read(fd, count) {
                        Some(data) => {
                            self.write_memory(buf_addr, &data);
                            SyscallResult::Continue(data.len() as i64)
                        }
                        None => SyscallResult::Continue(0), // EOF or would-block
                    }
                } else {
                    SyscallResult::Continue(0)
                }
            }

            // ── Open ─────────────────────────────────────────
            56 => {
                // openat(dirfd, path, flags, mode)
                let path = self.read_string(x1 as u64, 256);
                if let Some(ref mut vfs) = vfs {
                    let fd = vfs.open(&path, x2 as i32);
                    SyscallResult::Continue(if fd >= 0 { fd as i64 } else { -2 })
                // -ENOENT
                } else {
                    SyscallResult::Continue(-2)
                }
            }

            // ── Close ────────────────────────────────────────
            57 => {
                // close(fd)
                if let Some(ref mut vfs) = vfs {
                    let r = vfs.close(x0 as i32);
                    SyscallResult::Continue(r as i64)
                } else {
                    SyscallResult::Continue(0)
                }
            }

            // ── Lseek ────────────────────────────────────────
            62 => {
                // lseek(fd, offset, whence)
                if let Some(ref mut vfs) = vfs {
                    let r = vfs.lseek(x0 as i32, x1, x2 as i32);
                    SyscallResult::Continue(r)
                } else {
                    SyscallResult::Continue(-1)
                }
            }

            // ── Fstat ────────────────────────────────────────
            80 => {
                // fstat(fd, statbuf)
                if let Some(ref vfs) = vfs {
                    if let Some(info) = vfs.fstat(x0 as i32) {
                        let packed = GpuVfs::pack_stat64(&info);
                        self.write_memory(x1 as usize, &packed);
                        SyscallResult::Continue(0)
                    } else {
                        SyscallResult::Continue(-9) // -EBADF
                    }
                } else {
                    SyscallResult::Continue(-9)
                }
            }

            // ── Newfstatat ───────────────────────────────────
            79 => {
                // newfstatat(dirfd, path, statbuf, flags)
                let path = self.read_string(x1 as u64, 256);
                if let Some(ref vfs) = vfs {
                    let info = if !path.is_empty() {
                        vfs.stat(&path)
                    } else if (x3 & 0x1000) != 0 {
                        // AT_EMPTY_PATH
                        vfs.fstat(x0 as i32)
                    } else {
                        None
                    };
                    if let Some(info) = info {
                        let packed = GpuVfs::pack_stat64(&info);
                        self.write_memory(x2 as usize, &packed);
                        SyscallResult::Continue(0)
                    } else {
                        SyscallResult::Continue(-2) // -ENOENT
                    }
                } else {
                    SyscallResult::Continue(-2)
                }
            }

            // ── Getdents64 ───────────────────────────────────
            61 => {
                // getdents64(fd, dirp, count)
                if let Some(ref mut vfs) = vfs {
                    let fd_key = x0 as i32;
                    if let Some(entry) = vfs.fd_table.get(&fd_key) {
                        let dir_path = entry.path.clone();
                        let consumed = entry.offset;
                        let (packed, entry_count) =
                            vfs.pack_getdents64(&dir_path, x2 as usize, consumed);
                        if !packed.is_empty() {
                            self.write_memory(x1 as usize, &packed);
                        }
                        if let Some(e) = vfs.fd_table.get_mut(&fd_key) {
                            e.offset += entry_count;
                        }
                        SyscallResult::Continue(packed.len() as i64)
                    } else {
                        SyscallResult::Continue(-9) // -EBADF
                    }
                } else {
                    SyscallResult::Continue(-9)
                }
            }

            // ── Readlinkat ───────────────────────────────────
            78 => {
                let path = self.read_string(x1 as u64, 256);
                if let Some(ref vfs) = vfs {
                    if let Some(target) = vfs.readlink(&path) {
                        let target_bytes = target.as_bytes();
                        let len = target_bytes.len().min(x3 as usize);
                        self.write_memory(x2 as usize, &target_bytes[..len]);
                        SyscallResult::Continue(len as i64)
                    } else {
                        SyscallResult::Continue(-22) // -EINVAL
                    }
                } else {
                    SyscallResult::Continue(-22)
                }
            }

            // ── Faccessat ────────────────────────────────────
            48 => {
                let path = self.read_string(x1 as u64, 256);
                if let Some(ref vfs) = vfs {
                    let exists = vfs.exists(&path);
                    SyscallResult::Continue(if exists { 0 } else { -2 })
                } else {
                    SyscallResult::Continue(-2)
                }
            }

            // ── BRK ──────────────────────────────────────────
            214 => {
                // brk
                let addr = x0 as u64;
                if addr == 0 {
                    SyscallResult::Continue(*heap_break as i64)
                } else {
                    *heap_break = addr;
                    SyscallResult::Continue(*heap_break as i64)
                }
            }

            // ── MMAP ─────────────────────────────────────────
            222 => {
                // mmap(addr, len, prot, flags, fd, offset)
                let addr = x0 as u64;
                let length = x1 as u64;
                if addr == 0 {
                    let aligned = (*mmap_next + 0xFFF) & !0xFFF;
                    *mmap_next = aligned + length;
                    SyscallResult::Continue(aligned as i64)
                } else {
                    SyscallResult::Continue(addr as i64)
                }
            }

            // ── MPROTECT / MUNMAP ────────────────────────────
            226 | 215 => SyscallResult::Continue(0),

            // ── Dup3 ─────────────────────────────────────────
            24 => {
                // dup3(oldfd, newfd, flags)
                if let Some(ref mut vfs) = vfs {
                    let r = vfs.dup2(x0 as i32, x1 as i32);
                    SyscallResult::Continue(r as i64)
                } else {
                    SyscallResult::Continue(-9)
                }
            }

            // ── Pipe2 ────────────────────────────────────────
            59 => {
                // pipe2(pipefd, flags)
                if let Some(ref mut vfs) = vfs {
                    let (rfd, wfd) = vfs.create_pipe();
                    if rfd >= 0 {
                        self.write_memory(x0 as usize, &(rfd as i32).to_le_bytes());
                        self.write_memory(x0 as usize + 4, &(wfd as i32).to_le_bytes());
                        SyscallResult::Continue(0)
                    } else {
                        SyscallResult::Continue(-24) // -EMFILE
                    }
                } else {
                    SyscallResult::Continue(-24)
                }
            }

            // ── Chdir ────────────────────────────────────────
            49 => {
                let path = self.read_string(x0 as u64, 256);
                if let Some(ref mut vfs) = vfs {
                    let r = vfs.chdir(&path);
                    SyscallResult::Continue(r as i64)
                } else {
                    SyscallResult::Continue(-2)
                }
            }

            // ── Getcwd ───────────────────────────────────────
            17 => {
                if let Some(ref vfs) = vfs {
                    let cwd = format!("{}\0", vfs.getcwd());
                    self.write_memory(x0 as usize, cwd.as_bytes());
                    SyscallResult::Continue(x0)
                } else {
                    self.write_memory(x0 as usize, b"/\0");
                    SyscallResult::Continue(x0)
                }
            }

            // ── Mkdirat ──────────────────────────────────────
            34 => {
                let path = self.read_string(x1 as u64, 256);
                if let Some(ref mut vfs) = vfs {
                    let r = vfs.mkdir(&path);
                    SyscallResult::Continue(r as i64)
                } else {
                    SyscallResult::Continue(-1)
                }
            }

            // ── Unlinkat ─────────────────────────────────────
            35 => {
                let path = self.read_string(x1 as u64, 256);
                let flags = x2;  // Third argument (x2), not x3
                if let Some(ref mut vfs) = vfs {
                    let r = if (flags & 0x200) != 0 {
                        // AT_REMOVEDIR
                        vfs.rmdir(&path)
                    } else {
                        vfs.unlink(&path)
                    };
                    SyscallResult::Continue(r as i64)
                } else {
                    SyscallResult::Continue(-2)
                }
            }

            // ── Renameat ─────────────────────────────────────
            38 => {
                let old_path = self.read_string(x1 as u64, 256);
                let new_path = self.read_string(x3 as u64, 256);
                if let Some(ref mut vfs) = vfs {
                    let r = vfs.rename(&old_path, &new_path);
                    SyscallResult::Continue(r as i64)
                } else {
                    SyscallResult::Continue(-2)
                }
            }

            // ── Symlinkat ────────────────────────────────────
            36 => {
                let target = self.read_string(x0 as u64, 256);
                let link_path = self.read_string(x2 as u64, 256);
                if let Some(ref mut vfs) = vfs {
                    let r = vfs.symlink(&target, &link_path);
                    SyscallResult::Continue(r as i64)
                } else {
                    SyscallResult::Continue(-1)
                }
            }

            // ── Linkat ───────────────────────────────────────
            37 => {
                let old_path = self.read_string(x1 as u64, 256);
                let new_path = self.read_string(x3 as u64, 256);
                if let Some(ref mut vfs) = vfs {
                    let resolved_old = vfs.resolve_path(&old_path);
                    if let Some(content) = vfs.files.get(&resolved_old).cloned() {
                        let resolved_new = vfs.resolve_path(&new_path);
                        vfs.files.insert(resolved_new, content);
                        SyscallResult::Continue(0)
                    } else {
                        SyscallResult::Continue(-2)
                    }
                } else {
                    SyscallResult::Continue(-1)
                }
            }

            // ── Ioctl ────────────────────────────────────────
            29 => {
                let fd = x0 as i32;
                let request = x1 as u64;
                let argp = x2 as usize;

                // Common ioctl operations
                match request {
                    0x5413 => {
                        // TIOCGWINSZ - get window size
                        let winsize: [u8; 8] = {
                            let mut buf = [0u8; 8];
                            buf[0..2].copy_from_slice(&24u16.to_le_bytes()); // rows
                            buf[2..4].copy_from_slice(&80u16.to_le_bytes()); // cols
                            buf[4..6].copy_from_slice(&0u16.to_le_bytes()); // xpixel
                            buf[6..8].copy_from_slice(&0u16.to_le_bytes()); // ypixel
                            buf
                        };
                        self.write_memory(argp, &winsize);
                        SyscallResult::Continue(0)
                    }
                    0x5401 => {
                        // TCGETS - get terminal attributes
                        // Return dummy terminal settings
                        let termios: [u8; 36] = [0u8; 36];
                        self.write_memory(argp, &termios);
                        SyscallResult::Continue(0)
                    }
                    0x5402 => {
                        // TCSETS - set terminal attributes (no-op)
                        SyscallResult::Continue(0)
                    }
                    0x5403 => {
                        // TCSETSW - set terminal attributes (drain) (no-op)
                        SyscallResult::Continue(0)
                    }
                    0x5404 => {
                        // TCSETSF - set terminal attributes (flush) (no-op)
                        SyscallResult::Continue(0)
                    }
                    0x5405 => {
                        // TCFLUSH - flush buffers
                        SyscallResult::Continue(0)
                    }
                    0x5406 => {
                        // TCSBRK - break (no-op)
                        SyscallResult::Continue(0)
                    }
                    0x5407 => {
                        // TIOCMGET - get modem lines
                        SyscallResult::Continue(0)
                    }
                    0x5408 => {
                        // TIOCMSET - set modem lines
                        SyscallResult::Continue(0)
                    }
                    0x540C => {
                        // TIOCGSID - get session ID (return 0)
                        self.write_memory(argp, &0u32.to_le_bytes());
                        SyscallResult::Continue(0)
                    }
                    _ => {
                        // Unknown ioctl - return ENOTTY
                        SyscallResult::Continue(-25) // -ENOTTY
                    }
                }
            }

            // ── Writev ───────────────────────────────────────
            66 => {
                let iov_addr = x1 as usize;
                let iovcnt = x2 as usize;
                let mut total = 0i64;
                for i in 0..iovcnt {
                    let base = self.read_u64((iov_addr + i * 16) as usize);
                    let len = self.read_u64((iov_addr + i * 16 + 8) as usize) as usize;
                    if len > 0 && base > 0 {
                        let data = self.read_memory(base as usize, len);
                        match x0 {
                            1 => stdout_buf.extend_from_slice(&data),
                            2 => stderr_buf.extend_from_slice(&data),
                            _ => {
                                if let Some(ref mut vfs) = vfs {
                                    vfs.write(x0 as i32, &data);
                                }
                            }
                        }
                        total += len as i64;
                    }
                }
                SyscallResult::Continue(total)
            }

            // ── Fcntl ────────────────────────────────────────
            25 => {
                let fd = x0 as i32;
                let cmd = x1;
                let arg = x2;

                match cmd {
                    0 => {
                        // F_DUPFD - duplicate fd
                        if let Some(ref mut vfs) = vfs {
                            let r = vfs.dup2(fd, -1);
                            SyscallResult::Continue(r as i64)
                        } else {
                            SyscallResult::Continue(-9) // EBADF
                        }
                    }
                    1 => {
                        // F_GETFD - get close-on-exec flag
                        SyscallResult::Continue(0)
                    }
                    2 => {
                        // F_SETFD - set close-on-exec flag (no-op)
                        SyscallResult::Continue(0)
                    }
                    3 => {
                        // F_GETFL - get file status flags
                        // Return O_RDWR for valid fds
                        if let Some(vfs) = vfs.as_ref() {
                            if vfs.fd_table.contains_key(&fd) {
                                SyscallResult::Continue(2) // O_RDWR
                            } else {
                                SyscallResult::Continue(-9) // EBADF
                            }
                        } else {
                            SyscallResult::Continue(-9)
                        }
                    }
                    4 => {
                        // F_SETFL - set file status flags (no-op for now)
                        SyscallResult::Continue(0)
                    }
                    9 => {
                        // F_SETOWN - set process to receive SIGIO
                        SyscallResult::Continue(0)
                    }
                    8 => {
                        // F_GETOWN - get process for SIGIO
                        SyscallResult::Continue(0)
                    }
                    10 => {
                        // F_GETLK - get lock
                        SyscallResult::Continue(0)
                    }
                    11 => {
                        // F_SETLK - set lock (no-op)
                        SyscallResult::Continue(0)
                    }
                    12 => {
                        // F_SETLKW - set lock wait (no-op)
                        SyscallResult::Continue(0)
                    }
                    1030 => {
                        // F_DUPFD_CLOEXEC - duplicate with cloexec
                        if let Some(ref mut vfs) = vfs {
                            let r = vfs.dup2(fd, -1);
                            SyscallResult::Continue(r as i64)
                        } else {
                            SyscallResult::Continue(-9)
                        }
                    }
                    _ => SyscallResult::Continue(0),
                }
            }

            // ── Prctl ────────────────────────────────────────
            157 => {
                let option = x0 as i32;
                let arg2 = x1 as u64;
                let arg3 = x2 as u64;
                let arg4 = x3 as u64;
                let arg5 = self.get_register(4) as u64;

                match option {
                    0 => {
                        // PR_SET_NAME - set process name
                        // No-op for now
                        SyscallResult::Continue(0)
                    }
                    1 => {
                        // PR_GET_NAME - get process name
                        let addr = arg2 as usize;
                        let name = b"ncpu-gpu\0";
                        self.write_memory(addr, name);
                        SyscallResult::Continue(0)
                    }
                    16 => {
                        // PR_SET_PDEATHSIG - parent death signal
                        SyscallResult::Continue(0)
                    }
                    17 => {
                        // PR_GET_PDEATHSIG - get parent death signal
                        let addr = arg2 as usize;
                        self.write_memory(addr, &0u32.to_le_bytes());
                        SyscallResult::Continue(0)
                    }
                    15 => {
                        // PR_GET_TID_ADDRESS - get tid address
                        let addr = arg2 as usize;
                        // Return a valid pointer (use a dummy location)
                        self.write_memory(addr, &(0u64).to_le_bytes());
                        SyscallResult::Continue(0)
                    }
                    18 => {
                        // PR_SET_THP_DISABLE - transparent hugepage disable
                        SyscallResult::Continue(0)
                    }
                    38 => {
                        // PR_GET_SPECULATION_CTRL - speculation control
                        SyscallResult::Continue(0)
                    }
                    39 => {
                        // PR_SET_SPECULATION_CTRL - speculation control
                        SyscallResult::Continue(0)
                    }
                    _ => SyscallResult::Continue(0),
                }
            }

            // ── Uname ────────────────────────────────────────
            160 => {
                let buf_addr = x0 as usize;
                let fields: [&[u8]; 5] = [
                    b"Linux\0",
                    b"ncpu-gpu\0",
                    b"6.1.0-ncpu\0",
                    b"#1 SMP Metal GPU\0",
                    b"aarch64\0",
                ];
                let mut offset = 0;
                for field in &fields {
                    let mut padded = [0u8; 65];
                    let len = field.len().min(65);
                    padded[..len].copy_from_slice(&field[..len]);
                    self.write_memory(buf_addr + offset, &padded);
                    offset += 65;
                }
                SyscallResult::Continue(0)
            }

            // ── Clock_gettime ────────────────────────────────
            113 => {
                let ts_addr = x1 as usize;
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default();
                let sec = now.as_secs() as i64;
                let nsec = now.subsec_nanos() as i64;
                self.write_memory(ts_addr, &sec.to_le_bytes());
                self.write_memory(ts_addr + 8, &nsec.to_le_bytes());
                SyscallResult::Continue(0)
            }

            // ── Clock_getres ─────────────────────────────────
            114 => {
                let ts_addr = x1 as usize;
                if ts_addr != 0 {
                    self.write_memory(ts_addr, &0i64.to_le_bytes());
                    self.write_memory(ts_addr + 8, &1_000_000i64.to_le_bytes());
                    // 1ms
                }
                SyscallResult::Continue(0)
            }

            // ── Getrandom ────────────────────────────────────
            278 => {
                let buf_addr = x0 as usize;
                let length = (x1 as usize).min(256);
                // Deterministic "random" for reproducibility
                let data: Vec<u8> = (0..length)
                    .map(|i| (i as u8).wrapping_mul(37).wrapping_add(7))
                    .collect();
                self.write_memory(buf_addr, &data);
                SyscallResult::Continue(length as i64)
            }

            // ── Socketcall (socket, bind, listen, accept, etc.) ─
            // For aarch64 musl, socketcall passes args directly in x0, x1, x2 rather than in an array
            198 => {
                // x0 = domain (2 = AF_INET), x1 = type (1 = SOCK_STREAM, 2 = SOCK_DGRAM), x2 = protocol (0)
                let domain = x0 as i32;
                let sock_type = x1 as i32;
                let protocol = x2 as i32;

                // Accept AF_INET + SOCK_STREAM (TCP) or SOCK_DGRAM (UDP)
                if domain == 2 {
                    if let Some(ref mut vfs) = vfs {
                        use crate::vfs::HostSocketState;
                        let state = if sock_type == 1 {
                            // TCP
                            HostSocketState::PendingTcp {
                                bind_host: None,
                                bind_port: None,
                            }
                        } else if sock_type == 2 {
                            // UDP
                            HostSocketState::PendingUdp {
                                bind_host: None,
                                bind_port: None,
                            }
                        } else {
                            return SyscallResult::Continue(-97); // EAFNOSUPPORT
                        };
                        let fd = vfs.insert_host_socket(state, 0, "<socket>");
                        SyscallResult::Continue(if fd >= 0 { fd as i64 } else { -24 })
                    } else {
                        SyscallResult::Continue(-24) // -EMFILE
                    }
                } else {
                    SyscallResult::Continue(-97) // -EAFNOSUPPORT
                }
            }

            // ── Bind (syscall 200) ────────────────────────────
            200 => {
                // bind(sockfd, addr, addrlen)
                let sockfd = x0 as i32;
                let addr_ptr = x1 as usize;
                let _addrlen = x2 as usize;

                if let Some(ref vfs) = vfs {
                    // Read sockaddr_in: sin_family(2), sin_port(2), sin_addr(4), rest zeros
                    let data = self.read_memory(addr_ptr, 16);
                    let family = u16::from_le_bytes([data[0], data[1]]);
                    let port = u16::from_le_bytes([data[2], data[3]]);
                    let ip = format!(
                        "{}.{}.{}.{}",
                        data[4], data[5], data[6], data[7]
                    );

                    if family != 2 {
                        SyscallResult::Continue(-97) // -EAFNOSUPPORT
                    } else if let Some(entry) = vfs.fd_table.get(&sockfd) {
                        use crate::vfs::FdKind;
                        use crate::vfs::HostSocketState;
                        if let FdKind::HostSocket(socket) = &entry.kind {
                            let mut state = socket.lock().unwrap();
                            if let HostSocketState::PendingTcp { bind_host, bind_port } | HostSocketState::PendingUdp { bind_host, bind_port } = &mut *state {
                                *bind_host = Some(ip);
                                *bind_port = Some(port);
                                SyscallResult::Continue(0)
                            } else {
                                SyscallResult::Continue(-22) // -EINVAL
                            }
                        } else {
                            SyscallResult::Continue(-9) // -EBADF
                        }
                    } else {
                        SyscallResult::Continue(-9) // -EBADF
                    }
                } else {
                    SyscallResult::Continue(-9)
                }
            }

            // ── Listen (syscall 201) ───────────────────────────
            201 => {
                // listen(sockfd, backlog) OR connect(sockfd, addr, addrlen)
                // musl on aarch64 may use 201 for listen, check if second arg looks like backlog
                let sockfd = x0 as i32;
                let x1_val = x1 as i32;
                let x2_val = x2 as i32;

                // If x2 (addrlen) is small like 0 or 1, it's likely listen(backlog)
                if x2_val == 0 || x2_val == 1 || x1_val > 1000 {
                    // listen(backlog)
                    if let Some(ref vfs) = vfs {
                        if let Some(entry) = vfs.fd_table.get(&sockfd) {
                            use crate::vfs::FdKind;
                            use crate::vfs::HostSocketState;
                            if let FdKind::HostSocket(socket) = &entry.kind {
                                let mut state = socket.lock().unwrap();
                                if let HostSocketState::PendingTcp { bind_host, bind_port } | HostSocketState::PendingUdp { bind_host, bind_port } = &mut *state {
                                    let host = bind_host.clone().unwrap_or_else(|| "0.0.0.0".to_string());
                                    let port = bind_port.unwrap_or(0);
                                    match std::net::TcpListener::bind((host.as_str(), port)) {
                                        Ok(listener) => {
                                            *state = HostSocketState::Listener(listener);
                                            SyscallResult::Continue(0)
                                        }
                                        Err(_) => {
                                            SyscallResult::Continue(-98)
                                        }
                                    }
                                } else {
                                    SyscallResult::Continue(-22)
                                }
                            } else {
                                SyscallResult::Continue(-9)
                            }
                        } else {
                            SyscallResult::Continue(-9)
                        }
                    } else {
                        SyscallResult::Continue(-9)
                    }
                } else {
                    // connect(sockfd, addr, addrlen)
                    let addr_ptr = x1 as usize;
                    let _addrlen = x2 as usize;

                    if let Some(ref vfs) = vfs {
                        let data = self.read_memory(addr_ptr, 16);
                        let family = u16::from_le_bytes([data[0], data[1]]);
                        let port = u16::from_le_bytes([data[2], data[3]]);
                        let ip = format!(
                            "{}.{}.{}.{}",
                            data[4], data[5], data[6], data[7]
                        );

                        if family != 2 {
                            SyscallResult::Continue(-97)
                        } else if let Some(entry) = vfs.fd_table.get(&sockfd) {
                            use crate::vfs::FdKind;
                            use crate::vfs::HostSocketState;
                            if let FdKind::HostSocket(socket) = &entry.kind {
                                let mut state = socket.lock().unwrap();
                                if let HostSocketState::PendingTcp { .. } | HostSocketState::PendingUdp { .. } = &mut *state {
                                    match std::net::TcpStream::connect((ip.as_str(), port)) {
                                        Ok(stream) => {
                                            *state = HostSocketState::Stream(stream);
                                            SyscallResult::Continue(0)
                                        }
                                        Err(_) => {
                                            SyscallResult::Continue(-111)
                                        }
                                    }
                                } else {
                                    SyscallResult::Continue(-22)
                                }
                            } else {
                                SyscallResult::Continue(-9)
                            }
                        } else {
                            SyscallResult::Continue(-9)
                        }
                    } else {
                        SyscallResult::Continue(-9)
                    }
                }
            }

            // ── Listen (syscall 202) ───────────────────────────
            202 => {
                // listen(sockfd, backlog)
                let sockfd = x0 as i32;
                let backlog = x1 as i32;

                if let Some(ref vfs) = vfs {
                    if let Some(entry) = vfs.fd_table.get(&sockfd) {
                        use crate::vfs::FdKind;
                        use crate::vfs::HostSocketState;
                        if let FdKind::HostSocket(socket) = &entry.kind {
                            let mut state = socket.lock().unwrap();
                            if let HostSocketState::PendingTcp { bind_host, bind_port } | HostSocketState::PendingUdp { bind_host, bind_port } = &mut *state {
                                let host = bind_host.clone().unwrap_or_else(|| "0.0.0.0".to_string());
                                let port = bind_port.unwrap_or(0);
                                match std::net::TcpListener::bind((host.as_str(), port)) {
                                    Ok(listener) => {
                                        *state = HostSocketState::Listener(listener);
                                        SyscallResult::Continue(0)
                                    }
                                    Err(_) => {
                                        SyscallResult::Continue(-98)
                                    }
                                }
                            } else {
                                SyscallResult::Continue(-22) // -EINVAL
                            }
                        } else {
                            SyscallResult::Continue(-9)
                        }
                    } else {
                        SyscallResult::Continue(-9)
                    }
                } else {
                    SyscallResult::Continue(-9)
                }
            }

            // ── Accept (syscall 203) ────────────────────────────
            203 => {
                // accept(sockfd, addr, addrlen)
                let sockfd = x0 as i32;
                let _addr_ptr = x1 as usize;
                let _addrlen_ptr = x2 as usize;

                let fs = match vfs.as_mut() {
                    Some(fs) => fs,
                    None => return SyscallResult::Continue(-9),
                };
                let entry = match fs.fd_table.get(&sockfd).cloned() {
                    Some(e) => e,
                    None => return SyscallResult::Continue(-9),
                };
                use crate::vfs::FdKind;
                use crate::vfs::HostSocketState;
                if let FdKind::HostSocket(socket) = entry.kind {
                    let mut state = socket.lock().unwrap();
                    if let HostSocketState::Listener(listener) = &mut *state {
                        match listener.accept() {
                            Ok((stream, _addr)) => {
                                let new_fd = fs.insert_host_socket(
                                    HostSocketState::Stream(stream),
                                    0,
                                    "<accepted>",
                                );
                                SyscallResult::Continue(if new_fd >= 0 { new_fd as i64 } else { -24 })
                            }
                            Err(_) => SyscallResult::Continue(-11) // -EAGAIN
                        }
                    } else {
                        SyscallResult::Continue(-22) // -EINVAL
                    }
                } else {
                    SyscallResult::Continue(-9)
                }
            }

            // ── Getsockopt (syscall 205) ───────────────────────
            205 => {
                // getsockopt(sockfd, level, optname, optval, optlen)
                let sockfd = x0 as i32;
                let _level = x1 as i32;
                let _optname = x2 as i32;
                let _optval_ptr = x3 as usize;
                // For now, just return success with default options
                // Could be extended to return actual socket options
                SyscallResult::Continue(0)
            }

            // ── Setsockopt (syscall 206) ───────────────────────
            206 => {
                // setsockopt(sockfd, level, optname, optval, optlen)
                let _sockfd = x0 as i32;
                let _level = x1 as i32;
                let _optname = x2 as i32;
                let _optval_ptr = x3 as usize;
                // For now, just return success
                // Could be extended to set actual socket options
                SyscallResult::Continue(0)
            }

            // ── Getsockopt (syscall 209) ───────────────────────
            209 => {
                // getsockopt(sockfd, level, optname, optval, optlen)
                let _sockfd = x0 as i32;
                let _level = x1 as i32;
                let _optname = x2 as i32;
                let optval_addr = x3 as usize;
                let optlen_addr = self.get_register(4) as usize;
                // Return default options
                if optval_addr > 0 {
                    let mut optval = [0u8; 4];
                    self.write_memory(optval_addr, &optval);
                }
                if optlen_addr > 0 {
                    self.write_memory(optlen_addr, &4u32.to_le_bytes());
                }
                SyscallResult::Continue(0)
            }

            // ── Capget (syscall 90) ───────────────────────────
            90 => {
                // capget(header, data)
                let _header = x0 as usize;
                let _data = x1 as usize;
                // Return empty capabilities (no privs)
                SyscallResult::Continue(0)
            }

            // ── Capset (syscall 91) ───────────────────────────
            91 => {
                // capset(header, data)
                let _header = x0 as usize;
                let _data = x1 as usize;
                // Just return success
                SyscallResult::Continue(0)
            }

            // ── Pread64 ─────────────────────────────────────
            67 => {
                if let Some(ref mut vfs) = vfs {
                    let fd = x0 as i32;
                    if let Some(entry) = vfs.fd_table.get(&fd) {
                        let saved = entry.offset;
                        let path = entry.path.clone();
                        if let Some(data) = vfs.files.get(&path) {
                            let pos = x3 as usize;
                            let count = x2 as usize;
                            let end = (pos + count).min(data.len());
                            let chunk = if pos < data.len() {
                                data[pos..end].to_vec()
                            } else {
                                vec![]
                            };
                            self.write_memory(x1 as usize, &chunk);
                            SyscallResult::Continue(chunk.len() as i64)
                        } else {
                            SyscallResult::Continue(-9)
                        }
                    } else {
                        SyscallResult::Continue(-9)
                    }
                } else {
                    SyscallResult::Continue(-9)
                }
            }

            // ── Pwrite64 ────────────────────────────────────
            68 => {
                if let Some(ref mut vfs) = vfs {
                    let fd = x0 as i32;
                    let data = self.read_memory(x1 as usize, x2 as usize);
                    if let Some(entry) = vfs.fd_table.get_mut(&fd) {
                        let saved = entry.offset;
                        entry.offset = x3 as usize;
                        let written = vfs.write(fd, &data);
                        if let Some(entry) = vfs.fd_table.get_mut(&fd) {
                            entry.offset = saved;
                        }
                        SyscallResult::Continue(written as i64)
                    } else {
                        SyscallResult::Continue(-9)
                    }
                } else {
                    SyscallResult::Continue(-9)
                }
            }

            // ── Fchmod / Fchmodat / Fchown / Fchownat ───────
            52 | 53 | 55 | 54 => SyscallResult::Continue(0),

            // ── Utimensat ────────────────────────────────────
            88 => {
                if x1 != 0 {
                    let path = self.read_string(x1 as u64, 256);
                    if let Some(ref vfs) = vfs {
                        let resolved = vfs.resolve_path(&path);
                        if vfs.files.contains_key(&resolved) || vfs.directories.contains(&resolved)
                        {
                            SyscallResult::Continue(0)
                        } else {
                            SyscallResult::Continue(-2) // -ENOENT
                        }
                    } else {
                        SyscallResult::Continue(0)
                    }
                } else {
                    SyscallResult::Continue(0)
                }
            }

            // ── Statfs ───────────────────────────────────────
            43 => {
                let buf_addr = x1 as usize;
                let mut statfs = [0u8; 120];
                // f_type
                statfs[0..8].copy_from_slice(&0x4E435055u64.to_le_bytes());
                // f_bsize
                statfs[8..16].copy_from_slice(&4096u64.to_le_bytes());
                // f_blocks
                statfs[16..24].copy_from_slice(&1048576u64.to_le_bytes());
                // f_bfree, f_bavail
                statfs[24..32].copy_from_slice(&524288u64.to_le_bytes());
                statfs[32..40].copy_from_slice(&524288u64.to_le_bytes());
                // f_files
                statfs[40..48].copy_from_slice(&65536u64.to_le_bytes());
                // f_ffree
                statfs[48..56].copy_from_slice(&32768u64.to_le_bytes());
                // f_namelen
                statfs[64..72].copy_from_slice(&255u64.to_le_bytes());
                // f_frsize
                statfs[72..80].copy_from_slice(&4096u64.to_le_bytes());
                self.write_memory(buf_addr, &statfs);
                SyscallResult::Continue(0)
            }

            // ── Sysinfo ──────────────────────────────────────
            179 => {
                let buf_addr = x0 as usize;
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default();
                let uptime = (now.as_secs() % 86400) as i64;
                let mut sysinfo = [0u8; 112];
                sysinfo[0..8].copy_from_slice(&uptime.to_le_bytes());
                // totalram at offset 32
                sysinfo[32..40].copy_from_slice(&(256u64 * 1024 * 1024).to_le_bytes());
                // freeram at offset 40
                sysinfo[40..48].copy_from_slice(&(128u64 * 1024 * 1024).to_le_bytes());
                // procs at offset 72 (u16)
                sysinfo[72..74].copy_from_slice(&1u16.to_le_bytes());
                // mem_unit at offset 104 (u32)
                sysinfo[104..108].copy_from_slice(&1u32.to_le_bytes());
                self.write_memory(buf_addr, &sysinfo);
                SyscallResult::Continue(0)
            }

            // ── Nanosleep / clock_nanosleep ──────────────────
            101 | 115 => SyscallResult::Continue(0),

            // ── Sched_getaffinity / timer_settime / misc ─────
            123 | 110 | 122 | 158 => SyscallResult::Continue(0),

            // ── Getpid / getppid / gettid ────────────────────
            172 => SyscallResult::Continue(1), // getpid
            173 => SyscallResult::Continue(0), // getppid
            178 => SyscallResult::Continue(1), // gettid

            // ── Getuid/geteuid/getgid/getegid ────────────────
            174 | 175 | 176 | 177 => SyscallResult::Continue(0), // root

            // ── Setuid/Setgid ────────────────────────────────
            146 => SyscallResult::Continue(0), // setuid
            147 => SyscallResult::Continue(0), // setgid
            149 => SyscallResult::Continue(0), // setreuid
            150 => SyscallResult::Continue(0), // setregid
            120 => SyscallResult::Continue(0), // setresuid
            121 => SyscallResult::Continue(0), // setresgid

            // ── Musl init stubs ──────────────────────────────
            96 => SyscallResult::Continue(1), // set_tid_address → TID=1
            99 => SyscallResult::Continue(0), // set_robust_list
            134 | 135 | 139 => SyscallResult::Continue(0), // rt_sigaction, rt_sigprocmask, rt_sigreturn

            // ── Ppoll / clone ────────────────────────────────
            73 => SyscallResult::Continue(0),    // ppoll
            220 => SyscallResult::Continue(-38), // clone → ENOSYS

            // ── Sched_getaffinity ────────────────────────────
            // Already covered above

            // ── Shutdown (210) ────────────────────────────────
            // shutdown(sockfd, how) - shut down socket send/receive
            210 => {
                // Return 0 (success) - we don't actually track socket state
                SyscallResult::Continue(0)
            }

            // ── Sendmsg (211) ────────────────────────────────
            // sendmsg(sockfd, msg, flags)
            211 => {
                // Return bytes sent - we don't actually send
                let _sockfd = x0 as i32;
                let _msg = x1 as usize;
                let _flags = x2 as i32;
                // For now, return success with arbitrary positive value
                SyscallResult::Continue(0)
            }

            // ── Recvmsg (212) ────────────────────────────────
            // recvmsg(sockfd, msg, flags)
            212 => {
                // Return no data
                let _sockfd = x0 as i32;
                let _msg = x1 as usize;
                let _flags = x2 as i32;
                SyscallResult::Continue(0)
            }

            // ── Getpriority (140) ─────────────────────────────
            // getpriority(which, who) - get process scheduling priority
            140 => {
                // Return default priority (20 = nice default)
                SyscallResult::Continue(20)
            }

            // ── Setpriority (141) ─────────────────────────────
            // setpriority(which, who, prio) - set process scheduling priority
            141 => {
                // Return success
                SyscallResult::Continue(0)
            }

            // ── Nanosleep (101) ───────────────────────────────
            // nanosleep(req, rem) - high-resolution sleep
            101 => {
                let req_addr = x0 as usize;
                let _rem_addr = x1 as usize;
                if req_addr > 0 {
                    let req_data = self.read_memory(req_addr, 16);
                    if req_data.len() >= 16 {
                        let secs = i64::from_le_bytes(req_data[0..8].try_into().unwrap_or([0; 8]));
                        let nsecs = i64::from_le_bytes(req_data[8..16].try_into().unwrap_or([0; 8]));
                        let total_nanos = secs * 1_000_000_000 + nsecs;
                        // For nanosleep, we just return success immediately
                        // In a real implementation, we'd actually sleep
                        if total_nanos < 1_000_000 {  // Less than 1ms - just return
                            SyscallResult::Continue(0)
                        } else {
                            // For longer sleeps, return success anyway
                            // Real implementation would block
                            SyscallResult::Continue(0)
                        }
                    } else {
                        SyscallResult::Continue(-14) // -EFAULT
                    }
                } else {
                    SyscallResult::Continue(-14) // -EFAULT
                }
            }

            // ── Gettimeofday (169) ────────────────────────────
            // gettimeofday(tv, tz)
            169 => {
                let tv_addr = x0 as usize;
                let tz_addr = x1 as usize;
                if tv_addr > 0 {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default();
                    let mut tv = [0u8; 16];
                    // tv_sec at offset 0
                    tv[0..8].copy_from_slice(&now.as_secs().to_le_bytes());
                    // tv_usec at offset 8 (we have nanos, so convert)
                    tv[8..16].copy_from_slice(&(now.subsec_nanos() / 1000).to_le_bytes());
                    self.write_memory(tv_addr, &tv);
                }
                if tz_addr > 0 {
                    // timezone struct: 4 bytes for minutes west of UTC, 4 bytes for DST
                    let mut tz_data = [0u8; 8];
                    tz_data[0..4].copy_from_slice(&0i32.to_le_bytes()); // tz_minuteswest = 0 (UTC)
                    tz_data[4..8].copy_from_slice(&0i32.to_le_bytes()); // tz_dsttime = 0 (none)
                    self.write_memory(tz_addr, &tz_data);
                }
                SyscallResult::Continue(0)
            }

            // ── Geteuid (175) / Geteuid (207) ─────────────────
            // For programs that check real vs effective UID
            175 => SyscallResult::Continue(0), // geteuid
            207 => SyscallResult::Continue(0), // geteuid (32-bit)

            // ── Getegid (176) / Getegid (208) ─────────────────
            176 => SyscallResult::Continue(0), // getegid
            208 => SyscallResult::Continue(0), // getegid (32-bit)

            // ── Getuid (174) / Getuid (102) ────────────────────
            174 => SyscallResult::Continue(0), // getuid
            102 => SyscallResult::Continue(0), // getuid32

            // ── Getgid (173) / Getgid (104) ────────────────────
            173 => SyscallResult::Continue(0), // getgid (wait4 was here, but now handled)
            104 => SyscallResult::Continue(0), // getgid32

            // ── Getpid (172) ────────────────────────────────────
            172 => SyscallResult::Continue(1), // getpid returns 1 (init)

            // ── Getppid (163) ───────────────────────────────────
            163 => SyscallResult::Continue(0), // getppid - parent is shell

            // ── Getpgid (132) ───────────────────────────────────
            132 => SyscallResult::Continue(1), // getpgid - process group

            // ── Setpgid (123) ───────────────────────────────────
            123 => SyscallResult::Continue(0), // setpgid

            // ── Getsid (149) ────────────────────────────────────
            149 => SyscallResult::Continue(1), // getsid - session ID

            // ── Setsid (148) ────────────────────────────────────
            148 => SyscallResult::Continue(0), // setsid

            // ── Setreuid (136) ──────────────────────────────────
            136 => SyscallResult::Continue(0), // setreuid

            // ── Setregid (135) ──────────────────────────────────
            135 => SyscallResult::Continue(0), // setregid

            // ── Getgroups (115) ─────────────────────────────────
            115 => {
                // Return 1 group (root group)
                let size = x1 as usize;
                if size > 0 {
                    self.write_memory(x0 as usize, &0u32.to_le_bytes());
                }
                SyscallResult::Continue(1) // Return 1 group
            }

            // ── Setgroups (113) ─────────────────────────────────
            113 => SyscallResult::Continue(0), // setgroups

            // ── Uname (160) - Already handled above ─────────────

            // ── Getrlimit (163) ─────────────────────────────────
            163 => {
                let resource = x0 as u32;
                let rlim_addr = x1 as usize;
                // RLIM_INFINITY = (u64)-1
                let mut rlim = [0u8; 16];
                rlim[0..8].copy_from_slice(&u64::MAX.to_le_bytes()); // rlim_cur
                rlim[8..16].copy_from_slice(&u64::MAX.to_le_bytes()); // rlim_max
                self.write_memory(rlim_addr, &rlim);
                SyscallResult::Continue(0)
            }

            // ── Getrusage (165) ─────────────────────────────────
            165 => {
                let who = x0 as i32;
                let usage_addr = x1 as usize;
                // Return mostly zeros with some basic values
                let mut usage = [0u8; 128];
                // utime and stime at offset 0 and 16
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default();
                let secs = now.as_secs() as i64;
                usage[0..8].copy_from_slice(&secs.to_le_bytes());
                usage[16..24].copy_from_slice(&secs.to_le_bytes());
                self.write_memory(usage_addr, &usage);
                SyscallResult::Continue(0)
            }

            // ── Utime (132) ────────────────────────────────────
            132 => SyscallResult::Continue(0), // utime

            // ── Mknod (133) ─────────────────────────────────────
            133 => SyscallResult::Continue(0), // mknod

            // ── Umount (166) ────────────────────────────────────
            166 => SyscallResult::Continue(0), // umount2

            // ── Mount (162) ─────────────────────────────────────
            162 => SyscallResult::Continue(-1), // mount - return ENOENT

            // ── Umask (151) ─────────────────────────────────────
            151 => SyscallResult::Continue(0o022), // umask - return old mask

            // ── Chroot (161) ────────────────────────────────────
            161 => SyscallResult::Continue(0), // chroot

            // ── Modify_ldt (154) ────────────────────────────────
            154 => SyscallResult::Continue(0), // modify_ldt

            // ── Pivot_root (155) ────────────────────────────────
            155 => SyscallResult::Continue(0), // pivot_root

            // ── Prctl (157) - Already handled above ─────────────

            // ── Arch_prctl (165) ─────────────────────────────────
            165 => {
                // arch_prctl - for TLS setup
                let code = x0 as i32;
                let addr = x1 as usize;
                if code == 0x1002 || code == 0x1003 { // ARCH_GET_FS or ARCH_GET_GS
                    // Just return success
                    SyscallResult::Continue(0)
                } else {
                    SyscallResult::Continue(0)
                }
            }

            // ── Flock (73) ─────────────────────────────────────
            // flock(fd, operation) - file locking
            73 => {
                // Just return success - we don't support actual file locking
                // but many programs call this just to check fd validity
                SyscallResult::Continue(0)
            }

            // ── Fsync (74) ─────────────────────────────────────
            // fsync(fd) - sync file to disk
            74 => {
                // Just return success - we're in-memory anyway
                SyscallResult::Continue(0)
            }

            // ── Fdatasync (75) ─────────────────────────────────
            // fdatasync(fd) - sync data to disk
            75 => {
                SyscallResult::Continue(0)
            }

            // ── Truncate (76) ───────────────────────────────────
            // truncate(path, length)
            76 => {
                let path_addr = x0 as usize;
                let _length = x1 as i64;
                if path_addr > 0 {
                    // Just return success for truncate
                    SyscallResult::Continue(0)
                } else {
                    SyscallResult::Continue(-14) // -EFAULT
                }
            }

            // ── Ftruncate (77) ─────────────────────────────────
            // ftruncate(fd, length)
            77 => {
                // Just return success
                SyscallResult::Continue(0)
            }

            // ── Getdents64 (61) - Already handled above ────────

            // ── Getcwd (17) - Already handled above ────────────

            // ── Readlinkat (78) - Already handled above ────────

            // ── Mmap (222) - Already handled above ─────────────

            // ── Munmap (215) ───────────────────────────────────
            // munmap(addr, length)
            215 => {
                // Just return success
                SyscallResult::Continue(0)
            }

            // ── Mprotect (226) ─────────────────────────────────
            // mprotect(addr, len, prot)
            226 => {
                // Just return success - we're already allowing all access
                SyscallResult::Continue(0)
            }

            // ── Mremap (216) ───────────────────────────────────
            // mremap(old_address, old_size, new_size, flags)
            216 => {
                // Return the old address as the new address
                SyscallResult::Continue(x0)
            }

            // ── Msync (227) ────────────────────────────────────
            // msync(addr, length, flags)
            227 => {
                SyscallResult::Continue(0)
            }

            // ── Madvise (233) ───────────────────────────────────
            // madvise(addr, length, advice)
            233 => {
                SyscallResult::Continue(0)
            }

            // ── Shmget (194) ───────────────────────────────────
            // shmget(key, size, shmflg)
            194 => {
                // Return -1 (no shared memory)
                SyscallResult::Continue(-1)
            }

            // ── Shmat (196) ────────────────────────────────────
            // shmat(shmid, shmaddr, shmflg)
            196 => {
                SyscallResult::Continue(0)
            }

            // ── Shmctl (195) ───────────────────────────────────
            // shmctl(shmid, cmd, buf)
            195 => {
                SyscallResult::Continue(0)
            }

            // ── Dup (23) ────────────────────────────────────────
            // dup(oldfd) - duplicate file descriptor
            23 => {
                if let Some(ref mut vfs) = vfs {
                    let oldfd = x0 as i32;
                    // Use dup2 with -1 to get a new fd
                    let r = vfs.dup2(oldfd, -1);
                    SyscallResult::Continue(r as i64)
                } else {
                    SyscallResult::Continue(-9)
                }
            }

            // ── Dup2 (63) - Already handled above ───────────────

            // ── Pipe2 (59) - Already handled above ──────────────

            // ── Mkfifo (245) ───────────────────────────────────
            // mkfifo(pathname, mode)
            245 => {
                // Just return success
                SyscallResult::Continue(0)
            }

            // ── Utimes (88) - Already handled above ───────────

            // ── Futimesat (269) ────────────────────────────────
            269 => {
                // futimesat - same as utimensat for our purposes
                SyscallResult::Continue(0)
            }

            // ── Faccessat (48) - Already handled above ─────────

            // ── Chdir (49) - Already handled above ─────────────

            // ── Fchownat (55) ───────────────────────────────────
            55 => SyscallResult::Continue(0), // fchownat

            // ── Fstatat (79) - Already handled above ──────────

            // ── Fchmodat (55) - Same as fchmod ────────────────
            55 => SyscallResult::Continue(0),

            // ── Readlink (78) ───────────────────────────────────
            78 => {
                let path_addr = x0 as usize;
                let buf_addr = x1 as usize;
                let bufsiz = x2 as usize;
                if path_addr > 0 && buf_addr > 0 {
                    // Read symlink target (we don't track actual symlink targets well)
                    // Just return empty for now
                    SyscallResult::Continue(0)
                } else {
                    SyscallResult::Continue(-14)
                }
            }

            // ── Symlinkat (36) - Already handled above ─────────

            // ── Linkat (37) - Already handled above ─────────────

            // ── Unlinkat (35) - Already handled above ─────────

            // ── Renameat2 (276) ────────────────────────────────
            276 => {
                let olddirfd = x0 as i32;
                let old_path = self.read_string(x1 as u64, 256);
                let newdirfd = x2 as i32;
                let new_path = self.read_string(x3 as u64, 256);
                if let Some(ref mut vfs) = vfs {
                    let r = vfs.rename(&old_path, &new_path);
                    SyscallResult::Continue(r as i64)
                } else {
                    SyscallResult::Continue(-2)
                }
            }

            // ── Creat (85) ─────────────────────────────────────
            // creat(path, mode) - create and open a file
            // O_WRONLY | O_CREAT | O_TRUNC = 1 | 64 | 512 = 577
            85 => {
                let path = self.read_string(x0 as u64, 256);
                let _mode = x1 as i32;
                if let Some(ref mut vfs) = vfs {
                    let fd = vfs.open(&path, 577);
                    SyscallResult::Continue(fd as i64)
                } else {
                    SyscallResult::Continue(-2)
                }
            }

            // ── Link (85 + 1) ───────────────────────────────────
            // Note: link is syscall 86 but often confused
            86 => {
                let old_path = self.read_string(x0 as u64, 256);
                let new_path = self.read_string(x1 as u64, 256);
                if let Some(ref mut vfs) = vfs {
                    let resolved_old = vfs.resolve_path(&old_path);
                    if let Some(content) = vfs.files.get(&resolved_old).cloned() {
                        let resolved_new = vfs.resolve_path(&new_path);
                        vfs.files.insert(resolved_new, content);
                        SyscallResult::Continue(0)
                    } else {
                        SyscallResult::Continue(-2)
                    }
                } else {
                    SyscallResult::Continue(-2)
                }
            }

            // ── Unlink (87) ────────────────────────────────────
            87 => {
                let path = self.read_string(x0 as u64, 256);
                if let Some(ref mut vfs) = vfs {
                    let r = vfs.unlink(&path);
                    SyscallResult::Continue(r as i64)
                } else {
                    SyscallResult::Continue(-2)
                }
            }

            // ── Rmdir (84) ─────────────────────────────────────
            84 => {
                let path = self.read_string(x0 as u64, 256);
                if let Some(ref mut vfs) = vfs {
                    let r = vfs.rmdir(&path);
                    SyscallResult::Continue(r as i64)
                } else {
                    SyscallResult::Continue(-2)
                }
            }

            // ── Mkdir (83) ──────────────────────────────────────
            83 => {
                let path = self.read_string(x0 as u64, 256);
                let mode = x1 as i32;
                if let Some(ref mut vfs) = vfs {
                    let r = vfs.mkdir(&path);
                    SyscallResult::Continue(r as i64)
                } else {
                    SyscallResult::Continue(-2)
                }
            }

            // ── Access (48) ────────────────────────────────────
            // Note: access is handled above as faccessat

            // ── Readlinkat (78) - Already handled above ─────────

            // ── Pause (106) ────────────────────────────────────
            // pause() - sleep until signal
            106 => {
                // Return -EINTR immediately
                SyscallResult::Continue(-4)
            }

            // ── Alarm (107) ────────────────────────────────────
            // alarm(seconds) - set alarm clock
            107 => {
                SyscallResult::Continue(0)
            }

            // ── Getitimer (102) ─────────────────────────────────
            // getitimer(which)
            102 => {
                let which = x0 as i32;
                let addr = x1 as usize;
                if addr > 0 {
                    // Return zeros (no timer)
                    let mut itv = [0u8; 32];
                    self.write_memory(addr, &itv);
                }
                SyscallResult::Continue(0)
            }

            // ── Setitimer (103) ────────────────────────────────
            // setitimer(which, new, old)
            103 => {
                SyscallResult::Continue(0)
            }

            // ── Getpid (172) - Already handled above ────────────

            // ── Socket (198) - Already handled above ────────────

            // ── Connect (203) - Already handled above ───────────

            // ── Accept (202) - Already handled above ───────────

            // ── Listen (201) - Already handled above ───────────

            // ── Bind (200) - Already handled above ─────────────

            // ── Socketpair (232) ────────────────────────────────
            // socketpair(domain, type, protocol, sv)
            232 => {
                let sv_addr = x3 as usize;
                if sv_addr > 0 {
                    // Create a pair of connected sockets (as pipes)
                    if let Some(ref mut vfs) = vfs {
                        let (rfd, wfd) = vfs.create_pipe();
                        if rfd >= 0 && wfd >= 0 {
                            let mut fds = [0i32; 2];
                            fds[0] = rfd;
                            fds[1] = wfd;
                            let fds_bytes = unsafe {
                                std::slice::from_raw_parts(fds.as_ptr() as *const u8, 8)
                            };
                            self.write_memory(sv_addr, fds_bytes);
                            SyscallResult::Continue(0)
                        } else {
                            SyscallResult::Continue(-1)
                        }
                    } else {
                        SyscallResult::Continue(-1)
                    }
                } else {
                    SyscallResult::Continue(-14)
                }
            }

            // ── Sendto (205) - Already handled above ───────────

            // ── Recvfrom (206) - Already handled above ─────────

            // ── Setsockopt (206) - Already handled above ───────

            // ── Getsockopt (205) - Already handled above ───────

            // ── Shutdown (210) - Already handled above ─────────

            // ── Sendmsg (211) - Already handled above ─────────

            // ── Recvmsg (212) - Already handled above ─────────

            // ── Brk (214) - Already handled above ──────────────

            // ── Munmap (215) - Already handled above ────────────

            // ── Mremap (216) - Already handled above ───────────

            // ── Madvise (233) - Already handled above ───────────

            // ── Mprotect (226) - Already handled above ──────────

            // ── Rt_sigaction (134) ──────────────────────────────
            // Already handled above as 134 | 135 | 139

            // ── Rt_sigprocmask (135) ───────────────────────────
            // Already handled above

            // ── Rt_sigpending (136) ────────────────────────────
            136 => {
                // No pending signals
                SyscallResult::Continue(0)
            }

            // ── Rt_sigtimedwait (137) ──────────────────────────
            137 => {
                // No signals to wait for
                SyscallResult::Continue(-4) // -EAGAIN
            }

            // ── Rt_sigqueueinfo (138) ──────────────────────────
            138 => SyscallResult::Continue(0),

            // ── Rt_sigsuspend (139) ────────────────────────────
            // Already handled

            // ── Sigaltstack (131) ──────────────────────────────
            // sigaltstack(ss, oss)
            131 => {
                let ss_addr = x0 as usize;
                let oss_addr = x1 as usize;
                if oss_addr > 0 {
                    // Return default signal stack (disabled)
                    let mut ss = [0u8; 24];
                    // ss_flags at offset 0 = SS_DISABLE (1)
                    ss[0..4].copy_from_slice(&1u32.to_le_bytes());
                    self.write_memory(oss_addr, &ss);
                }
                SyscallResult::Continue(0)
            }

            // ── Ugetrlimit (163) ───────────────────────────────
            // Same as getrlimit - handled above

            // ── Vfork (221) ────────────────────────────────────
            // vfork - simplified version that just returns success
            // In handle_syscall we don't have access to proc_mgr, so we return
            // the same value as fork would
            221 => {
                // Return 2 as child PID (simulating fork behavior)
                SyscallResult::Continue(2)
            }

            // ── Ulimit syscalls ────────────────────────────────
            // getrlimit, setrlimit handled above

            // ── Syslog (103) ────────────────────────────────────
            // syslog(type, buf, len)
            103 => {
                let _typ = x0 as i32;
                let buf_addr = x1 as usize;
                let _len = x2 as usize;
                // Just return empty
                SyscallResult::Continue(0)
            }

            // ── Gethostname (170) ───────────────────────────────
            170 => {
                let addr = x0 as usize;
                let len = x1 as usize;
                if addr > 0 && len > 0 {
                    let hostname = b"ncpu-gpu\0";
                    let write_len = hostname.len().min(len);
                    self.write_memory(addr, &hostname[..write_len]);
                    SyscallResult::Continue(0)
                } else {
                    SyscallResult::Continue(-14)
                }
            }

            // ── Sethostname (171) ───────────────────────────────
            171 => {
                // Just return success
                SyscallResult::Continue(0)
            }

            // ── Setdomainname (171) ────────────────────────────
            172 => {
                SyscallResult::Continue(0)
            }

            // ── Getrdomain (170) ───────────────────────────────
            // getrdomain - not commonly used

            // ── Iopl (106) ──────────────────────────────────────
            106 => {
                // I/O privilege level - not applicable
                SyscallResult::Continue(-38) // ENOSYS
            }

            // ── Ioperm (108) ───────────────────────────────────
            108 => {
                // I/O port permissions - not applicable
                SyscallResult::Continue(-38)
            }

            // ── Init_module (175) ──────────────────────────────
            175 => SyscallResult::Continue(-38),

            // ── Delete_module (176) ────────────────────────────
            176 => SyscallResult::Continue(-38),

            // ── Quote (232) ────────────────────────────────────
            // Not a real syscall, just a placeholder

            // ── Remap_file_pages (234) ─────────────────────────
            234 => SyscallResult::Continue(0),

            // ── Timer_create (222) ──────────────────────────────
            222 => {
                // Return -1 (no timer support)
                SyscallResult::Continue(-1)
            }

            // ── Timer_settime (223) ────────────────────────────
            223 => SyscallResult::Continue(0),

            // ── Timer_gettime (224) ─────────────────────────────
            224 => {
                let addr = x1 as usize;
                if addr > 0 {
                    let mut itimerspec = [0u8; 32];
                    self.write_memory(addr, &itimerspec);
                }
                SyscallResult::Continue(0)
            }

            // ── Timer_getoverrun (225) ─────────────────────────
            225 => SyscallResult::Continue(0),

            // ── Timer_delete (227) ─────────────────────────────
            227 => SyscallResult::Continue(0),

            // ── Clock_settime (112) ────────────────────────────
            112 => SyscallResult::Continue(0),

            // ── Clock_gettime (113) - Already handled above ─────

            // ── Clock_getres (114) - Already handled above ─────

            // ── Clock_nanosleep (115) ──────────────────────────
            115 => {
                // Similar to nanosleep but using clock
                // Just return success for now
                SyscallResult::Continue(0)
            }

            // ── Exit_group (94) - Handled at top ───────────────

            // ── Epoll_create (213) ────────────────────────────
            213 => {
                // Return -1 (no epoll support)
                SyscallResult::Continue(-1)
            }

            // ── Epoll_ctl (214) ────────────────────────────────
            214 => SyscallResult::Continue(0),

            // ── Epoll_wait (215) ───────────────────────────────
            215 => {
                // No events - return timeout
                SyscallResult::Continue(0)
            }

            // ── Remap (216) - Already handled ──────────────────

            // ── Set_tid_address (96) - Handled above ──────────

            // ── Restart_syscall (128) ──────────────────────────
            128 => {
                // Restart interrupted syscall
                SyscallResult::Continue(-4)
            }

            // ── Semtimedop (220) ───────────────────────────────
            220 => {
                // Semaphore operations - return error
                SyscallResult::Continue(-1)
            }

            // ── Semop (219) ───────────────────────────────────
            219 => SyscallResult::Continue(0),

            // ── Semget (217) ───────────────────────────────────
            217 => SyscallResult::Continue(-1),

            // ── Semctl (218) ───────────────────────────────────
            218 => SyscallResult::Continue(0),

            // ── Shmdt (197) ───────────────────────────────────
            197 => SyscallResult::Continue(0),

            // ── Msgget (218) ───────────────────────────────────
            218 => SyscallResult::Continue(-1),

            // ── Msgsnd (219) ───────────────────────────────────
            219 => SyscallResult::Continue(0),

            // ── Msgrcv (220) ───────────────────────────────────
            220 => SyscallResult::Continue(0),

            // ── Msgctl (221) ───────────────────────────────────
            221 => SyscallResult::Continue(0),

            // ── Perf_event_open (241) ──────────────────────────
            // Performance monitoring event open
            241 => {
                // Return -1 (no performance monitoring support)
                SyscallResult::Continue(-1)
            }

            // ── Bpf (280) ──────────────────────────────────────
            // Berkeley Packet Filter - now used for many things
            280 => {
                let cmd = x0 as i32;
                let attr_addr = x1 as usize;
                let size = x2 as usize;
                // BPF commands - return error for all
                // Most programs check for -1 and fall back
                SyscallResult::Continue(-1)
            }

            // ── Fcntl (25) - Already handled above ────────────

            // ── Fsync (74) - Already handled above ─────────────

            // ── Fdatasync (75) - Already handled above ─────────

            // ── Ftruncate (77) - Already handled above ─────────

            // ── Getcwd (17) - Already handled above ────────────

            // ── Flush Framebuffer ─────────────────────────────
            // SYS_FLUSH_FB = 313 - sends framebuffer to display
            313 => {
                let width = x0 as u32;
                let height = x1 as u32;
                let fb_addr = x2 as usize;
                if width > 0 && height > 0 && fb_addr > 0 {
                    let fb_size = (width as usize) * (height as usize) * 4; // RGBA
                    let fb_data = self.read_memory(fb_addr, fb_size);
                    // Call the Python framebuffer callback if set
                    crate::call_framebuffer_callback(width, height, &fb_data);
                    SyscallResult::Continue(0)
                } else {
                    SyscallResult::Continue(-1)
                }
            }

            // ── Unknown ──────────────────────────────────────
            _ => {
                eprintln!("[launcher] unhandled syscall {}", num);
                SyscallResult::Continue(-38) // -ENOSYS
            }
        }
    }
}

// ── Microkernel Syscall Handler ─────────────────────────────────────────────

/// Handle syscalls using the GPU microkernel instead of Linux VFS
pub fn handle_microkernel_syscall(
    kernel: &mut crate::microkernel::GpuMicrokernel,
    num: i64,
    x0: i64,
    x1: i64,
    x2: i64,
    x3: i64,
    stdout_buf: &mut Vec<u8>,
    stderr_buf: &mut Vec<u8>,
) -> SyscallResult {
    use crate::microkernel::ProcessState;
    use crate::microkernel::FdKind;

    match num {
        // Exit
        93 | 94 => {
            // exit / exit_group
            kernel.exit(x0 as i32);
            SyscallResult::Exit(x0 as i32)
        }

        // Write
        64 => {
            let fd = x0 as i32;
            let buf_addr = x1 as u64;
            let count = x2 as usize;

            if count == 0 {
                return SyscallResult::Continue(0);
            }

            // Get current process
            if let Some(pid) = kernel.current_pid {
                if let Some(proc) = kernel.processes.get_mut(&pid) {
                    match proc.get_fd(fd) {
                        Some(entry) => {
                            match &entry.kind {
                                FdKind::Console => {
                                    // Write to console
                                    // For now, just count as written
                                    // In real impl, would buffer output
                                    SyscallResult::Continue(count as i64)
                                }
                                _ => {
                                    // Other fds - just return success
                                    SyscallResult::Continue(count as i64)
                                }
                            }
                        }
                        None => SyscallResult::Continue(-9), // EBADF
                    }
                } else {
                    SyscallResult::Continue(-9)
                }
            } else {
                SyscallResult::Continue(-9)
            }
        }

        // Read
        63 => {
            let fd = x0 as i32;
            let buf_addr = x1 as u64;
            let count = x2 as usize;

            if count == 0 {
                return SyscallResult::Continue(0);
            }

            if let Some(pid) = kernel.current_pid {
                if let Some(proc) = kernel.processes.get_mut(&pid) {
                    match proc.get_fd(fd) {
                        Some(entry) => {
                            match &entry.kind {
                                FdKind::Console => {
                                    // Console read - return EOF for now
                                    SyscallResult::Continue(0)
                                }
                                _ => SyscallResult::Continue(0),
                            }
                        }
                        None => SyscallResult::Continue(-9),
                    }
                } else {
                    SyscallResult::Continue(-9)
                }
            } else {
                SyscallResult::Continue(-9)
            }
        }

        // Fork
        220 => {
            if let Some((parent_pid, child_pid)) = kernel.fork() {
                // Parent returns child PID
                SyscallResult::Continue(child_pid as i64)
            } else {
                SyscallResult::Continue(-1)
            }
        }

        // Execve - handled by loading new program
        221 => {
            // Would need to load new program into memory
            // For now, just return success
            SyscallResult::Continue(0)
        }

        // Getpid
        172 => {
            if let Some(pid) = kernel.current_pid {
                SyscallResult::Continue(pid as i64)
            } else {
                SyscallResult::Continue(1)
            }
        }

        // Getppid
        173 => {
            if let Some(pid) = kernel.current_pid {
                if let Some(proc) = kernel.processes.get(&pid) {
                    return SyscallResult::Continue(proc.ppid as i64);
                }
            }
            SyscallResult::Continue(0)
        }

        // Wait4
        260 => {
            if let Some(code) = kernel.wait() {
                SyscallResult::Continue(code as i64)
            } else {
                // Would block - return ECHILD
                SyscallResult::Continue(-10)
            }
        }

        // Open (simplified - just creates a file)
        56 => {
            let path = read_cstring(kernel, x1 as u64, 256);
            let flags = x2;

            if let Some(pid) = kernel.current_pid {
                if let Some(proc) = kernel.processes.get_mut(&pid) {
                    // Simple implementation: create file if O_CREAT
                    if flags & 0x40 != 0 { // O_CREAT
                        kernel.vfs.create(&path, b"");
                    }
                    let fd = proc.alloc_fd(FdKind::File(path), flags as u32);
                    match fd {
                        Some(fd) => SyscallResult::Continue(fd as i64),
                        None => SyscallResult::Continue(-24), // EMFILE
                    }
                } else {
                    SyscallResult::Continue(-9)
                }
            } else {
                SyscallResult::Continue(-9)
            }
        }

        // Close
        57 => {
            let fd = x0 as i32;
            if let Some(pid) = kernel.current_pid {
                if let Some(proc) = kernel.processes.get_mut(&pid) {
                    if proc.close_fd(fd) {
                        proc.fd_table[fd as usize] = None;
                        SyscallResult::Continue(0)
                    } else {
                        SyscallResult::Continue(-9)
                    }
                } else {
                    SyscallResult::Continue(-9)
                }
            } else {
                SyscallResult::Continue(-9)
            }
        }

        // Brk
        214 => {
            let new_brk = x0 as u64;
            if let Some(pid) = kernel.current_pid {
                if let Some(proc) = kernel.processes.get_mut(&pid) {
                    if new_brk == 0 {
                        return SyscallResult::Continue(proc.heap_brk as i64);
                    }
                    if new_brk > proc.heap_base {
                        proc.heap_brk = new_brk;
                    }
                    SyscallResult::Continue(proc.heap_brk as i64)
                } else {
                    SyscallResult::Continue(0)
                }
            } else {
                SyscallResult::Continue(0)
            }
        }

        // Uname
        160 => {
            let addr = x0 as usize;
            // Write a minimal uname structure
            let uname = b"ncpu-gpu\0".to_vec();
            // Pad to 65 bytes (UTSNAME_LENGTH)
            let mut uname_buf = uname;
            while uname_buf.len() < 65 {
                uname_buf.push(0);
            }
            // Write sysname
            // This is a simplification - real uname has more fields
            unsafe {
                let ptr = uname_buf.as_ptr() as *mut u8;
                for (i, &byte) in uname_buf.iter().enumerate() {
                    *ptr.add(i) = byte;
                }
            }
            SyscallResult::Continue(0)
        }

        // Getcwd
        17 => {
            let buf_addr = x0 as usize;
            let size = x1 as usize;

            if let Some(pid) = kernel.current_pid {
                if let Some(proc) = kernel.processes.get(&pid) {
                    let cwd = proc.cwd.as_bytes();
                    if cwd.len() < size {
                        let mut buf = cwd.to_vec();
                        buf.push(0);
                        // Would need to actually write to memory
                        SyscallResult::Continue(buf_addr as i64)
                    } else {
                        SyscallResult::Continue(-34) // ERANGE
                    }
                } else {
                    SyscallResult::Continue(-9)
                }
            } else {
                SyscallResult::Continue(-9)
            }
        }

        // Clock gettime
        113 => {
            let clock_id = x0 as i32;
            let timespec_addr = x1 as usize;

            if timespec_addr > 0 {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default();
                let mut ts = [0u8; 16];
                ts[0..8].copy_from_slice(&now.as_secs().to_le_bytes());
                ts[8..16].copy_from_slice(&(now.subsec_nanos() as u64).to_le_bytes());
                // Would need to write to memory
                SyscallResult::Continue(0)
            } else {
                SyscallResult::Continue(-14)
            }
        }

        // Unknown - return ENOSYS
        _ => {
            eprintln!("[microkernel] unhandled syscall {}", num);
            SyscallResult::Continue(-38)
        }
    }
}

/// Helper to read a null-terminated string from process memory
fn read_cstring(_kernel: &crate::microkernel::GpuMicrokernel, _addr: u64, _max_len: usize) -> String {
    // This would need access to process memory
    // Simplified for now
    String::new()
}

enum SyscallResult {
    Continue(i64), // return value to put in x0
    Exit(i32),     // exit with code
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launcher_creation() {
        let launcher = GpuLauncher::new(4 * 1024 * 1024, 10_000_000);
        assert!(launcher.is_ok(), "GpuLauncher should create successfully");
    }

    #[test]
    fn memory_read_write() {
        let launcher = GpuLauncher::new(1024 * 1024, 10_000).unwrap();
        launcher.write_memory(0x1000, b"hello GPU");
        let data = launcher.read_memory(0x1000, 9);
        assert_eq!(data, b"hello GPU");
    }

    #[test]
    fn register_access() {
        let launcher = GpuLauncher::new(1024 * 1024, 10_000).unwrap();
        launcher.set_register(0, 42);
        assert_eq!(launcher.get_register(0), 42);
        launcher.set_register(31, 0xFF000);
        assert_eq!(launcher.get_register(31), 0xFF000);
    }

    #[test]
    fn run_halt_instruction() {
        // HLT #0 = 0xD4400000
        let launcher = GpuLauncher::new(1024 * 1024, 10_000).unwrap();
        // Write HLT at address 0x10000
        launcher.write_memory(0x10000, &0xD4400000u32.to_le_bytes());
        launcher.set_pc(0x10000);
        launcher.set_register(31, 0xF0000); // SP

        let mut vfs = None;
        let result = launcher.run(&mut vfs, 1_000_000, 5.0, true).unwrap();
        assert_eq!(result.stop_reason, "HALT");
        assert!(result.total_cycles <= 10);
    }

    #[test]
    fn run_minimal_program() {
        // MOV X0, #42 ; MOV X8, #93 ; SVC #0
        // The GPU shader handles exit(93) on-chip by signaling HALT.
        let launcher = GpuLauncher::new(4 * 1024 * 1024, 10_000_000).unwrap();

        // MOVZ X0, #42      = 0xD2800540
        // MOVZ X8, #93      = 0xD2800BA8
        // SVC  #0            = 0xD4000001
        launcher.write_memory(0x10000, &0xD2800540u32.to_le_bytes());
        launcher.write_memory(0x10004, &0xD2800BA8u32.to_le_bytes());
        launcher.write_memory(0x10008, &0xD4000001u32.to_le_bytes());
        launcher.set_pc(0x10000);
        launcher.set_register(31, 0xF0000);

        let mut vfs = None;
        let result = launcher.run(&mut vfs, 1_000_000, 5.0, true).unwrap();
        // GPU handles exit syscall on-chip, signals HALT
        assert!(
            result.stop_reason == "HALT" || result.stop_reason == "EXIT",
            "Expected HALT or EXIT, got: {}",
            result.stop_reason
        );
        // x0 should still hold 42
        assert_eq!(launcher.get_register(0), 42);
    }
}
