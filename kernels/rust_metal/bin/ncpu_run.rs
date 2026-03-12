//! ncpu_run — Standalone GPU launcher for ARM64 ELF binaries and boot images.
//!
//! Usage:
//!   ncpu_run --elf <path> [-- arg1 arg2 ...]
//!   ncpu_run <boot_image.bin>

use std::env;
use std::fs;
use std::path::Path;
use std::process;

use ncpu_metal::boot_image::{
    deserialize_region, deserialize_task, validate_image, BootImageHeader, BootRegionDesc,
    BootTaskDesc,
};
use ncpu_metal::elf_loader;
use ncpu_metal::launcher::GpuLauncher;
use ncpu_metal::rootfs::{create_standard_dirs, load_rootfs_into_vfs};
use ncpu_metal::vfs::GpuVfs;

const MIB: usize = 1024 * 1024;
const DEFAULT_MEMORY_MB: usize = 16;
const DEFAULT_CYCLES_PER_BATCH: u32 = 10_000_000;
const ZERO_CHUNK: [u8; 4096] = [0; 4096];

enum InputKind {
    Elf,
    BootImage(BootImageHeader),
}

struct BootTaskRuntime {
    task: BootTaskDesc,
    heap_base: u64,
    vfs: Option<GpuVfs>,
}

fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

fn memory_size_from_profile(profile_mb: u32) -> Result<usize, String> {
    let mb = usize::try_from(profile_mb)
        .map_err(|_| format!("invalid memory profile {}", profile_mb))?;
    mb.checked_mul(MIB)
        .ok_or_else(|| format!("memory profile too large: {} MB", profile_mb))
}

fn parse_region_table(
    image: &[u8],
    header: &BootImageHeader,
) -> Result<Vec<BootRegionDesc>, String> {
    let base = usize::try_from(header.region_table_offset)
        .map_err(|_| "region table offset out of range".to_string())?;
    let count = usize::try_from(header.region_count)
        .map_err(|_| "region count out of range".to_string())?;
    let bytes = count
        .checked_mul(64)
        .ok_or_else(|| "region table size overflow".to_string())?;
    let end = base
        .checked_add(bytes)
        .ok_or_else(|| "region table end overflow".to_string())?;
    if end > image.len() {
        return Err("region table extends beyond image".to_string());
    }

    let mut regions = Vec::with_capacity(count);
    for idx in 0..count {
        let offset = base + idx * 64;
        let mut buf = [0u8; 64];
        buf.copy_from_slice(&image[offset..offset + 64]);
        regions.push(deserialize_region(&buf));
    }
    Ok(regions)
}

fn parse_task_table(image: &[u8], header: &BootImageHeader) -> Result<Vec<BootTaskDesc>, String> {
    let base = usize::try_from(header.task_table_offset)
        .map_err(|_| "task table offset out of range".to_string())?;
    let count =
        usize::try_from(header.task_count).map_err(|_| "task count out of range".to_string())?;
    let bytes = count
        .checked_mul(64)
        .ok_or_else(|| "task table size overflow".to_string())?;
    let end = base
        .checked_add(bytes)
        .ok_or_else(|| "task table end overflow".to_string())?;
    if end > image.len() {
        return Err("task table extends beyond image".to_string());
    }

    let mut tasks = Vec::with_capacity(count);
    for idx in 0..count {
        let offset = base + idx * 64;
        let mut buf = [0u8; 64];
        buf.copy_from_slice(&image[offset..offset + 64]);
        tasks.push(deserialize_task(&buf));
    }
    Ok(tasks)
}

fn zero_guest_range(launcher: &GpuLauncher, addr: u64, len: u64) -> Result<(), String> {
    let mut offset = 0u64;
    while offset < len {
        let remaining = len - offset;
        let chunk = remaining.min(ZERO_CHUNK.len() as u64) as usize;
        let guest_addr =
            usize::try_from(addr + offset).map_err(|_| "guest address out of range".to_string())?;
        launcher.write_memory(guest_addr, &ZERO_CHUNK[..chunk]);
        offset += chunk as u64;
    }
    Ok(())
}

fn build_boot_vfs(
    image: &[u8],
    header: &BootImageHeader,
    with_rootfs_flag: bool,
) -> Result<Option<GpuVfs>, String> {
    let needs_vfs = header.rootfs_size > 0 || with_rootfs_flag;
    if !needs_vfs {
        return Ok(Some(GpuVfs::new()));
    }

    let mut vfs = GpuVfs::new();
    create_standard_dirs(&mut vfs);
    if header.rootfs_size > 0 {
        let rootfs_offset = usize::try_from(header.rootfs_offset)
            .map_err(|_| "rootfs offset out of range".to_string())?;
        let rootfs_size = usize::try_from(header.rootfs_size)
            .map_err(|_| "rootfs size out of range".to_string())?;
        let rootfs_end = rootfs_offset
            .checked_add(rootfs_size)
            .ok_or_else(|| "rootfs end overflow".to_string())?;
        if rootfs_end > image.len() {
            return Err("rootfs extends beyond image".to_string());
        }
        load_rootfs_into_vfs(&mut vfs, &image[rootfs_offset..rootfs_end])?;
    }
    Ok(Some(vfs))
}

fn load_boot_image(
    launcher: &GpuLauncher,
    image: &[u8],
    header: &BootImageHeader,
    with_rootfs_flag: bool,
) -> Result<BootTaskRuntime, String> {
    let regions = parse_region_table(image, header)?;
    let tasks = parse_task_table(image, header)?;
    let task_idx = usize::try_from(header.root_task_index)
        .map_err(|_| "root task index out of range".to_string())?;
    let task = tasks
        .get(task_idx)
        .cloned()
        .ok_or_else(|| "root task index points past task table".to_string())?;

    let mut max_region_end = 0u64;
    for region in &regions {
        if region.file_size > region.mem_size {
            return Err(format!(
                "region at guest 0x{:X} has file_size {} > mem_size {}",
                region.guest_base, region.file_size, region.mem_size
            ));
        }

        if region.mem_size > 0 {
            zero_guest_range(launcher, region.guest_base, region.mem_size)?;
        }

        let file_offset = usize::try_from(region.file_offset)
            .map_err(|_| "region file offset out of range".to_string())?;
        let file_size = usize::try_from(region.file_size)
            .map_err(|_| "region file size out of range".to_string())?;
        let file_end = file_offset
            .checked_add(file_size)
            .ok_or_else(|| "region data end overflow".to_string())?;
        if file_end > image.len() {
            return Err(format!(
                "region data at offset {} (size {}) extends beyond image",
                region.file_offset, region.file_size
            ));
        }

        if file_size > 0 {
            let guest_base = usize::try_from(region.guest_base)
                .map_err(|_| "region guest base out of range".to_string())?;
            launcher.write_memory(guest_base, &image[file_offset..file_end]);
        }

        max_region_end = max_region_end.max(region.guest_base.saturating_add(region.mem_size));
    }

    let heap_base = align_up(max_region_end.max(0x60000), 0x10000);
    let vfs = build_boot_vfs(image, header, with_rootfs_flag)?;
    Ok(BootTaskRuntime {
        task,
        heap_base,
        vfs,
    })
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: ncpu_run --elf <path> [-- arg1 arg2 ...]");
        eprintln!("       ncpu_run <boot_image.bin>");
        process::exit(1);
    }

    // Parse arguments
    let mut input_path: Option<String> = None;
    let mut program_args: Vec<String> = Vec::new();
    let mut memory_mb: usize = DEFAULT_MEMORY_MB;
    let mut memory_overridden = false;
    let mut max_cycles: u64 = 500_000_000;
    let mut timeout: f64 = 30.0;
    let mut quiet = false;
    let mut with_rootfs = false;
    let mut force_elf = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--elf" => {
                force_elf = true;
                i += 1;
                if i < args.len() {
                    input_path = Some(args[i].clone());
                }
            }
            "--memory" => {
                i += 1;
                if i < args.len() {
                    memory_mb = args[i].parse().unwrap_or(DEFAULT_MEMORY_MB);
                    memory_overridden = true;
                }
            }
            "--max-cycles" => {
                i += 1;
                if i < args.len() {
                    max_cycles = args[i].parse().unwrap_or(500_000_000);
                }
            }
            "--timeout" => {
                i += 1;
                if i < args.len() {
                    timeout = args[i].parse().unwrap_or(30.0);
                }
            }
            "--quiet" | "-q" => {
                quiet = true;
            }
            "--rootfs" => {
                with_rootfs = true;
            }
            "--" => {
                // Everything after -- is program args
                program_args = args[i + 1..].to_vec();
                break;
            }
            other => {
                if input_path.is_none() {
                    input_path = Some(other.to_string());
                } else {
                    program_args.push(other.to_string());
                }
            }
        }
        i += 1;
    }

    let input_path = match input_path {
        Some(p) => p,
        None => {
            eprintln!("Error: no input file specified");
            process::exit(1);
        }
    };

    let input_data = match fs::read(&input_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error reading {}: {}", input_path, e);
            process::exit(1);
        }
    };

    let input_kind = if force_elf {
        InputKind::Elf
    } else if let Ok(header) = validate_image(&input_data) {
        InputKind::BootImage(header)
    } else {
        InputKind::Elf
    };

    let memory_size = match &input_kind {
        InputKind::BootImage(header) if !memory_overridden => {
            match memory_size_from_profile(header.memory_profile) {
                Ok(bytes) => {
                    memory_mb = bytes / MIB;
                    bytes
                }
                Err(err) => {
                    eprintln!("Invalid boot image memory profile: {}", err);
                    process::exit(1);
                }
            }
        }
        _ => memory_mb * MIB,
    };

    // Create launcher
    let launcher = match GpuLauncher::new(memory_size, DEFAULT_CYCLES_PER_BATCH) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("GPU launcher creation failed: {}", e);
            process::exit(1);
        }
    };

    let mut vfs = match &input_kind {
        InputKind::BootImage(header) => {
            if !program_args.is_empty() {
                eprintln!("Error: boot image mode does not yet accept extra CLI args");
                process::exit(1);
            }

            let runtime = match load_boot_image(&launcher, &input_data, header, with_rootfs) {
                Ok(runtime) => runtime,
                Err(err) => {
                    eprintln!("Boot image load failed: {}", err);
                    process::exit(1);
                }
            };

            if !quiet {
                eprintln!(
                    "[ncpu_run] boot image loaded: entry=0x{:X}, SP=0x{:X}, heap=0x{:X}",
                    runtime.task.entry_pc, runtime.task.stack_top, runtime.heap_base
                );
                eprintln!(
                    "[ncpu_run] image profile: {}MB, max_cycles: {}, timeout: {}s",
                    memory_mb, max_cycles, timeout
                );
            }

            launcher.load_boot_task(
                runtime.task.entry_pc,
                runtime.task.stack_top,
                runtime.heap_base,
            );
            runtime.vfs
        }
        InputKind::Elf => {
            let binary_name = Path::new(&input_path)
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_else(|| input_path.clone());
            let mut argv: Vec<String> = vec![binary_name];
            argv.extend(program_args.clone());
            let argv_refs = argv.iter().map(String::as_str).collect::<Vec<_>>();

            let prepared = match elf_loader::prepare_elf(
                &input_data,
                &argv_refs,
                &elf_loader::DEFAULT_ENVP,
                memory_size as u64,
            ) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("ELF preparation failed: {:?}", e);
                    process::exit(1);
                }
            };

            if !quiet {
                eprintln!(
                    "[ncpu_run] ELF loaded: entry=0x{:X}, SP=0x{:X}, heap=0x{:X}",
                    prepared.entry_pc, prepared.stack_pointer, prepared.heap_base
                );
                eprintln!(
                    "[ncpu_run] Memory: {}MB, max_cycles: {}, timeout: {}s",
                    memory_mb, max_cycles, timeout
                );
            }

            launcher.load_prepared_elf(&prepared);

            if with_rootfs {
                let mut v = GpuVfs::new();
                create_standard_dirs(&mut v);
                v.files
                    .insert(format!("/bin/{}", argv[0]), input_data.clone());
                v.files
                    .insert("/etc/hostname".to_string(), b"ncpu-gpu\n".to_vec());
                v.files.insert(
                    "/etc/passwd".to_string(),
                    b"root:x:0:0:root:/root:/bin/sh\n".to_vec(),
                );
                v.files
                    .insert("/etc/group".to_string(), b"root:x:0:\n".to_vec());
                Some(v)
            } else {
                Some(GpuVfs::new())
            }
        }
    };

    // Run
    let result = match launcher.run(&mut vfs, max_cycles, timeout, quiet) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Execution failed: {}", e);
            process::exit(1);
        }
    };

    // Print stdout
    if !result.stdout.is_empty() {
        let stdout_str = String::from_utf8_lossy(&result.stdout);
        print!("{}", stdout_str);
    }

    // Print stderr to stderr
    if !result.stderr.is_empty() {
        let stderr_str = String::from_utf8_lossy(&result.stderr);
        eprint!("{}", stderr_str);
    }

    if !quiet {
        let ips = if result.elapsed_secs > 0.0 {
            result.total_cycles as f64 / result.elapsed_secs
        } else {
            0.0
        };
        eprintln!(
            "\n[ncpu_run] {} after {} cycles ({:.3}s, {:.0} IPS)",
            result.stop_reason, result.total_cycles, result.elapsed_secs, ips
        );
        eprintln!("[ncpu_run] exit_code={}", result.exit_code);
        eprintln!(
            "[ncpu_run] processes={} forks={} context_switches={}",
            result.processes_created, result.total_forks, result.total_context_switches
        );
    }

    process::exit(result.exit_code);
}
