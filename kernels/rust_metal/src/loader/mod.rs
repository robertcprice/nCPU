//! Program loading, ELF handling, boot images, root filesystem, and virtual filesystem.
//!
//! This module owns everything required to load and run real ARM64 binaries
//! (BusyBox, Alpine, self-hosting C compiler, etc.) on the Metal GPU substrate.

pub mod boot_image;
pub mod elf_loader;
pub mod rootfs;
pub mod vfs;

// Key re-exports
pub use elf_loader::{prepare_elf, PreparedElf, DEFAULT_ENVP};
pub use vfs::GpuVfs;