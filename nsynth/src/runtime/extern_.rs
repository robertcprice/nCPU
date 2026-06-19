//! External Function Support for nCPU/nSynth
//!
//! This module enables safe FFI (Foreign Function Interface) calls to C libraries
//! and system primitives. It provides a sandboxed wrapper around unsafe operations.

use crate::runtime::{Errno, FfiResult, Value};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// External function types that can be called from synthesized programs
#[derive(Clone)]
pub enum ExternFunc {
    /// Socket operations (connect, bind, listen, accept)
    Socket(SocketFunc),
    /// Process operations (fork, exec, wait, pipe)
    Process(ProcessFunc),
    /// Signal operations (kill, alarm, signal handler)
    Signal(SignalFunc),
    /// File operations (open, read, write, close)
    File(FileFunc),
    /// Custom external function registered by user
    Custom(
        String,
        Arc<dyn Fn(&[Value]) -> std::result::Result<Value, Errno> + Send + Sync>,
    ),
}

impl std::fmt::Debug for ExternFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExternFunc::Socket(s) => write!(f, "Socket({:?})", s),
            ExternFunc::Process(p) => write!(f, "Process({:?})", p),
            ExternFunc::Signal(s) => write!(f, "Signal({:?})", s),
            ExternFunc::File(fi) => write!(f, "File({:?})", fi),
            ExternFunc::Custom(name, _) => write!(f, "Custom({})", name),
        }
    }
}

/// Socket operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SocketFunc {
    /// socket(domain, type, protocol) -> fd
    Socket,
    /// bind(fd, addr, len) -> result
    Bind,
    /// listen(fd, backlog) -> result
    Listen,
    /// accept(fd, addr, len) -> client_fd
    Accept,
    /// connect(fd, addr, len) -> result
    Connect,
    /// send(fd, buf, len, flags) -> bytes_sent
    Send,
    /// recv(fd, buf, len, flags) -> bytes_received
    Recv,
    /// close(fd) -> result
    Close,
}

/// Process operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessFunc {
    /// fork() -> pid (0 in child, >0 in parent, <0 on error)
    Fork,
    /// exec(path, argv) -> result (never returns on success)
    Exec,
    /// wait(pid, status, options) -> pid
    Wait,
    /// pipe(fds) -> result (fds[0]=read, fds[1]=write)
    Pipe,
    /// exit(status) -> never returns
    Exit,
}

/// Signal operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalFunc {
    /// kill(pid, signal) -> result
    Kill,
    /// alarm(seconds) -> previous_alarm
    Alarm,
    /// signal(sig, handler) -> previous_handler
    Signal,
}

/// File operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFunc {
    /// open(path, flags, mode) -> fd
    Open,
    /// read(fd, buf, count) -> bytes_read
    Read,
    /// write(fd, buf, count) -> bytes_written
    Write,
    /// close(fd) -> result
    CloseFile,
    /// lseek(fd, offset, whence) -> new_offset
    Lseek,
}

/// Registry of external functions available to synthesized programs
#[derive(Debug, Clone)]
pub struct ExternRegistry {
    /// Map of function names to their external implementations
    funcs: HashMap<String, ExternFunc>,
    /// Whether unsafe operations are allowed
    allow_unsafe: bool,
    /// Resource limits for sandboxing
    limits: ResourceLimits,
}

/// Resource limits for sandboxing external operations
#[derive(Debug, Clone, Copy)]
pub struct ResourceLimits {
    /// Maximum number of open file descriptors
    max_fds: usize,
    /// Maximum number of sockets
    max_sockets: usize,
    /// Maximum number of child processes
    max_processes: usize,
    /// Maximum memory allocation for extern calls (bytes)
    max_memory: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_fds: 1024,
            max_sockets: 256,
            max_processes: 64,
            max_memory: 1024 * 1024, // 1MB
        }
    }
}

impl ExternRegistry {
    /// Create a new external function registry
    pub fn new() -> Self {
        Self {
            funcs: HashMap::new(),
            allow_unsafe: false,
            limits: ResourceLimits::default(),
        }
    }

    /// Create with unsafe operations allowed
    pub fn with_unsafe(mut self, allow: bool) -> Self {
        self.allow_unsafe = allow;
        self
    }

    /// Set resource limits
    pub fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Register a socket function
    pub fn register_socket(&mut self, name: String, func: SocketFunc) -> FfiResult<()> {
        if !self.allow_unsafe {
            return Err(Errno::PermissionDenied(
                "Unsafe operations not allowed".to_string(),
            ));
        }
        self.funcs.insert(name, ExternFunc::Socket(func));
        Ok(())
    }

    /// Register a process function
    pub fn register_process(&mut self, name: String, func: ProcessFunc) -> FfiResult<()> {
        if !self.allow_unsafe {
            return Err(Errno::PermissionDenied(
                "Unsafe operations not allowed".to_string(),
            ));
        }
        self.funcs.insert(name, ExternFunc::Process(func));
        Ok(())
    }

    /// Register a signal function
    pub fn register_signal(&mut self, name: String, func: SignalFunc) -> FfiResult<()> {
        if !self.allow_unsafe {
            return Err(Errno::PermissionDenied(
                "Unsafe operations not allowed".to_string(),
            ));
        }
        self.funcs.insert(name, ExternFunc::Signal(func));
        Ok(())
    }

    /// Register a file function
    pub fn register_file(&mut self, name: String, func: FileFunc) -> FfiResult<()> {
        if !self.allow_unsafe {
            return Err(Errno::PermissionDenied(
                "Unsafe operations not allowed".to_string(),
            ));
        }
        self.funcs.insert(name, ExternFunc::File(func));
        Ok(())
    }

    /// Register a custom external function
    pub fn register_custom<F>(&mut self, name: String, func: F) -> FfiResult<()>
    where
        F: Fn(&[Value]) -> std::result::Result<Value, Errno> + Send + Sync + 'static,
    {
        self.funcs
            .insert(name.clone(), ExternFunc::Custom(name, Arc::new(func)));
        Ok(())
    }

    /// Check if a function is registered
    pub fn has(&self, name: &str) -> bool {
        self.funcs.contains_key(name)
    }

    /// Get a registered function
    pub fn get(&self, name: &str) -> Option<ExternFunc> {
        self.funcs.get(name).cloned()
    }

    /// List all registered function names
    pub fn list(&self) -> Vec<String> {
        self.funcs.keys().cloned().collect()
    }

    /// Get resource limits
    pub fn limits(&self) -> ResourceLimits {
        self.limits
    }

    /// Check if unsafe operations are allowed
    pub fn allows_unsafe(&self) -> bool {
        self.allow_unsafe
    }
}

impl Default for ExternRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe external function registry
pub type SharedExternRegistry = Arc<RwLock<ExternRegistry>>;

/// Create a new shared registry
pub fn shared_registry() -> SharedExternRegistry {
    Arc::new(RwLock::new(ExternRegistry::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = ExternRegistry::new();
        assert!(!registry.allows_unsafe());
        assert_eq!(registry.list().len(), 0);
    }

    #[test]
    fn test_registry_with_unsafe() {
        let registry = ExternRegistry::new().with_unsafe(true);
        assert!(registry.allows_unsafe());
    }

    #[test]
    fn test_register_socket() {
        let mut registry = ExternRegistry::new().with_unsafe(true);
        registry
            .register_socket("socket".to_string(), SocketFunc::Socket)
            .unwrap();
        assert!(registry.has("socket"));
    }

    #[test]
    fn test_register_custom() {
        let mut registry = ExternRegistry::new();
        registry
            .register_custom("double".to_string(), |args| {
                if let Some(&Value::Int(n)) = args.first() {
                    Ok(Value::Int(n * 2))
                } else {
                    Err(Errno::InvalidArgument("Expected integer".to_string()))
                }
            })
            .unwrap();

        assert!(registry.has("double"));
    }

    #[test]
    fn test_register_without_unsafe_fails() {
        let mut registry = ExternRegistry::new();
        let result = registry.register_socket("socket".to_string(), SocketFunc::Socket);
        assert!(result.is_err());
    }

    #[test]
    fn test_resource_limits() {
        let limits = ResourceLimits::default();
        assert_eq!(limits.max_fds, 1024);
        assert_eq!(limits.max_sockets, 256);
        assert_eq!(limits.max_processes, 64);
    }
}
