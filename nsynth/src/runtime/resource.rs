//! Resource Management for nCPU/nSynth
//!
//! RAII-style automatic cleanup for file descriptors, sockets, and other system resources.
//! Prevents resource leaks by ensuring resources are released when dropped.

use crate::runtime::{Errno, FfiResult};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

/// Resource types that can be tracked and auto-cleaned
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceType {
    /// File descriptor
    FileDesc(i32),
    /// Socket handle
    Socket(i32),
    /// Process ID
    Process(i32),
    /// Pipe (read_fd, write_fd)
    Pipe(i32, i32),
    /// Mutex ID
    Mutex(i64),
    /// Channel ID
    Channel(i64),
}

impl ResourceType {
    /// Get the raw file descriptor if applicable
    pub fn as_fd(&self) -> Option<i32> {
        match self {
            ResourceType::FileDesc(fd) => Some(*fd),
            ResourceType::Socket(fd) => Some(*fd),
            _ => None,
        }
    }

    /// Check if this resource is still valid
    pub fn is_valid(&self) -> bool {
        match self {
            ResourceType::FileDesc(fd) => *fd >= 0,
            ResourceType::Socket(fd) => *fd >= 0,
            ResourceType::Process(pid) => *pid > 0,
            ResourceType::Pipe(read_fd, write_fd) => *read_fd >= 0 && *write_fd >= 0,
            ResourceType::Mutex(id) => *id >= 0,
            ResourceType::Channel(id) => *id >= 0,
        }
    }
}

/// Resource manager for tracking and auto-cleanup
#[derive(Debug)]
pub struct ResourceManager {
    /// Currently active resources
    resources: HashSet<ResourceType>,
    /// Maximum allowed resources of each type
    limits: ResourceLimits,
}

/// Resource limits to prevent exhaustion
#[derive(Debug, Clone, Copy)]
pub struct ResourceLimits {
    /// Maximum open file descriptors
    max_files: usize,
    /// Maximum open sockets
    max_sockets: usize,
    /// Maximum child processes
    max_processes: usize,
    /// Maximum pipes
    max_pipes: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_files: 1024,
            max_sockets: 256,
            max_processes: 64,
            max_pipes: 128,
        }
    }
}

impl ResourceManager {
    /// Create a new resource manager
    pub fn new() -> Self {
        Self {
            resources: HashSet::new(),
            limits: ResourceLimits::default(),
        }
    }

    /// Create with custom limits
    pub fn with_limits(limits: ResourceLimits) -> Self {
        Self {
            resources: HashSet::new(),
            limits,
        }
    }

    /// Register a file descriptor
    pub fn register_file(&mut self, fd: i32) -> FfiResult<()> {
        if fd < 0 {
            return Err(Errno::InvalidArgument(
                "Invalid file descriptor".to_string(),
            ));
        }
        if self.count_files() >= self.limits.max_files {
            return Err(Errno::ResourceExhausted("Too many open files".to_string()));
        }
        self.resources.insert(ResourceType::FileDesc(fd));
        Ok(())
    }

    /// Register a socket
    pub fn register_socket(&mut self, fd: i32) -> FfiResult<()> {
        if fd < 0 {
            return Err(Errno::InvalidArgument(
                "Invalid socket descriptor".to_string(),
            ));
        }
        if self.count_sockets() >= self.limits.max_sockets {
            return Err(Errno::ResourceExhausted(
                "Too many open sockets".to_string(),
            ));
        }
        self.resources.insert(ResourceType::Socket(fd));
        Ok(())
    }

    /// Register a process
    pub fn register_process(&mut self, pid: i32) -> FfiResult<()> {
        if pid <= 0 {
            return Err(Errno::InvalidArgument("Invalid process ID".to_string()));
        }
        if self.count_processes() >= self.limits.max_processes {
            return Err(Errno::ResourceExhausted("Too many processes".to_string()));
        }
        self.resources.insert(ResourceType::Process(pid));
        Ok(())
    }

    /// Register a pipe
    pub fn register_pipe(&mut self, read_fd: i32, write_fd: i32) -> FfiResult<()> {
        if read_fd < 0 || write_fd < 0 {
            return Err(Errno::InvalidArgument(
                "Invalid pipe file descriptors".to_string(),
            ));
        }
        if self.count_pipes() >= self.limits.max_pipes {
            return Err(Errno::ResourceExhausted("Too many pipes".to_string()));
        }
        self.resources.insert(ResourceType::Pipe(read_fd, write_fd));
        Ok(())
    }

    /// Unregister a resource
    pub fn unregister(&mut self, resource: ResourceType) {
        self.resources.remove(&resource);
    }

    /// Unregister by file descriptor
    pub fn unregister_fd(&mut self, fd: i32) {
        self.resources.remove(&ResourceType::FileDesc(fd));
        self.resources.remove(&ResourceType::Socket(fd));
    }

    /// Check if a resource is registered
    pub fn contains(&self, resource: &ResourceType) -> bool {
        self.resources.contains(resource)
    }

    /// Get count of open files
    fn count_files(&self) -> usize {
        self.resources
            .iter()
            .filter(|r| matches!(r, ResourceType::FileDesc(_)))
            .count()
    }

    /// Get count of open sockets
    fn count_sockets(&self) -> usize {
        self.resources
            .iter()
            .filter(|r| matches!(r, ResourceType::Socket(_)))
            .count()
    }

    /// Get count of processes
    fn count_processes(&self) -> usize {
        self.resources
            .iter()
            .filter(|r| matches!(r, ResourceType::Process(_)))
            .count()
    }

    /// Get count of pipes
    fn count_pipes(&self) -> usize {
        self.resources
            .iter()
            .filter(|r| matches!(r, ResourceType::Pipe(..)))
            .count()
    }

    /// Get total resource count
    pub fn len(&self) -> usize {
        self.resources.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }

    /// Clear all resources (for cleanup)
    pub fn clear(&mut self) {
        self.resources.clear();
    }

    /// Get resource statistics
    pub fn stats(&self) -> ResourceStats {
        ResourceStats {
            total: self.resources.len(),
            files: self.count_files(),
            sockets: self.count_sockets(),
            processes: self.count_processes(),
            pipes: self.count_pipes(),
        }
    }
}

impl Default for ResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ResourceManager {
    fn drop(&mut self) {
        // Auto-close all resources on drop
        for resource in &self.resources {
            if let Some(fd) = resource.as_fd() {
                // Safe to call close during cleanup
                unsafe {
                    // Ignore errors during cleanup
                    libc::close(fd);
                }
            }
        }
    }
}

/// Resource statistics
#[derive(Debug, Clone, Copy)]
pub struct ResourceStats {
    pub total: usize,
    pub files: usize,
    pub sockets: usize,
    pub processes: usize,
    pub pipes: usize,
}

/// RAII guard for a single resource
#[derive(Debug)]
pub struct ResourceGuard<T> {
    manager: Option<Arc<Mutex<ResourceManager>>>,
    resource: Option<T>,
}

impl<T> ResourceGuard<T> {
    /// Create a new resource guard
    pub fn new(manager: Arc<Mutex<ResourceManager>>, resource: T) -> Self {
        Self {
            manager: Some(manager),
            resource: Some(resource),
        }
    }

    /// Get the resource
    pub fn get(&self) -> Option<&T> {
        self.resource.as_ref()
    }

    /// Get mutable reference to the resource
    pub fn get_mut(&mut self) -> Option<&mut T> {
        self.resource.as_mut()
    }

    /// Release the resource early
    pub fn release(&mut self) -> Option<T> {
        self.resource.take()
    }
}

impl<T> Drop for ResourceGuard<T> {
    fn drop(&mut self) {
        if let Some(_resource) = self.resource.take() {
            // Auto-cleanup logic would go here
            // For now, just mark as released
        }
    }
}

/// Shared resource manager type
pub type SharedResourceManager = Arc<Mutex<ResourceManager>>;

/// Create a new shared resource manager
pub fn shared_manager() -> SharedResourceManager {
    Arc::new(Mutex::new(ResourceManager::new()))
}

/// Create a shared manager with custom limits
pub fn shared_manager_with_limits(limits: ResourceLimits) -> SharedResourceManager {
    Arc::new(Mutex::new(ResourceManager::with_limits(limits)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_creation() {
        let manager = ResourceManager::new();
        assert!(manager.is_empty());
        assert_eq!(manager.len(), 0);
    }

    #[test]
    fn test_register_file() {
        let mut manager = ResourceManager::new();
        manager.register_file(3).unwrap();
        assert_eq!(manager.len(), 1);
        assert!(manager.contains(&ResourceType::FileDesc(3)));
    }

    #[test]
    fn test_register_invalid_fd() {
        let mut manager = ResourceManager::new();
        let result = manager.register_file(-1);
        assert!(result.is_err());
    }

    #[test]
    fn test_resource_limits() {
        let limits = ResourceLimits {
            max_files: 2,
            ..Default::default()
        };
        let mut manager = ResourceManager::with_limits(limits);
        manager.register_file(1).unwrap();
        manager.register_file(2).unwrap();
        let result = manager.register_file(3);
        assert!(result.is_err());
    }

    #[test]
    fn test_unregister_fd() {
        let mut manager = ResourceManager::new();
        manager.register_file(3).unwrap();
        manager.unregister_fd(3);
        assert!(!manager.contains(&ResourceType::FileDesc(3)));
        assert_eq!(manager.len(), 0);
    }

    #[test]
    fn test_stats() {
        let mut manager = ResourceManager::new();
        manager.register_file(1).unwrap();
        manager.register_socket(4).unwrap();
        manager.register_process(100).unwrap();

        let stats = manager.stats();
        assert_eq!(stats.files, 1);
        assert_eq!(stats.sockets, 1);
        assert_eq!(stats.processes, 1);
        assert_eq!(stats.total, 3);
    }

    #[test]
    fn test_pipe_registration() {
        let mut manager = ResourceManager::new();
        manager.register_pipe(5, 6).unwrap();
        assert!(manager.contains(&ResourceType::Pipe(5, 6)));
    }

    #[test]
    fn test_shared_manager() {
        let manager = shared_manager();
        {
            let mut guard = manager.lock().unwrap();
            guard.register_file(3).unwrap();
        }
        // Manager still accessible
        let guard = manager.lock().unwrap();
        assert_eq!(guard.len(), 1);
    }
}
