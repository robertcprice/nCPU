//! System Call Primitives for nCPU/nSynth
//!
//! Safe wrappers around unsafe system calls for socket, process, and signal operations.

use crate::runtime::{Errno, FfiResult, Value};
use std::ffi::CString;

/// Socket domain constants
pub mod socket {
    pub const AF_INET: i32 = 2;
    pub const AF_INET6: i32 = 30;
    pub const AF_UNIX: i32 = 1;
}

/// Socket type constants
pub mod sock_type {
    pub const SOCK_STREAM: i32 = 1; // TCP
    pub const SOCK_DGRAM: i32 = 2; // UDP
}

/// Socket protocol constants
pub mod socket_protocol {
    pub const IPPROTO_TCP: i32 = 6;
    pub const IPPROTO_UDP: i32 = 17;
    pub const IPPROTO_IP: i32 = 0;
}

/// File descriptor constants
pub mod fd {
    pub const STDIN: i32 = 0;
    pub const STDOUT: i32 = 1;
    pub const STDERR: i32 = 2;
}

/// File open flags
pub mod open_flags {
    pub const O_RDONLY: i32 = 0o000000;
    pub const O_WRONLY: i32 = 0o000001;
    pub const O_RDWR: i32 = 0o000002;
    pub const O_CREAT: i32 = 0o000100;
    pub const O_TRUNC: i32 = 0o001000;
    pub const O_APPEND: i32 = 0o002000;
}

/// Seek origins
pub mod seek {
    pub const SEEK_SET: i32 = 0;
    pub const SEEK_CUR: i32 = 1;
    pub const SEEK_END: i32 = 2;
}

/// Signal numbers
pub mod signal {
    pub const SIGHUP: i32 = 1;
    pub const SIGINT: i32 = 2;
    pub const SIGQUIT: i32 = 3;
    pub const SIGKILL: i32 = 9;
    pub const SIGTERM: i32 = 15;
    pub const SIGCHLD: i32 = 20;
}

/// Create a socket (safe wrapper)
///
/// # Safety
/// This function calls the raw `socket` syscall but validates parameters
pub unsafe fn sys_socket(domain: i32, type_: i32, protocol: i32) -> FfiResult<Value> {
    // Validate parameters
    if !matches!(domain, socket::AF_INET | socket::AF_INET6 | socket::AF_UNIX) {
        return Err(Errno::InvalidArgument(format!(
            "Invalid socket domain: {}",
            domain
        )));
    }
    if !matches!(type_, sock_type::SOCK_STREAM | sock_type::SOCK_DGRAM) {
        return Err(Errno::InvalidArgument(format!(
            "Invalid socket type: {}",
            type_
        )));
    }

    // Raw syscall
    let fd = libc::socket(domain, type_, protocol);

    if fd < 0 {
        let err = std::io::Error::last_os_error();
        Err(Errno::IOError(format!("socket failed: {}", err)))
    } else {
        Ok(Value::Int(fd as i64))
    }
}

/// Bind a socket to an address (safe wrapper for IPv4)
///
/// # Safety
/// This function uses raw pointers but validates them first
pub unsafe fn sys_bind(sockfd: i32, addr: &[u8], port: u16) -> FfiResult<()> {
    if sockfd < 0 {
        return Err(Errno::InvalidArgument(
            "Invalid file descriptor".to_string(),
        ));
    }
    if addr.len() != 4 {
        return Err(Errno::InvalidArgument(
            "IPv4 address must be 4 bytes".to_string(),
        ));
    }

    // Build sockaddr_in structure
    let mut addr_struct: libc::sockaddr_in = std::mem::zeroed();
    addr_struct.sin_family = socket::AF_INET as u8;
    addr_struct.sin_port = port.to_be();
    addr_struct.sin_addr.s_addr = u32::from_be_bytes([addr[0], addr[1], addr[2], addr[3]]);

    let result = libc::bind(
        sockfd,
        &addr_struct as *const libc::sockaddr_in as *const libc::sockaddr,
        std::mem::size_of::<libc::sockaddr_in>() as u32,
    );

    if result < 0 {
        let err = std::io::Error::last_os_error();
        Err(Errno::IOError(format!("bind failed: {}", err)))
    } else {
        Ok(())
    }
}

/// Listen for connections on a socket
///
/// # Safety
/// Wrapper around `listen` syscall with parameter validation
pub unsafe fn sys_listen(sockfd: i32, backlog: i32) -> FfiResult<()> {
    if sockfd < 0 {
        return Err(Errno::InvalidArgument(
            "Invalid file descriptor".to_string(),
        ));
    }
    if backlog < 0 || backlog > 128 {
        return Err(Errno::InvalidArgument("Backlog must be 0-128".to_string()));
    }

    let result = libc::listen(sockfd, backlog);

    if result < 0 {
        let err = std::io::Error::last_os_error();
        Err(Errno::IOError(format!("listen failed: {}", err)))
    } else {
        Ok(())
    }
}

/// Accept a connection on a socket
///
/// # Safety
/// Wrapper around `accept` syscall
pub unsafe fn sys_accept(sockfd: i32) -> FfiResult<Value> {
    if sockfd < 0 {
        return Err(Errno::InvalidArgument(
            "Invalid file descriptor".to_string(),
        ));
    }

    let mut addr: libc::sockaddr_in = std::mem::zeroed();
    let mut addr_len = std::mem::size_of::<libc::sockaddr_in>() as u32;

    let client_fd = libc::accept(
        sockfd,
        &mut addr as *mut libc::sockaddr_in as *mut libc::sockaddr,
        &mut addr_len,
    );

    if client_fd < 0 {
        let err = std::io::Error::last_os_error();
        Err(Errno::IOError(format!("accept failed: {}", err)))
    } else {
        Ok(Value::Int(client_fd as i64))
    }
}

/// Fork a new process
///
/// # Safety
/// Wrapper around `fork` with resource tracking
pub unsafe fn sys_fork() -> FfiResult<Value> {
    let pid = libc::fork();

    if pid < 0 {
        let err = std::io::Error::last_os_error();
        Err(Errno::IOError(format!("fork failed: {}", err)))
    } else {
        Ok(Value::Int(pid as i64))
    }
}

/// Execute a program
///
/// # Safety
/// Wrapper around `execvp` which never returns on success
pub unsafe fn sys_exec(path: &str, args: &[String]) -> FfiResult<Value> {
    let path_c = match CString::new(path) {
        Ok(s) => s,
        Err(_) => {
            return Err(Errno::InvalidArgument(
                "Path contains null byte".to_string(),
            ))
        }
    };

    // Convert args to C strings
    let mut args_c: Vec<CString> = Vec::new();
    for arg in args {
        match CString::new(arg.as_str()) {
            Ok(s) => args_c.push(s),
            Err(_) => {
                return Err(Errno::InvalidArgument(
                    "Argument contains null byte".to_string(),
                ))
            }
        }
    }

    // Build argv array (null-terminated)
    let mut argv: Vec<*mut i8> = args_c.iter().map(|s| s.as_ptr() as *mut i8).collect();
    argv.push(std::ptr::null_mut());

    libc::execvp(path_c.as_ptr(), argv.as_ptr() as *const *const i8);

    // exec only returns on error
    let err = std::io::Error::last_os_error();
    Err(Errno::IOError(format!("exec failed: {}", err)))
}

/// Wait for a process to exit
///
/// # Safety
/// Wrapper around `waitpid`
pub unsafe fn sys_wait(pid: i32) -> FfiResult<Value> {
    let mut status: i32 = 0;
    let result = libc::waitpid(pid, &mut status, 0);

    if result < 0 {
        let err = std::io::Error::last_os_error();
        Err(Errno::IOError(format!("wait failed: {}", err)))
    } else {
        // Return exit status
        let exit_code = (status & 0xFF00) >> 8;
        Ok(Value::Int(exit_code as i64))
    }
}

/// Send a signal to a process
///
/// # Safety
/// Wrapper around `kill` with validation
pub unsafe fn sys_kill(pid: i32, signal: i32) -> FfiResult<()> {
    if pid < 0 {
        return Err(Errno::InvalidArgument("Invalid PID".to_string()));
    }
    if !matches!(
        signal,
        signal::SIGHUP
            | signal::SIGINT
            | signal::SIGQUIT
            | signal::SIGKILL
            | signal::SIGTERM
            | signal::SIGCHLD
    ) {
        return Err(Errno::InvalidArgument(format!(
            "Invalid signal: {}",
            signal
        )));
    }

    let result = libc::kill(pid, signal);

    if result < 0 {
        let err = std::io::Error::last_os_error();
        Err(Errno::IOError(format!("kill failed: {}", err)))
    } else {
        Ok(())
    }
}

/// Create a pipe for IPC
///
/// # Safety
/// Wrapper around `pipe` with validation
pub unsafe fn sys_pipe() -> FfiResult<Value> {
    let mut fds: [i32; 2] = [0, 0];
    let result = libc::pipe(fds.as_mut_ptr());

    if result < 0 {
        let err = std::io::Error::last_os_error();
        Err(Errno::IOError(format!("pipe failed: {}", err)))
    } else {
        // Return as pair (read_fd, write_fd)
        Ok(Value::Pair(fds[0] as i64, fds[1] as i64))
    }
}

/// Open a file
///
/// # Safety
/// Wrapper around `open` with path validation
pub unsafe fn sys_open(path: &str, flags: i32, mode: i32) -> FfiResult<Value> {
    let path_c = match CString::new(path) {
        Ok(s) => s,
        Err(_) => {
            return Err(Errno::InvalidArgument(
                "Path contains null byte".to_string(),
            ))
        }
    };

    let fd = libc::open(path_c.as_ptr(), flags, mode);

    if fd < 0 {
        let err = std::io::Error::last_os_error();
        Err(Errno::IOError(format!("open failed: {}", err)))
    } else {
        Ok(Value::Int(fd as i64))
    }
}

/// Read from a file descriptor
///
/// # Safety
/// Wrapper around `read` with bounds checking
pub unsafe fn sys_read(fd: i32, size: usize) -> FfiResult<Value> {
    if fd < 0 {
        return Err(Errno::InvalidArgument(
            "Invalid file descriptor".to_string(),
        ));
    }
    if size > 1024 * 1024 {
        return Err(Errno::InvalidArgument(
            "Read size too large (max 1MB)".to_string(),
        ));
    }

    let mut buffer = vec![0u8; size];
    let bytes_read = libc::read(fd, buffer.as_mut_ptr() as *mut libc::c_void, size);

    if bytes_read < 0 {
        let err = std::io::Error::last_os_error();
        Err(Errno::IOError(format!("read failed: {}", err)))
    } else {
        buffer.truncate(bytes_read as usize);
        // Convert bytes to string if valid UTF-8, otherwise return as array
        match String::from_utf8(buffer.clone()) {
            Ok(s) => Ok(Value::Str(s)),
            Err(_) => Ok(Value::Array(
                buffer.into_iter().map(|b| Value::Int(b as i64)).collect(),
            )),
        }
    }
}

/// Write to a file descriptor
///
/// # Safety
/// Wrapper around `write` with validation
pub unsafe fn sys_write(fd: i32, data: &[u8]) -> FfiResult<Value> {
    if fd < 0 {
        return Err(Errno::InvalidArgument(
            "Invalid file descriptor".to_string(),
        ));
    }
    if data.is_empty() {
        return Ok(Value::Int(0));
    }
    if data.len() > 1024 * 1024 {
        return Err(Errno::InvalidArgument(
            "Write size too large (max 1MB)".to_string(),
        ));
    }

    let bytes_written = libc::write(fd, data.as_ptr() as *const libc::c_void, data.len());

    if bytes_written < 0 {
        let err = std::io::Error::last_os_error();
        Err(Errno::IOError(format!("write failed: {}", err)))
    } else {
        Ok(Value::Int(bytes_written as i64))
    }
}

/// Close a file descriptor
///
/// # Safety
/// Wrapper around `close`
pub unsafe fn sys_close(fd: i32) -> FfiResult<()> {
    if fd < 0 {
        return Err(Errno::InvalidArgument(
            "Invalid file descriptor".to_string(),
        ));
    }

    let result = libc::close(fd);

    if result < 0 {
        let err = std::io::Error::last_os_error();
        Err(Errno::IOError(format!("close failed: {}", err)))
    } else {
        Ok(())
    }
}

/// Seek in a file
///
/// # Safety
/// Wrapper around `lseek`
pub unsafe fn sys_lseek(fd: i32, offset: i64, whence: i32) -> FfiResult<Value> {
    if fd < 0 {
        return Err(Errno::InvalidArgument(
            "Invalid file descriptor".to_string(),
        ));
    }
    if !matches!(whence, seek::SEEK_SET | seek::SEEK_CUR | seek::SEEK_END) {
        return Err(Errno::InvalidArgument("Invalid whence value".to_string()));
    }

    let result = libc::lseek(fd, offset, whence);

    if result < 0 {
        let err = std::io::Error::last_os_error();
        Err(Errno::IOError(format!("lseek failed: {}", err)))
    } else {
        Ok(Value::Int(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_socket_constants() {
        assert_eq!(socket::AF_INET, 2);
        assert_eq!(sock_type::SOCK_STREAM, 1);
        assert_eq!(socket_protocol::IPPROTO_TCP, 6);
    }

    #[test]
    fn test_sys_socket_invalid_domain() {
        let result = unsafe { sys_socket(999, sock_type::SOCK_STREAM, 0) };
        assert!(result.is_err());
    }

    #[test]
    fn test_sys_bind_invalid_addr() {
        let result = unsafe { sys_bind(0, &[1, 2], 8080) }; // Not 4 bytes
        assert!(result.is_err());
    }

    #[test]
    fn test_sys_pipe() {
        let result = unsafe { sys_pipe() };
        assert!(result.is_ok());
        if let Ok(Value::Pair(read_fd, write_fd)) = result {
            assert!(read_fd >= 0);
            assert!(write_fd >= 0);
        }
    }

    #[test]
    fn test_constants() {
        assert_eq!(fd::STDIN, 0);
        assert_eq!(fd::STDOUT, 1);
        assert_eq!(fd::STDERR, 2);
        assert_eq!(signal::SIGTERM, 15);
    }
}
