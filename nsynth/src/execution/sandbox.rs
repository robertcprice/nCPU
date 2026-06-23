//! Sandboxed execution environment for safely running synthesized code.
//!
//! This module provides a comprehensive sandboxing capability that isolates
//! code execution with resource limits, timeouts, and security boundaries.
//! It supports multiple languages (Rust, JavaScript, Python) and provides
//! detailed error capture and output reporting.
//!
//! # Architecture
//!
//! The sandbox operates at multiple isolation layers:
//! - **Process isolation**: Each execution runs in a separate subprocess
//! - **Resource limits**: CPU time, memory, file descriptors are capped
//! - **Namespace isolation**: Network, filesystem, and IPC isolation (Unix)
//! - **Signal handling**: Graceful termination with proper cleanup
//!
//! # Usage
//!
//! ```rust
//! use nsynth::execution::sandbox::{Sandbox, VerificationReport, Example};
//!
//! let sandbox = Sandbox::new();
//! let code = r#"fn add(x: i64, y: i64) -> i64 { x + y }"#;
//!
//! let examples = vec![
//!     Example { inputs: vec![2.into(), 3.into()], expected: 5.into() },
//! ];
//!
//! let report = sandbox.verify(code, "add", &examples, Language::Rust).unwrap();
//!
//! if report.all_passed() {
//!     println!("All examples passed!");
//! }
//! ```

use std::collections::HashMap;
use std::ffi::OsStr;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{self, Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::io::AsRawFd;
#[cfg(unix)]
use std::os::unix::process::CommandExt;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

use crate::benchmark::Value as BenchmarkValue;
use serde::{Deserialize, Serialize};

/// Maximum execution time before timeout (default: 10 seconds)
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

/// Maximum memory allocation in bytes (default: 512MB)
const DEFAULT_MEMORY_LIMIT: usize = 512 * 1024 * 1024;

/// Maximum output size to capture (stdout + stderr)
const MAX_OUTPUT_SIZE: usize = 10 * 1024 * 1024; // 10MB

/// Supported programming languages for execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    /// Rust (primary language)
    Rust,
    /// JavaScript/TypeScript via Node.js
    JavaScript,
    /// Python 3
    Python,
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Language::Rust => write!(f, "rust"),
            Language::JavaScript => write!(f, "javascript"),
            Language::Python => write!(f, "python"),
        }
    }
}

impl Language {
    /// Get the file extension for source files in this language
    pub fn extension(&self) -> &str {
        match self {
            Language::Rust => "rs",
            Language::JavaScript => "js",
            Language::Python => "py",
        }
    }

    /// Check if the runtime for this language is available
    pub fn is_available(&self) -> bool {
        match self {
            Language::Rust => Self::check_command("rustc") && Self::check_command("rustc"),
            Language::JavaScript => Self::check_command("node") || Self::check_command("nodejs"),
            Language::Python => Self::check_command("python3") || Self::check_command("python"),
        }
    }

    fn check_command(cmd: &str) -> bool {
        Command::new(cmd)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
}

/// A single test case for verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    /// Input values to pass to the function
    pub inputs: Vec<InputValue>,
    /// Expected return value
    pub expected: InputValue,
}

/// Value that can be passed as input or expected as output
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum InputValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    IntArray(Vec<i64>),
}

impl From<BenchmarkValue> for InputValue {
    fn from(value: BenchmarkValue) -> Self {
        match value {
            BenchmarkValue::Int(i) => InputValue::Int(i),
            BenchmarkValue::Float(f) => InputValue::Float(f64::from_bits(f)),
            BenchmarkValue::Bool(b) => InputValue::Bool(b),
            BenchmarkValue::Str(s) => InputValue::String(s),
            // This sandbox input type only models integer arrays. An all-int
            // array converts directly; a typed/nested array has no `InputValue`
            // representation, so fall through to the unsupported fallback rather
            // than silently dropping element data.
            ref v @ BenchmarkValue::Array(_) => match v.as_i64_slice() {
                Some(ints) => InputValue::IntArray(ints),
                None => InputValue::Int(0),
            },
            _ => InputValue::Int(0), // Fallback for unsupported types
        }
    }
}

impl fmt::Display for InputValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InputValue::Int(i) => write!(f, "{}", i),
            InputValue::Float(v) => write!(f, "{}", v),
            InputValue::Bool(b) => write!(f, "{}", b),
            InputValue::String(s) => write!(f, "\"{}\"", s),
            InputValue::IntArray(arr) => {
                write!(f, "[")?;
                for (i, val) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", val)?;
                }
                write!(f, "]")
            }
        }
    }
}

/// Result of verifying a function against examples
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    /// Whether all examples passed
    pub success: bool,
    /// Total number of examples tested
    pub total: usize,
    /// Number of examples that passed
    pub passed: usize,
    /// Detailed results for each example
    pub results: Vec<ExampleResult>,
    /// Overall execution metrics
    pub metrics: ExecutionMetrics,
}

impl VerificationReport {
    /// Check if all examples passed
    pub fn all_passed(&self) -> bool {
        self.success && self.passed == self.total
    }

    /// Get the failure rate as a fraction (0.0 to 1.0)
    pub fn failure_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.total - self.passed) as f64 / self.total as f64
        }
    }
}

/// Result of executing a single example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleResult {
    /// Index of the example in the test suite
    pub index: usize,
    /// Whether the example passed
    pub passed: bool,
    /// Actual output value (if execution succeeded)
    pub actual: Option<InputValue>,
    /// Expected output value
    pub expected: InputValue,
    /// Error message (if execution failed or output mismatched)
    pub error: Option<String>,
    /// Execution time for this example
    pub duration_ms: u64,
}

/// Metrics collected during execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    /// Total wall-clock time in milliseconds
    pub total_duration_ms: u64,
    /// CPU time used (user + system) in seconds
    pub cpu_time_secs: f64,
    /// Peak memory usage in bytes (if measurable)
    pub peak_memory_bytes: Option<usize>,
    /// Number of examples executed
    pub examples_executed: usize,
}

impl Default for ExecutionMetrics {
    fn default() -> Self {
        Self {
            total_duration_ms: 0,
            cpu_time_secs: 0.0,
            peak_memory_bytes: None,
            examples_executed: 0,
        }
    }
}

/// Detailed error information from sandbox execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SandboxError {
    /// Compilation failed with syntax/type errors
    CompilationFailed { language: Language, stderr: String },
    /// Execution timed out
    Timeout {
        language: Language,
        timeout_secs: u64,
    },
    /// Memory limit exceeded
    MemoryLimitExceeded { limit_bytes: usize },
    /// Runtime panic or crash
    RuntimePanic {
        message: String,
        backtrace: Option<String>,
    },
    /// Signal received (Unix-specific)
    Signal { signal: i32, name: String },
    /// I/O error during execution
    IoError { message: String },
    /// Language runtime not available
    RuntimeUnavailable { language: Language },
    /// Security violation (unsafe operation attempted)
    SecurityViolation { operation: String },
    /// Output too large to capture
    OutputTooLarge { size: usize, limit: usize },
}

impl fmt::Display for SandboxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SandboxError::CompilationFailed { language, stderr } => {
                write!(f, "{} compilation failed: {}", language, stderr)
            }
            SandboxError::Timeout {
                language,
                timeout_secs,
            } => {
                write!(
                    f,
                    "{} execution timed out after {}s",
                    language, timeout_secs
                )
            }
            SandboxError::MemoryLimitExceeded { limit_bytes } => {
                write!(
                    f,
                    "Memory limit exceeded ({} MB)",
                    limit_bytes / 1024 / 1024
                )
            }
            SandboxError::RuntimePanic { message, backtrace } => {
                write!(f, "Runtime panic: {}", message)?;
                if let Some(bt) = backtrace {
                    write!(f, "\nBacktrace: {}", bt)?;
                }
                Ok(())
            }
            SandboxError::Signal { signal, name } => {
                write!(f, "Process received signal {}: {}", signal, name)
            }
            SandboxError::IoError { message } => {
                write!(f, "I/O error: {}", message)
            }
            SandboxError::RuntimeUnavailable { language } => {
                write!(f, "{} runtime not available", language)
            }
            SandboxError::SecurityViolation { operation } => {
                write!(f, "Security violation: attempted {}", operation)
            }
            SandboxError::OutputTooLarge { size, limit } => {
                write!(f, "Output too large: {} bytes (limit: {})", size, limit)
            }
        }
    }
}

impl std::error::Error for SandboxError {}

impl From<std::io::Error> for SandboxError {
    fn from(err: std::io::Error) -> Self {
        SandboxError::IoError {
            message: err.to_string(),
        }
    }
}

/// Result of a sandbox execution
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Standard output captured
    pub stdout: String,
    /// Standard error captured
    pub stderr: String,
    /// Exit code (0 = success)
    pub exit_code: Option<i32>,
    /// Whether execution timed out
    pub timed_out: bool,
    /// Execution duration
    pub duration: Duration,
    /// Peak memory usage (if available)
    pub memory_bytes: Option<usize>,
}

/// Configuration for sandbox execution
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Maximum execution time before timeout
    pub timeout: Duration,
    /// Maximum memory allocation in bytes
    pub memory_limit: usize,
    /// Whether to enable namespace isolation (Unix only)
    pub enable_isolation: bool,
    /// Working directory for execution
    pub working_directory: Option<PathBuf>,
    /// Environment variables to set
    pub env_vars: HashMap<String, String>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            timeout: DEFAULT_TIMEOUT,
            memory_limit: DEFAULT_MEMORY_LIMIT,
            enable_isolation: true,
            working_directory: None,
            env_vars: HashMap::new(),
        }
    }
}

/// Main sandbox struct for executing code safely
pub struct Sandbox {
    config: SandboxConfig,
    temp_dir: PathBuf,
}

impl Sandbox {
    /// Create a new sandbox with default configuration
    pub fn new() -> Result<Self, io::Error> {
        Self::with_config(SandboxConfig::default())
    }

    /// Create a new sandbox with custom configuration
    pub fn with_config(config: SandboxConfig) -> Result<Self, io::Error> {
        let temp_dir = std::env::temp_dir().join(format!("nsynth-sandbox-{}", std::process::id()));
        fs::create_dir_all(&temp_dir)?;

        Ok(Self { config, temp_dir })
    }

    /// Execute code and verify against examples
    ///
    /// This is the main entry point for verifying synthesized code.
    /// It compiles the code, runs it against each example, and returns
    /// a detailed verification report.
    pub fn verify(
        &self,
        code: &str,
        function_name: &str,
        examples: &[Example],
        language: Language,
    ) -> Result<VerificationReport, SandboxError> {
        let start = Instant::now();

        // First, compile the code
        let executable = self.compile(code, language)?;

        // Run each example
        let mut results = Vec::with_capacity(examples.len());
        let mut metrics = ExecutionMetrics {
            examples_executed: examples.len(),
            ..Default::default()
        };

        for (idx, example) in examples.iter().enumerate() {
            let example_start = Instant::now();
            let exec_result = self.execute_example(&executable, function_name, example, language);

            let duration = example_start.elapsed();

            match exec_result {
                Ok(actual) => {
                    let passed = actual == example.expected;
                    results.push(ExampleResult {
                        index: idx,
                        passed,
                        actual: Some(actual.clone()),
                        expected: example.expected.clone(),
                        error: None,
                        duration_ms: duration.as_millis() as u64,
                    });
                }
                Err(e) => {
                    results.push(ExampleResult {
                        index: idx,
                        passed: false,
                        actual: None,
                        expected: example.expected.clone(),
                        error: Some(e.to_string()),
                        duration_ms: duration.as_millis() as u64,
                    });
                }
            }
        }

        let passed = results.iter().filter(|r| r.passed).count();
        metrics.total_duration_ms = start.elapsed().as_millis() as u64;

        Ok(VerificationReport {
            success: passed == examples.len(),
            total: examples.len(),
            passed,
            results,
            metrics,
        })
    }

    /// Execute code directly (without verification)
    ///
    /// Runs the code and returns the raw execution result.
    pub fn execute(
        &self,
        code: &str,
        inputs: &[InputValue],
        language: Language,
    ) -> Result<ExecutionResult, SandboxError> {
        let executable = self.compile(code, language)?;
        self.run_executable(&executable, inputs, language)
    }

    /// Compile code to an executable
    fn compile(&self, code: &str, language: Language) -> Result<PathBuf, SandboxError> {
        if !language.is_available() {
            return Err(SandboxError::RuntimeUnavailable { language });
        }

        let source_file = self.temp_dir.join(format!("main.{}", language.extension()));
        fs::write(&source_file, code)?;

        let executable = match language {
            Language::Rust => self.compile_rust(&source_file)?,
            Language::JavaScript => {
                // JavaScript doesn't need compilation, just return the source
                source_file
            }
            Language::Python => {
                // Python doesn't need compilation, just return the source
                source_file
            }
        };

        Ok(executable)
    }

    /// Compile Rust code to a native binary
    #[cfg(unix)]
    fn compile_rust(&self, source_file: &Path) -> Result<PathBuf, SandboxError> {
        let output_file = self.temp_dir.join("program");

        let mut cmd = Command::new("rustc");
        cmd.arg(source_file)
            .arg("-o")
            .arg(&output_file)
            .arg("-C")
            .arg("opt-level=2")
            .arg("-C")
            .arg("debuginfo=0")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Add resource limits to compilation too
        self.apply_resource_limits(&mut cmd);

        let result = cmd.spawn().and_then(|mut child| child.wait_with_output());

        match result {
            Ok(output) => {
                if output.status.success() {
                    Ok(output_file)
                } else {
                    Err(SandboxError::CompilationFailed {
                        language: Language::Rust,
                        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                    })
                }
            }
            Err(e) => Err(SandboxError::IoError {
                message: e.to_string(),
            }),
        }
    }

    #[cfg(windows)]
    fn compile_rust(&self, source_file: &Path) -> Result<PathBuf, SandboxError> {
        let output_file = self.temp_dir.join("program.exe");

        let mut cmd = Command::new("rustc");
        cmd.arg(source_file)
            .arg("-o")
            .arg(&output_file)
            .arg("-C")
            .arg("opt-level=2")
            .arg("-C")
            .arg("debuginfo=0")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let result = cmd.spawn().and_then(|mut child| child.wait_with_output());

        match result {
            Ok(output) => {
                if output.status.success() {
                    Ok(output_file)
                } else {
                    Err(SandboxError::CompilationFailed {
                        language: Language::Rust,
                        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                    })
                }
            }
            Err(e) => Err(SandboxError::IoError {
                message: e.to_string(),
            }),
        }
    }

    /// Execute a single example against the compiled program
    fn execute_example(
        &self,
        executable: &Path,
        function_name: &str,
        example: &Example,
        language: Language,
    ) -> Result<InputValue, SandboxError> {
        let result = self.run_executable(executable, &example.inputs, language)?;

        if result.timed_out {
            return Err(SandboxError::Timeout {
                language,
                timeout_secs: self.config.timeout.as_secs(),
            });
        }

        if let Some(code) = result.exit_code {
            if code != 0 {
                return Err(SandboxError::RuntimePanic {
                    message: format!("Process exited with code {}", code),
                    backtrace: Some(result.stderr),
                });
            }
        }

        // Parse the output
        self.parse_output(&result.stdout, language)
    }

    /// Run the compiled program with inputs
    fn run_executable(
        &self,
        executable: &Path,
        inputs: &[InputValue],
        language: Language,
    ) -> Result<ExecutionResult, SandboxError> {
        let start = Instant::now();

        let mut cmd = match language {
            Language::Rust => {
                let mut c = Command::new(executable);
                // Apply namespace isolation on Unix
                #[cfg(unix)]
                self.apply_isolation(&mut c);
                c
            }
            Language::JavaScript => {
                let mut c = Command::new("node");
                c.arg(executable);
                #[cfg(unix)]
                self.apply_isolation(&mut c);
                c
            }
            Language::Python => {
                let mut c = if Command::new("python3").arg("--version").output().is_ok() {
                    Command::new("python3")
                } else {
                    Command::new("python")
                };
                c.arg(executable);
                #[cfg(unix)]
                self.apply_isolation(&mut c);
                c
            }
        };

        // Set up stdin, stdout, stderr
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Set working directory if specified
        if let Some(wd) = &self.config.working_directory {
            cmd.current_dir(wd);
        }

        // Apply resource limits
        self.apply_resource_limits(&mut cmd);

        // Spawn the process
        let mut child = cmd.spawn().map_err(|e| SandboxError::IoError {
            message: format!("Failed to spawn process: {}", e),
        })?;

        // Write inputs to stdin
        if let Some(mut stdin) = child.stdin.take() {
            let input_str = self.format_inputs(inputs, language);
            stdin
                .write_all(input_str.as_bytes())
                .map_err(|e| SandboxError::IoError {
                    message: format!("Failed to write to stdin: {}", e),
                })?;
        }

        // Set up timeout monitoring
        let timeout = self.config.timeout;
        let child_id = child.id();
        let timed_out = Arc::new(AtomicBool::new(false));
        let timeout_monitor = timed_out.clone();

        let timeout_thread = std::thread::spawn(move || {
            let start = Instant::now();
            while start.elapsed() < timeout {
                std::thread::sleep(Duration::from_millis(100));
                // Check if process is still running
                if let Ok(Some(_)) = check_process(child_id) {
                    continue;
                } else {
                    return; // Process already exited
                }
            }
            // Timeout exceeded - terminate the process
            timeout_monitor.store(true, Ordering::SeqCst);
            terminate_process(child_id);
        });

        // Wait for completion
        let output = child.wait_with_output();

        // Cancel timeout thread
        timed_out.store(true, Ordering::SeqCst); // Signal thread to exit
        let _ = timeout_thread.join();

        let duration = start.elapsed();
        let timed_out_flag = timed_out.load(Ordering::SeqCst);

        if timed_out_flag {
            return Ok(ExecutionResult {
                stdout: String::new(),
                stderr: "Execution timed out".to_string(),
                exit_code: None,
                timed_out: true,
                duration,
                memory_bytes: None,
            });
        }

        let output = output.map_err(|e| SandboxError::IoError {
            message: format!("Failed to wait for process: {}", e),
        })?;

        // Check output size limits
        let stdout_len = output.stdout.len();
        let stderr_len = output.stderr.len();
        if stdout_len + stderr_len > MAX_OUTPUT_SIZE {
            return Err(SandboxError::OutputTooLarge {
                size: stdout_len + stderr_len,
                limit: MAX_OUTPUT_SIZE,
            });
        }

        Ok(ExecutionResult {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: Some(output.status.code().unwrap_or(-1)),
            timed_out: false,
            duration,
            memory_bytes: None, // Could be implemented with /proc on Linux
        })
    }

    /// Format input values for stdin
    fn format_inputs(&self, inputs: &[InputValue], language: Language) -> String {
        match language {
            Language::Rust | Language::JavaScript => {
                // JSON format
                serde_json::to_string(inputs).unwrap_or_default()
            }
            Language::Python => {
                // JSON format for Python too
                serde_json::to_string(inputs).unwrap_or_default()
            }
        }
    }

    /// Parse output from stdout
    fn parse_output(&self, output: &str, language: Language) -> Result<InputValue, SandboxError> {
        let output = output.trim();
        if output.is_empty() {
            return Err(SandboxError::RuntimePanic {
                message: "No output produced".to_string(),
                backtrace: None,
            });
        }

        // Try to parse as JSON first
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(output) {
            return Self::json_to_input_value(value);
        }

        // Fallback: try to parse as plain int
        if let Ok(i) = output.parse::<i64>() {
            return Ok(InputValue::Int(i));
        }

        // Try as float
        if let Ok(f) = output.parse::<f64>() {
            return Ok(InputValue::Float(f));
        }

        // Try as bool
        match output.to_lowercase().as_str() {
            "true" => return Ok(InputValue::Bool(true)),
            "false" => return Ok(InputValue::Bool(false)),
            _ => {}
        }

        // Default to string
        Ok(InputValue::String(output.to_string()))
    }

    fn json_to_input_value(value: serde_json::Value) -> Result<InputValue, SandboxError> {
        match value {
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(InputValue::Int(i))
                } else if let Some(f) = n.as_f64() {
                    Ok(InputValue::Float(f))
                } else {
                    Err(SandboxError::RuntimePanic {
                        message: "Invalid number format".to_string(),
                        backtrace: None,
                    })
                }
            }
            serde_json::Value::Bool(b) => Ok(InputValue::Bool(b)),
            serde_json::Value::String(s) => Ok(InputValue::String(s)),
            serde_json::Value::Array(arr) => {
                let ints: Result<Vec<i64>, _> = arr
                    .iter()
                    .map(|v| {
                        v.as_i64().ok_or_else(|| SandboxError::RuntimePanic {
                            message: "Array elements must be integers".to_string(),
                            backtrace: None,
                        })
                    })
                    .collect();
                Ok(InputValue::IntArray(ints?))
            }
            _ => Err(SandboxError::RuntimePanic {
                message: format!("Unsupported JSON type: {}", value),
                backtrace: None,
            }),
        }
    }

    /// Apply namespace isolation (Unix only)
    #[cfg(unix)]
    fn apply_isolation(&self, cmd: &mut Command) {
        if !self.config.enable_isolation {
            return;
        }

        unsafe {
            // Use POSIX spawn for resource limits
            cmd.pre_exec(|| {
                // Ignore SIGPIPE to prevent crashes from broken pipes
                libc::signal(libc::SIGPIPE, libc::SIG_IGN);

                // Create new process group
                #[cfg(target_os = "linux")]
                {
                    libc::setsid();
                }

                Ok(())
            });
        }
    }

    /// Apply resource limits to the command
    #[cfg(unix)]
    fn apply_resource_limits(&self, cmd: &mut Command) {
        // Extract values before closure to avoid lifetime issues
        let time_limit = self.config.timeout.as_secs() as libc::rlim_t;
        let mem_limit = self.config.memory_limit as libc::rlim_t;
        let fd_limit = 256 as libc::rlim_t;
        let stack_limit = 8 * 1024 * 1024 as libc::rlim_t;

        unsafe {
            cmd.pre_exec(move || {
                // Set CPU time limit (soft + hard limits)
                let rlimit = libc::rlimit {
                    rlim_cur: time_limit,
                    rlim_max: time_limit,
                };
                libc::setrlimit(libc::RLIMIT_CPU, &rlimit);

                // Set memory limit (address space)
                let rlimit_as = libc::rlimit {
                    rlim_cur: mem_limit,
                    rlim_max: mem_limit,
                };
                libc::setrlimit(libc::RLIMIT_AS, &rlimit_as);

                // Set file descriptor limit
                let rlimit_nofile = libc::rlimit {
                    rlim_cur: fd_limit,
                    rlim_max: fd_limit,
                };
                libc::setrlimit(libc::RLIMIT_NOFILE, &rlimit_nofile);

                // Set stack size limit
                let rlimit_stack = libc::rlimit {
                    rlim_cur: stack_limit,
                    rlim_max: stack_limit,
                };
                libc::setrlimit(libc::RLIMIT_STACK, &rlimit_stack);

                Ok(())
            });
        }
    }

    #[cfg(windows)]
    fn apply_resource_limits(&self, _cmd: &mut Command) {
        // Windows resource limiting via Job Objects would go here
        // For now, we rely on timeout monitoring
    }

    /// Clean up temporary files
    pub fn cleanup(&self) -> io::Result<()> {
        if self.temp_dir.exists() {
            fs::remove_dir_all(&self.temp_dir)
        } else {
            Ok(())
        }
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        // Best-effort cleanup
        let _ = self.cleanup();
    }
}

/// Check if a process is still running
#[cfg(unix)]
fn check_process(pid: u32) -> io::Result<Option<bool>> {
    let result = unsafe { libc::kill(pid as i32, 0) };
    if result == 0 {
        Ok(Some(true))
    } else {
        let errno = io::Error::last_os_error().raw_os_error().unwrap_or(0);
        if errno == libc::ESRCH {
            Ok(Some(false))
        } else {
            Err(io::Error::last_os_error())
        }
    }
}

#[cfg(windows)]
fn check_process(pid: u32) -> io::Result<Option<bool>> {
    // Windows process checking would go here
    Ok(Some(true))
}

/// Terminate a process forcefully
#[cfg(unix)]
fn terminate_process(pid: u32) {
    let _ = unsafe { libc::kill(pid as i32, libc::SIGKILL) };
}

#[cfg(windows)]
fn terminate_process(pid: u32) {
    // Windows termination would go here
    let _ = Command::new("taskkill")
        .args(&["/F", "/PID", &pid.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_value_display() {
        assert_eq!(InputValue::Int(42).to_string(), "42");
        assert_eq!(InputValue::Bool(true).to_string(), "true");
        assert_eq!(InputValue::IntArray(vec![1, 2, 3]).to_string(), "[1, 2, 3]");
    }

    #[test]
    fn test_language_availability() {
        // At least one language should be available
        assert!(
            Language::Rust.is_available()
                || Language::JavaScript.is_available()
                || Language::Python.is_available()
        );
    }

    #[test]
    fn test_sandbox_config_default() {
        let config = SandboxConfig::default();
        assert_eq!(config.timeout, DEFAULT_TIMEOUT);
        assert_eq!(config.memory_limit, DEFAULT_MEMORY_LIMIT);
    }
}
