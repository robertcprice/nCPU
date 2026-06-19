//! Web Workers and Worklets for nCPU/nSynth
//!
//! Comprehensive implementation of browser worker APIs including:
//! - WebWorker: Dedicated workers for off-main-thread computation
//! - WorkerPool: Pool of workers for load balancing
//! - SharedWorker: Workers shared across multiple browsing contexts
//! - AudioWorklet: Audio processing thread with custom processors
//! - PaintWorklet: Custom painting for CSS Houdini
//!
//! # Example
//!
//! ```rust
//! use nsynth::http::workers::*;
//!
//! // Create a worker pool
//! let mut pool = WorkerPool::new(4);
//! pool.add_worker("worker-1", "const add = (a, b) => a + b;");
//!
//! // Execute work
//! let result = pool.execute("worker-1", WorkerMessage::compute([2, 3])).await?;
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Mutex, RwLock};

// ============================================================================
// Common Types
// ============================================================================

/// Worker message for communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerMessage {
    /// Message ID for correlation
    pub id: String,
    /// Message type
    pub message_type: WorkerMessageType,
    /// Message payload
    pub payload: serde_json::Value,
}

impl WorkerMessage {
    /// Create a new worker message
    pub fn new(message_type: WorkerMessageType, payload: serde_json::Value) -> Self {
        Self {
            id: uuid_v4(),
            message_type,
            payload,
        }
    }

    /// Create a compute message
    pub fn compute(data: Vec<i32>) -> Self {
        Self::new(
            WorkerMessageType::Compute,
            serde_json::json!({ "data": data }),
        )
    }

    /// Create a result message
    pub fn result(data: serde_json::Value) -> Self {
        Self::new(WorkerMessageType::Result, data)
    }

    /// Create an error message
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(
            WorkerMessageType::Error,
            serde_json::json!({ "message": message.into() }),
        )
    }

    /// Create a terminate message
    pub fn terminate() -> Self {
        Self::new(WorkerMessageType::Terminate, serde_json::Value::Null)
    }

    /// Create a ping message
    pub fn ping() -> Self {
        Self::new(WorkerMessageType::Ping, serde_json::Value::Null)
    }

    /// Create a pong message
    pub fn pong() -> Self {
        Self::new(WorkerMessageType::Pong, serde_json::Value::Null)
    }
}

/// Worker message types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerMessageType {
    /// Compute request
    Compute,
    /// Compute result
    Result,
    /// Error occurred
    Error,
    /// Terminate worker
    Terminate,
    /// Ping/heartbeat
    Ping,
    /// Pong response
    Pong,
    /// Custom message
    Custom,
}

/// Worker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    /// Worker is being initialized
    Initializing,
    /// Worker is ready to process
    Ready,
    /// Worker is processing a task
    Busy,
    /// Worker is terminating
    Terminating,
    /// Worker has terminated
    Terminated,
    /// Worker encountered an error
    Errored,
}

// ============================================================================
// Web Worker
// ============================================================================

/// Dedicated Web Worker for off-main-thread computation
#[derive(Debug)]
pub struct WebWorker {
    /// Unique worker ID
    pub id: String,
    /// Worker URL or script content
    pub script: String,
    /// Current worker state
    pub state: WorkerState,
    /// Message channel for sending messages to worker
    pub sender: mpsc::UnboundedSender<WorkerMessage>,
    /// Message channel for receiving messages from worker
    pub receiver: mpsc::UnboundedReceiver<WorkerMessage>,
    /// Number of messages processed
    pub messages_processed: u64,
    /// Number of errors encountered
    pub errors: u64,
    /// Worker creation timestamp
    pub created_at: i64,
    /// Last activity timestamp
    pub last_activity: i64,
}

impl Clone for WebWorker {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            script: self.script.clone(),
            state: self.state,
            sender: self.sender.clone(),
            receiver: mpsc::unbounded_channel().1,
            messages_processed: self.messages_processed,
            errors: self.errors,
            created_at: self.created_at,
            last_activity: self.last_activity,
        }
    }
}

impl WebWorker {
    /// Create a new WebWorker with the given script
    pub fn new(script: impl Into<String>) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let script = script.into();

        Self {
            id: uuid_v4(),
            script,
            state: WorkerState::Initializing,
            sender: tx,
            receiver: rx,
            messages_processed: 0,
            errors: 0,
            created_at: chrono_timestamp(),
            last_activity: chrono_timestamp(),
        }
    }

    /// Create a WebWorker from a URL
    pub fn from_url(url: impl Into<String>) -> Self {
        Self::new(url)
    }

    /// Set worker state
    pub fn set_state(&mut self, state: WorkerState) {
        self.state = state;
        self.last_activity = chrono_timestamp();
    }

    /// Post a message to the worker
    pub fn post_message(&self, message: WorkerMessage) -> Result<(), WorkerError> {
        if self.state == WorkerState::Terminated || self.state == WorkerState::Terminating {
            return Err(WorkerError::WorkerTerminated);
        }

        self.sender
            .send(message)
            .map_err(|_| WorkerError::ChannelClosed)?;
        Ok(())
    }

    /// Receive a message from the worker (blocking)
    pub async fn receive_message(&mut self) -> Option<WorkerMessage> {
        self.receiver.recv().await
    }

    /// Terminate the worker
    pub fn terminate(&mut self) {
        self.set_state(WorkerState::Terminating);
        let _ = self.post_message(WorkerMessage::terminate());
        self.set_state(WorkerState::Terminated);
    }

    /// Send a ping message
    pub fn ping(&self) -> Result<(), WorkerError> {
        self.post_message(WorkerMessage::ping())
    }

    /// Check if worker is alive
    pub fn is_alive(&self) -> bool {
        matches!(
            self.state,
            WorkerState::Ready | WorkerState::Busy | WorkerState::Initializing
        )
    }

    /// Check if worker is available for work
    pub fn is_available(&self) -> bool {
        self.state == WorkerState::Ready
    }

    /// Get uptime in milliseconds
    pub fn uptime(&self) -> i64 {
        chrono_timestamp() - self.created_at
    }

    /// Get idle time in milliseconds
    pub fn idle_time(&self) -> i64 {
        chrono_timestamp() - self.last_activity
    }

    /// Increment processed counter
    pub fn increment_processed(&mut self) {
        self.messages_processed += 1;
        self.last_activity = chrono_timestamp();
    }

    /// Increment error counter
    pub fn increment_errors(&mut self) {
        self.errors += 1;
    }
}

/// Worker error types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerError {
    /// Worker has been terminated
    WorkerTerminated,
    /// Communication channel closed
    ChannelClosed,
    /// Script evaluation failed
    ScriptError(String),
    /// Message serialization failed
    SerializationError,
    /// Worker initialization failed
    InitializationFailed(String),
    /// Worker not found
    WorkerNotFound,
}

// ============================================================================
// Worker Pool
// ============================================================================

/// Pool of WebWorkers for load balancing
#[derive(Debug)]
pub struct WorkerPool {
    /// Workers in the pool
    workers: Arc<RwLock<HashMap<String, WebWorker>>>,
    /// Task queue
    task_queue: Arc<Mutex<VecDeque<WorkerTask>>>,
    /// Maximum pool size
    max_size: usize,
    /// Task result channel
    result_tx: mpsc::UnboundedSender<WorkerTaskResult>,
}

/// Task to be executed by a worker
#[derive(Debug, Clone)]
pub struct WorkerTask {
    /// Task ID
    pub id: String,
    /// Target worker ID (optional, will be assigned if None)
    pub worker_id: Option<String>,
    /// Task message
    pub message: WorkerMessage,
    /// Task priority
    pub priority: TaskPriority,
}

/// Task priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Task execution result
#[derive(Debug, Clone)]
pub struct WorkerTaskResult {
    /// Task ID
    pub task_id: String,
    /// Worker ID that executed the task
    pub worker_id: String,
    /// Result message
    pub result: Option<WorkerMessage>,
    /// Error if execution failed
    pub error: Option<WorkerError>,
}

impl WorkerPool {
    /// Create a new worker pool
    pub fn new(max_size: usize) -> Self {
        let (result_tx, _) = mpsc::unbounded_channel();

        Self {
            workers: Arc::new(RwLock::new(HashMap::new())),
            task_queue: Arc::new(Mutex::new(VecDeque::new())),
            max_size,
            result_tx,
        }
    }

    /// Add a worker to the pool
    pub async fn add_worker(&self, id: impl Into<String>, script: impl Into<String>) -> Result<(), WorkerError> {
        let mut workers = self.workers.write().await;
        let id = id.into();

        if workers.len() >= self.max_size {
            return Err(WorkerError::InitializationFailed("Pool is full".to_string()));
        }

        let mut worker = WebWorker::new(script);
        worker.set_state(WorkerState::Ready);
        workers.insert(id.clone(), worker);

        Ok(())
    }

    /// Remove a worker from the pool
    pub async fn remove_worker(&self, id: &str) -> Result<(), WorkerError> {
        let mut workers = self.workers.write().await;

        if let Some(mut worker) = workers.remove(id) {
            worker.terminate();
            Ok(())
        } else {
            Err(WorkerError::WorkerNotFound)
        }
    }

    /// Get a worker by ID
    pub async fn get_worker(&self, id: &str) -> Option<WebWorker> {
        self.workers.read().await.get(id).cloned()
    }

    /// Get all workers
    pub async fn get_all_workers(&self) -> Vec<WebWorker> {
        self.workers
            .read()
            .await
            .values()
            .cloned()
            .collect()
    }

    /// Get available (ready) workers
    pub async fn available_workers(&self) -> Vec<WebWorker> {
        self.workers
            .read()
            .await
            .values()
            .filter(|w| w.is_available())
            .cloned()
            .collect()
    }

    /// Get worker count
    pub async fn worker_count(&self) -> usize {
        self.workers.read().await.len()
    }

    /// Submit a task to the pool
    pub async fn submit(&self, task: WorkerTask) -> Result<(), WorkerError> {
        let mut queue = self.task_queue.lock().await;
        queue.push_back(task);
        Ok(())
    }

    /// Execute a task on a specific worker
    pub async fn execute(
        &self,
        worker_id: &str,
        message: WorkerMessage,
    ) -> Result<WorkerMessage, WorkerError> {
        let workers = self.workers.read().await;
        let worker = workers.get(worker_id).ok_or(WorkerError::WorkerNotFound)?;

        if !worker.is_available() {
            return Err(WorkerError::WorkerTerminated);
        }

        worker.post_message(message)?;

        // In a real implementation, we would wait for the result
        // For now, return a placeholder
        Ok(WorkerMessage::result(serde_json::json!({})))
    }

    /// Execute a task and return the result asynchronously
    pub async fn execute_async(&self, task: WorkerTask) -> Result<WorkerTaskResult, WorkerError> {
        let worker_id = if let Some(id) = task.worker_id.clone() {
            id
        } else {
            // Find an available worker
            let available = self.available_workers().await;
            if available.is_empty() {
                // Queue the task
                self.submit(task).await?;
                return Err(WorkerError::WorkerTerminated); // No workers available
            }
            available[0].id.clone()
        };

        let result = self.execute(&worker_id, task.message).await?;

        Ok(WorkerTaskResult {
            task_id: task.id,
            worker_id,
            result: Some(result),
            error: None,
        })
    }

    /// Get pending task count
    pub async fn pending_count(&self) -> usize {
        self.task_queue.lock().await.len()
    }

    /// Process queued tasks
    pub async fn process_queue(&self) -> Result<usize, WorkerError> {
        let mut processed = 0;
        let mut queue = self.task_queue.lock().await;

        while let Some(task) = queue.pop_front() {
            if let Ok(_) = self.execute_async(task).await {
                processed += 1;
            }
        }

        Ok(processed)
    }

    /// Shutdown all workers in the pool
    pub async fn shutdown(&self) -> Result<(), WorkerError> {
        let mut workers = self.workers.write().await;

        for (_, mut worker) in workers.iter_mut() {
            worker.terminate();
        }

        workers.clear();
        Ok(())
    }

    /// Get pool statistics
    pub async fn stats(&self) -> WorkerPoolStats {
        let workers = self.workers.read().await;
        let total = workers.len();
        let available = workers.values().filter(|w| w.is_available()).count();
        let busy = workers.values().filter(|w| w.state == WorkerState::Busy).count();
        let errored = workers.values().filter(|w| w.state == WorkerState::Errored).count();

        WorkerPoolStats {
            total_workers: total,
            available_workers: available,
            busy_workers: busy,
            errored_workers: errored,
            pending_tasks: self.pending_count().await,
        }
    }
}

/// Worker pool statistics
#[derive(Debug, Clone)]
pub struct WorkerPoolStats {
    pub total_workers: usize,
    pub available_workers: usize,
    pub busy_workers: usize,
    pub errored_workers: usize,
    pub pending_tasks: usize,
}

// ============================================================================
// Shared Worker
// ============================================================================

/// Shared worker accessible from multiple browsing contexts
#[derive(Debug, Clone)]
pub struct SharedWorker {
    /// Unique shared worker ID
    pub id: String,
    /// Worker name
    pub name: String,
    /// Worker script
    pub script: String,
    /// Connected ports (browsing contexts)
    pub ports: Vec<SharedWorkerPort>,
    /// Worker state
    pub state: WorkerState,
    /// Message broadcast channel
    pub broadcast: mpsc::UnboundedSender<SharedWorkerMessage>,
    /// Creation timestamp
    pub created_at: i64,
}

/// Port for communicating with a shared worker
#[derive(Debug, Clone)]
pub struct SharedWorkerPort {
    /// Unique port ID
    pub id: String,
    /// Origin of the connected context
    pub origin: String,
    /// Message channel
    pub channel: mpsc::UnboundedSender<SharedWorkerMessage>,
    /// Whether the port is started
    pub started: bool,
}

/// Message for shared worker communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedWorkerMessage {
    /// Source port ID
    pub port_id: String,
    /// Message data
    pub data: serde_json::Value,
    /// Message timestamp
    pub timestamp: i64,
}

impl SharedWorker {
    /// Create a new SharedWorker
    pub fn new(name: impl Into<String>, script: impl Into<String>) -> Self {
        let (tx, _) = mpsc::unbounded_channel();

        Self {
            id: uuid_v4(),
            name: name.into(),
            script: script.into(),
            ports: Vec::new(),
            state: WorkerState::Initializing,
            broadcast: tx,
            created_at: chrono_timestamp(),
        }
    }

    /// Connect a new port to this shared worker
    pub fn connect(&mut self, origin: impl Into<String>) -> SharedWorkerPort {
        let (tx, _) = mpsc::unbounded_channel();
        let port = SharedWorkerPort {
            id: uuid_v4(),
            origin: origin.into(),
            channel: tx,
            started: false,
        };
        self.ports.push(port.clone());
        port
    }

    /// Disconnect a port
    pub fn disconnect(&mut self, port_id: &str) -> bool {
        if let Some(pos) = self.ports.iter().position(|p| p.id == port_id) {
            self.ports.remove(pos);
            true
        } else {
            false
        }
    }

    /// Broadcast a message to all ports
    pub fn broadcast(&self, message: SharedWorkerMessage) -> Result<usize, WorkerError> {
        let mut sent = 0;

        for port in &self.ports {
            if port.started {
                if port.channel.send(message.clone()).is_ok() {
                    sent += 1;
                }
            }
        }

        Ok(sent)
    }

    /// Send a message to a specific port
    pub fn send_to_port(
        &self,
        port_id: &str,
        message: SharedWorkerMessage,
    ) -> Result<(), WorkerError> {
        let port = self
            .ports
            .iter()
            .find(|p| p.id == port_id)
            .ok_or(WorkerError::WorkerNotFound)?;

        if !port.started {
            return Err(WorkerError::WorkerTerminated);
        }

        port.channel
            .send(message)
            .map_err(|_| WorkerError::ChannelClosed)?;
        Ok(())
    }

    /// Start a port (enable message receiving)
    pub fn start_port(&mut self, port_id: &str) -> Result<(), WorkerError> {
        if let Some(port) = self.ports.iter_mut().find(|p| p.id == port_id) {
            port.started = true;
            self.state = WorkerState::Ready;
            Ok(())
        } else {
            Err(WorkerError::WorkerNotFound)
        }
    }

    /// Get connected port count
    pub fn port_count(&self) -> usize {
        self.ports.len()
    }

    /// Get active (started) port count
    pub fn active_port_count(&self) -> usize {
        self.ports.iter().filter(|p| p.started).count()
    }

    /// Terminate the shared worker
    pub fn terminate(&mut self) {
        self.state = WorkerState::Terminating;
        self.ports.clear();
        self.state = WorkerState::Terminated;
    }
}

// ============================================================================
// Audio Worklet
// ============================================================================

/// Audio worklet for custom audio processing
#[derive(Debug, Clone)]
pub struct AudioWorklet {
    /// Worklet name
    pub name: String,
    /// Processor definitions
    pub processors: HashMap<String, AudioWorkletProcessor>,
    /// Active processor instances
    pub instances: HashMap<String, AudioWorkletInstance>,
    /// Worklet state
    pub state: WorkerState,
}

/// Audio worklet processor definition
#[derive(Debug, Clone)]
pub struct AudioWorkletProcessor {
    /// Processor name
    pub name: String,
    /// Processor code (JavaScript)
    pub code: String,
    /// Parameter descriptors
    pub parameter_descriptors: Vec<ParameterDescriptor>,
    /// Input channel count
    pub input_channels: u32,
    /// Output channel count
    pub output_channels: u32,
}

/// Audio parameter descriptor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDescriptor {
    /// Parameter name
    pub name: String,
    /// Minimum value
    pub min_value: f32,
    /// Maximum value
    pub max_value: f32,
    /// Default value
    pub default_value: f32,
}

/// Active audio worklet processor instance
#[derive(Debug, Clone)]
pub struct AudioWorkletInstance {
    /// Instance ID
    pub id: String,
    /// Processor type
    pub processor_name: String,
    /// Current parameter values
    pub parameters: HashMap<String, f32>,
    /// Instance state
    pub state: ProcessorState,
    /// Audio buffer size
    pub buffer_size: usize,
}

/// Processor instance state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessorState {
    Running,
    Suspended,
    Closed,
}

impl AudioWorklet {
    /// Create a new AudioWorklet
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            processors: HashMap::new(),
            instances: HashMap::new(),
            state: WorkerState::Initializing,
        }
    }

    /// Add a processor definition
    pub fn add_processor(&mut self, processor: AudioWorkletProcessor) -> Result<(), WorkerError> {
        self.processors.insert(processor.name.clone(), processor);
        Ok(())
    }

    /// Load a processor from code
    pub fn load_processor(
        &mut self,
        name: impl Into<String>,
        code: impl Into<String>,
    ) -> Result<(), WorkerError> {
        let processor = AudioWorkletProcessor {
            name: name.into(),
            code: code.into(),
            parameter_descriptors: Vec::new(),
            input_channels: 2,
            output_channels: 2,
        };

        self.add_processor(processor)
    }

    /// Create a processor instance
    pub fn create_instance(
        &mut self,
        processor_name: &str,
        buffer_size: usize,
    ) -> Result<AudioWorkletInstance, WorkerError> {
        let processor = self
            .processors
            .get(processor_name)
            .ok_or(WorkerError::WorkerNotFound)?;

        let instance = AudioWorkletInstance {
            id: uuid_v4(),
            processor_name: processor_name.to_string(),
            parameters: processor
                .parameter_descriptors
                .iter()
                .map(|p| (p.name.clone(), p.default_value))
                .collect(),
            state: ProcessorState::Running,
            buffer_size,
        };

        self.instances.insert(instance.id.clone(), instance.clone());
        Ok(instance)
    }

    /// Process audio through an instance
    pub fn process(
        &mut self,
        instance_id: &str,
        input: &[f32],
    ) -> Result<Vec<f32>, WorkerError> {
        let instance = self
            .instances
            .get_mut(instance_id)
            .ok_or(WorkerError::WorkerNotFound)?;

        if instance.state != ProcessorState::Running {
            return Err(WorkerError::WorkerTerminated);
        }

        // In a real implementation, this would execute the processor code
        // For now, pass through
        Ok(input.to_vec())
    }

    /// Set a parameter value
    pub fn set_parameter(
        &mut self,
        instance_id: &str,
        name: &str,
        value: f32,
    ) -> Result<(), WorkerError> {
        let instance = self
            .instances
            .get_mut(instance_id)
            .ok_or(WorkerError::WorkerNotFound)?;

        instance.parameters.insert(name.to_string(), value);
        Ok(())
    }

    /// Get a parameter value
    pub fn get_parameter(&self, instance_id: &str, name: &str) -> Option<f32> {
        self.instances
            .get(instance_id)
            .and_then(|i| i.parameters.get(name).copied())
    }

    /// Suspend a processor instance
    pub fn suspend_instance(&mut self, instance_id: &str) -> Result<(), WorkerError> {
        let instance = self
            .instances
            .get_mut(instance_id)
            .ok_or(WorkerError::WorkerNotFound)?;

        instance.state = ProcessorState::Suspended;
        Ok(())
    }

    /// Resume a processor instance
    pub fn resume_instance(&mut self, instance_id: &str) -> Result<(), WorkerError> {
        let instance = self
            .instances
            .get_mut(instance_id)
            .ok_or(WorkerError::WorkerNotFound)?;

        instance.state = ProcessorState::Running;
        Ok(())
    }

    /// Close a processor instance
    pub fn close_instance(&mut self, instance_id: &str) -> Result<(), WorkerError> {
        let instance = self
            .instances
            .get_mut(instance_id)
            .ok_or(WorkerError::WorkerNotFound)?;

        instance.state = ProcessorState::Closed;
        self.instances.remove(instance_id);
        Ok(())
    }

    /// Get all instances
    pub fn instances(&self) -> Vec<AudioWorkletInstance> {
        self.instances.values().cloned().collect()
    }
}

// ============================================================================
// Paint Worklet
// ============================================================================

/// Paint worklet for custom CSS painting
#[derive(Debug, Clone)]
pub struct PaintWorklet {
    /// Worklet name
    pub name: String,
    /// Registered paint classes
    pub paint_classes: HashMap<String, PaintClass>,
    /// Worklet state
    pub state: WorkerState,
}

/// Paint class definition
#[derive(Debug, Clone)]
pub struct PaintClass {
    /// Class name
    pub name: String,
    /// Painter code (JavaScript)
    pub code: String,
    /// Input properties
    pub input_properties: Vec<String>,
    /// Input arguments
    pub input_arguments: Vec<ArgumentDescriptor>,
    /// Alpha flag
    pub alpha: bool,
    /// Context alpha flag
    pub context_alpha: bool,
}

/// Argument descriptor for paint worklet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgumentDescriptor {
    /// Argument name
    pub name: String,
    /// Argument type
    pub type_name: String,
    /// Whether argument is required
    pub required: bool,
    /// Default value
    pub default_value: Option<serde_json::Value>,
}

impl PaintWorklet {
    /// Create a new PaintWorklet
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            paint_classes: HashMap::new(),
            state: WorkerState::Initializing,
        }
    }

    /// Register a paint class
    pub fn register_class(&mut self, class: PaintClass) -> Result<(), WorkerError> {
        self.paint_classes.insert(class.name.clone(), class);
        self.state = WorkerState::Ready;
        Ok(())
    }

    /// Register a paint class from code
    pub fn register_from_code(
        &mut self,
        name: impl Into<String>,
        code: impl Into<String>,
    ) -> Result<(), WorkerError> {
        let class = PaintClass {
            name: name.into(),
            code: code.into(),
            input_properties: Vec::new(),
            input_arguments: Vec::new(),
            alpha: false,
            context_alpha: false,
        };

        self.register_class(class)
    }

    /// Get a paint class by name
    pub fn get_class(&self, name: &str) -> Option<&PaintClass> {
        self.paint_classes.get(name)
    }

    /// Paint a rectangle
    pub fn paint(
        &self,
        class_name: &str,
        width: f32,
        height: f32,
        properties: &HashMap<String, String>,
    ) -> Result<PaintOutput, WorkerError> {
        let class = self
            .paint_classes
            .get(class_name)
            .ok_or(WorkerError::WorkerNotFound)?;

        // In a real implementation, this would execute the paint code
        // For now, return a placeholder
        Ok(PaintOutput {
            class_name: class_name.to_string(),
            width,
            height,
            data: vec![0; (width as usize * height as usize * 4)], // RGBA
            alpha: class.alpha,
        })
    }

    /// Get all registered class names
    pub fn class_names(&self) -> Vec<String> {
        self.paint_classes.keys().cloned().collect()
    }

    /// Check if a class is registered
    pub fn has_class(&self, name: &str) -> bool {
        self.paint_classes.contains_key(name)
    }
}

/// Output from a paint operation
#[derive(Debug, Clone)]
pub struct PaintOutput {
    /// Class name that produced the output
    pub class_name: String,
    /// Output width
    pub width: f32,
    /// Output height
    pub height: f32,
    /// Pixel data (RGBA)
    pub data: Vec<u8>,
    /// Whether output has alpha
    pub alpha: bool,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate UUID v4
fn uuid_v4() -> String {
    format!(
        "{:04x}{:04x}-{:04x}-{:04x}-{:04x}-{:04x}{:04x}{:04x}",
        rand::random::<u16>(),
        rand::random::<u16>(),
        rand::random::<u16>(),
        rand::random::<u16>(),
        rand::random::<u16>(),
        rand::random::<u16>(),
        rand::random::<u16>(),
        rand::random::<u16>()
    )
}

/// Get current timestamp (milliseconds since epoch)
fn chrono_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_message_creation() {
        let msg = WorkerMessage::compute(vec![1, 2, 3]);
        assert_eq!(msg.message_type, WorkerMessageType::Compute);

        let result = WorkerMessage::result(serde_json::json!({"value": 42}));
        assert_eq!(result.message_type, WorkerMessageType::Result);

        let err = WorkerMessage::error("Test error");
        assert_eq!(err.message_type, WorkerMessageType::Error);

        let term = WorkerMessage::terminate();
        assert_eq!(term.message_type, WorkerMessageType::Terminate);

        let ping = WorkerMessage::ping();
        assert_eq!(ping.message_type, WorkerMessageType::Ping);

        let pong = WorkerMessage::pong();
        assert_eq!(pong.message_type, WorkerMessageType::Pong);
    }

    #[test]
    fn test_web_worker_creation() {
        let worker = WebWorker::new("const add = (a, b) => a + b;");
        assert_eq!(worker.script, "const add = (a, b) => a + b;");
        assert_eq!(worker.state, WorkerState::Initializing);
        assert!(worker.is_alive());
        assert!(!worker.is_available()); // Still initializing

        let mut worker = worker;
        worker.set_state(WorkerState::Ready);
        assert!(worker.is_available());
    }

    #[test]
    fn test_web_worker_post_message() {
        let worker = WebWorker::new("test");
        let msg = WorkerMessage::ping();

        assert!(worker.post_message(msg).is_ok());
    }

    #[test]
    fn test_web_worker_terminate() {
        let mut worker = WebWorker::new("test");
        worker.terminate();
        assert_eq!(worker.state, WorkerState::Terminated);
        assert!(!worker.is_alive());
    }

    #[test]
    fn test_web_worker_from_url() {
        let worker = WebWorker::from_url("/worker.js");
        assert_eq!(worker.script, "/worker.js");
    }

    #[test]
    fn test_web_worker_uptime() {
        let worker = WebWorker::new("test");
        assert!(worker.uptime() >= 0);
    }

    #[test]
    fn test_web_worker_counters() {
        let mut worker = WebWorker::new("test");
        worker.increment_processed();
        assert_eq!(worker.messages_processed, 1);

        worker.increment_errors();
        assert_eq!(worker.errors, 1);
    }

    #[test]
    fn test_worker_pool_creation() {
        let pool = WorkerPool::new(4);
        assert_eq!(pool.max_size, 4);
    }

    #[tokio::test]
    async fn test_worker_pool_add_worker() {
        let pool = WorkerPool::new(4);
        assert!(pool.add_worker("worker-1", "test script").await.is_ok());
        assert_eq!(pool.worker_count().await, 1);

        let worker = pool.get_worker("worker-1").await;
        assert!(worker.is_some());
    }

    #[tokio::test]
    async fn test_worker_pool_remove_worker() {
        let pool = WorkerPool::new(4);
        pool.add_worker("worker-1", "test script").await.unwrap();

        assert!(pool.remove_worker("worker-1").await.is_ok());
        assert_eq!(pool.worker_count().await, 0);
    }

    #[tokio::test]
    async fn test_worker_pool_available_workers() {
        let pool = WorkerPool::new(4);
        pool.add_worker("worker-1", "test script").await.unwrap();

        let available = pool.available_workers().await;
        assert_eq!(available.len(), 1);
    }

    #[tokio::test]
    async fn test_worker_pool_stats() {
        let pool = WorkerPool::new(4);
        pool.add_worker("worker-1", "test script").await.unwrap();

        let stats = pool.stats().await;
        assert_eq!(stats.total_workers, 1);
        assert_eq!(stats.available_workers, 1);
        assert_eq!(stats.busy_workers, 0);
    }

    #[tokio::test]
    async fn test_worker_pool_shutdown() {
        let pool = WorkerPool::new(4);
        pool.add_worker("worker-1", "test script").await.unwrap();

        assert!(pool.shutdown().await.is_ok());
        assert_eq!(pool.worker_count().await, 0);
    }

    #[test]
    fn test_shared_worker_creation() {
        let worker = SharedWorker::new("test-worker", "const x = 42;");
        assert_eq!(worker.name, "test-worker");
        assert_eq!(worker.script, "const x = 42;");
        assert_eq!(worker.state, WorkerState::Initializing);
    }

    #[test]
    fn test_shared_worker_connect() {
        let mut worker = SharedWorker::new("test", "script");
        let port = worker.connect("https://example.com");

        assert_eq!(port.origin, "https://example.com");
        assert!(!port.started);
        assert_eq!(worker.port_count(), 1);
    }

    #[test]
    fn test_shared_worker_start_port() {
        let mut worker = SharedWorker::new("test", "script");
        let port = worker.connect("https://example.com");
        let port_id = port.id.clone();

        assert!(worker.start_port(&port_id).is_ok());
        assert!(worker.ports.iter().any(|p| p.id == port_id && p.started));
    }

    #[test]
    fn test_shared_worker_disconnect() {
        let mut worker = SharedWorker::new("test", "script");
        let port = worker.connect("https://example.com");
        let port_id = port.id.clone();

        assert!(worker.disconnect(&port_id));
        assert_eq!(worker.port_count(), 0);
    }

    #[test]
    fn test_shared_worker_active_port_count() {
        let mut worker = SharedWorker::new("test", "script");
        let port1 = worker.connect("https://example.com");
        let port2 = worker.connect("https://another.com");

        let _ = worker.start_port(&port1.id);

        assert_eq!(worker.active_port_count(), 1);
        assert_eq!(worker.port_count(), 2);
    }

    #[test]
    fn test_shared_worker_terminate() {
        let mut worker = SharedWorker::new("test", "script");
        worker.connect("https://example.com");
        worker.terminate();

        assert_eq!(worker.state, WorkerState::Terminated);
        assert_eq!(worker.port_count(), 0);
    }

    #[test]
    fn test_audio_worklet_creation() {
        let worklet = AudioWorklet::new("test-worklet");
        assert_eq!(worklet.name, "test-worklet");
        assert_eq!(worklet.state, WorkerState::Initializing);
    }

    #[test]
    fn test_audio_worklet_add_processor() {
        let mut worklet = AudioWorklet::new("test");
        let processor = AudioWorkletProcessor {
            name: "gain".to_string(),
            code: "class GainProcessor extends AudioWorkletProcessor {}".to_string(),
            parameter_descriptors: vec![ParameterDescriptor {
                name: "gain".to_string(),
                min_value: 0.0,
                max_value: 2.0,
                default_value: 1.0,
            }],
            input_channels: 2,
            output_channels: 2,
        };

        assert!(worklet.add_processor(processor).is_ok());
        assert!(worklet.processors.contains_key("gain"));
    }

    #[test]
    fn test_audio_worklet_load_processor() {
        let mut worklet = AudioWorklet::new("test");
        assert!(worklet
            .load_processor("gain", "class GainProcessor extends AudioWorkletProcessor {}")
            .is_ok());

        assert!(worklet.processors.contains_key("gain"));
    }

    #[test]
    fn test_audio_worklet_create_instance() {
        let mut worklet = AudioWorklet::new("test");
        worklet.load_processor("gain", "class GainProcessor {}").unwrap();

        let instance = worklet.create_instance("gain", 128);
        assert!(instance.is_ok());

        let instance = instance.unwrap();
        assert_eq!(instance.processor_name, "gain");
        assert_eq!(instance.buffer_size, 128);
        assert_eq!(instance.state, ProcessorState::Running);
    }

    #[test]
    fn test_audio_worklet_process() {
        let mut worklet = AudioWorklet::new("test");
        worklet.load_processor("gain", "class GainProcessor {}").unwrap();

        let instance = worklet.create_instance("gain", 128).unwrap();
        let instance_id = instance.id.clone();

        let input = vec![0.5, 0.6, 0.7, 0.8];
        let result = worklet.process(&instance_id, &input);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), input.len());
    }

    #[test]
    fn test_audio_worklet_set_parameter() {
        let mut worklet = AudioWorklet::new("test");
        let mut processor = AudioWorkletProcessor {
            name: "gain".to_string(),
            code: String::new(),
            parameter_descriptors: vec![ParameterDescriptor {
                name: "gain".to_string(),
                min_value: 0.0,
                max_value: 2.0,
                default_value: 1.0,
            }],
            input_channels: 2,
            output_channels: 2,
        };
        processor.parameter_descriptors[0].default_value = 1.0;

        worklet.add_processor(processor).unwrap();
        let instance = worklet.create_instance("gain", 128).unwrap();
        let instance_id = instance.id.clone();

        assert!(worklet.set_parameter(&instance_id, "gain", 0.5).is_ok());
        assert_eq!(worklet.get_parameter(&instance_id, "gain"), Some(0.5));
    }

    #[test]
    fn test_audio_worklet_suspend_resume() {
        let mut worklet = AudioWorklet::new("test");
        worklet.load_processor("gain", "class GainProcessor {}").unwrap();

        let instance = worklet.create_instance("gain", 128).unwrap();
        let instance_id = instance.id.clone();

        assert!(worklet.suspend_instance(&instance_id).is_ok());
        let instance = worklet.instances.get(&instance_id).unwrap();
        assert_eq!(instance.state, ProcessorState::Suspended);

        assert!(worklet.resume_instance(&instance_id).is_ok());
        let instance = worklet.instances.get(&instance_id).unwrap();
        assert_eq!(instance.state, ProcessorState::Running);
    }

    #[test]
    fn test_audio_worklet_close_instance() {
        let mut worklet = AudioWorklet::new("test");
        worklet.load_processor("gain", "class GainProcessor {}").unwrap();

        let instance = worklet.create_instance("gain", 128).unwrap();
        let instance_id = instance.id.clone();

        assert!(worklet.close_instance(&instance_id).is_ok());
        assert!(worklet.instances.get(&instance_id).is_none());
    }

    #[test]
    fn test_paint_worklet_creation() {
        let worklet = PaintWorklet::new("test-worklet");
        assert_eq!(worklet.name, "test-worklet");
        assert_eq!(worklet.state, WorkerState::Initializing);
    }

    #[test]
    fn test_paint_worklet_register_class() {
        let mut worklet = PaintWorklet::new("test");
        let class = PaintClass {
            name: "checkerboard".to_string(),
            code: "class Checkerboard {}".to_string(),
            input_properties: vec!["color".to_string()],
            input_arguments: vec![],
            alpha: false,
            context_alpha: false,
        };

        assert!(worklet.register_class(class).is_ok());
        assert_eq!(worklet.state, WorkerState::Ready);
        assert!(worklet.has_class("checkerboard"));
    }

    #[test]
    fn test_paint_worklet_register_from_code() {
        let mut worklet = PaintWorklet::new("test");
        assert!(worklet
            .register_from_code("circle", "class Circle extends PaintWorklet {}")
            .is_ok());

        assert!(worklet.has_class("circle"));
    }

    #[test]
    fn test_paint_worklet_get_class() {
        let mut worklet = PaintWorklet::new("test");
        worklet.register_from_code("circle", "class Circle {}").unwrap();

        let class = worklet.get_class("circle");
        assert!(class.is_some());
        assert_eq!(class.unwrap().name, "circle");
    }

    #[test]
    fn test_paint_worklet_paint() {
        let mut worklet = PaintWorklet::new("test");
        worklet.register_from_code("circle", "class Circle {}").unwrap();

        let props = HashMap::new();
        let result = worklet.paint("circle", 100.0, 100.0, &props);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.class_name, "circle");
        assert_eq!(output.width, 100.0);
        assert_eq!(output.height, 100.0);
    }

    #[test]
    fn test_paint_worklet_class_names() {
        let mut worklet = PaintWorklet::new("test");
        worklet.register_from_code("circle", "class Circle {}").unwrap();
        worklet.register_from_code("rect", "class Rect {}").unwrap();

        let names = worklet.class_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"circle".to_string()));
        assert!(names.contains(&"rect".to_string()));
    }

    #[test]
    fn test_task_priority_ord() {
        assert!(TaskPriority::Critical > TaskPriority::High);
        assert!(TaskPriority::High > TaskPriority::Normal);
        assert!(TaskPriority::Normal > TaskPriority::Low);
    }

    #[test]
    fn test_worker_message_id_unique() {
        let msg1 = WorkerMessage::compute(vec![1]);
        let msg2 = WorkerMessage::compute(vec![2]);

        assert_ne!(msg1.id, msg2.id);
    }

    #[test]
    fn test_shared_worker_message() {
        let msg = SharedWorkerMessage {
            port_id: uuid_v4(),
            data: serde_json::json!({"test": true}),
            timestamp: chrono_timestamp(),
        };

        assert_eq!(msg.data["test"], true);
    }

    #[test]
    fn test_parameter_descriptor() {
        let desc = ParameterDescriptor {
            name: "gain".to_string(),
            min_value: 0.0,
            max_value: 2.0,
            default_value: 1.0,
        };

        assert_eq!(desc.name, "gain");
        assert_eq!(desc.min_value, 0.0);
        assert_eq!(desc.max_value, 2.0);
        assert_eq!(desc.default_value, 1.0);
    }

    #[test]
    fn test_argument_descriptor() {
        let desc = ArgumentDescriptor {
            name: "size".to_string(),
            type_name: "float".to_string(),
            required: true,
            default_value: None,
        };

        assert_eq!(desc.name, "size");
        assert!(desc.required);
        assert!(desc.default_value.is_none());
    }

    #[test]
    fn test_paint_output() {
        let output = PaintOutput {
            class_name: "test".to_string(),
            width: 100.0,
            height: 100.0,
            data: vec![0u8; 40000],
            alpha: false,
        };

        assert_eq!(output.width, 100.0);
        assert_eq!(output.height, 100.0);
        assert_eq!(output.data.len(), 40000);
    }

    #[tokio::test]
    async fn test_worker_pool_execute() {
        let pool = WorkerPool::new(4);
        pool.add_worker("worker-1", "const add = (a, b) => a + b;").await.unwrap();

        let msg = WorkerMessage::compute(vec![2, 3]);
        let result = pool.execute("worker-1", msg).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_worker_pool_submit() {
        let pool = WorkerPool::new(4);

        let task = WorkerTask {
            id: uuid_v4(),
            worker_id: None,
            message: WorkerMessage::compute(vec![1, 2]),
            priority: TaskPriority::Normal,
        };

        assert!(pool.submit(task).await.is_ok());
        assert_eq!(pool.pending_count().await, 1);
    }

    #[tokio::test]
    async fn test_worker_pool_process_queue() {
        let pool = WorkerPool::new(4);
        pool.add_worker("worker-1", "test").await.unwrap();

        let task = WorkerTask {
            id: uuid_v4(),
            worker_id: Some("worker-1".to_string()),
            message: WorkerMessage::compute(vec![1]),
            priority: TaskPriority::Normal,
        };

        pool.submit(task).await.unwrap();
        let processed = pool.process_queue().await;

        assert!(processed.is_ok());
        assert_eq!(pool.pending_count().await, 0);
    }

    #[test]
    fn test_worker_error_variants() {
        let _ = WorkerError::WorkerTerminated;
        let _ = WorkerError::ChannelClosed;
        let _ = WorkerError::ScriptError("test".to_string());
        let _ = WorkerError::SerializationError;
        let _ = WorkerError::InitializationFailed("test".to_string());
        let _ = WorkerError::WorkerNotFound;
    }

    #[test]
    fn test_worker_state_transitions() {
        let mut worker = WebWorker::new("test");
        assert_eq!(worker.state, WorkerState::Initializing);

        worker.set_state(WorkerState::Ready);
        assert!(worker.is_available());

        worker.set_state(WorkerState::Busy);
        assert!(!worker.is_available());
        assert!(worker.is_alive());

        worker.set_state(WorkerState::Errored);
        assert!(!worker.is_alive());
    }

    #[tokio::test]
    async fn test_worker_pool_stats_detailed() {
        let pool = WorkerPool::new(10);
        let stats = pool.stats().await;

        assert_eq!(stats.total_workers, 0);
        assert_eq!(stats.available_workers, 0);
        assert_eq!(stats.busy_workers, 0);
        assert_eq!(stats.errored_workers, 0);
        assert_eq!(stats.pending_tasks, 0);
    }

    #[test]
    fn test_paint_class_alpha() {
        let mut worklet = PaintWorklet::new("test");
        let mut class = PaintClass {
            name: "transparent".to_string(),
            code: String::new(),
            input_properties: Vec::new(),
            input_arguments: Vec::new(),
            alpha: true,
            context_alpha: true,
        };

        worklet.register_class(class.clone()).unwrap();

        let retrieved = worklet.get_class("transparent").unwrap();
        assert!(retrieved.alpha);
        assert!(retrieved.context_alpha);
    }

    #[test]
    fn test_audio_worklet_instances() {
        let mut worklet = AudioWorklet::new("test");
        worklet.load_processor("p1", "code1").unwrap();
        worklet.load_processor("p2", "code2").unwrap();

        let i1 = worklet.create_instance("p1", 128).unwrap();
        let i2 = worklet.create_instance("p2", 256).unwrap();

        let instances = worklet.instances();
        assert_eq!(instances.len(), 2);

        let instance_ids: Vec<_> = instances.iter().map(|i| &i.id).collect();
        assert!(instance_ids.contains(&&i1.id));
        assert!(instance_ids.contains(&&i2.id));
    }

    #[tokio::test]
    async fn test_shared_worker_send_to_port() {
        let worker = SharedWorker::new("test", "script");
        let mut worker_clone = worker.clone();

        let port = worker_clone.connect("https://example.com");
        let port_id = port.id.clone();
        let _ = worker_clone.start_port(&port_id);

        let msg = SharedWorkerMessage {
            port_id: uuid_v4(),
            data: serde_json::json!({"test": true}),
            timestamp: chrono_timestamp(),
        };

        let result = worker.send_to_port(&port_id, msg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_worker_message_serialization() {
        let msg = WorkerMessage::compute(vec![1, 2, 3]);

        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: WorkerMessage = serde_json::from_str(&json).unwrap();

        assert_eq!(msg.id, deserialized.id);
        assert_eq!(msg.message_type, deserialized.message_type);
    }
}
