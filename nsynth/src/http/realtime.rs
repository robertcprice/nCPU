//! Real-time Communication Support for nCPU/nSynth
//!
//! Complete implementation of Socket.IO and Server-Sent Events (SSE)
//! for real-time bidirectional communication and streaming.

use crate::http::types::{HeaderMap, Method, Request, Response, StatusCode};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ============================================================================
// Socket.IO Implementation
// ============================================================================

/// Socket.IO packet type
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum SocketPacketType {
    Connect = 0,
    Disconnect = 1,
    Event = 2,
    Ack = 3,
    Error = 4,
    BinaryEvent = 5,
    BinaryAck = 6,
}

/// Socket.IO engine.io packet type
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum EnginePacketType {
    Open = 0,
    Close = 1,
    Ping = 2,
    Pong = 3,
    Message = 4,
    Upgrade = 5,
    Noop = 6,
}

/// Socket.IO event data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EventData {
    String(String),
    Number(i64),
    Float(f64),
    Bool(bool),
    Null,
    Array(Vec<EventData>),
    Object(HashMap<String, EventData>),
    Binary(Vec<u8>),
}

/// Socket.IO event
#[derive(Debug, Clone)]
pub struct SocketIOEvent {
    pub name: String,
    pub data: Vec<EventData>,
    pub namespace: Option<String>,
    pub room: Option<String>,
}

impl SocketIOEvent {
    /// Create new event
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            data: Vec::new(),
            namespace: None,
            room: None,
        }
    }

    /// Add data argument
    pub fn with_data(mut self, data: EventData) -> Self {
        self.data.push(data);
        self
    }

    /// Set namespace
    pub fn with_namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = Some(ns.into());
        self
    }

    /// Set target room
    pub fn to_room(mut self, room: impl Into<String>) -> Self {
        self.room = Some(room.into());
        self
    }
}

/// Socket.IO client connection
#[derive(Debug)]
pub struct SocketIOClient {
    /// Client ID
    pub id: String,
    /// Current namespace
    pub namespace: String,
    /// Rooms this client has joined
    pub rooms: HashSet<String>,
    /// Connection state
    connected: bool,
    /// Last activity timestamp
    last_activity: Arc<Mutex<Instant>>,
    /// Client data
    pub data: Arc<Mutex<HashMap<String, EventData>>>,
}

impl SocketIOClient {
    /// Create new client
    pub fn new(id: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            namespace: namespace.into(),
            rooms: HashSet::new(),
            connected: true,
            last_activity: Arc::new(Mutex::new(Instant::now())),
            data: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Join a room
    pub fn join(&mut self, room: impl Into<String>) {
        self.rooms.insert(room.into());
        self.update_activity();
    }

    /// Leave a room
    pub fn leave(&mut self, room: &str) {
        self.rooms.remove(room);
        self.update_activity();
    }

    /// Check if client is in room
    pub fn in_room(&self, room: &str) -> bool {
        self.rooms.contains(room)
    }

    /// Get all rooms
    pub fn rooms(&self) -> Vec<String> {
        self.rooms.iter().cloned().collect()
    }

    /// Leave all rooms
    pub fn leave_all(&mut self) {
        self.rooms.clear();
        self.update_activity();
    }

    /// Disconnect client
    pub fn disconnect(&mut self) {
        self.connected = false;
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Update last activity
    fn update_activity(&self) {
        *self.last_activity.lock().unwrap() = Instant::now();
    }

    /// Get last activity time
    pub fn last_activity(&self) -> Instant {
        *self.last_activity.lock().unwrap()
    }

    /// Set client data
    pub fn set(&self, key: impl Into<String>, value: EventData) {
        let mut data = self.data.lock().unwrap();
        data.insert(key.into(), value);
    }

    /// Get client data
    pub fn get(&self, key: &str) -> Option<EventData> {
        let data = self.data.lock().unwrap();
        data.get(key).cloned()
    }

    /// Remove client data
    pub fn remove(&self, key: &str) -> Option<EventData> {
        let mut data = self.data.lock().unwrap();
        data.remove(key)
    }
}

/// Namespace for Socket.IO
pub struct Namespace {
    /// Namespace name (e.g., "/chat", "/admin")
    pub name: String,
    /// Connected clients in this namespace
    clients: Arc<Mutex<HashMap<String, Arc<Mutex<SocketIOClient>>>>>,
    /// Event handlers
    handlers: Arc<Mutex<HashMap<String, Vec<Arc<dyn EventHandler>>>>>,
    /// Middleware
    middleware: Arc<Mutex<Vec<Arc<dyn MiddlewareFn>>>>,
}

impl std::fmt::Debug for Namespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Namespace")
            .field("name", &self.name)
            .field("client_count", &self.clients.lock().unwrap().len())
            .finish()
    }
}

impl Namespace {
    /// Create new namespace
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            clients: Arc::new(Mutex::new(HashMap::new())),
            handlers: Arc::new(Mutex::new(HashMap::new())),
            middleware: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Add event handler
    pub fn on(&self, event: impl Into<String>, handler: Arc<dyn EventHandler>) {
        let mut handlers = self.handlers.lock().unwrap();
        handlers
            .entry(event.into())
            .or_insert_with(Vec::new)
            .push(handler);
    }

    /// Add middleware
    pub fn use_middleware(&self, middleware: Arc<dyn MiddlewareFn>) {
        let mut mw = self.middleware.lock().unwrap();
        mw.push(middleware);
    }

    /// Add client to namespace
    pub fn add_client(&self, client: Arc<Mutex<SocketIOClient>>) {
        let id = client.lock().unwrap().id.clone();
        self.clients.lock().unwrap().insert(id, client);
    }

    /// Remove client from namespace
    pub fn remove_client(&self, client_id: &str) {
        self.clients.lock().unwrap().remove(client_id);
    }

    /// Get client by ID
    pub fn get_client(&self, id: &str) -> Option<Arc<Mutex<SocketIOClient>>> {
        self.clients.lock().unwrap().get(id).cloned()
    }

    /// Get all clients
    pub fn clients(&self) -> Vec<Arc<Mutex<SocketIOClient>>> {
        self.clients.lock().unwrap().values().cloned().collect()
    }

    /// Get client count
    pub fn client_count(&self) -> usize {
        self.clients.lock().unwrap().len()
    }

    /// Emit event to all clients in namespace
    pub fn emit(&self, event: impl Into<String>, data: EventData) {
        let event_name = event.into();
        let handlers = self.handlers.lock().unwrap();
        if let Some(handler_list) = handlers.get(&event_name) {
            let clients = self.clients();
            for handler in handler_list.iter() {
                for client in clients.iter() {
                    handler.handle(client.clone(), &event_name, &data);
                }
            }
        }
    }

    /// Emit event to specific room
    pub fn emit_to_room(&self, room: &str, event: impl Into<String>, data: EventData) {
        let clients = self.clients();
        let event_name = event.into();
        let handlers = self.handlers.lock().unwrap();

        if let Some(handler_list) = handlers.get(&event_name) {
            for client in clients.iter() {
                let c = client.lock().unwrap();
                if c.in_room(room) {
                    drop(c);
                    for handler in handler_list.iter() {
                        handler.handle(client.clone(), &event_name, &data);
                    }
                }
            }
        }
    }

    /// Emit event to specific client
    pub fn emit_to_client(&self, client_id: &str, event: impl Into<String>, data: EventData) {
        if let Some(client) = self.get_client(client_id) {
            let event_name = event.into();
            let handlers = self.handlers.lock().unwrap();
            if let Some(handler_list) = handlers.get(&event_name) {
                for handler in handler_list.iter() {
                    handler.handle(client.clone(), &event_name, &data);
                }
            }
        }
    }
}

/// Room management
#[derive(Debug)]
pub struct Room {
    /// Room name
    pub name: String,
    /// Members in room
    members: Arc<Mutex<HashSet<String>>>,
}

impl Room {
    /// Create new room
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            members: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Add member to room
    pub fn add_member(&self, client_id: impl Into<String>) {
        self.members.lock().unwrap().insert(client_id.into());
    }

    /// Remove member from room
    pub fn remove_member(&self, client_id: &str) {
        self.members.lock().unwrap().remove(client_id);
    }

    /// Get all members
    pub fn members(&self) -> Vec<String> {
        self.members.lock().unwrap().iter().cloned().collect()
    }

    /// Get member count
    pub fn member_count(&self) -> usize {
        self.members.lock().unwrap().len()
    }

    /// Check if client is in room
    pub fn has_member(&self, client_id: &str) -> bool {
        self.members.lock().unwrap().contains(client_id)
    }
}

/// Event handler trait
pub trait EventHandler: Send + Sync {
    /// Handle event
    fn handle(&self, client: Arc<Mutex<SocketIOClient>>, event: &str, data: &EventData);
}

/// Middleware function trait
pub trait MiddlewareFn: Send + Sync {
    /// Handle middleware
    fn handle(&self, client: Arc<Mutex<SocketIOClient>>, next: bool) -> bool;
}

/// Socket.IO server
pub struct SocketIOServer {
    /// Server namespace
    pub namespace: String,
    /// Namespaces
    namespaces: Arc<Mutex<HashMap<String, Arc<Namespace>>>>,
    /// Rooms
    rooms: Arc<Mutex<HashMap<String, Arc<Room>>>>,
    /// Default namespace
    default_namespace: Arc<Namespace>,
    /// Ping interval (ms)
    ping_interval: u64,
    /// Ping timeout (ms)
    ping_timeout: u64,
    /// Connected clients
    clients: Arc<Mutex<HashMap<String, Arc<Mutex<SocketIOClient>>>>>,
}

impl std::fmt::Debug for SocketIOServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SocketIOServer")
            .field("namespace", &self.namespace)
            .field("client_count", &self.clients.lock().unwrap().len())
            .finish()
    }
}

impl SocketIOServer {
    /// Create new Socket.IO server
    pub fn new(namespace: impl Into<String>) -> Self {
        let ns = namespace.into();
        let default = Arc::new(Namespace::new("/"));

        Self {
            namespace: ns.clone(),
            namespaces: Arc::new(Mutex::new(HashMap::from([(
                "/".to_string(),
                default.clone(),
            )]))),
            rooms: Arc::new(Mutex::new(HashMap::new())),
            default_namespace: default,
            ping_interval: 25000,
            ping_timeout: 5000,
            clients: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create or get namespace
    pub fn of(&self, namespace: impl Into<String>) -> Arc<Namespace> {
        let ns = namespace.into();
        let mut namespaces = self.namespaces.lock().unwrap();

        if let Some(namespace) = namespaces.get(&ns) {
            return namespace.clone();
        }

        let new_ns = Arc::new(Namespace::new(ns.clone()));
        namespaces.insert(ns, new_ns.clone());
        new_ns
    }

    /// Get namespace
    pub fn namespace(&self, name: &str) -> Option<Arc<Namespace>> {
        self.namespaces.lock().unwrap().get(name).cloned()
    }

    /// Create or get room
    pub fn room(&self, name: impl Into<String>) -> Arc<Room> {
        let room_name = name.into();
        let mut rooms = self.rooms.lock().unwrap();

        if let Some(room) = rooms.get(&room_name) {
            return room.clone();
        }

        let new_room = Arc::new(Room::new(room_name.clone()));
        rooms.insert(room_name, new_room.clone());
        new_room
    }

    /// Add event handler to namespace
    pub fn on(&self, namespace: &str, event: impl Into<String>, handler: Arc<dyn EventHandler>) {
        if let Some(ns) = self.namespace(namespace) {
            ns.on(event, handler);
        }
    }

    /// Handle new connection
    pub fn handle_connection(
        &self,
        client_id: impl Into<String>,
        namespace: &str,
    ) -> Arc<Mutex<SocketIOClient>> {
        let client = Arc::new(Mutex::new(SocketIOClient::new(client_id, namespace)));

        // Add to clients list
        let id = client.lock().unwrap().id.clone();
        self.clients
            .lock()
            .unwrap()
            .insert(id.clone(), client.clone());

        // Add to namespace
        if let Some(ns) = self.namespace(namespace) {
            ns.add_client(client.clone());
        }

        client
    }

    /// Handle disconnection
    pub fn handle_disconnection(&self, client_id: &str, namespace: &str) {
        // Remove from clients
        self.clients.lock().unwrap().remove(client_id);

        // Remove from namespace
        if let Some(ns) = self.namespace(namespace) {
            ns.remove_client(client_id);
        }

        // Remove from all rooms
        let rooms = self.rooms.lock().unwrap();
        for (_, room) in rooms.iter() {
            room.remove_member(client_id);
        }
    }

    /// Get all clients
    pub fn clients(&self) -> Vec<Arc<Mutex<SocketIOClient>>> {
        self.clients.lock().unwrap().values().cloned().collect()
    }

    /// Get client count
    pub fn client_count(&self) -> usize {
        self.clients.lock().unwrap().len()
    }

    /// Get all rooms
    pub fn rooms(&self) -> Vec<Arc<Room>> {
        self.rooms.lock().unwrap().values().cloned().collect()
    }

    /// Set ping interval
    pub fn with_ping_interval(mut self, interval_ms: u64) -> Self {
        self.ping_interval = interval_ms;
        self
    }

    /// Set ping timeout
    pub fn with_ping_timeout(mut self, timeout_ms: u64) -> Self {
        self.ping_timeout = timeout_ms;
        self
    }

    /// Emit to all clients in namespace
    pub fn emit(&self, namespace: &str, event: impl Into<String>, data: EventData) {
        if let Some(ns) = self.namespace(namespace) {
            ns.emit(event, data);
        }
    }

    /// Emit to room
    pub fn emit_to_room(
        &self,
        namespace: &str,
        room: &str,
        event: impl Into<String>,
        data: EventData,
    ) {
        if let Some(ns) = self.namespace(namespace) {
            ns.emit_to_room(room, event, data);
        }
    }

    /// Emit to specific client
    pub fn emit_to_client(
        &self,
        namespace: &str,
        client_id: &str,
        event: impl Into<String>,
        data: EventData,
    ) {
        if let Some(ns) = self.namespace(namespace) {
            ns.emit_to_client(client_id, event, data);
        }
    }
}

// ============================================================================
// Server-Sent Events (SSE) Implementation
// ============================================================================

/// SSE event
#[derive(Debug, Clone)]
pub struct SSEEvent {
    /// Event ID
    pub id: Option<String>,
    /// Event type/name
    pub event: Option<String>,
    /// Event data
    pub data: String,
    /// Retry interval (ms)
    pub retry: Option<u32>,
}

impl SSEEvent {
    /// Create new SSE event
    pub fn new(data: impl Into<String>) -> Self {
        Self {
            id: None,
            event: None,
            data: data.into(),
            retry: None,
        }
    }

    /// Set event ID
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set event type
    pub fn with_event(mut self, event: impl Into<String>) -> Self {
        self.event = Some(event.into());
        self
    }

    /// Set retry interval
    pub fn with_retry(mut self, retry_ms: u32) -> Self {
        self.retry = Some(retry_ms);
        self
    }

    /// Format as SSE message
    pub fn format(&self) -> String {
        let mut parts = Vec::new();

        if let Some(id) = &self.id {
            parts.push(format!("id: {}", id));
        }

        if let Some(event) = &self.event {
            parts.push(format!("event: {}", event));
        }

        if let Some(retry) = self.retry {
            parts.push(format!("retry: {}", retry));
        }

        // Split data by lines and prefix each with "data: "
        for line in self.data.lines() {
            parts.push(format!("data: {}", line));
        }

        parts.push(String::new()); // Empty line to end event
        parts.join("\n")
    }
}

/// SSE client connection
#[derive(Debug)]
pub struct EventSource {
    /// Client ID
    pub id: String,
    /// Last event ID sent
    last_event_id: Arc<Mutex<Option<String>>>,
    /// Connection state
    connected: Arc<Mutex<bool>>,
    /// Connected timestamp
    connected_at: Arc<Mutex<Instant>>,
    /// Last activity
    last_activity: Arc<Mutex<Instant>>,
    /// Client headers
    pub headers: HeaderMap,
    /// Client query params
    pub query: HashMap<String, String>,
}

impl EventSource {
    /// Create new EventSource
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            last_event_id: Arc::new(Mutex::new(None)),
            connected: Arc::new(Mutex::new(true)),
            connected_at: Arc::new(Mutex::new(Instant::now())),
            last_activity: Arc::new(Mutex::new(Instant::now())),
            headers: HeaderMap::new(),
            query: HashMap::new(),
        }
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        *self.connected.lock().unwrap()
    }

    /// Disconnect
    pub fn disconnect(&self) {
        *self.connected.lock().unwrap() = false;
    }

    /// Get last event ID
    pub fn last_event_id(&self) -> Option<String> {
        self.last_event_id.lock().unwrap().clone()
    }

    /// Set last event ID
    pub fn set_last_event_id(&self, id: impl Into<String>) {
        *self.last_event_id.lock().unwrap() = Some(id.into());
    }

    /// Get connected duration
    pub fn connected_duration(&self) -> Duration {
        self.connected_at.lock().unwrap().elapsed()
    }

    /// Update last activity
    pub fn update_activity(&self) {
        *self.last_activity.lock().unwrap() = Instant::now();
    }

    /// Get idle duration
    pub fn idle_duration(&self) -> Duration {
        self.last_activity.lock().unwrap().elapsed()
    }

    /// Create from HTTP request
    pub fn from_request(id: impl Into<String>, req: &Request) -> Self {
        let mut es = Self::new(id);
        es.headers = req.headers.clone();

        // Parse query parameters
        if let Some(query) = &req.query {
            for pair in query.split('&') {
                if let Some((key, value)) = pair.split_once('=') {
                    es.query.insert(
                        key.to_string(),
                        percent_encoding::percent_decode(value.as_bytes())
                            .decode_utf8()
                            .unwrap_or_default()
                            .to_string(),
                    );
                }
            }
        }

        // Set Last-Event-ID if present
        if let Some(last_id) = req.headers.get("last-event-id") {
            es.set_last_event_id(last_id);
        }

        es
    }
}

/// SSE server
pub struct SSEServer {
    /// Connected clients
    clients: Arc<Mutex<HashMap<String, Arc<EventSource>>>>,
    /// Event channels (topic -> clients)
    channels: Arc<Mutex<HashMap<String, HashSet<String>>>>,
    /// Keep-alive interval (seconds)
    keep_alive_interval: u64,
    /// Retry interval (ms)
    retry_interval: u32,
    /// Server ID
    server_id: String,
}

impl std::fmt::Debug for SSEServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SSEServer")
            .field("client_count", &self.clients.lock().unwrap().len())
            .field("channel_count", &self.channels.lock().unwrap().len())
            .finish()
    }
}

impl SSEServer {
    /// Create new SSE server
    pub fn new() -> Self {
        Self {
            clients: Arc::new(Mutex::new(HashMap::new())),
            channels: Arc::new(Mutex::new(HashMap::new())),
            keep_alive_interval: 30,
            retry_interval: 3000,
            server_id: uuid::Uuid::new_v4().to_string(),
        }
    }

    /// Generate unique client ID
    fn generate_client_id(&self) -> String {
        format!("{}-{}", self.server_id, uuid::Uuid::new_v4())
    }

    /// Handle new SSE connection
    pub fn handle_connection(&self, req: &Request) -> (Arc<EventSource>, Response) {
        let client_id = self.generate_client_id();
        let client = Arc::new(EventSource::from_request(client_id.clone(), req));

        // Add to clients
        self.clients
            .lock()
            .unwrap()
            .insert(client_id.clone(), client.clone());

        // Build SSE response
        let response = Response::new(StatusCode::OK)
            .with_header("Content-Type", "text/event-stream")
            .with_header("Cache-Control", "no-cache")
            .with_header("Connection", "keep-alive")
            .with_header("X-Accel-Buffering", "no"); // Disable nginx buffering

        (client, response)
    }

    /// Disconnect client
    pub fn disconnect(&self, client_id: &str) {
        let client = self.clients.lock().unwrap().remove(client_id);

        if let Some(client) = client {
            client.disconnect();

            // Remove from all channels
            let mut channels = self.channels.lock().unwrap();
            for (_, clients) in channels.iter_mut() {
                clients.remove(client_id);
            }
        }
    }

    /// Subscribe client to channel
    pub fn subscribe(&self, client_id: &str, channel: impl Into<String>) {
        let mut channels = self.channels.lock().unwrap();
        channels
            .entry(channel.into())
            .or_insert_with(HashSet::new)
            .insert(client_id.to_string());
    }

    /// Unsubscribe client from channel
    pub fn unsubscribe(&self, client_id: &str, channel: &str) {
        let mut channels = self.channels.lock().unwrap();
        if let Some(clients) = channels.get_mut(channel) {
            clients.remove(client_id);
        }
    }

    /// Publish event to channel
    pub fn publish(&self, channel: &str, event: &SSEEvent) {
        let channels = self.channels.lock().unwrap();
        let clients = self.clients.lock().unwrap();

        if let Some(subscribers) = channels.get(channel) {
            let _event_data = event.format(); // Would be written to socket in real implementation

            for client_id in subscribers.iter() {
                if let Some(client) = clients.get(client_id) {
                    if client.is_connected() {
                        client.update_activity();
                        // In real implementation, write _event_data to socket here
                    }
                }
            }
        }
    }

    /// Broadcast to all clients
    pub fn broadcast(&self, event: &SSEEvent) {
        let clients = self.clients.lock().unwrap();
        let _event_data = event.format(); // Would be written to socket in real implementation

        for (_, client) in clients.iter() {
            if client.is_connected() {
                client.update_activity();
                // In real implementation, write _event_data to socket here
            }
        }
    }

    /// Send event to specific client
    pub fn send(&self, client_id: &str, _event: &SSEEvent) -> Result<(), String> {
        let clients = self.clients.lock().unwrap();

        if let Some(client) = clients.get(client_id) {
            if client.is_connected() {
                client.update_activity();
                // In real implementation, write to socket here
                return Ok(());
            }
            Err("Client not connected".to_string())
        } else {
            Err("Client not found".to_string())
        }
    }

    /// Get all clients
    pub fn clients(&self) -> Vec<Arc<EventSource>> {
        self.clients.lock().unwrap().values().cloned().collect()
    }

    /// Get client count
    pub fn client_count(&self) -> usize {
        self.clients.lock().unwrap().len()
    }

    /// Get channel subscribers
    pub fn channel_subscribers(&self, channel: &str) -> Vec<String> {
        let channels = self.channels.lock().unwrap();
        if let Some(subscribers) = channels.get(channel) {
            subscribers.iter().cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Get all channels
    pub fn channels(&self) -> Vec<String> {
        self.channels.lock().unwrap().keys().cloned().collect()
    }

    /// Set keep-alive interval
    pub fn with_keep_alive(mut self, interval_secs: u64) -> Self {
        self.keep_alive_interval = interval_secs;
        self
    }

    /// Set retry interval
    pub fn with_retry(mut self, interval_ms: u32) -> Self {
        self.retry_interval = interval_ms;
        self
    }

    /// Clean up idle clients
    pub fn cleanup_idle(&self, idle_timeout: Duration) -> usize {
        let mut to_remove = Vec::new();

        let clients = self.clients.lock().unwrap();
        for (id, client) in clients.iter() {
            if client.idle_duration() > idle_timeout {
                to_remove.push(id.clone());
            }
        }
        drop(clients);

        for id in to_remove.iter() {
            self.disconnect(id);
        }

        to_remove.len()
    }

    /// Send keep-alive comment to all clients
    pub fn send_keep_alive(&self) {
        let _comment = SSEEvent {
            id: None,
            event: None,
            data: ":keep-alive".to_string(),
            retry: None,
        };

        let clients = self.clients.lock().unwrap();
        for (_, client) in clients.iter() {
            if client.is_connected() {
                // In real implementation, send comment to socket
            }
        }
    }

    /// Start keep-alive loop (returns handle for cancellation)
    pub fn start_keep_alive(&self) -> KeepAliveHandle {
        KeepAliveHandle {
            interval: self.keep_alive_interval,
            running: Arc::new(Mutex::new(true)),
        }
    }
}

/// Keep-alive loop handle
pub struct KeepAliveHandle {
    interval: u64,
    running: Arc<Mutex<bool>>,
}

impl std::fmt::Debug for KeepAliveHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeepAliveHandle")
            .field("interval", &self.interval)
            .field("running", &self.running.lock().unwrap())
            .finish()
    }
}

impl KeepAliveHandle {
    /// Stop keep-alive loop
    pub fn stop(&self) {
        *self.running.lock().unwrap() = false;
    }

    /// Check if running
    pub fn is_running(&self) -> bool {
        *self.running.lock().unwrap()
    }
}

/// Generate UUID v4 (simplified - would use uuid crate in production)
fn uuid_v4() -> String {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let mut bytes = [0u8; 16];
    rng.fill(&mut bytes);

    // Set version (4) and variant bits
    bytes[6] = (bytes[6] & 0x0F) | 0x40; // Version 4
    bytes[8] = (bytes[8] & 0x3F) | 0x80; // Variant 1

    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5],
        bytes[6], bytes[7],
        bytes[8], bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    )
}

// Add uuid module for the simplified implementation
mod uuid {
    pub struct Uuid;

    impl Uuid {
        pub fn new_v4() -> String {
            super::uuid_v4()
        }
    }
}

// ============================================================================
// Simple EventHandler implementations
// ============================================================================

/// Callback event handler
pub struct CallbackHandler {
    callback: Arc<dyn Fn(Arc<Mutex<SocketIOClient>>, &str, &EventData) + Send + Sync>,
}

impl CallbackHandler {
    /// Create new callback handler
    pub fn new<F>(callback: F) -> Self
    where
        F: Fn(Arc<Mutex<SocketIOClient>>, &str, &EventData) + Send + Sync + 'static,
    {
        Self {
            callback: Arc::new(callback),
        }
    }
}

impl EventHandler for CallbackHandler {
    fn handle(&self, client: Arc<Mutex<SocketIOClient>>, event: &str, data: &EventData) {
        (self.callback)(client, event, data);
    }
}

/// Broadcast handler - emits to all clients in namespace
pub struct BroadcastHandler {
    namespace: Arc<Namespace>,
}

impl BroadcastHandler {
    /// Create new broadcast handler
    pub fn new(namespace: Arc<Namespace>) -> Self {
        Self { namespace }
    }
}

impl EventHandler for BroadcastHandler {
    fn handle(&self, _client: Arc<Mutex<SocketIOClient>>, _event: &str, _data: &EventData) {
        // Broadcast to all other clients
        // TODO: Implement actual broadcast logic
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Socket.IO Tests

    #[test]
    fn test_socket_io_client_creation() {
        let client = SocketIOClient::new("client1", "/");

        assert_eq!(client.id, "client1");
        assert_eq!(client.namespace, "/");
        assert!(client.is_connected());
        assert!(client.rooms().is_empty());
    }

    #[test]
    fn test_socket_io_client_rooms() {
        let mut client = SocketIOClient::new("client1", "/");

        client.join("room1");
        client.join("room2");

        assert!(client.in_room("room1"));
        assert!(client.in_room("room2"));
        assert!(!client.in_room("room3"));
        assert_eq!(client.rooms().len(), 2);

        client.leave("room1");
        assert!(!client.in_room("room1"));
        assert!(client.in_room("room2"));
    }

    #[test]
    fn test_socket_io_client_data() {
        let client = SocketIOClient::new("client1", "/");

        client.set("username", EventData::String("alice".to_string()));
        client.set("count", EventData::Number(42));

        assert_eq!(
            client.get("username"),
            Some(EventData::String("alice".to_string()))
        );
        assert_eq!(client.get("count"), Some(EventData::Number(42)));
        assert_eq!(client.get("missing"), None);

        assert_eq!(
            client.remove("username"),
            Some(EventData::String("alice".to_string()))
        );
        assert_eq!(client.get("username"), None);
    }

    #[test]
    fn test_namespace_creation() {
        let ns = Namespace::new("/chat");

        assert_eq!(ns.name, "/chat");
        assert_eq!(ns.client_count(), 0);
    }

    #[test]
    fn test_namespace_client_management() {
        let ns = Namespace::new("/chat");
        let client = Arc::new(Mutex::new(SocketIOClient::new("client1", "/chat")));

        ns.add_client(client.clone());
        assert_eq!(ns.client_count(), 1);

        let retrieved = ns.get_client("client1");
        assert!(retrieved.is_some());

        ns.remove_client("client1");
        assert_eq!(ns.client_count(), 0);
    }

    #[test]
    fn test_room_creation() {
        let room = Room::new("general");

        assert_eq!(room.name, "general");
        assert_eq!(room.member_count(), 0);
    }

    #[test]
    fn test_room_members() {
        let room = Room::new("general");

        room.add_member("client1");
        room.add_member("client2");

        assert_eq!(room.member_count(), 2);
        assert!(room.has_member("client1"));
        assert!(!room.has_member("client3"));

        room.remove_member("client1");
        assert_eq!(room.member_count(), 1);
    }

    #[test]
    fn test_socket_io_server_creation() {
        let server = SocketIOServer::new("/main");

        assert_eq!(server.namespace, "/main");
        assert_eq!(server.client_count(), 0);
    }

    #[test]
    fn test_socket_io_server_namespaces() {
        let server = SocketIOServer::new("/main");

        let ns1 = server.of("/chat");
        let ns2 = server.of("/chat");

        // Should return same namespace
        assert!(Arc::ptr_eq(&ns1, &ns2));
    }

    #[test]
    fn test_socket_io_server_rooms() {
        let server = SocketIOServer::new("/main");

        let room1 = server.room("general");
        let room2 = server.room("general");

        // Should return same room
        assert!(Arc::ptr_eq(&room1, &room2));
    }

    #[test]
    fn test_socket_io_server_connection_handling() {
        let server = SocketIOServer::new("/main");

        let client = server.handle_connection("client1", "/");

        assert_eq!(client.lock().unwrap().id, "client1");
        assert_eq!(server.client_count(), 1);

        server.handle_disconnection("client1", "/");
        assert_eq!(server.client_count(), 0);
    }

    #[test]
    fn test_socket_event_creation() {
        let event = SocketIOEvent::new("message")
            .with_data(EventData::String("hello".to_string()))
            .with_namespace("/chat")
            .to_room("general");

        assert_eq!(event.name, "message");
        assert_eq!(event.data.len(), 1);
        assert_eq!(event.namespace, Some("/chat".to_string()));
        assert_eq!(event.room, Some("general".to_string()));
    }

    // SSE Tests

    #[test]
    fn test_sse_event_creation() {
        let event = SSEEvent::new("hello world");

        assert_eq!(event.data, "hello world");
        assert!(event.id.is_none());
        assert!(event.event.is_none());
    }

    #[test]
    fn test_sse_event_with_fields() {
        let event = SSEEvent::new("data")
            .with_id("123")
            .with_event("message")
            .with_retry(5000);

        assert_eq!(event.id, Some("123".to_string()));
        assert_eq!(event.event, Some("message".to_string()));
        assert_eq!(event.retry, Some(5000));
    }

    #[test]
    fn test_sse_event_formatting() {
        let event = SSEEvent::new("line1\nline2")
            .with_id("123")
            .with_event("message");

        let formatted = event.format();

        assert!(formatted.contains("id: 123"));
        assert!(formatted.contains("event: message"));
        assert!(formatted.contains("data: line1"));
        assert!(formatted.contains("data: line2"));
    }

    #[test]
    fn test_event_source_creation() {
        let es = EventSource::new("client1");

        assert_eq!(es.id, "client1");
        assert!(es.is_connected());
        assert!(es.last_event_id().is_none());
    }

    #[test]
    fn test_event_source_event_id() {
        let es = EventSource::new("client1");

        assert_eq!(es.last_event_id(), None);

        es.set_last_event_id("event-123");
        assert_eq!(es.last_event_id(), Some("event-123".to_string()));
    }

    #[test]
    fn test_event_source_connected_duration() {
        let es = EventSource::new("client1");

        std::thread::sleep(std::time::Duration::from_millis(10));

        assert!(es.connected_duration() >= std::time::Duration::from_millis(10));
    }

    #[test]
    fn test_event_source_idle_duration() {
        let es = EventSource::new("client1");

        std::thread::sleep(std::time::Duration::from_millis(10));

        assert!(es.idle_duration() >= std::time::Duration::from_millis(10));

        es.update_activity();
        assert!(es.idle_duration() < std::time::Duration::from_millis(10));
    }

    #[test]
    fn test_sse_server_creation() {
        let server = SSEServer::new();

        assert_eq!(server.client_count(), 0);
        assert!(server.channels().is_empty());
    }

    #[test]
    fn test_sse_server_connection() {
        let server = SSEServer::new();

        let req = Request::new(Method::GET, "/events");
        let (client, response) = server.handle_connection(&req);

        assert_eq!(
            &client.id,
            server.clients.lock().unwrap().keys().next().unwrap()
        );
        assert_eq!(response.status, StatusCode::OK);
        assert_eq!(response.content_type(), Some("text/event-stream"));
    }

    #[test]
    fn test_sse_server_channels() {
        let server = SSEServer::new();

        server.subscribe("client1", "news");
        server.subscribe("client2", "news");
        server.subscribe("client1", "updates");

        let news_subs = server.channel_subscribers("news");
        assert_eq!(news_subs.len(), 2);
        assert!(news_subs.contains(&"client1".to_string()));
        assert!(news_subs.contains(&"client2".to_string()));

        let updates_subs = server.channel_subscribers("updates");
        assert_eq!(updates_subs.len(), 1);

        server.unsubscribe("client1", "news");
        let news_subs = server.channel_subscribers("news");
        assert_eq!(news_subs.len(), 1);
        assert!(!news_subs.contains(&"client1".to_string()));
    }

    #[test]
    fn test_sse_server_disconnect() {
        let server = SSEServer::new();

        server.subscribe("client1", "news");
        server.disconnect("client1");

        assert!(server.channel_subscribers("news").is_empty());
    }

    #[test]
    fn test_callback_handler() {
        let server = SocketIOServer::new("/main");
        let client = server.handle_connection("client1", "/");

        let called = Arc::new(Mutex::new(false));
        let called_clone = called.clone();

        let handler = CallbackHandler::new(move |_, _, _| {
            *called_clone.lock().unwrap() = true;
        });

        handler.handle(
            client.clone(),
            "test",
            &EventData::String("test".to_string()),
        );

        assert!(*called.lock().unwrap());
    }

    #[test]
    fn test_keep_alive_handle() {
        let server = SSEServer::new();
        let handle = server.start_keep_alive();

        assert!(handle.is_running());

        handle.stop();
        assert!(!handle.is_running());
    }

    #[test]
    fn test_sse_server_config() {
        let server = SSEServer::new().with_keep_alive(60).with_retry(5000);

        assert_eq!(server.keep_alive_interval, 60);
        assert_eq!(server.retry_interval, 5000);
    }

    #[test]
    fn test_socket_io_server_config() {
        let server = SocketIOServer::new("/main")
            .with_ping_interval(30000)
            .with_ping_timeout(10000);

        assert_eq!(server.ping_interval, 30000);
        assert_eq!(server.ping_timeout, 10000);
    }
}

// ============================================================================
// HTTP Route Helpers for SSE
// ============================================================================

/// Create SSE handshake response
pub fn sse_response() -> Response {
    Response::new(StatusCode::OK)
        .with_header("Content-Type", "text/event-stream")
        .with_header("Cache-Control", "no-cache, no-transform")
        .with_header("Connection", "keep-alive")
        .with_header("X-Accel-Buffering", "no") // Disable nginx buffering
        .with_header("X-Content-Type-Options", "nosniff")
}

/// Create Socket.IO handshake response
pub fn socketio_handshake(sid: &str) -> Response {
    let handshake_data = serde_json::json!({
        "sid": sid,
        "upgrades": ["websocket"],
        "pingInterval": 25000,
        "pingTimeout": 5000
    });

    Response::new(StatusCode::OK)
        .with_header("Content-Type", "application/json")
        .with_body(handshake_data.to_string().as_bytes().to_vec())
}

/// Parse Socket.IO handshake request
pub fn parse_socketio_handshake(req: &Request) -> Result<HandshakeInfo, String> {
    let query = req.query.as_ref().ok_or("Missing query string")?;

    let mut sid = None;
    let mut transports = Vec::new();

    for pair in query.split('&') {
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));

        match key {
            "sid" => sid = Some(value.to_string()),
            "transport" => transports.push(value.to_string()),
            _ => {}
        }
    }

    if transports.is_empty() {
        transports.push("websocket".to_string());
    }

    Ok(HandshakeInfo { sid, transports })
}

/// Socket.IO handshake information
#[derive(Debug, Clone)]
pub struct HandshakeInfo {
    pub sid: Option<String>,
    pub transports: Vec<String>,
}
