//! WebSocket Support for nCPU/nSynth
//!
//! WebSocket protocol implementation for real-time communication.

use crate::http::types::{Request, Response, StatusCode};
use std::sync::{Arc, Mutex};

/// WebSocket message
#[derive(Debug, Clone)]
pub enum Message {
    Text(String),
    Binary(Vec<u8>),
    Ping(Vec<u8>),
    Pong(Vec<u8>),
    Close,
}

impl Message {
    /// Create text message
    pub fn text(s: impl Into<String>) -> Self {
        Message::Text(s.into())
    }

    /// Create binary message
    pub fn binary(data: Vec<u8>) -> Self {
        Message::Binary(data)
    }

    /// Check if message is text
    pub fn is_text(&self) -> bool {
        matches!(self, Message::Text(_))
    }

    /// Check if message is binary
    pub fn is_binary(&self) -> bool {
        matches!(self, Message::Binary(_))
    }

    /// Get text content if text message
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Message::Text(s) => Some(s),
            _ => None,
        }
    }

    /// Get binary content if binary message
    pub fn as_binary(&self) -> Option<&[u8]> {
        match self {
            Message::Binary(data) => Some(data),
            _ => None,
        }
    }
}

/// WebSocket frame opcode
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
enum Opcode {
    Continuation = 0x0,
    Text = 0x1,
    Binary = 0x2,
    Close = 0x8,
    Ping = 0x9,
    Pong = 0xA,
}

/// WebSocket connection state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum State {
    Connecting,
    Open,
    Closing,
    Closed,
}

/// WebSocket connection
#[derive(Debug)]
pub struct WebSocket {
    /// Connection state
    state: Arc<Mutex<State>>,
    /// Socket file descriptor
    fd: Arc<Mutex<Option<i32>>>,
    /// Receive buffer
    rx_buffer: Arc<Mutex<Vec<u8>>>,
    /// Send buffer
    tx_buffer: Arc<Mutex<Vec<u8>>>,
}

impl WebSocket {
    /// Create new WebSocket
    pub fn new(fd: i32) -> Self {
        Self {
            state: Arc::new(Mutex::new(State::Connecting)),
            fd: Arc::new(Mutex::new(Some(fd))),
            rx_buffer: Arc::new(Mutex::new(Vec::new())),
            tx_buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Perform WebSocket handshake
    pub fn handshake(&self, req: &Request) -> Result<Response, String> {
        let mut state = self.state.lock().unwrap();
        if *state != State::Connecting {
            return Err("WebSocket already connected".to_string());
        }

        // Extract WebSocket key from request
        let key = req
            .headers
            .get("Sec-WebSocket-Key")
            .ok_or("Missing Sec-WebSocket-Key header")?;

        // Compute accept key (in real implementation: base64(sha1(key + GUID)))
        let accept_key = Self::compute_accept_key(key);

        // Build handshake response
        let response = Response::new(StatusCode::SwitchingProtocols)
            .with_header("Upgrade", "websocket")
            .with_header("Connection", "Upgrade")
            .with_header("Sec-WebSocket-Accept", &accept_key);

        *state = State::Open;
        Ok(response)
    }

    /// Compute WebSocket accept key from client key
    fn compute_accept_key(key: &str) -> String {
        // In real implementation:
        // let guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
        // let hash = sha1(key + guid);
        // base64_encode(hash)
        format!("{}={}", key, "computed")
    }

    /// Send message
    pub fn send(&self, msg: Message) -> Result<(), String> {
        let state = self.state.lock().unwrap();
        if *state != State::Open {
            return Err("WebSocket not open".to_string());
        }
        drop(state);

        // In real implementation:
        // 1. Frame the message according to WebSocket protocol
        // 2. Write to socket

        Ok(())
    }

    /// Receive message (blocking)
    pub fn recv(&self) -> Result<Message, String> {
        let state = self.state.lock().unwrap();
        if *state != State::Open {
            return Err("WebSocket not open".to_string());
        }
        drop(state);

        // In real implementation:
        // 1. Read from socket
        // 2. Parse WebSocket frame
        // 3. Return message

        Ok(Message::Text(String::new()))
    }

    /// Receive message with timeout
    pub fn recv_timeout(&self, timeout_ms: u64) -> Result<Option<Message>, String> {
        // In real implementation: poll with timeout
        Ok(None)
    }

    /// Close connection
    pub fn close(&self) -> Result<(), String> {
        let mut state = self.state.lock().unwrap();
        if *state == State::Closed {
            return Ok(());
        }

        // Send close frame
        *state = State::Closing;

        // In real implementation: send Close frame

        *state = State::Closed;
        Ok(())
    }

    /// Check if connection is open
    pub fn is_open(&self) -> bool {
        *self.state.lock().unwrap() == State::Open
    }

    /// Get current state
    pub fn state(&self) -> State {
        *self.state.lock().unwrap()
    }
}

/// WebSocket server
#[derive(Debug)]
pub struct WebSocketServer {
    /// Listen address
    addr: String,
    /// Listen port
    port: u16,
    /// Active connections
    connections: Arc<Mutex<Vec<WebSocket>>>,
}

impl WebSocketServer {
    /// Create new WebSocket server
    pub fn new(addr: impl Into<String>, port: u16) -> Self {
        Self {
            addr: addr.into(),
            port,
            connections: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Start WebSocket server
    pub fn listen(&self) -> Result<(), String> {
        // In real implementation:
        // 1. Listen on TCP port
        // 2. Accept connections
        // 3. Perform WebSocket handshake
        // 4. Manage connections
        Err("WebSocket server not fully implemented".to_string())
    }

    /// Broadcast message to all connections
    pub fn broadcast(&self, msg: &Message) -> Result<(), String> {
        let conns = self.connections.lock().unwrap();
        for conn in conns.iter() {
            conn.send(msg.clone())?;
        }
        Ok(())
    }

    /// Get connection count
    pub fn connection_count(&self) -> usize {
        self.connections.lock().unwrap().len()
    }
}

/// WebSocket client
#[derive(Debug)]
pub struct WebSocketClient {
    /// Server URL
    url: String,
    /// WebSocket connection
    ws: Option<WebSocket>,
}

impl WebSocketClient {
    /// Create new WebSocket client
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            ws: None,
        }
    }

    /// Connect to WebSocket server
    pub fn connect(&mut self) -> Result<(), String> {
        // In real implementation:
        // 1. Parse URL (ws:// or wss://)
        // 2. Connect to server
        // 3. Send WebSocket handshake
        // 4. Wait for handshake response

        Ok(())
    }

    /// Send message
    pub fn send(&mut self, msg: Message) -> Result<(), String> {
        if let Some(ref ws) = self.ws {
            ws.send(msg)
        } else {
            Err("Not connected".to_string())
        }
    }

    /// Receive message
    pub fn recv(&mut self) -> Result<Message, String> {
        if let Some(ref ws) = self.ws {
            ws.recv()
        } else {
            Err("Not connected".to_string())
        }
    }

    /// Close connection
    pub fn close(&mut self) -> Result<(), String> {
        if let Some(ref ws) = self.ws {
            ws.close()
        } else {
            Ok(())
        }
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.ws.as_ref().map(|ws| ws.is_open()).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_text() {
        let msg = Message::text("hello");
        assert!(msg.is_text());
        assert_eq!(msg.as_text(), Some("hello"));
    }

    #[test]
    fn test_message_binary() {
        let data = vec![1, 2, 3];
        let msg = Message::binary(data.clone());
        assert!(msg.is_binary());
        assert_eq!(msg.as_binary(), Some(&data[..]));
    }

    #[test]
    fn test_websocket_states() {
        let ws = WebSocket::new(1);
        assert_eq!(ws.state(), State::Connecting);
    }

    #[test]
    fn test_websocket_client_new() {
        let client = WebSocketClient::new("ws://example.com");
        assert!(!client.is_connected());
        assert_eq!(client.url, "ws://example.com");
    }

    #[test]
    fn test_websocket_server() {
        let server = WebSocketServer::new("127.0.0.1", 8080);
        assert_eq!(server.connection_count(), 0);
        assert_eq!(server.port, 8080);
    }

    #[test]
    fn test_websocket_handshake_missing_key() {
        let ws = WebSocket::new(1);
        let req = Request::new(crate::http::types::Method::GET, "/ws");

        assert!(ws.handshake(&req).is_err());
    }
}
