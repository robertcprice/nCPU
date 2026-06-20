//! Event Loop for nCPU/nSynth
//!
//! Async event loop for handling HTTP requests, timers, and signals.

use crate::runtime::{Errno, FfiResult, Value};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Event types that can be handled by the event loop
#[derive(Debug, Clone)]
pub enum Event {
    /// HTTP request event
    Http(HttpEvent),
    /// Timer event
    Timer(TimerEvent),
    /// Signal event
    Signal(SignalEvent),
    /// Custom event
    Custom(String, Vec<u8>),
}

#[derive(Debug, Clone)]
pub struct HttpEvent {
    pub request_id: u64,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct TimerEvent {
    pub timer_id: u64,
    pub deadline: Instant,
}

#[derive(Debug, Clone)]
pub struct SignalEvent {
    pub signal: i32,
    pub source_pid: i32,
}

/// Event handler function type
pub type EventHandler = fn(Event) -> Result<Vec<Event>, String>;

/// Event loop configuration
#[derive(Debug, Clone)]
pub struct EventLoopConfig {
    /// Maximum number of events to process per iteration
    pub max_events_per_tick: usize,
    /// Timeout for polling
    pub poll_timeout: Duration,
    /// Queue size
    pub queue_size: usize,
}

impl Default for EventLoopConfig {
    fn default() -> Self {
        Self {
            max_events_per_tick: 100,
            poll_timeout: Duration::from_millis(100),
            queue_size: 10000,
        }
    }
}

/// Event loop
pub struct EventLoop {
    /// Event queue
    events: Arc<Mutex<VecDeque<Event>>>,
    /// Registered handlers
    handlers: Vec<EventHandler>,
    /// Running state
    running: Arc<Mutex<bool>>,
    /// Configuration
    config: EventLoopConfig,
}

impl EventLoop {
    /// Create new event loop
    pub fn new() -> Self {
        Self::with_config(EventLoopConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: EventLoopConfig) -> Self {
        Self {
            events: Arc::new(Mutex::new(VecDeque::with_capacity(config.queue_size))),
            handlers: Vec::new(),
            running: Arc::new(Mutex::new(false)),
            config,
        }
    }

    /// Register an event handler
    pub fn register_handler(&mut self, handler: EventHandler) {
        self.handlers.push(handler);
    }

    /// Push event to queue
    pub fn push_event(&self, event: Event) -> Result<(), String> {
        let mut events = self
            .events
            .lock()
            .map_err(|e| format!("Lock failed: {}", e))?;

        if events.len() >= self.config.queue_size {
            return Err("Event queue full".to_string());
        }

        events.push_back(event);
        Ok(())
    }

    /// Get next event (blocking with timeout)
    pub fn next_event(&self, timeout: Duration) -> Option<Event> {
        let start = Instant::now();
        let remaining = timeout;

        loop {
            // Try to get an event
            {
                let mut events = self.events.lock().ok()?;
                if let Some(event) = events.pop_front() {
                    return Some(event);
                }
            }

            // Check timeout
            if start.elapsed() >= remaining {
                return None;
            }

            // Small sleep to avoid busy waiting
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Run the event loop
    pub fn run(&mut self) -> Result<(), String> {
        *self
            .running
            .lock()
            .map_err(|e| format!("Lock failed: {}", e))? = true;

        while *self
            .running
            .lock()
            .map_err(|e| format!("Lock failed: {}", e))?
        {
            // Process events
            let mut processed = 0;

            for _ in 0..self.config.max_events_per_tick {
                let event = match self.next_event(self.config.poll_timeout) {
                    Some(e) => e,
                    None => break,
                };

                // Handle event
                for handler in &self.handlers {
                    match handler(event.clone()) {
                        Ok(new_events) => {
                            for new_event in new_events {
                                let _ = self.push_event(new_event);
                            }
                        }
                        Err(e) => {
                            eprintln!("Handler error: {}", e);
                        }
                    }
                }

                processed += 1;
            }

            // If no events processed, small sleep
            if processed == 0 {
                std::thread::sleep(self.config.poll_timeout);
            }
        }

        Ok(())
    }

    /// Stop the event loop
    pub fn stop(&self) -> Result<(), String> {
        *self
            .running
            .lock()
            .map_err(|e| format!("Lock failed: {}", e))? = false;
        Ok(())
    }

    /// Check if running
    pub fn is_running(&self) -> bool {
        self.running.lock().map(|r| *r).unwrap_or(false)
    }

    /// Get event queue size
    pub fn queue_size(&self) -> usize {
        self.events.lock().map(|e| e.len()).unwrap_or(0)
    }

    /// Clear event queue
    pub fn clear_queue(&self) -> Result<(), String> {
        let mut events = self
            .events
            .lock()
            .map_err(|e| format!("Lock failed: {}", e))?;
        events.clear();
        Ok(())
    }
}

/// Convenience function to run event loop with handlers
pub fn run_loop(handlers: Vec<EventHandler>) -> Result<(), String> {
    let mut loop_ = EventLoop::new();
    for handler in handlers {
        loop_.register_handler(handler);
    }
    loop_.run()
}

/// Create a timer event
pub fn timer_event(id: u64, delay: Duration) -> TimerEvent {
    TimerEvent {
        timer_id: id,
        deadline: Instant::now() + delay,
    }
}

/// Create an HTTP event
pub fn http_event(id: u64, data: Vec<u8>) -> Event {
    Event::Http(HttpEvent {
        request_id: id,
        data,
    })
}

/// Create a signal event
pub fn signal_event(signal: i32, source_pid: i32) -> Event {
    Event::Signal(SignalEvent { signal, source_pid })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_loop_creation() {
        let loop_ = EventLoop::new();
        assert!(!loop_.is_running());
        assert_eq!(loop_.queue_size(), 0);
    }

    #[test]
    fn test_push_and_get_event() {
        let loop_ = EventLoop::new();
        let event = Event::Custom("test".to_string(), vec![1, 2, 3]);

        loop_.push_event(event.clone()).unwrap();
        assert_eq!(loop_.queue_size(), 1);

        let retrieved = loop_.next_event(Duration::from_millis(10));
        assert!(retrieved.is_some());

        if let Some(Event::Custom(name, data)) = retrieved {
            assert_eq!(name, "test");
            assert_eq!(data, vec![1, 2, 3]);
        } else {
            panic!("Wrong event type");
        }
    }

    #[test]
    fn test_event_queue_full() {
        let config = EventLoopConfig {
            queue_size: 2,
            ..Default::default()
        };
        let loop_ = EventLoop::with_config(config);

        loop_
            .push_event(Event::Custom("1".to_string(), vec![]))
            .unwrap();
        loop_
            .push_event(Event::Custom("2".to_string(), vec![]))
            .unwrap();

        let result = loop_.push_event(Event::Custom("3".to_string(), vec![]));
        assert!(result.is_err());
    }

    #[test]
    fn test_clear_queue() {
        let loop_ = EventLoop::new();
        loop_
            .push_event(Event::Custom("test".to_string(), vec![]))
            .unwrap();
        assert_eq!(loop_.queue_size(), 1);

        loop_.clear_queue().unwrap();
        assert_eq!(loop_.queue_size(), 0);
    }

    #[test]
    fn test_timer_event_creation() {
        let event = timer_event(1, Duration::from_secs(5));
        assert_eq!(event.timer_id, 1);
        assert!(event.deadline > Instant::now());
        assert!(event.deadline < Instant::now() + Duration::from_secs(6));
    }

    #[test]
    fn test_http_event_creation() {
        let event = http_event(123, vec![1, 2, 3]);
        match event {
            Event::Http(http) => {
                assert_eq!(http.request_id, 123);
                assert_eq!(http.data, vec![1, 2, 3]);
            }
            _ => panic!("Wrong event type"),
        }
    }

    #[test]
    fn test_signal_event_creation() {
        let event = signal_event(9, 1234);
        match event {
            Event::Signal(sig) => {
                assert_eq!(sig.signal, 9);
                assert_eq!(sig.source_pid, 1234);
            }
            _ => panic!("Wrong event type"),
        }
    }

    #[test]
    fn test_handler_registration() {
        let mut loop_ = EventLoop::new();
        let handler: EventHandler = |event| match event {
            Event::Custom(name, _) => Ok(vec![Event::Custom(name.clone() + "_resp", vec![])]),
            _ => Ok(vec![]),
        };

        loop_.register_handler(handler);
        assert_eq!(loop_.handlers.len(), 1);
    }
}
