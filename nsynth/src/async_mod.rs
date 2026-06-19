//! Async Primitives for nCPU/nSynth
//!
//! Basic async/await support and future types.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// A simple future that completes with a value
#[derive(Debug)]
pub struct Ready<T> {
    value: Option<T>,
}

impl<T> Ready<T> {
    /// Create a new ready future
    pub fn new(value: T) -> Self {
        Self { value: Some(value) }
    }

    /// Extract the value (consumes the future)
    pub fn into_inner(mut self) -> T {
        self.value.take().expect("Ready already consumed")
    }
}

impl<T> Future for Ready<T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<T> {
        // SAFETY: We're not moving the value, just taking from Option
        let value = unsafe { &mut *self.as_mut().get_unchecked_mut() }
            .value
            .take();
        Poll::Ready(value.expect("Ready polled after completion"))
    }
}

/// Create a ready future
pub fn ready<T>(value: T) -> Ready<T> {
    Ready::new(value)
}

/// A simple future that never completes
#[derive(Debug)]
pub struct Pending<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Pending<T> {
    /// Create a new pending future
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> Default for Pending<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Future for Pending<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<T> {
        Poll::Pending
    }
}

/// Create a pending future
pub fn pending<T>() -> Pending<T> {
    Pending::new()
}

/// A simple shared state for async coordination
#[derive(Debug)]
pub struct SharedState<T> {
    inner: Arc<Mutex<StateInner<T>>>,
}

impl<T> Clone for SharedState<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[derive(Debug)]
struct StateInner<T> {
    value: Option<T>,
    waker: Option<Waker>,
}

impl<T: Clone> StateInner<T> {
    fn clone_value(&self) -> Option<T> {
        self.value.clone()
    }
}

impl<T> SharedState<T> {
    /// Create new shared state
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(StateInner {
                value: None,
                waker: None,
            })),
        }
    }

    /// Set the value and wake the waiter
    pub fn set(&self, value: T) {
        let mut inner = self.inner.lock().unwrap();
        inner.value = Some(value);
        if let Some(waker) = inner.waker.take() {
            waker.wake();
        }
    }

    /// Get the value if available (requires T: Clone)
    pub fn get_clone(&self) -> Option<T>
    where
        T: Clone,
    {
        let inner = self.inner.lock().unwrap();
        inner.value.clone()
    }

    /// Try to get the value, consuming it from the state
    pub fn take(&self) -> Option<T> {
        let mut inner = self.inner.lock().unwrap();
        inner.value.take()
    }
}

/// A future that resolves when the shared state is set
pub struct WaitForState<T> {
    shared: SharedState<T>,
}

impl<T> WaitForState<T> {
    /// Create new wait future
    pub fn new(shared: SharedState<T>) -> Self {
        Self { shared }
    }
}

impl<T: Clone> Future for WaitForState<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        if let Some(value) = self.shared.get_clone() {
            Poll::Ready(value)
        } else {
            let mut inner = self.shared.inner.lock().unwrap();
            inner.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ready_future() {
        let future = ready(42);
        assert_eq!(future.into_inner(), 42);
    }

    #[test]
    fn test_pending_future() {
        let future: Pending<i32> = pending();
        let _ = future;
    }

    #[test]
    fn test_shared_state() {
        let state = SharedState::new();
        assert_eq!(state.get_clone(), None);

        state.set(42);
        assert_eq!(state.get_clone(), Some(42));
    }

    #[test]
    fn test_shared_state_take() {
        let state = SharedState::new();
        state.set(42);
        assert_eq!(state.take(), Some(42));
        assert_eq!(state.take(), None);
    }

    #[test]
    fn test_wait_for_state() {
        let state = SharedState::new();
        let future = WaitForState::new(state.clone());
        let _ = future;

        state.set(123);
        assert_eq!(state.get_clone(), Some(123));
    }

    #[test]
    fn test_shared_state_clone() {
        let state1 = SharedState::new();
        let state2 = state1.clone();

        state1.set(42);
        assert_eq!(state2.get_clone(), Some(42));
    }
}
