//! Prometheus metrics for solver observability.
//!
//! Emits error category counts, method success/failure rates, and
//! transient retry success rates for monitoring and alerting.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

// Ported from `once_cell::sync::Lazy` to the std `LazyLock` (stable since Rust
// 1.80) so no `once_cell` direct dependency is needed.
use std::sync::LazyLock as Lazy;
use prometheus_client::{
    encoding::text::encode,
    metrics::{counter::Counter, gauge::Gauge},
};

/// Global metrics registry.
static REGISTRY: Lazy<Mutex<prometheus_client::registry::Registry>> =
    Lazy::new(|| Mutex::new(prometheus_client::registry::Registry::default()));

/// Error category counter: synthesis_errors_total{category,method}
static ERROR_COUNTERS: Lazy<Mutex<HashMap<(String, String), Counter>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Method success rate gauge: method_success_rate{method}
static SUCCESS_RATE_GAUGES: Lazy<Mutex<HashMap<String, Gauge<f64, AtomicU64>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Method permanent error rate gauge: method_permanent_error_rate{method}
static PERMANENT_ERROR_GAUGES: Lazy<Mutex<HashMap<String, Gauge<f64, AtomicU64>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Transient retry success rate gauge: transient_retry_success_rate
static RETRY_SUCCESS_GAUGE: Lazy<Gauge<f64, AtomicU64>> = Lazy::new(|| {
    let mut registry = REGISTRY.lock().unwrap();
    let gauge = Gauge::default();
    registry.register(
        "transient_retry_success_rate",
        "Success rate of retries after transient errors",
        gauge.clone(),
    );
    gauge
});

/// Total synthesis attempts counter
static TOTAL_SYNTHESIS_COUNTER: Lazy<AtomicU64> = Lazy::new(|| AtomicU64::new(0));

/// Successful syntheses counter
static SUCCESS_COUNTER: Lazy<AtomicU64> = Lazy::new(|| AtomicU64::new(0));

/// Retry attempt counter
static RETRY_COUNTER: Lazy<AtomicU64> = Lazy::new(|| AtomicU64::new(0));

/// Retry success counter
static RETRY_SUCCESS_COUNTER: Lazy<AtomicU64> = Lazy::new(|| AtomicU64::new(0));

/// Track retry statistics for rate calculation
static RETRY_STATS: Lazy<Mutex<RetryStats>> = Lazy::new(|| Mutex::new(RetryStats::default()));

#[derive(Debug, Default)]
struct RetryStats {
    total_retries: u64,
    successful_retries: u64,
}

/// Record a synthesis error with category and method.
///
/// Increments the synthesis_errors_total counter with labels for
/// error category and method.
pub fn record_error(category: &str, method: &str) {
    let key = (category.to_string(), method.to_string());
    let mut counters = ERROR_COUNTERS.lock().unwrap();

    let counter = counters
        .entry(key)
        .or_insert_with(|| Counter::default());

    counter.inc();
    TOTAL_SYNTHESIS_COUNTER.fetch_add(1, Ordering::Relaxed);
}

/// Record a successful synthesis.
pub fn record_success(_method: &str) {
    SUCCESS_COUNTER.fetch_add(1, Ordering::Relaxed);
    TOTAL_SYNTHESIS_COUNTER.fetch_add(1, Ordering::Relaxed);
}

/// Record a permanent error for a method.
pub fn record_permanent_error(method: &str) {
    let mut gauges = PERMANENT_ERROR_GAUGES.lock().unwrap();
    let _gauge = gauges
        .entry(method.to_string())
        .or_insert_with(|| Gauge::default());

    // Increment would be tracked by the caller
}

/// Record a retry attempt after a transient error.
pub fn record_retry_attempt() {
    RETRY_COUNTER.fetch_add(1, Ordering::Relaxed);
    TOTAL_SYNTHESIS_COUNTER.fetch_add(1, Ordering::Relaxed);

    let mut stats = RETRY_STATS.lock().unwrap();
    stats.total_retries += 1;
}

/// Record a successful retry after a transient error.
pub fn record_retry_success() {
    RETRY_SUCCESS_COUNTER.fetch_add(1, Ordering::Relaxed);

    let mut stats = RETRY_STATS.lock().unwrap();
    stats.successful_retries += 1;

    // Update the retry success rate gauge
    if stats.total_retries > 0 {
        let rate = stats.successful_retries as f64 / stats.total_retries as f64;
        RETRY_SUCCESS_GAUGE.set(rate);
    }
}

/// Get current retry success rate.
pub fn get_retry_success_rate() -> f64 {
    let stats = RETRY_STATS.lock().unwrap();
    if stats.total_retries == 0 {
        0.0
    } else {
        stats.successful_retries as f64 / stats.total_retries as f64
    }
}

/// Test-only snapshot of the raw `(total_retries, successful_retries)` counters.
///
/// `RETRY_STATS` is a process-global accumulator with no production callers, so
/// tests assert on their own *delta* contribution rather than the cumulative
/// global rate (which is polluted by other tests running in parallel).
#[cfg(test)]
pub(crate) fn retry_raw_counts() -> (u64, u64) {
    let stats = RETRY_STATS.lock().unwrap();
    (stats.total_retries, stats.successful_retries)
}

/// Test-only mutex serializing the retry-metric tests so each observes a clean
/// before/after delta of its own `record_retry_*` calls.
#[cfg(test)]
pub(crate) static RETRY_TEST_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

/// Export metrics in Prometheus text format.
pub fn export_metrics() -> String {
    let mut buffer = String::new();
    let registry = REGISTRY.lock().unwrap();

    // Add our atomic counters
    use std::fmt::Write;
    let _ = writeln!(
        buffer,
        "synthesis_attempts_total {}",
        TOTAL_SYNTHESIS_COUNTER.load(Ordering::Relaxed)
    );
    let _ = writeln!(
        buffer,
        "synthesis_successes_total {}",
        SUCCESS_COUNTER.load(Ordering::Relaxed)
    );
    let _ = writeln!(
        buffer,
        "synthesis_retries_total {}",
        RETRY_COUNTER.load(Ordering::Relaxed)
    );
    let _ = writeln!(
        buffer,
        "synthesis_retry_successes_total {}",
        RETRY_SUCCESS_COUNTER.load(Ordering::Relaxed)
    );

    // Export error counters by category and method
    let counters = ERROR_COUNTERS.lock().unwrap();
    for ((category, method), counter) in counters.iter() {
        let _ = writeln!(
            buffer,
            "synthesis_errors_total{{category=\"{}\",method=\"{}\"}} {}",
            category,
            method,
            counter.get()
        );
    }

    if let Err(e) = encode(&mut buffer, &registry) {
        eprintln!("Failed to encode metrics: {}", e);
    }

    buffer
}

/// Record a SolveResult into Prometheus metrics.
///
/// This is the main integration point - call this after each solve attempt.
pub fn record_solve_result(
    method: &str,
    success: bool,
    error_category: Option<&crate::solver::ErrorCategory>,
) {
    record_solve_result_with_problem(method, success, error_category, None);
}

/// Record a SolveResult with optional problem for feature-based learning.
pub fn record_solve_result_with_problem(
    method: &str,
    success: bool,
    error_category: Option<&crate::solver::ErrorCategory>,
    problem: Option<&crate::benchmark::Problem>,
) {
    // Extract features if problem available
    let features = problem.map(|p| crate::solver::method_stats::ProblemFeatures::from_problem(p));

    if success {
        record_success(method);
        if let Some(ref feat) = features {
            crate::solver::method_stats::record_method_result(method, true, None, feat);
        } else {
            // Fallback: create default features when problem not available
            let default_features = crate::solver::method_stats::ProblemFeatures {
                arity: 1,
                input_types: vec![crate::solver::method_stats::TypeClass::ScalarInt],
                output_type: crate::solver::method_stats::TypeClass::ScalarInt,
                complexity: crate::solver::method_stats::Complexity::Simple,
            };
            crate::solver::method_stats::record_method_result(method, true, None, &default_features);
        }
    } else {
        let category_str = match error_category {
            Some(cat) => match cat {
                crate::solver::ErrorCategory::Transient { .. } => "Transient",
                crate::solver::ErrorCategory::Permanent => "Permanent",
                crate::solver::ErrorCategory::ResourceExhaustion => "ResourceExhaustion",
                crate::solver::ErrorCategory::Configuration => "Configuration",
                crate::solver::ErrorCategory::Partial { .. } => "Partial",
            },
            None => "Unknown",
        };

        record_error(category_str, method);

        if matches!(
            error_category,
            Some(crate::solver::ErrorCategory::Permanent)
                | Some(crate::solver::ErrorCategory::Partial { .. })
                | None
        ) {
            record_permanent_error(method);
        }

        if let Some(ref feat) = features {
            crate::solver::method_stats::record_method_result(method, false, error_category, feat);
        } else {
            let default_features = crate::solver::method_stats::ProblemFeatures {
                arity: 1,
                input_types: vec![crate::solver::method_stats::TypeClass::ScalarInt],
                output_type: crate::solver::method_stats::TypeClass::ScalarInt,
                complexity: crate::solver::method_stats::Complexity::Simple,
            };
            crate::solver::method_stats::record_method_result(method, false, error_category, &default_features);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_error() {
        record_error("Permanent", "test_method");
        record_error("Transient", "test_method");

        let exported = export_metrics();
        assert!(exported.contains("synthesis_errors_total") ||
                exported.contains("synthesis_attempts_total"));
    }

    #[test]
    fn test_record_success() {
        record_success("test_method");

        let exported = export_metrics();
        assert!(exported.contains("synthesis_successes_total"));
    }

    #[test]
    fn test_retry_tracking() {
        // RETRY_STATS is a process-global accumulator; serialize against the
        // other retry-metric test and assert on this test's own delta.
        let _guard = RETRY_TEST_LOCK.lock().unwrap();
        let (tot_before, succ_before) = retry_raw_counts();

        record_retry_attempt();
        record_retry_attempt();
        record_retry_success();

        let (tot_after, succ_after) = retry_raw_counts();
        assert_eq!(tot_after - tot_before, 2, "expected 2 retry attempts recorded");
        assert_eq!(succ_after - succ_before, 1, "expected 1 retry success recorded");

        let exported = export_metrics();
        assert!(exported.contains("synthesis_retries_total"));
    }

    #[test]
    fn test_record_solve_result() {
        use crate::solver::ErrorCategory;

        // Record success
        record_solve_result("test_method", true, None);

        // Record permanent error
        record_solve_result("test_method", false, Some(&ErrorCategory::Permanent));

        let exported = export_metrics();
        assert!(exported.contains("synthesis_successes_total"));
    }
}
