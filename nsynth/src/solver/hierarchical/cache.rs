//! Cache for hierarchical synthesis
//!
//! Provides efficient caching and reuse of synthesized components.

use std::collections::HashMap;
use std::hash::Hash;

/// Generic cache for synthesis results
#[derive(Debug, Clone)]
pub struct SynthCache<K, V>
where
    K: Hash + Eq,
{
    data: HashMap<K, CacheEntry<V>>,
    max_size: usize,
}

/// Cache entry with metadata
#[derive(Debug, Clone)]
struct CacheEntry<V> {
    value: V,
    hits: usize,
    last_access: std::time::Instant,
}

impl<K, V> SynthCache<K, V>
where
    K: Hash + Eq + Clone,
{
    /// Create new cache with max size
    pub fn new(max_size: usize) -> Self {
        Self {
            data: HashMap::new(),
            max_size,
        }
    }

    /// Get value from cache
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some(entry) = self.data.get_mut(key) {
            entry.hits += 1;
            entry.last_access = std::time::Instant::now();
            Some(&entry.value)
        } else {
            None
        }
    }

    /// Insert value into cache
    pub fn insert(&mut self, key: K, value: V) {
        // Evict if at capacity
        if self.data.len() >= self.max_size {
            self.evict_lru();
        }

        self.data.insert(
            key,
            CacheEntry {
                value,
                hits: 0,
                last_access: std::time::Instant::now(),
            },
        );
    }

    /// Check if key exists
    pub fn contains_key(&self, key: &K) -> bool {
        self.data.contains_key(key)
    }

    /// Remove entry from cache
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.data.remove(key).map(|e| e.value)
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Evict least recently used entry
    fn evict_lru(&mut self) {
        // Find LRU key manually to avoid borrow conflicts
        let mut lru_key: Option<K> = None;
        let mut lru_time = std::time::Instant::now();

        for (key, entry) in &self.data {
            if entry.last_access < lru_time {
                lru_time = entry.last_access;
                lru_key = Some((*key).clone());
            }
        }

        if let Some(key) = lru_key {
            self.data.remove(&key);
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let total_hits: usize = self.data.values().map(|e| e.hits).sum();
        CacheStats {
            entries: self.data.len(),
            total_hits,
            hit_rate: if total_hits > 0 {
                total_hits as f64 / (self.data.len() as f64)
            } else {
                0.0
            },
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub total_hits: usize,
    pub hit_rate: f64,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub max_size: usize,
    pub ttl: Option<std::time::Duration>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_size: 1000,
            ttl: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic() {
        let mut cache: SynthCache<String, i32> = SynthCache::new(2);

        assert!(cache.get(&"key".to_string()).is_none());

        cache.insert("key".to_string(), 42);
        assert_eq!(cache.get(&"key".to_string()), Some(&42));
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache: SynthCache<String, i32> = SynthCache::new(2);

        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);
        cache.insert("c".to_string(), 3); // Should evict LRU

        assert!(cache.contains_key(&"a".to_string()) || cache.contains_key(&"b".to_string()));
        assert!(cache.contains_key(&"c".to_string()));
    }

    #[test]
    fn test_cache_stats() {
        let mut cache: SynthCache<String, i32> = SynthCache::new(10);

        cache.insert("key".to_string(), 42);
        cache.get(&"key".to_string());
        cache.get(&"key".to_string());

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.total_hits, 2);
    }
}
