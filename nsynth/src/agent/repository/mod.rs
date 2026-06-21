//! Repository indexing and retrieval (Package E seed).

pub mod index;
pub mod retrieval;

pub use index::RepoIndex;
pub use retrieval::{
    localization_confidence, retrieve_paths, RetrievalBenchmarkReport, RetrievalCase,
    run_retrieval_benchmark,
};
