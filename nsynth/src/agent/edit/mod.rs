//! Transactional structured edits (Package G).

pub mod transaction;
pub mod worktree;

pub use transaction::EditTransaction;
pub use worktree::IsolatedRepoSession;
