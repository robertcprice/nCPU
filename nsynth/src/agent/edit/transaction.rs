//! Transactional file edits (Package G seed).

use crate::agent::repo::RepairPatch;
use std::collections::HashMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditTransaction {
    root: PathBuf,
    originals: HashMap<PathBuf, String>,
    working: HashMap<PathBuf, String>,
    /// Files this transaction CREATES (absent before apply). Rollback deletes
    /// them instead of restoring content.
    created: std::collections::HashSet<PathBuf>,
    committed: bool,
}

impl EditTransaction {
    pub fn begin(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            originals: HashMap::new(),
            working: HashMap::new(),
            created: std::collections::HashSet::new(),
            committed: false,
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn snapshot_file(&mut self, relative: &str) -> Result<(), String> {
        let path = self.resolve_relative(relative)?;
        if self.originals.contains_key(&path) {
            return Ok(());
        }
        let content = fs::read_to_string(&path).map_err(|e| format!("read {}: {}", relative, e))?;
        self.originals.insert(path.clone(), content.clone());
        self.working.insert(path, content);
        Ok(())
    }

    pub fn apply_repair_patch(&mut self, patch: &RepairPatch) -> Result<(), String> {
        for edit in &patch.edits {
            let path = self.resolve_relative(&edit.path)?;
            // NEW-FILE edit: the path does not exist yet (a coordinated patch may
            // create a module and wire it elsewhere). Track for delete-on-rollback
            // and seed the working copy with old_text so the exactly-one-occurrence
            // rule below degenerates to "write new_text".
            if !path.exists() && !self.working.contains_key(&path) {
                self.created.insert(path.clone());
                self.working.insert(path.clone(), edit.old_text.clone());
            } else {
                self.snapshot_file(&edit.path)?;
            }
            let original = self
                .working
                .get(&path)
                .cloned()
                .unwrap_or_else(|| {
                    fs::read_to_string(&path).unwrap_or_else(|_| edit.old_text.clone())
                });
            let occurrences = original.matches(&edit.old_text).count();
            if occurrences != 1 {
                return Err(format!(
                    "expected exactly one occurrence of old_text in {}, found {}",
                    edit.path, occurrences
                ));
            }
            let updated = original.replacen(&edit.old_text, &edit.new_text, 1);
            self.working.insert(path, updated);
        }
        Ok(())
    }

    pub fn commit(mut self) -> Result<(), String> {
        for (path, content) in &self.working {
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            if let Err(error) = fs::write(path, content) {
                let _ = self.rollback();
                return Err(error.to_string());
            }
        }
        self.committed = true;
        Ok(())
    }

    pub fn rollback(&self) -> Result<(), String> {
        if self.committed {
            return Err("transaction already committed".to_string());
        }
        for (path, content) in &self.originals {
            fs::write(path, content).map_err(|e| e.to_string())?;
        }
        for path in &self.created {
            let _ = fs::remove_file(path);
        }
        Ok(())
    }

    pub fn is_committed(&self) -> bool {
        self.committed
    }

    /// Peek at the pending (not-yet-committed) content this transaction would
    /// write for `relative`. Returns `Ok(None)` if the transaction has not
    /// touched that path. Used by the repair loop to validate manifest edits
    /// against the *resulting* file before committing.
    pub fn working_content(&self, relative: &str) -> Result<Option<String>, String> {
        let path = self.resolve_relative(relative)?;
        Ok(self.working.get(&path).cloned())
    }

    fn resolve_relative(&self, relative: &str) -> Result<PathBuf, String> {
        let path = Path::new(relative);
        if path.is_absolute() {
            return Err(format!("absolute paths are not allowed: {relative}"));
        }
        for component in path.components() {
            match component {
                Component::ParentDir => {
                    return Err(format!("parent-directory traversal is not allowed: {relative}"));
                }
                Component::RootDir | Component::Prefix(_) => {
                    return Err(format!("rooted paths are not allowed: {relative}"));
                }
                _ => {}
            }
        }
        Ok(self.root.join(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::repo::RepairEdit;
    use std::fs;

    #[test]
    fn transaction_apply_and_rollback_restores_snapshot() {
        let root = std::env::temp_dir().join(format!("nsynth_edit_tx_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/lib.rs"), "wrong\n").unwrap();

        let mut tx = EditTransaction::begin(&root);
        let patch = RepairPatch::new().with_edit(RepairEdit::new(
            "src/lib.rs",
            "wrong\n",
            "fixed\n",
            "test",
        ));
        tx.apply_repair_patch(&patch).expect("apply");
        tx.commit().expect("commit");
        assert_eq!(fs::read_to_string(root.join("src/lib.rs")).unwrap(), "fixed\n");

        let mut tx2 = EditTransaction::begin(&root);
        let revert = RepairPatch::new().with_edit(RepairEdit::new(
            "src/lib.rs",
            "fixed\n",
            "broken\n",
            "test rollback",
        ));
        tx2.apply_repair_patch(&revert).expect("apply revert");
        tx2.rollback().expect("rollback");
        assert_eq!(fs::read_to_string(root.join("src/lib.rs")).unwrap(), "fixed\n");

        let _ = fs::remove_dir_all(root);
    }
}
