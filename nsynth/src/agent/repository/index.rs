//! Deterministic, ignore-aware repository file index.

use crate::agent::repo::{GuardrailDecision, GuardrailPolicy};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoIndex {
    root: PathBuf,
    /// Repo-relative paths sorted for deterministic retrieval.
    pub files: Vec<String>,
}

impl RepoIndex {
    pub fn build(root: impl AsRef<Path>, policy: &GuardrailPolicy) -> Result<Self, String> {
        let root = root.as_ref();
        let mut files = Vec::new();
        walk(root, root, policy, &mut files)?;
        files.sort();
        files.dedup();
        Ok(Self {
            root: root.to_path_buf(),
            files,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn contains(&self, relative: &str) -> bool {
        self.files.iter().any(|path| path == relative)
    }

    pub fn filter_prefix(&self, prefix: &str) -> Vec<String> {
        self.files
            .iter()
            .filter(|path| path.starts_with(prefix))
            .cloned()
            .collect()
    }

    pub fn rust_sources(&self) -> Vec<String> {
        self.files
            .iter()
            .filter(|path| path.ends_with(".rs"))
            .cloned()
            .collect()
    }
}

fn walk(
    root: &Path,
    dir: &Path,
    policy: &GuardrailPolicy,
    files: &mut Vec<String>,
) -> Result<(), String> {
    let entries = fs::read_dir(dir).map_err(|e| e.to_string())?;
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        let relative = path
            .strip_prefix(root)
            .map_err(|e| e.to_string())?
            .to_string_lossy()
            .replace('\\', "/");
        if matches!(policy.check_path(&relative, false), GuardrailDecision::Deny(_)) {
            continue;
        }
        if path.is_dir() {
            walk(root, &path, policy, files)?;
        } else if path.is_file() {
            files.push(relative);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn index_skips_git_and_sorts_paths() {
        let root = std::env::temp_dir().join(format!("nsynth_repo_index_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join(".git/objects")).unwrap();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join(".git/objects/pack"), "pack").unwrap();
        fs::write(root.join("src/a.rs"), "a").unwrap();
        fs::write(root.join("src/b.rs"), "b").unwrap();
        fs::write(root.join("README.md"), "readme").unwrap();

        let policy = GuardrailPolicy::default();
        let index = RepoIndex::build(&root, &policy).expect("build");
        assert!(!index.contains(".git/objects/pack"));
        assert_eq!(index.files, vec!["README.md", "src/a.rs", "src/b.rs"]);
        assert_eq!(index.rust_sources(), vec!["src/a.rs", "src/b.rs"]);
        let _ = fs::remove_dir_all(root);
    }
}
