//! Isolated repository sessions for transactional repair (Package G).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IsolationMode {
    GitWorktree,
    TempCopy,
}

/// Edits run in an isolated worktree or temp copy; promote merges success to parent.
pub struct IsolatedRepoSession {
    parent: PathBuf,
    work_root: PathBuf,
    mode: IsolationMode,
}

impl IsolatedRepoSession {
    /// Open an isolated workspace for repair (git worktree when available, else temp copy).
    pub fn open(parent: impl Into<PathBuf>) -> Result<Self, String> {
        let parent = parent.into();
        if is_git_repo(&parent) {
            if let Ok(session) = Self::open_git_worktree(&parent) {
                return Ok(session);
            }
        }
        Self::open_temp_copy(&parent)
    }

    pub fn parent(&self) -> &Path {
        &self.parent
    }

    pub fn work_root(&self) -> &Path {
        &self.work_root
    }

    pub fn mode(&self) -> &'static str {
        match self.mode {
            IsolationMode::GitWorktree => "git_worktree",
            IsolationMode::TempCopy => "temp_copy",
        }
    }

    /// Copy repaired files from the isolated workspace back to the parent repo.
    pub fn promote(&self) -> Result<(), String> {
        copy_tree(&self.work_root, &self.parent, true)
    }

    /// Tear down the isolated workspace without promoting changes.
    pub fn discard(self) -> Result<(), String> {
        match self.mode {
            IsolationMode::GitWorktree => {
                let output = Command::new("git")
                    .arg("-C")
                    .arg(&self.parent)
                    .args(["worktree", "remove", "--force"])
                    .arg(&self.work_root)
                    .output()
                    .map_err(|e| format!("git worktree remove failed: {e}"))?;
                if output.status.success() {
                    Ok(())
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    Err(format!("git worktree remove: {}", stderr.trim()))
                }
            }
            IsolationMode::TempCopy => {
                let _ = fs::remove_dir_all(&self.work_root);
                Ok(())
            }
        }
    }

    fn open_git_worktree(parent: &Path) -> Result<Self, String> {
        let work_root = std::env::temp_dir().join(format!(
            "nsynth_wt_{}_{}",
            parent.file_name()
                .map(|s| s.to_string_lossy())
                .unwrap_or_default(),
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&work_root);
        let output = Command::new("git")
            .arg("-C")
            .arg(parent)
            .args(["worktree", "add", "--detach"])
            .arg(&work_root)
            .output()
            .map_err(|e| format!("git worktree add failed: {e}"))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("git worktree add: {}", stderr.trim()));
        }
        Ok(Self {
            parent: parent.to_path_buf(),
            work_root,
            mode: IsolationMode::GitWorktree,
        })
    }

    fn open_temp_copy(parent: &Path) -> Result<Self, String> {
        let work_root = std::env::temp_dir().join(format!(
            "nsynth_iso_{}_{}",
            parent.file_name()
                .map(|s| s.to_string_lossy())
                .unwrap_or_default(),
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&work_root);
        copy_tree(parent, &work_root, false)?;
        Ok(Self {
            parent: parent.to_path_buf(),
            work_root,
            mode: IsolationMode::TempCopy,
        })
    }
}

fn is_git_repo(path: &Path) -> bool {
    Command::new("git")
        .arg("-C")
        .arg(path)
        .arg("rev-parse")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn copy_tree(from: &Path, to: &Path, skip_git: bool) -> Result<(), String> {
    if !from.is_dir() {
        return Err(format!("copy source is not a directory: {}", from.display()));
    }
    fs::create_dir_all(to).map_err(|e| e.to_string())?;
    for entry in fs::read_dir(from).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        let file_name = entry.file_name();
        if skip_git && file_name == ".git" {
            continue;
        }
        let dest = to.join(file_name);
        if path.is_dir() {
            copy_tree(&path, &dest, skip_git)?;
        } else if path.is_file() {
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            fs::copy(&path, &dest).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn isolated_session_promote_applies_repair_to_parent() {
        let parent = std::env::temp_dir().join(format!("nsynth_iso_parent_{}", std::process::id()));
        let _ = fs::remove_dir_all(&parent);
        fs::create_dir_all(parent.join("src")).unwrap();
        fs::write(parent.join("src/lib.rs"), "wrong\n").unwrap();

        let session = IsolatedRepoSession::open(&parent).expect("open");
        let broken = session.work_root().join("src/lib.rs");
        fs::write(&broken, "fixed\n").unwrap();
        session.promote().expect("promote");
        session.discard().expect("discard");

        assert_eq!(fs::read_to_string(parent.join("src/lib.rs")).unwrap(), "fixed\n");
        let _ = fs::remove_dir_all(parent);
    }

    #[test]
    fn isolated_session_opens_git_worktree_when_repo_initialized() {
        if !Command::new("git")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return;
        }
        let parent = std::env::temp_dir().join(format!(
            "nsynth_iso_git_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = fs::remove_dir_all(&parent);
        fs::create_dir_all(parent.join("src")).unwrap();
        fs::write(parent.join("src/lib.rs"), "ok\n").unwrap();
        Command::new("git")
            .arg("-C")
            .arg(&parent)
            .args(["init", "-q"])
            .status()
            .expect("git init");
        Command::new("git")
            .arg("-C")
            .arg(&parent)
            .args(["add", "src/lib.rs"])
            .status()
            .expect("git add");
        Command::new("git")
            .arg("-C")
            .arg(&parent)
            .args(["commit", "-q", "-m", "init", "--allow-empty"])
            .status()
            .expect("git commit");

        let session = IsolatedRepoSession::open(&parent).expect("open");
        assert_eq!(session.mode(), "git_worktree");
        fs::write(session.work_root().join("src/lib.rs"), "fixed\n").unwrap();
        session.promote().expect("promote");
        session.discard().expect("discard");
        assert_eq!(fs::read_to_string(parent.join("src/lib.rs")).unwrap(), "fixed\n");
        let _ = fs::remove_dir_all(parent);
    }
}
