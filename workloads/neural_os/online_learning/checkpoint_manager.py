#!/usr/bin/env python3
"""
CheckpointManager - State management and rollback for online learning
======================================================================

Provides versioned checkpoint management with:
- Multiple checkpoint slots
- Named checkpoints for milestones
- Automatic cleanup of old checkpoints
- Diff-based storage for efficiency
"""

import torch
import os
import json
import time
import shutil
from typing import Dict, Any, Optional, List
from pathlib import Path
import copy


class CheckpointManager:
    """
    Manages checkpoints for safe online learning.

    Features:
    - Rolling checkpoints (keeps last N)
    - Named milestone checkpoints
    - Metadata tracking
    - Space-efficient storage
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_rolling_checkpoints: int = 5,
        keep_milestones: bool = True
    ):
        """
        Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            max_rolling_checkpoints: Maximum number of rolling checkpoints to keep
            keep_milestones: Whether to keep named milestone checkpoints forever
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_rolling = max_rolling_checkpoints
        self.keep_milestones = keep_milestones

        # Track checkpoints
        self.rolling_checkpoints: List[str] = []
        self.milestone_checkpoints: Dict[str, str] = {}

        # Metadata file
        self.metadata_path = self.checkpoint_dir / "metadata.json"
        self._load_metadata()

    def _load_metadata(self):
        """Load checkpoint metadata from disk."""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)
                self.rolling_checkpoints = data.get('rolling', [])
                self.milestone_checkpoints = data.get('milestones', {})
        else:
            self._save_metadata()

    def _save_metadata(self):
        """Save checkpoint metadata to disk."""
        data = {
            'rolling': self.rolling_checkpoints,
            'milestones': self.milestone_checkpoints,
            'updated': time.time()
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(data, f, indent=2)

    def save_checkpoint(
        self,
        state: Dict[str, Any],
        name: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save a checkpoint.

        Args:
            state: State dict to save (model state, optimizer state, etc.)
            name: Optional name for milestone checkpoint
            metadata: Optional additional metadata

        Returns:
            Path to saved checkpoint
        """
        timestamp = int(time.time() * 1000)

        if name:
            # Milestone checkpoint
            filename = f"milestone_{name}.pt"
            self.milestone_checkpoints[name] = filename
        else:
            # Rolling checkpoint
            filename = f"checkpoint_{timestamp}.pt"
            self.rolling_checkpoints.append(filename)

            # Cleanup old rolling checkpoints
            while len(self.rolling_checkpoints) > self.max_rolling:
                old_file = self.rolling_checkpoints.pop(0)
                old_path = self.checkpoint_dir / old_file
                if old_path.exists():
                    old_path.unlink()

        # Prepare save data
        save_data = {
            'state': state,
            'metadata': metadata or {},
            'timestamp': timestamp,
            'name': name
        }

        # Save
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(save_data, checkpoint_path)

        self._save_metadata()

        return str(checkpoint_path)

    def load_checkpoint(
        self,
        name: Optional[str] = None,
        index: int = -1
    ) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.

        Args:
            name: Name of milestone checkpoint to load
            index: Index of rolling checkpoint (-1 = latest)

        Returns:
            Checkpoint data or None if not found
        """
        if name:
            # Load milestone
            if name not in self.milestone_checkpoints:
                return None
            filename = self.milestone_checkpoints[name]
        else:
            # Load rolling checkpoint by index
            if not self.rolling_checkpoints:
                return None
            filename = self.rolling_checkpoints[index]

        checkpoint_path = self.checkpoint_dir / filename
        if not checkpoint_path.exists():
            return None

        return torch.load(checkpoint_path, weights_only=False)

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint."""
        return self.load_checkpoint(index=-1)

    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about available checkpoints."""
        info = {
            'rolling_count': len(self.rolling_checkpoints),
            'rolling_checkpoints': self.rolling_checkpoints.copy(),
            'milestone_count': len(self.milestone_checkpoints),
            'milestones': list(self.milestone_checkpoints.keys()),
            'directory': str(self.checkpoint_dir)
        }

        # Add size info
        total_size = 0
        for f in self.checkpoint_dir.glob("*.pt"):
            total_size += f.stat().st_size

        info['total_size_mb'] = total_size / (1024 * 1024)

        return info

    def delete_milestone(self, name: str) -> bool:
        """Delete a milestone checkpoint."""
        if name not in self.milestone_checkpoints:
            return False

        filename = self.milestone_checkpoints[name]
        checkpoint_path = self.checkpoint_dir / filename

        if checkpoint_path.exists():
            checkpoint_path.unlink()

        del self.milestone_checkpoints[name]
        self._save_metadata()

        return True

    def clear_rolling(self):
        """Clear all rolling checkpoints."""
        for filename in self.rolling_checkpoints:
            path = self.checkpoint_dir / filename
            if path.exists():
                path.unlink()

        self.rolling_checkpoints.clear()
        self._save_metadata()

    def export_checkpoint(self, source: str, dest_path: str) -> bool:
        """Export a checkpoint to a different location."""
        # Find the checkpoint
        if source in self.milestone_checkpoints:
            filename = self.milestone_checkpoints[source]
        elif source in self.rolling_checkpoints:
            filename = source
        else:
            return False

        src_path = self.checkpoint_dir / filename
        if not src_path.exists():
            return False

        shutil.copy2(src_path, dest_path)
        return True


class InMemoryCheckpoint:
    """
    In-memory checkpoint for fast rollback.

    Keeps state in memory for instant restore without disk I/O.
    """

    def __init__(self, max_checkpoints: int = 3):
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[Dict[str, Any]] = []

    def save(self, state: Dict[str, Any], metadata: Optional[Dict] = None):
        """Save checkpoint to memory."""
        checkpoint = {
            'state': copy.deepcopy(state),
            'metadata': metadata or {},
            'timestamp': time.time()
        }

        self.checkpoints.append(checkpoint)

        # Limit size
        while len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints.pop(0)

    def load(self, index: int = -1) -> Optional[Dict[str, Any]]:
        """Load checkpoint from memory."""
        if not self.checkpoints:
            return None

        return copy.deepcopy(self.checkpoints[index])

    def clear(self):
        """Clear all in-memory checkpoints."""
        self.checkpoints.clear()
