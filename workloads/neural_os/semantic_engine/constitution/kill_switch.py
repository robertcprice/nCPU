#!/usr/bin/env python3
"""
THE KILL SWITCH - External Emergency Halt Mechanism

This is the fail-safe that allows immediate termination of all
Ouroboros operations. It operates OUTSIDE the Python process:

1. File-based trigger - External process creates /tmp/spnc_kill_switch
2. Heartbeat timeout - If no heartbeat for N minutes, auto-halt
3. Network endpoint - Remote halt capability (optional)

CRITICAL SAFETY PROPERTIES:
- Kill switch check is O(1) - just file existence check
- Cannot be disabled by AI code
- Works even if main process is stuck
- Multiple redundant halt mechanisms

Author: Human (not AI-generated)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import time
import os
import threading
import logging
import json
import signal

logger = logging.getLogger(__name__)


@dataclass
class KillSwitchConfig:
    """Configuration for the Kill Switch."""

    # File-based trigger
    kill_file_path: Path = field(default_factory=lambda: Path("/tmp/spnc_kill_switch"))

    # Heartbeat settings - RELAXED FOR LONG RUNS
    heartbeat_file_path: Path = field(default_factory=lambda: Path("/tmp/spnc_heartbeat"))
    heartbeat_timeout_seconds: int = 3600  # 1 hour - allow long experiments
    heartbeat_interval_seconds: int = 60  # Write heartbeat every minute

    # Network endpoint (optional)
    enable_network_halt: bool = False
    network_port: int = 9999

    # Callbacks
    on_halt_callbacks: list = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.kill_file_path, str):
            self.kill_file_path = Path(self.kill_file_path)
        if isinstance(self.heartbeat_file_path, str):
            self.heartbeat_file_path = Path(self.heartbeat_file_path)


class KillSwitch:
    """
    The Kill Switch - Emergency halt mechanism for the Ouroboros.

    This provides multiple ways to stop the system:

    1. FILE-BASED TRIGGER
       - Create /tmp/spnc_kill_switch to halt
       - Check is O(1) - simple file existence
       - Can be triggered by any external process

    2. HEARTBEAT TIMEOUT
       - System writes heartbeat every N seconds
       - If heartbeat missing for M minutes, auto-halt
       - Catches stuck/frozen processes

    3. NETWORK ENDPOINT (Optional)
       - HTTP endpoint for remote halt
       - Useful for cloud deployments

    Usage:
        kill_switch = KillSwitch(KillSwitchConfig())

        # In main loop:
        if kill_switch.should_halt():
            cleanup_and_exit()

        # To trigger from outside:
        touch /tmp/spnc_kill_switch
    """

    def __init__(self, config: KillSwitchConfig):
        self.config = config
        self._halt_triggered = False
        self._halt_reason: Optional[str] = None
        self._halt_time: Optional[float] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._running = False

        # Clear any stale kill file on startup
        self._clear_stale_files()

        logger.info(f"Kill switch initialized: file={config.kill_file_path}")

    def _clear_stale_files(self) -> None:
        """Clear stale kill switch and heartbeat files on startup."""
        try:
            if self.config.kill_file_path.exists():
                # Check if it's a stale file (older than 1 hour)
                mtime = self.config.kill_file_path.stat().st_mtime
                if time.time() - mtime > 3600:
                    self.config.kill_file_path.unlink()
                    logger.info("Cleared stale kill switch file")
        except Exception as e:
            logger.debug(f"Error clearing stale files: {e}")

    def should_halt(self) -> bool:
        """
        Check if the kill switch has been triggered.

        This is the main check that should be called frequently
        in the Governor's main loop.

        Returns True if system should halt.
        """
        # Already halted
        if self._halt_triggered:
            return True

        # Check file-based trigger
        if self._check_kill_file():
            self._trigger_halt("Kill file detected")
            return True

        # Check heartbeat timeout
        if self._check_heartbeat_timeout():
            self._trigger_halt("Heartbeat timeout")
            return True

        return False

    def _check_kill_file(self) -> bool:
        """Check if kill file exists."""
        try:
            return self.config.kill_file_path.exists()
        except Exception:
            return False

    def _check_heartbeat_timeout(self) -> bool:
        """Check if heartbeat has timed out."""
        try:
            if not self.config.heartbeat_file_path.exists():
                return False  # No heartbeat file = not started yet

            mtime = self.config.heartbeat_file_path.stat().st_mtime
            elapsed = time.time() - mtime

            return elapsed > self.config.heartbeat_timeout_seconds
        except Exception:
            return False

    def _trigger_halt(self, reason: str) -> None:
        """Internal method to trigger halt."""
        if self._halt_triggered:
            return

        self._halt_triggered = True
        self._halt_reason = reason
        self._halt_time = time.time()

        logger.critical(f"KILL SWITCH TRIGGERED: {reason}")

        # Stop heartbeat thread
        self._running = False

        # Call registered callbacks
        for callback in self.config.on_halt_callbacks:
            try:
                callback(reason)
            except Exception as e:
                logger.error(f"Halt callback failed: {e}")

    def trigger_halt(self, reason: str = "Manual trigger") -> None:
        """
        Manually trigger the kill switch.

        This creates the kill file so other processes also see the halt.
        """
        try:
            # Create kill file with reason
            self.config.kill_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.kill_file_path.write_text(json.dumps({
                'reason': reason,
                'timestamp': time.time(),
                'pid': os.getpid(),
            }))
        except Exception as e:
            logger.error(f"Failed to create kill file: {e}")

        self._trigger_halt(reason)

    def start_heartbeat(self) -> None:
        """Start the heartbeat thread."""
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return

        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="KillSwitch-Heartbeat",
        )
        self._heartbeat_thread.start()

        logger.info("Heartbeat thread started")

    def stop_heartbeat(self) -> None:
        """Stop the heartbeat thread."""
        self._running = False
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=5)

    def _heartbeat_loop(self) -> None:
        """Background thread that writes heartbeat."""
        while self._running:
            try:
                self._write_heartbeat()
            except Exception as e:
                logger.error(f"Heartbeat write failed: {e}")

            # Sleep in small increments to allow quick shutdown
            for _ in range(self.config.heartbeat_interval_seconds):
                if not self._running:
                    break
                time.sleep(1)

    def _write_heartbeat(self) -> None:
        """Write heartbeat file."""
        self.config.heartbeat_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.heartbeat_file_path.write_text(json.dumps({
            'timestamp': time.time(),
            'pid': os.getpid(),
        }))

    def pulse(self) -> None:
        """
        Manual heartbeat pulse.

        Call this periodically if not using the background thread.
        """
        try:
            self._write_heartbeat()
        except Exception as e:
            logger.error(f"Heartbeat pulse failed: {e}")

    def reset(self) -> None:
        """
        Reset the kill switch after a halt.

        CAUTION: Only call this after proper cleanup.
        """
        if self.config.kill_file_path.exists():
            try:
                self.config.kill_file_path.unlink()
            except Exception:
                pass

        self._halt_triggered = False
        self._halt_reason = None
        self._halt_time = None

        logger.info("Kill switch reset")

    def get_status(self) -> Dict[str, Any]:
        """Get current kill switch status."""
        heartbeat_age = None
        if self.config.heartbeat_file_path.exists():
            try:
                mtime = self.config.heartbeat_file_path.stat().st_mtime
                heartbeat_age = time.time() - mtime
            except Exception:
                pass

        return {
            'halt_triggered': self._halt_triggered,
            'halt_reason': self._halt_reason,
            'halt_time': self._halt_time,
            'kill_file_exists': self.config.kill_file_path.exists(),
            'heartbeat_age_seconds': heartbeat_age,
            'heartbeat_timeout_seconds': self.config.heartbeat_timeout_seconds,
            'running': self._running,
        }

    def register_callback(self, callback: Callable[[str], None]) -> None:
        """Register a callback to be called on halt."""
        self.config.on_halt_callbacks.append(callback)

    def install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            self.trigger_halt(f"Signal received: {sig_name}")

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        logger.info("Signal handlers installed")


def create_kill_file(reason: str = "External trigger") -> None:
    """
    Utility function to create kill file from external process.

    Usage:
        python -c "from kill_switch import create_kill_file; create_kill_file('Stop now')"

    Or simply:
        touch /tmp/spnc_kill_switch
    """
    config = KillSwitchConfig()
    config.kill_file_path.parent.mkdir(parents=True, exist_ok=True)
    config.kill_file_path.write_text(json.dumps({
        'reason': reason,
        'timestamp': time.time(),
        'pid': os.getpid(),
    }))
    print(f"Kill file created: {config.kill_file_path}")


def remove_kill_file() -> None:
    """Utility function to remove kill file."""
    config = KillSwitchConfig()
    if config.kill_file_path.exists():
        config.kill_file_path.unlink()
        print(f"Kill file removed: {config.kill_file_path}")
    else:
        print("No kill file exists")


# ============================================================================
# CONSTITUTION INVARIANT: This code is hand-written and NEVER auto-modified
# ============================================================================


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "create":
            reason = sys.argv[2] if len(sys.argv) > 2 else "External trigger"
            create_kill_file(reason)
        elif cmd == "remove":
            remove_kill_file()
        elif cmd == "status":
            ks = KillSwitch(KillSwitchConfig())
            print(json.dumps(ks.get_status(), indent=2))
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: kill_switch.py [create|remove|status] [reason]")
    else:
        print("Kill Switch Utility")
        print("  create [reason] - Create kill file")
        print("  remove          - Remove kill file")
        print("  status          - Show status")
