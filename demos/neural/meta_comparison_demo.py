#!/usr/bin/env python3
"""Interactive neural-rendered comparison demo against Meta-style Neural Computers.

Left pane:
  A real interactive shell attached to a PTY. Every shell byte is rendered
  through NeuralDisplayV2, so the visible terminal pixels are produced by the
  neural display pipeline.

Right pane:
  A neural-rendered comparison terminal that explains how this repo differs
  from the "Neural Computers" paper discussed in `paper/ncpu_paper.md`.
  This is a textual reference panel, not their model output.

The point of the demo is to make two facts visually obvious:
  1. the displayed screen is being produced neurally here, and
  2. the left pane remains a real interactive computer rather than a screen
     prediction model.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import re
import sys
import time
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pygame
except ImportError:
    print("ERROR: pygame is required for the Meta comparison demo.")
    print("Install with: pip install pygame")
    sys.exit(1)

from demos.neural.meta_comparison_runtime import build_left_runtime
from demos.neural.neural_terminal_pty import _key_to_bytes
from ncpu.neural.neural_terminal_renderer import FRAME_H, FRAME_W, TERM_COLS, TERM_ROWS
from ncpu.neural.neural_terminal_renderer_v2 import NeuralDisplayV2


PANEL_GAP = 0
LEFT_RUNTIME_CHOICES = ("pty", "neural-os")


def _comparison_text(shell: str, backend: str, left_runtime: str, left_runtime_label: str) -> str:
    if left_runtime == "neural-os":
        left_runtime_lines = (
            "  - nCPU-managed ARM64 shell on our GPU OS\n"
            "  - Keyboard input routed into the nCPU runtime\n"
            "  - SYS_WRITE bytes update the left terminal state\n"
            "  - NeuralDisplayV2 renders every visible pixel\n"
            "  - Try: pwd, help, cc hello.c, run /bin/hello\n"
        )
        why_line = "  - Left pane is now our own computer path, not a host PTY\n"
    else:
        left_runtime_lines = (
            "  - Real PTY shell process\n"
            "  - Keyboard input forwarded to shell\n"
            "  - Bytes update terminal state\n"
            "  - NeuralDisplayV2 renders every visible pixel\n"
            "  - Try: ls, pwd, echo 2+2, python3 --version\n"
        )
        why_line = "  - Left pane stays interactive like a computer\n"

    return (
        "\x1b[2J\x1b[H"
        "\x1b[1;36mnCPU vs Meta Neural Computers\x1b[0m\n"
        "\x1b[90mRight pane is a reference summary, not their model output.\x1b[0m\n"
        "\n"
        "\x1b[1;32mnCPU (left pane)\x1b[0m\n"
        + left_runtime_lines
        + "\n"
        + "\x1b[1;33mMeta / Zhuge Neural Computers\x1b[0m\n"
        + "  - Paper goal: a completely neural computer\n"
        + "  - Runtime framed as screen generation / video prediction\n"
        + "  - Learned latent state instead of explicit ISA execution\n"
        + "  - In this repo we do not ship their model\n"
        + "\n"
        + "\x1b[1;35mWhy this demo matters\x1b[0m\n"
        + why_line
        + "  - The displayed text is neurally rasterized here\n"
        + "  - Computation and display are separated cleanly\n"
        + "  - This makes the claim auditable, not just visual\n"
        + "\n"
        + "\x1b[1;37mVisible proof points\x1b[0m\n"
        + f"  - Shell: {shell}\n"
        + f"  - Left runtime: {left_runtime_label}\n"
        + f"  - Terminal size: {TERM_COLS}x{TERM_ROWS}\n"
        + f"  - Backend: {backend}\n"
        + "  - Renderer: NeuralDisplayV2 (390,916 params)\n"
        + "  - Panel pixels: 640x384 each\n"
        + "  - Visible window content: neural-rendered panels only\n"
        + "\n"
        + "\x1b[90mCtrl+Q quits. Type in the left pane.\x1b[0m\n"
    )


def _load_display(device: str | None) -> NeuralDisplayV2:
    model_path = PROJECT_ROOT / "models" / "display" / "terminal_renderer_v2.pt"
    return NeuralDisplayV2(str(model_path), device=device)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:40] or "step"


def _save_surface(screen: pygame.Surface, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(screen, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive neural-rendered comparison demo against Meta Neural Computers"
    )
    parser.add_argument(
        "--left-runtime",
        choices=LEFT_RUNTIME_CHOICES,
        default="pty",
        help="Left pane compute backend: host PTY or nCPU GPU shell",
    )
    parser.add_argument("--shell", type=str, default=None, help="Shell to spawn in the interactive pane")
    parser.add_argument("--device", type=str, default=None, help="Neural display device override")
    parser.add_argument("--scale", type=int, default=1, help="Window scale (default: 1)")
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        help="Scripted shell command to send to the left pane; may be repeated",
    )
    parser.add_argument("--capture-dir", type=str, default=None, help="Directory for per-step PNG captures")
    parser.add_argument("--summary-json", type=str, default=None, help="Write a machine-readable summary JSON")
    parser.add_argument("--shell-log", type=str, default=None, help="Write decoded shell output to this log path")
    parser.add_argument("--boot-delay-ms", type=int, default=1200, help="Delay before the first scripted capture/send")
    parser.add_argument("--step-delay-ms", type=int, default=1000, help="Delay between scripted command send and capture")
    parser.add_argument("--final-hold-ms", type=int, default=800, help="Delay after the final scripted capture")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Exit after N rendered frames (useful for smoke tests or artifact capture)",
    )
    parser.add_argument("--output", type=str, default=None, help="Save the composed neural frame as a PNG")
    args = parser.parse_args()

    shell = args.shell or os.environ.get("SHELL", "/bin/zsh")
    scale = max(1, args.scale)
    left_runtime_name = args.left_runtime
    script_commands = list(args.command or [])
    capture_dir = Path(args.capture_dir).resolve() if args.capture_dir else None
    summary_path = Path(args.summary_json).resolve() if args.summary_json else None
    shell_log_path = Path(args.shell_log).resolve() if args.shell_log else None
    if shell_log_path is None and capture_dir is not None:
        shell_log_path = capture_dir / "shell.log"
    script_mode = bool(script_commands)
    boot_delay_s = max(0, args.boot_delay_ms) / 1000.0
    step_delay_s = max(0, args.step_delay_ms) / 1000.0
    final_hold_s = max(0, args.final_hold_ms) / 1000.0

    left_runtime = build_left_runtime(
        left_runtime_name,
        shell=shell,
        device=args.device,
        script_mode=script_mode,
    )
    left = left_runtime.display
    if left_runtime_name == "neural-os":
        left.terminal.write_str(
            "\x1b[1;32mnCPU neural OS pane\x1b[0m - our shell path, neural pixels only.\r\n"
            "\x1b[90mBooting ARM64 shell on the nCPU GPU runtime...\x1b[0m\r\n\r\n"
        )
    else:
        left.terminal.write_str(
            "\x1b[1;32mnCPU PTY pane\x1b[0m - host shell, neural pixels only.\r\n"
            "\x1b[90mCtrl+Q quits this demo.\x1b[0m\r\n\r\n"
        )
    left_runtime.start()
    left_runtime.wait_until_ready(timeout=30.0 if left_runtime_name == "neural-os" else None)
    right = _load_display(args.device)

    backend = "Metal GPU" if getattr(left, "metal_available", False) else "PyTorch"
    renderer_params = int(left.renderer.count_params())
    right.reset()
    shell_label = left_runtime.metadata().get("left_runtime_shell", shell)
    right.terminal.write_str(
        _comparison_text(
            shell_label,
            backend,
            left_runtime_name,
            str(left_runtime.metadata().get("left_runtime_label", left_runtime_name)),
        )
    )

    panel_w = FRAME_W * scale
    panel_h = FRAME_H * scale
    win_w = panel_w * 2 + PANEL_GAP
    win_h = panel_h

    pygame.init()
    pygame.display.set_caption(f"nCPU Neural Comparison Demo [{backend}]")
    screen = pygame.display.set_mode((win_w, win_h))
    clock = pygame.time.Clock()
    pygame.key.set_repeat(400, 50)

    frame_times: deque[float] = deque(maxlen=60)
    running = True
    frame_count = 0
    saved_output = False
    start_time = time.perf_counter()
    next_script_at = start_time + boot_delay_s if script_mode else None
    boot_capture_done = not script_mode
    next_command_index = 0
    pending_capture: dict[str, object] | None = None
    finish_at: float | None = None
    capture_records: list[dict[str, object]] = []
    command_records: list[dict[str, object]] = []
    shell_log_chunks: list[str] = []
    summary_written = False

    def _elapsed_s() -> float:
        return time.perf_counter() - start_time

    def _write_shell_log() -> None:
        if shell_log_path is None:
            return
        shell_log_path.parent.mkdir(parents=True, exist_ok=True)
        shell_log_path.write_text("".join(shell_log_chunks), encoding="utf-8")

    def _write_summary(output_path_value: str | None) -> None:
        nonlocal summary_written
        if summary_written or summary_path is None:
            return
        first_output_latencies = [
            float(record["first_output_latency_ms"])
            for record in command_records
            if record.get("first_output_latency_ms") is not None
        ]
        capture_latencies = [
            float(record["capture_latency_ms"])
            for record in command_records
            if record.get("capture_latency_ms") is not None
        ]
        payload = {
            "demo": "meta_comparison_demo",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "scripted" if script_mode else "interactive",
            "shell": shell_label,
            "device": left.device,
            "backend": backend,
            "metal_active": bool(getattr(left, "metal_available", False)),
            "renderer_params": renderer_params,
            "scale": scale,
            "panel_width_px": FRAME_W,
            "panel_height_px": FRAME_H,
            "window_width_px": win_w,
            "window_height_px": win_h,
            "frames_rendered": frame_count,
            "elapsed_s": _elapsed_s(),
            "interactive_left_pane": True,
            "visible_content_neural_only": True,
            "reference_right_pane_not_meta_output": True,
            "left_pane_computation_owned_by_ncpu": bool(
                left_runtime.metadata().get("left_runtime_owned_by_ncpu", False)
            ),
            "avg_first_output_latency_ms": (
                sum(first_output_latencies) / len(first_output_latencies)
                if first_output_latencies
                else None
            ),
            "avg_capture_latency_ms": (
                sum(capture_latencies) / len(capture_latencies)
                if capture_latencies
                else None
            ),
            "capture_dir": str(capture_dir) if capture_dir is not None else None,
            "captures": capture_records,
            "commands": command_records,
            "output_path": output_path_value,
            "shell_log_path": str(shell_log_path) if shell_log_path is not None else None,
        }
        payload.update(left_runtime.metadata())
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        summary_written = True

    def _finish_now(output_path_value: str | None = None) -> None:
        _write_shell_log()
        _write_summary(output_path_value)
        left_runtime.terminate()
        pygame.quit()
        os._exit(0)

    try:
        while running:
            t0 = time.perf_counter()

            chunks = left_runtime.poll()
            if chunks:
                for chunk in chunks:
                    shell_log_chunks.append(chunk.decode("utf-8", errors="replace"))
                if pending_capture is not None and pending_capture.get("first_output_at_s") is None:
                    pending_capture["first_output_at_s"] = _elapsed_s()
                    pending_capture["first_output_latency_ms"] = (
                        float(pending_capture["first_output_at_s"])
                        - float(pending_capture["sent_at_s"])
                    ) * 1000.0

            if not left_runtime.alive:
                left.terminal.write_str(
                    "\r\n\x1b[1;31m[shell exited]\x1b[0m\r\n"
                    "\x1b[90mClose window or press any key.\x1b[0m"
                )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                    payload = _key_to_bytes(event)
                    if payload is None:
                        running = False
                        break
                    if payload and left_runtime.alive:
                        left_runtime.send(payload)
                    elif not left_runtime.alive:
                        running = False
                        break

            if not running:
                break

            left_frame = left_runtime.render()
            right_frame = right.render()
            left_surf = pygame.surfarray.make_surface(left_frame.swapaxes(0, 1))
            right_surf = pygame.surfarray.make_surface(right_frame.swapaxes(0, 1))
            if scale != 1:
                left_surf = pygame.transform.scale(left_surf, (panel_w, panel_h))
                right_surf = pygame.transform.scale(right_surf, (panel_w, panel_h))

            screen.blit(left_surf, (0, 0))
            screen.blit(right_surf, (panel_w + PANEL_GAP, 0))

            t1 = time.perf_counter()
            frame_times.append(t1 - t0)
            avg = sum(frame_times) / len(frame_times) if frame_times else 1.0
            fps = 1.0 / avg if avg > 0 else 0.0
            pygame.display.set_caption(
                f"nCPU Neural Comparison Demo [{backend}] - {fps:.1f} FPS - Ctrl+Q quits"
            )

            pygame.display.flip()
            frame_count += 1
            now = time.perf_counter()

            if script_mode:
                if not boot_capture_done and now >= (next_script_at or now):
                    if capture_dir is not None:
                        boot_path = capture_dir / "00_boot.png"
                        _save_surface(screen, boot_path)
                        capture_records.append(
                            {
                                "label": "boot",
                                "path": str(boot_path),
                                "captured_at_s": _elapsed_s(),
                            }
                        )
                    boot_capture_done = True
                    next_script_at = now + 0.05
                elif pending_capture is None and next_command_index < len(script_commands) and now >= (next_script_at or now):
                    command = script_commands[next_command_index]
                    if left_runtime.alive:
                        left_runtime.send(command.encode("utf-8") + b"\r")
                        record = {
                            "index": next_command_index + 1,
                            "command": command,
                            "sent_at_s": _elapsed_s(),
                            "first_output_at_s": None,
                            "first_output_latency_ms": None,
                        }
                        command_records.append(record)
                        pending_capture = record
                        next_command_index += 1
                        next_script_at = now + step_delay_s
                elif pending_capture is not None and now >= (next_script_at or now):
                    if capture_dir is not None:
                        label = f"{int(pending_capture['index']):02d}_{_slugify(str(pending_capture['command']))}.png"
                        capture_path = capture_dir / label
                        _save_surface(screen, capture_path)
                        pending_capture["capture_path"] = str(capture_path)
                        capture_records.append(
                            {
                                "label": label.removesuffix(".png"),
                                "path": str(capture_path),
                                "captured_at_s": _elapsed_s(),
                            }
                        )
                    pending_capture["captured_at_s"] = _elapsed_s()
                    pending_capture["capture_latency_ms"] = (
                        float(pending_capture["captured_at_s"])
                        - float(pending_capture["sent_at_s"])
                    ) * 1000.0
                    pending_capture = None
                    if next_command_index >= len(script_commands):
                        finish_at = now + final_hold_s
                    else:
                        next_script_at = now + 0.05
                elif finish_at is not None and now >= finish_at:
                    output_path_value = None
                    if args.output:
                        output_path = Path(args.output).resolve()
                        _save_surface(screen, output_path)
                        saved_output = True
                        output_path_value = str(output_path)
                    _finish_now(output_path_value)

            should_save = args.output and not saved_output and not script_mode
            if should_save and args.max_frames is not None:
                should_save = frame_count >= args.max_frames
            if should_save:
                output_path = Path(args.output).resolve()
                _save_surface(screen, output_path)
                saved_output = True
            if args.max_frames is not None and frame_count >= args.max_frames:
                output_path_value = str(Path(args.output).resolve()) if args.output and saved_output else None
                _finish_now(output_path_value)
            clock.tick(60)
    finally:
        _write_shell_log()
        output_path_value = str(Path(args.output).resolve()) if args.output and saved_output else None
        _write_summary(output_path_value)
        left_runtime.terminate()
        pygame.quit()


if __name__ == "__main__":
    main()
