"""Tests for the Neural Terminal Renderer — TerminalState, models, and NeuralDisplay."""

import sys
import pytest
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ncpu.neural.neural_terminal_renderer import (
    TerminalState, NeuralGlyphGenerator, NeuralColorPalette,
    NeuralCompositor, NeuralTerminalRenderer, NeuralDisplay,
    TERM_ROWS, TERM_COLS, CELL_H, CELL_W, FRAME_H, FRAME_W,
    N_CHARS, N_COLORS, ANSI_PALETTE,
)

MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'display' / 'terminal_renderer.pt'


# ═══════════════════════════════════════════════════════════════════════════
# TerminalState Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTerminalState:
    def test_initial_state(self):
        ts = TerminalState()
        assert ts.rows == TERM_ROWS
        assert ts.cols == TERM_COLS
        assert ts.cr == 0 and ts.cc == 0
        assert ts.cur_fg == 7 and ts.cur_bg == 0
        assert (ts.chars == ord(' ')).all()
        assert (ts.fg == 7).all()
        assert (ts.bg == 0).all()

    def test_write_ascii(self):
        ts = TerminalState()
        ts.write(b'Hello')
        assert ts.chars[0, 0] == ord('H')
        assert ts.chars[0, 1] == ord('e')
        assert ts.chars[0, 2] == ord('l')
        assert ts.chars[0, 3] == ord('l')
        assert ts.chars[0, 4] == ord('o')
        assert ts.cc == 5

    def test_newline(self):
        ts = TerminalState()
        ts.write(b'A\nB')
        assert ts.chars[0, 0] == ord('A')
        assert ts.chars[1, 0] == ord('B')
        assert ts.cr == 1 and ts.cc == 1

    def test_carriage_return(self):
        ts = TerminalState()
        ts.write(b'ABC\rD')
        assert ts.chars[0, 0] == ord('D')
        assert ts.chars[0, 1] == ord('B')  # B wasn't overwritten
        assert ts.cc == 1

    def test_tab(self):
        ts = TerminalState()
        ts.write(b'A\tB')
        assert ts.chars[0, 0] == ord('A')
        assert ts.cc > 2  # tab advanced past A
        # B should be at the tab stop
        assert ts.chars[0, 8] == ord('B')

    def test_backspace(self):
        ts = TerminalState()
        ts.write(b'AB\x08C')
        assert ts.chars[0, 0] == ord('A')
        assert ts.chars[0, 1] == ord('C')  # C overwrote B's position
        assert ts.cc == 2

    def test_line_wrap(self):
        ts = TerminalState()
        ts.write(b'A' * 81)  # 80 cols + 1
        assert ts.cr == 1
        assert ts.cc == 1
        assert ts.chars[1, 0] == ord('A')

    def test_scroll(self):
        ts = TerminalState()
        # Write 25 lines (terminal has 24 rows)
        for i in range(25):
            ts.write(f'line{i}\n'.encode())
        # After 25 lines with newlines, two scrolls occur
        assert ts.cr == TERM_ROWS - 1
        line0_text = bytes(ts.chars[0, :5]).decode()
        assert line0_text == 'line2'

    def test_reset(self):
        ts = TerminalState()
        ts.write(b'Hello World')
        ts.reset()
        assert ts.cr == 0 and ts.cc == 0
        assert (ts.chars == ord(' ')).all()

    def test_write_str(self):
        ts = TerminalState()
        ts.write_str('Hello')
        assert ts.chars[0, 0] == ord('H')

    # ── ANSI escape sequences ──

    def test_sgr_fg_color(self):
        ts = TerminalState()
        ts.write(b'\x1b[31mR')  # red foreground
        assert ts.chars[0, 0] == ord('R')
        assert ts.fg[0, 0] == 1  # red = color 1

    def test_sgr_bg_color(self):
        ts = TerminalState()
        ts.write(b'\x1b[42mG')  # green background
        assert ts.chars[0, 0] == ord('G')
        assert ts.bg[0, 0] == 2  # green = color 2

    def test_sgr_bold_bright(self):
        ts = TerminalState()
        ts.write(b'\x1b[33;1mB')  # yellow then bold → 3+8 = 11
        assert ts.chars[0, 0] == ord('B')
        assert ts.fg[0, 0] == 11  # yellow (3) + bold (+8) = 11

    def test_sgr_reset(self):
        ts = TerminalState()
        ts.write(b'\x1b[31mR\x1b[0mN')
        assert ts.fg[0, 0] == 1   # red
        assert ts.fg[0, 1] == 7   # reset to default white

    def test_sgr_bright_fg(self):
        ts = TerminalState()
        ts.write(b'\x1b[91mR')  # bright red
        assert ts.fg[0, 0] == 9  # 91 - 82 = 9

    def test_sgr_bright_bg(self):
        ts = TerminalState()
        ts.write(b'\x1b[101mR')  # bright red bg
        assert ts.bg[0, 0] == 9  # 101 - 92 = 9

    def test_cursor_position(self):
        ts = TerminalState()
        ts.write(b'\x1b[5;10H*')  # move to row 5, col 10 (1-based)
        assert ts.chars[4, 9] == ord('*')  # 0-based: row 4, col 9

    def test_erase_display(self):
        ts = TerminalState()
        ts.write(b'Hello')
        ts.write(b'\x1b[2J')  # erase all
        assert (ts.chars == ord(' ')).all()

    def test_erase_line_to_end(self):
        ts = TerminalState()
        ts.write(b'Hello World')
        ts.write(b'\x1b[5;1H')   # move to beginning of line 5 (but we wrote on line 0)
        ts.write(b'\x1b[1;6H')   # cursor to row 1, col 6 (after "Hello")
        ts.write(b'\x1b[0K')     # erase to end of line
        assert ts.chars[0, 0] == ord('H')
        assert ts.chars[0, 5] == ord(' ')  # erased

    def test_cursor_movement(self):
        ts = TerminalState()
        ts.write(b'\x1b[5;5H')   # go to 5,5
        ts.write(b'\x1b[2A')     # up 2
        assert ts.cr == 2        # 4-2 = 2 (0-based)
        ts.write(b'\x1b[3B')     # down 3
        assert ts.cr == 5
        ts.write(b'\x1b[2C')     # forward 2
        assert ts.cc == 6        # 4+2 = 6 (0-based)
        ts.write(b'\x1b[3D')     # backward 3
        assert ts.cc == 3

    # ── UTF-8 / Unicode ──

    def test_utf8_box_drawing(self):
        ts = TerminalState()
        ts.write('═══'.encode('utf-8'))
        # Should map to '=' via fallback
        assert ts.chars[0, 0] == ord('=')
        assert ts.chars[0, 1] == ord('=')
        assert ts.chars[0, 2] == ord('=')

    def test_utf8_corners(self):
        ts = TerminalState()
        ts.write('╔╗╚╝'.encode('utf-8'))
        for i in range(4):
            assert ts.chars[0, i] == ord('+')

    def test_utf8_pipe(self):
        ts = TerminalState()
        ts.write('║'.encode('utf-8'))
        assert ts.chars[0, 0] == ord('|')

    def test_utf8_arrows(self):
        ts = TerminalState()
        ts.write('←↑→↓'.encode('utf-8'))
        assert ts.chars[0, 0] == ord('<')
        assert ts.chars[0, 1] == ord('^')
        assert ts.chars[0, 2] == ord('>')
        assert ts.chars[0, 3] == ord('v')

    def test_utf8_unknown_codepoint(self):
        ts = TerminalState()
        ts.write('日'.encode('utf-8'))  # CJK, no fallback
        assert ts.chars[0, 0] == ord('?')

    def test_utf8_invalid_continuation(self):
        ts = TerminalState()
        # Start a 2-byte sequence then send non-continuation
        ts.write(bytes([0xC0, 0x41]))  # 0xC0 starts 2-byte, 0x41='A' is not continuation
        # Should recover and output 'A'
        assert ts.chars[0, 0] == ord('A')

    def test_to_tensors(self):
        ts = TerminalState()
        ts.write(b'Hello')
        chars, fg, bg, cursor = ts.to_tensors('cpu')
        assert chars.shape == (TERM_ROWS, TERM_COLS)
        assert fg.shape == (TERM_ROWS, TERM_COLS)
        assert bg.shape == (TERM_ROWS, TERM_COLS)
        assert cursor.shape == (TERM_ROWS, TERM_COLS)
        assert chars[0, 0].item() == ord('H')
        assert cursor[0, 5].item() == True  # cursor at col 5
        assert cursor.sum().item() == 1


# ═══════════════════════════════════════════════════════════════════════════
# Neural Model Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestNeuralGlyphGenerator:
    def test_output_shape(self):
        g = NeuralGlyphGenerator()
        codes = torch.randint(0, N_CHARS, (4, 24, 80))
        out = g(codes)
        assert out.shape == (4, 24, 80, CELL_H, CELL_W)

    def test_output_range(self):
        g = NeuralGlyphGenerator()
        codes = torch.randint(0, N_CHARS, (2, 24, 80))
        out = g(codes)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_unbatched(self):
        g = NeuralGlyphGenerator()
        codes = torch.randint(0, N_CHARS, (24, 80))
        out = g(codes)
        assert out.shape == (24, 80, CELL_H, CELL_W)

    def test_different_chars_different_glyphs(self):
        g = NeuralGlyphGenerator()
        a = g(torch.tensor([[ord('A')]]))
        b = g(torch.tensor([[ord('B')]]))
        # Different characters should produce different glyphs
        assert not torch.allclose(a, b)

    def test_space_is_nearly_blank(self):
        """Space glyph (untrained) should have some structure but test it doesn't crash."""
        g = NeuralGlyphGenerator()
        sp = g(torch.tensor([ord(' ')]))
        assert sp.shape == (1, CELL_H, CELL_W)


class TestNeuralColorPalette:
    def test_output_shape(self):
        c = NeuralColorPalette()
        codes = torch.randint(0, N_COLORS, (4, 24, 80))
        out = c(codes)
        assert out.shape == (4, 24, 80, 3)

    def test_initial_palette_matches_ansi(self):
        c = NeuralColorPalette()
        for i, (r, g, b) in enumerate(ANSI_PALETTE):
            rgb = c(torch.tensor([i]))
            expected = torch.tensor([r / 255.0, g / 255.0, b / 255.0])
            assert torch.allclose(rgb.squeeze(), expected, atol=1e-5)

    def test_black_is_zero(self):
        c = NeuralColorPalette()
        black = c(torch.tensor([0]))
        assert torch.allclose(black.squeeze(), torch.zeros(3), atol=1e-5)

    def test_white_is_one(self):
        c = NeuralColorPalette()
        white = c(torch.tensor([15]))
        assert torch.allclose(white.squeeze(), torch.ones(3), atol=1e-5)


class TestNeuralCompositor:
    def test_output_shape(self):
        comp = NeuralCompositor()
        frame = torch.randn(2, 3, FRAME_H, FRAME_W)
        out = comp(frame)
        assert out.shape == (2, 3, FRAME_H, FRAME_W)

    def test_near_identity_at_init(self):
        comp = NeuralCompositor()
        frame = torch.rand(1, 3, 64, 64)
        out = comp(frame)
        # Zero-initialized, so output should equal clamped input
        assert torch.allclose(out, frame.clamp(0, 1), atol=1e-5)

    def test_output_clamped(self):
        comp = NeuralCompositor()
        frame = torch.randn(1, 3, 32, 32)  # may have negatives
        out = comp(frame)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestNeuralTerminalRenderer:
    def test_batched_output_shape(self):
        r = NeuralTerminalRenderer()
        chars = torch.randint(0, N_CHARS, (2, TERM_ROWS, TERM_COLS))
        fg = torch.randint(0, N_COLORS, (2, TERM_ROWS, TERM_COLS))
        bg = torch.zeros(2, TERM_ROWS, TERM_COLS, dtype=torch.long)
        out = r(chars, fg, bg)
        assert out.shape == (2, FRAME_H, FRAME_W, 3)

    def test_unbatched_output_shape(self):
        r = NeuralTerminalRenderer()
        chars = torch.randint(0, N_CHARS, (TERM_ROWS, TERM_COLS))
        fg = torch.randint(0, N_COLORS, (TERM_ROWS, TERM_COLS))
        bg = torch.zeros(TERM_ROWS, TERM_COLS, dtype=torch.long)
        out = r(chars, fg, bg)
        assert out.shape == (FRAME_H, FRAME_W, 3)

    def test_output_range(self):
        r = NeuralTerminalRenderer()
        chars = torch.randint(0, N_CHARS, (TERM_ROWS, TERM_COLS))
        fg = torch.full((TERM_ROWS, TERM_COLS), 7, dtype=torch.long)
        bg = torch.zeros(TERM_ROWS, TERM_COLS, dtype=torch.long)
        out = r(chars, fg, bg)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_cursor_mask(self):
        r = NeuralTerminalRenderer()
        chars = torch.full((TERM_ROWS, TERM_COLS), ord(' '), dtype=torch.long)
        fg = torch.full((TERM_ROWS, TERM_COLS), 7, dtype=torch.long)
        bg = torch.zeros(TERM_ROWS, TERM_COLS, dtype=torch.long)
        cursor = torch.zeros(TERM_ROWS, TERM_COLS, dtype=torch.bool)
        cursor[0, 0] = True
        out_with = r(chars, fg, bg, cursor)
        out_without = r(chars, fg, bg, None)
        # Cursor position should differ
        cell_pixels_with = out_with[:CELL_H, :CELL_W]
        cell_pixels_without = out_without[:CELL_H, :CELL_W]
        assert not torch.allclose(cell_pixels_with, cell_pixels_without)

    def test_render_state(self):
        r = NeuralTerminalRenderer()
        ts = TerminalState()
        ts.write(b'Test')
        frame = r.render_state(ts, 'cpu')
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (FRAME_H, FRAME_W, 3)
        assert frame.dtype == np.uint8

    def test_count_params(self):
        r = NeuralTerminalRenderer()
        n = r.count_params()
        assert n == 143539  # known param count


# ═══════════════════════════════════════════════════════════════════════════
# NeuralDisplay Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestNeuralDisplay:
    def test_init_without_model(self):
        """NeuralDisplay should work even without a trained model."""
        d = NeuralDisplay(model_path='/nonexistent/path.pt', device='cpu')
        assert d.device == 'cpu'
        assert isinstance(d.terminal, TerminalState)

    def test_write_and_render(self):
        d = NeuralDisplay(model_path='/nonexistent/path.pt', device='cpu')
        d.write(b'Hello World')
        frame = d.render()
        assert frame.shape == (FRAME_H, FRAME_W, 3)
        assert frame.dtype == np.uint8

    def test_render_rgba(self):
        d = NeuralDisplay(model_path='/nonexistent/path.pt', device='cpu')
        d.write(b'Test')
        data, w, h = d.render_rgba()
        assert w == FRAME_W
        assert h == FRAME_H
        assert len(data) == FRAME_W * FRAME_H * 4

    def test_render_text(self):
        d = NeuralDisplay(model_path='/nonexistent/path.pt', device='cpu')
        frame = d.render_text('Hello')
        assert frame.shape == (FRAME_H, FRAME_W, 3)

    def test_reset(self):
        d = NeuralDisplay(model_path='/nonexistent/path.pt', device='cpu')
        d.write(b'Hello')
        d.reset()
        assert d.terminal.cr == 0
        assert d.terminal.cc == 0
        assert (d.terminal.chars == ord(' ')).all()

    @pytest.mark.skipif(not MODEL_PATH.exists(), reason="Trained model not found")
    def test_with_trained_model(self):
        d = NeuralDisplay(str(MODEL_PATH), device='cpu')
        d.write(b'Hello World')
        frame = d.render()
        assert frame.shape == (FRAME_H, FRAME_W, 3)
        # Trained model should produce non-uniform output
        assert frame.std() > 5.0

    @pytest.mark.skipif(not MODEL_PATH.exists(), reason="Trained model not found")
    def test_ansi_colors_render_differently(self):
        d = NeuralDisplay(str(MODEL_PATH), device='cpu')
        # Red text
        d.write(b'\x1b[31mRed\x1b[0m')
        red_frame = d.render().copy()
        # Green text
        d.reset()
        d.write(b'\x1b[32mGrn\x1b[0m')
        green_frame = d.render()
        # Text region should differ in color
        text_region_red = red_frame[:CELL_H, :CELL_W*3]
        text_region_green = green_frame[:CELL_H, :CELL_W*3]
        assert not np.array_equal(text_region_red, text_region_green)


# ═══════════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_write(self):
        ts = TerminalState()
        ts.write(b'')
        assert ts.cr == 0 and ts.cc == 0

    def test_full_screen(self):
        ts = TerminalState()
        # Fill entire screen
        for r in range(TERM_ROWS):
            ts.write(b'X' * TERM_COLS)
            if r < TERM_ROWS - 1:
                ts.write(b'\n')
        assert (ts.chars == ord('X')).all()

    def test_many_scrolls(self):
        ts = TerminalState()
        for i in range(100):
            ts.write(f'Line {i}\n'.encode())
        # Should not crash, cursor should be at last row
        assert ts.cr == TERM_ROWS - 1

    def test_rapid_color_changes(self):
        ts = TerminalState()
        for color in range(8):
            ts.write(f'\x1b[{30+color}m{color}'.encode())
        # Check each char has correct color
        for i in range(8):
            assert ts.fg[0, i] == i

    def test_cursor_boundary_clamping(self):
        ts = TerminalState()
        ts.write(b'\x1b[999;999H')  # way out of bounds
        assert ts.cr == TERM_ROWS - 1
        assert ts.cc == TERM_COLS - 1

    def test_backspace_at_origin(self):
        ts = TerminalState()
        ts.write(b'\x08')  # backspace at 0,0
        assert ts.cc == 0  # clamped

    def test_all_printable_ascii(self):
        ts = TerminalState()
        chars = bytes(range(0x20, 0x7F))  # 95 chars, wraps at col 80
        ts.write(chars)
        for i, ch in enumerate(range(0x20, 0x7F)):
            r, c = divmod(i, TERM_COLS)
            assert ts.chars[r, c] == ch

    def test_mixed_ansi_and_text(self):
        ts = TerminalState()
        ts.write(b'\x1b[36;1m=====\x1b[0m\n\x1b[33mHello\x1b[0m')
        # First row: cyan then bold '='
        assert ts.chars[0, 0] == ord('=')
        assert ts.fg[0, 0] == 14  # cyan (6) + bold (+8) = 14
        # Second row: yellow 'H'
        assert ts.chars[1, 0] == ord('H')
        assert ts.fg[1, 0] == 3  # yellow = 3 (33-30)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
