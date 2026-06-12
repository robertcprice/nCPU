"""Tests for the driver's FIX-4 library-growth hook (compounding-loop closure).

Uses a tiny fake model/tokenizer with a *real* ``ArrayExecutableThoughtHead``
and ``ArrayProgramLibrary`` on CPU torch, so the hook's full path is
exercised: hidden-state capture → 5-tuple translation →
``record_successful_generation`` (or direct record fallback) → library save.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ncpu.autoresearch.driver import (
    _capture_target_hidden_state,
    _find_array_thought_coprocessor,
    _first_usable_pair,
    grow_library_from_solve,
)
from ncpu.autoresearch.types import IoPair, SolvedItem, WorkItem
from ncpu.self_optimizing.array_executable_thought_head import (
    ArrayExecutableThoughtHead,
    ArrayExecutableThoughtHeadConfig,
)
from ncpu.self_optimizing.array_program_library import ArrayProgramLibrary

_HIDDEN_DIM = 16


class _FakeTokenizerOutput(dict):
    def to(self, device):  # noqa: ARG002 — mirror BatchEncoding.to
        return self


class _FakeTokenizer:
    """Minimal tokenizer stand-in (no apply_chat_template → raw prompt path)."""

    def __call__(self, text: str, return_tensors: str = "pt"):  # noqa: ARG002
        n = max(2, min(len(text.split()), 6))
        return _FakeTokenizerOutput(
            input_ids=torch.ones(1, n, dtype=torch.long),
            attention_mask=torch.ones(1, n, dtype=torch.long),
        )


class _FakeArrayThought(torch.nn.Module):
    """Duck-typed stand-in for ArrayThoughtCoprocessor: carries a real head
    and a real attached library, receives [B, S, H] hidden states."""

    def __init__(self, head, library):
        super().__init__()
        self.array_head = head
        self.program_library = library

    def forward(self, hidden_states):
        return torch.zeros_like(hidden_states)


class _FakeModel(torch.nn.Module):
    def __init__(self, coproc):
        super().__init__()
        self.coproc = coproc
        self.embed = torch.nn.Embedding(8, _HIDDEN_DIM)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):  # noqa: ARG002
        hidden = self.embed(input_ids)
        return self.coproc(hidden)


def _build_stack():
    torch.manual_seed(42)
    head = ArrayExecutableThoughtHead(
        ArrayExecutableThoughtHeadConfig(hidden_dim=_HIDDEN_DIM, array_max_len=8)
    )
    library = ArrayProgramLibrary()
    coproc = _FakeArrayThought(head, library)
    model = _FakeModel(coproc).eval()
    return model, coproc, library


def _sum_work_item() -> WorkItem:
    prompt = (
        "def add_all(arr):\n"
        '    """Return the sum of all elements."""\n'
    )
    pairs = [
        IoPair(args=[[1, 2, 3]], kwargs={}, expected=6),
        IoPair(args=[[0, -4]], kwargs={}, expected=-4),
    ]
    return WorkItem(
        task_id="HumanEval/999",
        source_benchmark="humaneval",
        prompt=prompt,
        entry_point="add_all",
        test_source="",
        io_pairs=pairs,
    )


def _sum_solved_item() -> SolvedItem:
    return SolvedItem(
        task_id="HumanEval/999",
        source_benchmark="humaneval",
        solver="llm_resample",
        program_python="    return sum(arr)\n",
    )


class TestFindCoprocessor(unittest.TestCase):
    def test_finds_duck_typed_module(self):
        model, coproc, _ = _build_stack()
        self.assertIs(_find_array_thought_coprocessor(model), coproc)

    def test_none_without_library(self):
        model, coproc, _ = _build_stack()
        coproc.program_library = None
        self.assertIsNone(_find_array_thought_coprocessor(model))

    def test_none_on_plain_model(self):
        self.assertIsNone(_find_array_thought_coprocessor(torch.nn.Linear(4, 4)))


class TestHiddenCapture(unittest.TestCase):
    def test_captures_last_token_hidden(self):
        model, coproc, _ = _build_stack()
        hidden = _capture_target_hidden_state(
            model, _FakeTokenizer(), "cpu", "def add_all(arr):\n", coproc
        )
        self.assertIsNotNone(hidden)
        self.assertEqual(tuple(hidden.shape), (_HIDDEN_DIM,))
        # Deterministic: same prompt → same capture.
        again = _capture_target_hidden_state(
            model, _FakeTokenizer(), "cpu", "def add_all(arr):\n", coproc
        )
        self.assertTrue(torch.equal(hidden, again))


class TestFirstUsablePair(unittest.TestCase):
    def test_picks_first_int_array_pair(self):
        pairs = [
            IoPair(args=["abc"], kwargs={}, expected=1),       # wrong shape
            IoPair(args=[list(range(20))], kwargs={}, expected=190),  # too long
            IoPair(args=[[1, 2]], kwargs={}, expected=3),
        ]
        self.assertEqual(_first_usable_pair(pairs, 8), ([1, 2], 3.0))

    def test_none_when_no_usable_pair(self):
        self.assertIsNone(_first_usable_pair(
            [IoPair(args=["abc"], kwargs={}, expected="x")], 8
        ))


class TestGrowLibraryFromSolve(unittest.TestCase):
    def test_solve_grows_library_and_saves_json(self):
        model, _, library = _build_stack()
        solved = _sum_solved_item()
        with tempfile.TemporaryDirectory() as td:
            library_path = Path(td) / "library.json"
            msg = grow_library_from_solve(
                model=model,
                tokenizer=_FakeTokenizer(),
                device="cpu",
                work_item=_sum_work_item(),
                solved_item=solved,
                library_path=library_path,
            )
            self.assertIn("entries 0→1", msg)
            self.assertIn("library grew", msg)
            self.assertEqual(len(library), 1)
            self.assertTrue(library_path.exists())

            # The in-memory solved item now carries the 5-tuple.
            self.assertEqual(solved.program_5tuple, {
                "init_idx": 0, "transform_idx": 0, "reduce_idx": 0,
                "post_scale_idx": 0, "offset": 0.0,
            })

            # Round-trip: the saved JSON loads and the cached program
            # reproduces the solved behavior (sum) regardless of which
            # record path fired.
            reloaded = ArrayProgramLibrary.load(library_path)
            self.assertEqual(len(reloaded), 1)
            entry = reloaded.entries[0]
            self.assertEqual(entry.task_name, "HumanEval/999")
            arrays = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            lengths = torch.tensor([3])
            self.assertAlmostEqual(
                float(entry.program.execute(arrays, lengths).item()), 6.0,
                delta=0.5,
            )

            # The keyed entry is immediately usable: lookup with the same
            # captured hidden state fires.
            model_coproc = _find_array_thought_coprocessor(model)
            hidden = _capture_target_hidden_state(
                model, _FakeTokenizer(), "cpu",
                _sum_work_item().prompt, model_coproc,
            )
            self.assertIsNotNone(reloaded.lookup(hidden))

    def test_refuses_untranslatable_solve(self):
        model, _, library = _build_stack()
        solved = SolvedItem(
            task_id="HumanEval/998",
            source_benchmark="humaneval",
            solver="llm_resample",
            program_python="    return s[::-1]\n",
        )
        work = WorkItem(
            task_id="HumanEval/998",
            source_benchmark="humaneval",
            prompt="def rev(s):\n",
            entry_point="rev",
            test_source="",
            io_pairs=[IoPair(args=["abc"], kwargs={}, expected="cba")],
        )
        with tempfile.TemporaryDirectory() as td:
            library_path = Path(td) / "library.json"
            msg = grow_library_from_solve(
                model=model,
                tokenizer=_FakeTokenizer(),
                device="cpu",
                work_item=work,
                solved_item=solved,
                library_path=library_path,
            )
            self.assertIn("not translatable", msg)
            self.assertEqual(len(library), 0)
            self.assertFalse(library_path.exists())

    def test_refuses_without_work_item(self):
        model, _, library = _build_stack()
        msg = grow_library_from_solve(
            model=model,
            tokenizer=_FakeTokenizer(),
            device="cpu",
            work_item=None,
            solved_item=_sum_solved_item(),
            library_path=Path("/tmp/never_written.json"),
        )
        self.assertIn("no WorkItem", msg)
        self.assertEqual(len(library), 0)

    def test_refuses_without_coprocessor(self):
        msg = grow_library_from_solve(
            model=torch.nn.Linear(4, 4),
            tokenizer=_FakeTokenizer(),
            device="cpu",
            work_item=_sum_work_item(),
            solved_item=_sum_solved_item(),
            library_path=Path("/tmp/never_written.json"),
        )
        self.assertIn("no array-thought coprocessor", msg)


if __name__ == "__main__":
    unittest.main()
