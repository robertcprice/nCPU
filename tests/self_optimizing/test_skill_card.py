"""Tests for library-as-CV skill cards (NV-CV)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    DiscreteArrayProgram,
)
from ncpu.self_optimizing.skill_card import (
    SkillCardConfig,
    build_skill_card,
    load_skill_card,
    render_skill_card_markdown,
    save_skill_card,
)


def _make_library() -> ArrayProgramLibrary:
    lib = ArrayProgramLibrary()
    lib.record(
        torch.tensor([1.0, 0.0, 0.0]),
        DiscreteArrayProgram(0, 0, 0, 0, 0.0),
        task_name="sum",
    )
    lib.record(
        torch.tensor([0.0, 1.0, 0.0]),
        DiscreteArrayProgram(2, 0, 2, 0, 0.0),
        task_name="max",
    )
    return lib


class TestBuildSkillCard(unittest.TestCase):
    def test_basic_card(self):
        lib = _make_library()
        card = build_skill_card(
            lib,
            config=SkillCardConfig(owner="Alice"),
        )
        self.assertEqual(card["owner"], "Alice")
        self.assertEqual(card["entry_count"], 2)
        self.assertTrue(card["fingerprint"].startswith("npcot1:"))
        self.assertIn("compliance", card)
        self.assertEqual(len(card["skills"]), 2)

    def test_unsigned_card_has_no_signature(self):
        card = build_skill_card(_make_library(), config=SkillCardConfig(owner="A"))
        self.assertNotIn("signature", card)

    def test_signed_card_has_digest(self):
        card = build_skill_card(
            _make_library(),
            config=SkillCardConfig(owner="A", signing_secret=b"my-key"),
        )
        self.assertIn("signature", card)
        self.assertEqual(card["signature"]["algorithm"], "hmac-sha256")
        self.assertEqual(len(card["signature"]["digest"]), 64)

    def test_dp_card_has_certificate(self):
        card = build_skill_card(
            _make_library(),
            config=SkillCardConfig(owner="A", dp_epsilon=1.0, dp_delta=1e-5),
        )
        self.assertIsNotNone(card["dp_certificate"])
        self.assertEqual(card["dp_certificate"]["epsilon"], 1.0)
        # After perturbation the card fingerprint differs from the original.
        self.assertNotEqual(card["fingerprint"], card["underlying_fingerprint"])

    def test_compliance_optional(self):
        card = build_skill_card(
            _make_library(),
            config=SkillCardConfig(owner="A", include_compliance=False),
        )
        self.assertNotIn("compliance", card)


class TestRenderMarkdown(unittest.TestCase):
    def test_markdown_has_title_and_skills(self):
        card = build_skill_card(_make_library(), config=SkillCardConfig(owner="Alice"))
        md = render_skill_card_markdown(card)
        self.assertIn("Alice", md)
        self.assertIn("Fingerprint:", md)
        self.assertIn("`sum`", md)
        self.assertIn("`max`", md)
        self.assertIn("```rust", md)


class TestSaveLoadRoundTrip(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "card.json"
            card = build_skill_card(_make_library(), config=SkillCardConfig(owner="A"))
            save_skill_card(card, path=path)
            loaded = load_skill_card(path)
            self.assertEqual(loaded["fingerprint"], card["fingerprint"])
            self.assertEqual(loaded["entry_count"], card["entry_count"])


if __name__ == "__main__":
    unittest.main()
