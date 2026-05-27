"""Library-as-CV skill cards.

A "skill card" is a public, portable, signed artifact derived from an
`ArrayProgramLibrary`. It contains:

* The library's fingerprint (stable content hash).
* A short list of each skill's discrete program and what operation it
  implements.
* The compliance aggregate (safe / warn / high).
* An optional HMAC signature.
* A DP-perturbed version of the signatures if privacy is enabled.

The intended use is as a recruiting / publication artifact: a candidate
or research group shares a skill card publicly; readers can verify the
card's fingerprint against the underlying library (if they have it) and
the signature against a public key. The DP perturbation makes it
infeasible to recover training data from the card.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
)
from ncpu.self_optimizing.compliance_report import (
    ComplianceReportConfig,
    build_compliance_report,
)
from ncpu.self_optimizing.library_privacy import dp_perturb_library
from ncpu.self_optimizing.library_signing import sign_library


@dataclass
class SkillCardConfig:
    """How to produce a skill card."""

    owner: str
    title: Optional[str] = None
    dp_epsilon: Optional[float] = None      # None = no perturbation
    dp_delta: float = 1e-5
    signing_secret: Optional[bytes] = None   # None = no signature
    include_compliance: bool = True


def build_skill_card(
    library: ArrayProgramLibrary,
    *,
    config: SkillCardConfig,
) -> dict[str, Any]:
    """Build a publishable skill card from a library."""
    exported_library = library
    dp_certificate = None
    if config.dp_epsilon is not None:
        exported_library, dp_cert = dp_perturb_library(
            library,
            epsilon=config.dp_epsilon,
            delta=config.dp_delta,
            seed=0,
        )
        dp_certificate = {
            "epsilon": dp_cert.epsilon,
            "delta": dp_cert.delta,
            "sigma": dp_cert.sigma,
            "sensitivity": dp_cert.sensitivity,
        }

    card: dict[str, Any] = {
        "schema": "npcot-skill-card/1.0",
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "owner": config.owner,
        "title": config.title or f"{config.owner}'s reasoning skills",
        "fingerprint": exported_library.fingerprint(),
        "underlying_fingerprint": library.fingerprint(),
        "entry_count": len(exported_library),
        "dp_certificate": dp_certificate,
        "skills": [
            {
                "task_name": entry.task_name,
                "program": entry.program.to_dict(),
                "hit_count": int(entry.hit_count),
            }
            for entry in exported_library.entries
        ],
    }

    if config.include_compliance:
        card["compliance"] = build_compliance_report(
            exported_library,
            config=ComplianceReportConfig(
                library_name=config.title or config.owner,
            ),
        )["aggregate"]

    if config.signing_secret is not None:
        sig = sign_library(exported_library, config.signing_secret)
        card["signature"] = {
            "algorithm": sig["algorithm"],
            "digest": sig["digest"],
            "byte_length": sig["byte_length"],
        }

    return card


def render_skill_card_markdown(card: dict[str, Any]) -> str:
    """Human-readable skill card — the kind you'd put on a personal site."""
    lines: list[str] = [
        f"# {card['title']}",
        "",
        f"**Owner:** {card['owner']}",
        f"**Issued:** {card['issued_at']}",
        f"**Fingerprint:** `{card['fingerprint']}`",
        f"**Entries:** {card['entry_count']}",
    ]
    if card.get("compliance"):
        agg = card["compliance"]
        lines.append(
            f"**Compliance:** `{agg['aggregate_risk']}` "
            f"({agg['safe_entries']}/{agg['entry_count']} safe)"
        )
    if card.get("signature"):
        lines.append(
            f"**Signed:** `{card['signature']['algorithm']}` "
            f"`{card['signature']['digest'][:16]}…`"
        )
    if card.get("dp_certificate"):
        cert = card["dp_certificate"]
        lines.append(
            f"**Privacy:** ({cert['epsilon']}, {cert['delta']})-DP "
            f"(σ={cert['sigma']:.3f})"
        )

    lines.extend(["", "## Reasoning skills", ""])
    for idx, skill in enumerate(card["skills"], start=1):
        task = skill["task_name"] or "(unnamed)"
        lines.append(f"### {idx}. `{task}` · {skill['hit_count']} uses")
        lines.append("")
        lines.append("```rust")
        lines.append(skill["program"]["program_text"])
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def save_skill_card(card: dict[str, Any], *, path: Path) -> None:
    """Write a card to disk as canonical JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(card, indent=2, sort_keys=False),
        encoding="utf-8",
    )


def load_skill_card(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "SkillCardConfig",
    "build_skill_card",
    "render_skill_card_markdown",
    "save_skill_card",
    "load_skill_card",
]
