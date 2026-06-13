"""The resolve pipeline: English → verified program, with an honest boundary.

``resolve(english, proposer)`` runs:

1. **propose** — proposer (LLM or deterministic) emits a RequirementsIR.
   Untrusted.
2. **split** — the IR's io_examples are split into a *train* set (fed to the
   synthesizer) and a *holdout* set (never seen by synthesis). This is the
   CEGIS/holdout discipline used elsewhere in nCPU: matching held-out
   examples is evidence of real generalization, not memorization.
3. **synthesize** — the train examples go to the nsynth backend via the
   embeddable synthesis handler. The returned program is already verified by
   the Rust side against every train example.
4. **verify on holdout** — the synthesized program (its transpiled Python) is
   run against the holdout examples in a restricted namespace. A program that
   reproduces unseen examples generalized.
5. **cross-check** — if the proposer supplied a reference implementation, it
   is run against the holdout too; agreement between two independently-derived
   programs (bottom-up synthesis vs. the proposer's reference) is the
   strongest signal short of a proof.

The result states plainly what is and isn't trusted: the program is verified
against examples that *came from an untrusted proposer*. Holdout match and
reference agreement raise confidence; the pipeline never claims more than it
checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ncpu.requirements.ir import IoExample, RequirementsIR
from ncpu.requirements.proposer import Proposer, ProposerError


@dataclass
class ResolvedRequirement:
    english: str
    ir: Optional[RequirementsIR]
    status: str  # "synthesized" | "refused" | "unsupported" | "no_ir" | "no_examples"
    program: Optional[str] = None
    program_lang: Optional[str] = None  # "mog"
    transpiled: dict[str, str] = field(default_factory=dict)
    method: Optional[str] = None
    train_count: int = 0
    holdout_count: int = 0
    holdout_passed: int = 0
    reference_holdout_passed: Optional[int] = None
    synth_vs_reference_agree: Optional[bool] = None
    confidence: str = "none"  # none | low | medium | high
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "english": self.english,
            "status": self.status,
            "program": self.program,
            "program_lang": self.program_lang,
            "transpiled": self.transpiled,
            "method": self.method,
            "train_count": self.train_count,
            "holdout_count": self.holdout_count,
            "holdout_passed": self.holdout_passed,
            "reference_holdout_passed": self.reference_holdout_passed,
            "synth_vs_reference_agree": self.synth_vs_reference_agree,
            "confidence": self.confidence,
            "notes": self.notes,
            "ir": self.ir.to_dict() if self.ir else None,
        }


def _split(examples: list[IoExample]) -> tuple[list[IoExample], list[IoExample]]:
    """Hold out ~1/3 (at least 1, but never all) for generalization testing.

    Strided, not tail-sliced: every third example goes to the holdout set.
    A tail slice would put a whole region of the input domain (e.g. all the
    large-x examples of a piecewise rule) exclusively in the holdout, leaving
    train unable to pin the rule there — so even the correct program can't be
    distinguished from an overfit, and generalization looks like failure. A
    stride keeps both train and holdout spanning the full domain, which is the
    honest test: train on a representative sample, check unseen points drawn
    from the same spread."""
    n = len(examples)
    if n <= 2:
        return examples, []  # too few to spare any; train on all
    train, holdout = [], []
    for i, ex in enumerate(examples):
        (holdout if i % 3 == 2 else train).append(ex)
    if not holdout:  # n==2 handled above, but guard anyway
        return examples, []
    return train, holdout


def _safe_callable(source: str, entry_point: str):
    """Exec `source` in a minimal namespace and return the entry function.

    The synthesized Python and the proposer's reference are both executed
    here only to *check* outputs on small integer/list/string inputs. The
    namespace is restricted (a small builtins allowlist); these are not raw
    end-user strings, but we keep it tight regardless. Returns None on any
    failure."""
    allowed = {
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
        "range": range, "sorted": sorted, "map": map, "filter": filter,
        "list": list, "tuple": tuple, "set": set, "dict": dict, "enumerate": enumerate,
        "int": int, "float": float, "str": str, "bool": bool, "zip": zip,
        "all": all, "any": any, "reversed": reversed, "round": round,
        "divmod": divmod, "pow": pow,
    }
    ns: dict[str, Any] = {"__builtins__": allowed}
    try:
        exec(source, ns)  # noqa: S102 — restricted namespace, verification only
    except Exception:  # noqa: BLE001
        return None
    fn = ns.get(entry_point)
    return fn if callable(fn) else None


def _run_on(fn, examples: list[IoExample]) -> int:
    """Count examples where fn(*inputs) == expected (exact). Errors → no match."""
    passed = 0
    for ex in examples:
        try:
            got = fn(*ex.inputs)
        except Exception:  # noqa: BLE001
            continue
        if got == ex.expected:
            passed += 1
    return passed


def _confidence(
    *, holdout: int, holdout_pass: int, ref_pass: Optional[int], agree: Optional[bool]
) -> str:
    if holdout == 0:
        # nothing held out; verified only on train (synth guarantees that)
        return "low"
    if holdout_pass < holdout:
        return "low"  # failed to generalize on at least one unseen case
    # generalized on all holdout:
    if agree is True and ref_pass == holdout:
        return "high"  # synth + independent reference both match all holdout
    return "medium"


def resolve(
    english: str,
    *,
    proposer: Proposer,
    synth_timeout_s: float = 20.0,
    cross_check: bool = True,
) -> ResolvedRequirement:
    """Run the full English → verified-program pipeline."""
    try:
        ir = proposer.propose(english)
    except ProposerError as exc:
        return ResolvedRequirement(
            english=english, ir=None, status="no_ir", notes=[str(exc)]
        )

    problems = ir.validate()
    if any("no io_examples" in p for p in problems):
        return ResolvedRequirement(
            english=english, ir=ir, status="no_examples",
            notes=problems or ["proposer produced no examples"],
        )

    if not ir.synth_supported():
        return ResolvedRequirement(
            english=english, ir=ir, status="unsupported",
            notes=[
                "examples use types outside the synthesizer's i64/[i64]/string "
                "domain (e.g. list or tuple output) — the contract is captured "
                "but bottom-up synthesis can't target it yet."
            ] + problems,
        )

    train, holdout = _split(ir.io_examples)

    # --- synthesize from the train examples via the nsynth backend ----------
    from ncpu.synthesis_api.server import SynthConfig, handle_synthesize_request

    config = SynthConfig(timeout_s=synth_timeout_s, max_timeout_s=synth_timeout_s)
    if not config.backend.is_file():
        return ResolvedRequirement(
            english=english, ir=ir, status="no_ir",
            notes=["synth backend binary not found — build nsynth release"],
        )

    request = {
        "name": ir.entry_point,
        "signature": ir.signature_str(),
        "examples": [e.to_dict() for e in train],
    }
    status_code, payload = handle_synthesize_request(request, config)
    if status_code != 200 or not payload.get("success"):
        return ResolvedRequirement(
            english=english, ir=ir, status="refused",
            train_count=len(train), holdout_count=len(holdout),
            notes=[
                "synthesizer found no program reproducing the train examples — "
                "honest refusal.",
                payload.get("error") or "",
            ],
        )

    transpiled = payload.get("transpiled") or {}
    result = ResolvedRequirement(
        english=english, ir=ir, status="synthesized",
        program=payload.get("code"), program_lang="mog",
        transpiled=transpiled, method=payload.get("method"),
        train_count=len(train), holdout_count=len(holdout),
    )

    # --- verify the synthesized program on the held-out examples ------------
    synth_py = transpiled.get("python")
    if holdout and synth_py:
        fn = _safe_callable(synth_py, ir.entry_point)
        result.holdout_passed = _run_on(fn, holdout) if fn else 0

    # --- cross-check against the proposer's reference implementation --------
    ref_pass: Optional[int] = None
    agree: Optional[bool] = None
    if cross_check and ir.reference_impl and ir.reference_lang == "python" and holdout:
        ref_fn = _safe_callable(ir.reference_impl, ir.entry_point)
        synth_fn = _safe_callable(synth_py, ir.entry_point) if synth_py else None
        if ref_fn is not None:
            ref_pass = _run_on(ref_fn, holdout)
        if ref_fn is not None and synth_fn is not None:
            agree = all(
                _safe_eq(ref_fn, synth_fn, ex.inputs) for ex in holdout
            )
    result.reference_holdout_passed = ref_pass
    result.synth_vs_reference_agree = agree

    result.confidence = _confidence(
        holdout=len(holdout),
        holdout_pass=result.holdout_passed,
        ref_pass=ref_pass,
        agree=agree,
    )
    result.notes.append(
        "Program verified against the train examples by the synthesizer; "
        f"reproduced {result.holdout_passed}/{len(holdout)} held-out examples. "
        "Examples originated from an untrusted proposer."
    )
    return result


def _safe_eq(fn_a, fn_b, inputs: list[Any]) -> bool:
    try:
        return fn_a(*inputs) == fn_b(*inputs)
    except Exception:  # noqa: BLE001
        return False
