"""JSON bridge for the pure gradient Mog solver.

Reads a synthesis request from stdin as JSON and writes a JSON response to stdout.
This lets the Rust crate invoke the existing differentiable solver without
re-implementing the gradient path in Rust first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from egdc.mog.lang import interpret
from egdc.mog.solvers.gradient_solver import gradient_solve
from egdc.mog.solvers.soft_programs import (
    SoftInteractiveFilterProgram,
    SoftInteractivePairProgram,
    SoftInteractiveStateEmitProgram,
    SoftLatentEmitProgram,
    SoftLatentOutputProgram,
    SoftInteractiveTwoRegisterProgram,
    SoftInteractiveTwoRegisterEmitProgram,
    SoftInteractiveProgram,
    train_interactive_filter_program,
    train_interactive_pair_program,
    train_interactive_state_emit_program,
    train_latent_emit_program,
    train_latent_output_program,
    train_interactive_two_register_program,
    train_interactive_two_register_emit_program,
    train_interactive_program,
)


def _extract_arg_names(signature: str) -> list[str]:
    params = (
        signature.split("fn ", 1)[1]
        .split("(", 1)[1]
        .split(")", 1)[0]
        .strip()
    )
    if not params:
        return []
    names: list[str] = []
    for param in params.split(","):
        name = param.split(":", 1)[0].strip()
        if name:
            names.append(name)
    return names


def _coerce_examples(raw_examples: list[dict[str, Any]]) -> list[tuple[tuple[float, ...], float]] | None:
    out: list[tuple[tuple[float, ...], float]] = []
    for example in raw_examples:
        inputs = example.get("inputs", [])
        expected = example.get("expected")
        if not isinstance(inputs, list) or not isinstance(expected, int | float):
            return None
        scalar_inputs: list[float] = []
        for value in inputs:
            if not isinstance(value, int | float):
                return None
            scalar_inputs.append(float(value))
        out.append((tuple(scalar_inputs), float(expected)))
    return out


def _coerce_interactive_traces(
    raw_traces: list[dict[str, Any]],
) -> list[tuple[list[int], list[int]]] | None:
    traces: list[tuple[list[int], list[int]]] = []
    for raw_trace in raw_traces:
        input_stream = raw_trace.get("input_stream")
        expected_output = raw_trace.get("expected_output")
        if not isinstance(input_stream, list) or not isinstance(expected_output, list):
            return None
        coerced_input: list[int] = []
        coerced_output: list[int] = []
        for inp in input_stream:
            if not isinstance(inp, int):
                return None
            coerced_input.append(inp)
        for expected in expected_output:
            if not isinstance(expected, int):
                return None
            coerced_output.append(expected)
        traces.append((coerced_input, coerced_output))
    return traces


def _state_update_traces(
    traces: list[tuple[list[int], list[int]]],
) -> list[list[tuple[int, int]]] | None:
    if any(len(inputs) != len(outputs) for inputs, outputs in traces):
        return None
    return [
        list(zip(inputs, outputs, strict=True))
        for inputs, outputs in traces
    ]


def _pair_group_traces(
    traces: list[tuple[list[int], list[int]]],
) -> list[list[tuple[int, int, int]]] | None:
    grouped: list[list[tuple[int, int, int]]] = []
    for inputs, outputs in traces:
        if len(outputs) != len(inputs) // 2:
            return None
        pairs: list[tuple[int, int, int]] = []
        for idx, expected in enumerate(outputs):
            pairs.append((inputs[idx * 2], inputs[idx * 2 + 1], expected))
        grouped.append(pairs)
    return grouped


def _derive_passthrough_emit_targets(
    traces: list[tuple[list[int], list[int]]],
) -> list[list[int]] | None:
    emit_targets: list[list[int]] = []
    for inputs, outputs in traces:
        out_idx = 0
        targets: list[int] = []
        for inp in inputs:
            should_emit = out_idx < len(outputs) and inp == outputs[out_idx]
            targets.append(1 if should_emit else 0)
            if should_emit:
                out_idx += 1
        if out_idx != len(outputs):
            return None
        emit_targets.append(targets)
    return emit_targets


def _interactive_structure_name(mode: int) -> str:
    return {
        0: "interactive_state_add_input",
        1: "interactive_state_sub_input",
        2: "interactive_state_mul_input",
        3: "interactive_passthrough",
        4: "interactive_state_add_const",
        5: "interactive_scale_input",
        6: "interactive_constant_output",
        7: "interactive_running_max",
        8: "interactive_running_min",
    }.get(mode, "interactive_unknown")


def _interactive_filter_structure_name(mode: int) -> str:
    return {
        0: "interactive_filter_positive_passthrough",
        1: "interactive_filter_negative_passthrough",
        2: "interactive_filter_zero_passthrough",
        3: "interactive_filter_even_passthrough",
        4: "interactive_filter_odd_passthrough",
        5: "interactive_filter_always_passthrough",
        6: "interactive_filter_never_passthrough",
    }.get(mode, "interactive_filter_unknown")


def _interactive_pair_structure_name(mode: int) -> str:
    return {
        0: "interactive_pair_add",
        1: "interactive_pair_sub",
        2: "interactive_pair_rev_sub",
        3: "interactive_pair_mul",
        4: "interactive_pair_max",
        5: "interactive_pair_min",
        6: "interactive_pair_abs_diff",
        7: "interactive_pair_first",
        8: "interactive_pair_second",
    }.get(mode, "interactive_pair_unknown")


def _interactive_state_update_label(mode: int) -> str:
    return {
        0: "state_add_input",
        1: "state_sub_input",
        2: "state_mul_input",
        3: "passthrough",
        4: "state_add_const",
        5: "scale_input",
        6: "constant_output",
        7: "running_max",
        8: "running_min",
    }.get(mode, "unknown")


def _interactive_emit_label(mode: int) -> str:
    return {
        0: "always",
        1: "first_or_changed",
        2: "first_or_increased",
        3: "first_or_decreased",
        4: "input_positive",
        5: "state_positive",
        6: "first_only",
        7: "never",
    }.get(mode, "unknown")


def _interactive_state_emit_structure_name(update_mode: int, emit_mode: int) -> str:
    return (
        "interactive_state_emit_"
        f"{_interactive_state_update_label(update_mode)}_"
        f"{_interactive_emit_label(emit_mode)}"
    )


def _interactive_two_register_structure_name(a_mode: int, b_mode: int, out_mode: int) -> str:
    a_label = {
        0: "accum_input",
        1: "sub_input",
        2: "input",
        3: "running_max",
        4: "running_min",
        5: "add_const",
        6: "kadane_step",
        7: "anti_kadane_step",
    }.get(a_mode, "a_unknown")
    b_label = {
        0: "keep",
        1: "add_const",
        2: "input",
        3: "accum_input",
        4: "global_max_a",
        5: "global_min_a",
    }.get(b_mode, "b_unknown")
    out_label = {
        0: "out_a",
        1: "out_b",
        2: "out_add",
        3: "out_sub",
        4: "out_div",
    }.get(out_mode, "out_unknown")
    return f"interactive_two_register_{a_label}_{b_label}_{out_label}"


def _interactive_two_register_emit_structure_name(
    a_mode: int,
    b_mode: int,
    out_mode: int,
    emit_mode: int,
) -> str:
    emit_label = {
        0: "always",
        1: "input_positive",
        2: "input_negative",
        3: "output_positive",
        4: "reg_a_positive",
        5: "reg_b_positive",
        6: "first_only",
        7: "never",
        8: "first_or_output_changed",
        9: "first_or_output_increased",
        10: "first_or_output_decreased",
        11: "output_above_threshold",
        12: "output_crosses_above_threshold",
        13: "output_crosses_below_threshold",
        14: "reg_a_gt_reg_b",
        15: "reg_a_lt_reg_b",
        16: "output_gt_reg_b",
        17: "output_lt_reg_b",
        18: "reg_a_crosses_above_reg_b",
        19: "reg_a_crosses_below_reg_b",
        20: "output_crosses_above_reg_b",
        21: "output_crosses_below_reg_b",
        22: "reg_a_minus_reg_b_above_threshold",
        23: "reg_a_minus_reg_b_crosses_above_threshold",
        24: "output_minus_reg_b_above_threshold",
        25: "output_minus_reg_b_crosses_above_threshold",
    }.get(emit_mode, "emit_unknown")
    return (
        f"{_interactive_two_register_structure_name(a_mode, b_mode, out_mode)}_{emit_label}"
    )


def _two_register_emit_candidate_key(
    candidate: tuple[float, int, int, int, int, int, int, int, str],
) -> tuple[int, int, int, int, int, int, int, str]:
    return (
        candidate[1],
        candidate[2],
        candidate[3],
        candidate[4],
        candidate[5],
        candidate[6],
        candidate[7],
        candidate[8],
    )


def _latent_output_structure_name(out_mode: int) -> str:
    return {
        0: "latent_output_out_a",
        1: "latent_output_out_b",
        2: "latent_output_out_add",
        3: "latent_output_out_sub",
        4: "latent_output_out_div",
    }.get(out_mode, "latent_output_unknown")


def _latent_emit_structure_name(emit_mode: int) -> str:
    return {
        0: "latent_emit_always",
        1: "latent_emit_input_positive",
        2: "latent_emit_input_negative",
        3: "latent_emit_output_positive",
        4: "latent_emit_reg_a_positive",
        5: "latent_emit_reg_b_positive",
        6: "latent_emit_first_only",
        7: "latent_emit_never",
        8: "latent_emit_first_or_output_changed",
        9: "latent_emit_first_or_output_increased",
        10: "latent_emit_first_or_output_decreased",
        11: "latent_emit_output_above_threshold",
        12: "latent_emit_output_crosses_above_threshold",
        13: "latent_emit_output_crosses_below_threshold",
        14: "latent_emit_reg_a_gt_reg_b",
        15: "latent_emit_reg_a_lt_reg_b",
        16: "latent_emit_output_gt_reg_b",
        17: "latent_emit_output_lt_reg_b",
        18: "latent_emit_reg_a_crosses_above_reg_b",
        19: "latent_emit_reg_a_crosses_below_reg_b",
        20: "latent_emit_output_crosses_above_reg_b",
        21: "latent_emit_output_crosses_below_reg_b",
        22: "latent_emit_reg_a_minus_reg_b_above_threshold",
        23: "latent_emit_reg_a_minus_reg_b_crosses_above_threshold",
        24: "latent_emit_output_minus_reg_b_above_threshold",
        25: "latent_emit_output_minus_reg_b_crosses_above_threshold",
    }.get(emit_mode, "latent_emit_unknown")


def _two_register_emit_candidate_affinity(
    prog: SoftInteractiveTwoRegisterEmitProgram,
    *,
    a_mode: int,
    b_mode: int,
    out_mode: int,
    emit_mode: int,
    const_a: int,
    const_b: int,
    emit_threshold: int,
) -> float:
    a_log_probs = torch.log_softmax(prog.a_logits.detach(), dim=0)
    b_log_probs = torch.log_softmax(prog.b_logits.detach(), dim=0)
    out_log_probs = torch.log_softmax(prog.out_logits.detach(), dim=0)
    emit_log_probs = torch.log_softmax(prog.emit_logits.detach(), dim=0)
    affinity = float(
        a_log_probs[a_mode].item()
        + b_log_probs[b_mode].item()
        + out_log_probs[out_mode].item()
        + emit_log_probs[emit_mode].item()
    )
    if a_mode == 5:
        affinity -= abs(float(const_a) - float(prog.const_a.detach().item()))
    if b_mode == 1:
        affinity -= abs(float(const_b) - float(prog.const_b.detach().item()))
    if emit_mode in {11, 12, 13, 22, 23, 24, 25}:
        affinity -= abs(float(emit_threshold) - float(prog.emit_threshold.detach().item()))
    return affinity


_LATENT_EMIT_ARG_NAMES = [
    "inp",
    "reg_a",
    "reg_b",
    "out",
    "prev_out",
    "prev_reg_a",
    "prev_reg_b",
    "is_first",
]


def _round_candidates(value: float) -> list[int]:
    center = int(round(value))
    candidates = [center - 2, center - 1, center, center + 1, center + 2, 0, 1, -1]
    deduped: list[int] = []
    seen: set[int] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return deduped


def _data_candidates_for_mode(mode: int, traces: list[list[tuple[int, int]]]) -> list[int]:
    candidates: list[int] = []
    if mode == 4:
        for trace in traces:
            if trace:
                candidates.append(trace[0][1])
                if len(trace) > 1:
                    candidates.append(trace[1][1] - trace[0][1])
    elif mode == 5:
        for trace in traces:
            for inp, expected in trace:
                if inp != 0 and expected % inp == 0:
                    candidates.append(expected // inp)
    elif mode == 6:
        for trace in traces:
            for _, expected in trace:
                candidates.append(expected)
    deduped: list[int] = []
    seen: set[int] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return deduped


def _simulate_interactive_candidate(
    mode: int,
    trace: list[tuple[int, int]],
    *,
    const_a: int,
    const_b: int,
) -> list[int]:
    state = 0
    outputs: list[int] = []
    for idx, (inp, _) in enumerate(trace):
        if mode == 0:
            state = state + inp
        elif mode == 1:
            state = state - inp
        elif mode == 2:
            state = state * inp
        elif mode == 3:
            state = inp
        elif mode == 4:
            state = state + const_a
        elif mode == 5:
            state = inp * const_a
        elif mode == 6:
            state = const_b
        elif mode == 7:
            state = inp if idx == 0 else max(state, inp)
        elif mode == 8:
            state = inp if idx == 0 else min(state, inp)
        else:
            raise ValueError(f"unknown interactive mode {mode}")
        outputs.append(state)
    return outputs


def _step_interactive_update(
    mode: int,
    *,
    state: int,
    inp: int,
    is_first: bool,
    const_a: int,
    const_b: int,
) -> int:
    if mode == 0:
        return state + inp
    if mode == 1:
        return state - inp
    if mode == 2:
        return state * inp
    if mode == 3:
        return inp
    if mode == 4:
        return state + const_a
    if mode == 5:
        return inp * const_a
    if mode == 6:
        return const_b
    if mode == 7:
        return inp if is_first else max(state, inp)
    if mode == 8:
        return inp if is_first else min(state, inp)
    raise ValueError(f"unknown interactive mode {mode}")


def _should_emit_state(
    emit_mode: int,
    *,
    old_state: int,
    new_state: int,
    inp: int,
    is_first: bool,
) -> bool:
    return {
        0: True,
        1: is_first or new_state != old_state,
        2: is_first or new_state > old_state,
        3: is_first or new_state < old_state,
        4: inp > 0,
        5: new_state > 0,
        6: is_first,
        7: False,
    }[emit_mode]


def _simulate_interactive_state_emit_candidate(
    update_mode: int,
    emit_mode: int,
    trace: tuple[list[int], list[int]],
    *,
    const_a: int,
    const_b: int,
) -> list[int]:
    inputs, _ = trace
    state = 0
    outputs: list[int] = []
    for idx, inp in enumerate(inputs):
        old_state = state
        state = _step_interactive_update(
            update_mode,
            state=state,
            inp=inp,
            is_first=idx == 0,
            const_a=const_a,
            const_b=const_b,
        )
        if _should_emit_state(
            emit_mode,
            old_state=old_state,
            new_state=state,
            inp=inp,
            is_first=idx == 0,
        ):
            outputs.append(state)
    return outputs


def _step_two_registers(
    a_mode: int,
    b_mode: int,
    *,
    reg_a: int,
    reg_b: int,
    inp: int,
    is_first: bool,
    const_a: int,
    const_b: int,
) -> tuple[int, int]:
    next_a = {
        0: reg_a + inp,
        1: reg_a - inp,
        2: inp,
        3: inp if is_first else max(reg_a, inp),
        4: inp if is_first else min(reg_a, inp),
        5: reg_a + const_a,
        6: max(reg_a + inp, inp),   # Kadane step: restart if negative prefix
        7: min(reg_a + inp, inp),   # anti-Kadane step: restart if positive prefix
    }[a_mode]
    next_b = {
        0: reg_b,
        1: reg_b + const_b,
        2: inp,
        3: reg_b + inp,
        4: next_a if is_first else max(reg_b, next_a),  # global max of reg_a, init on first step
        5: next_a if is_first else min(reg_b, next_a),  # global min of reg_a, init on first step
    }[b_mode]
    return next_a, next_b


def _two_register_output(out_mode: int, *, reg_a: int, reg_b: int) -> int | None:
    if out_mode == 0:
        return reg_a
    if out_mode == 1:
        return reg_b
    if out_mode == 2:
        return reg_a + reg_b
    if out_mode == 3:
        return reg_a - reg_b
    if out_mode == 4:
        if reg_b == 0:
            return None
        return int(reg_a / reg_b)
    raise ValueError(f"unknown two-register output mode {out_mode}")


def _simulate_interactive_two_register_candidate(
    a_mode: int,
    b_mode: int,
    out_mode: int,
    trace: list[tuple[int, int]],
    *,
    const_a: int,
    const_b: int,
) -> list[int] | None:
    reg_a = 0
    reg_b = 0
    outputs: list[int] = []
    for idx, (inp, _) in enumerate(trace):
        reg_a, reg_b = _step_two_registers(
            a_mode,
            b_mode,
            reg_a=reg_a,
            reg_b=reg_b,
            inp=inp,
            is_first=idx == 0,
            const_a=const_a,
            const_b=const_b,
        )
        out = _two_register_output(out_mode, reg_a=reg_a, reg_b=reg_b)
        if out is None:
            return None
        outputs.append(out)
    return outputs


def _should_emit_two_register(
    emit_mode: int,
    *,
    inp: int,
    output: int,
    prev_output: int | None,
    prev_reg_a: int,
    prev_reg_b: int,
    reg_a: int,
    reg_b: int,
    emit_threshold: int,
    is_first: bool,
) -> bool:
    return {
        0: True,
        1: inp > 0,
        2: inp < 0,
        3: output > 0,
        4: reg_a > 0,
        5: reg_b > 0,
        6: is_first,
        7: False,
        8: is_first or (prev_output is not None and output != prev_output),
        9: is_first or (prev_output is not None and output > prev_output),
        10: is_first or (prev_output is not None and output < prev_output),
        11: output > emit_threshold,
        12: (not is_first) and (prev_output is not None) and prev_output <= emit_threshold and output > emit_threshold,
        13: (not is_first) and (prev_output is not None) and prev_output >= emit_threshold and output < emit_threshold,
        14: reg_a > reg_b,
        15: reg_a < reg_b,
        16: output > reg_b,
        17: output < reg_b,
        18: (not is_first) and prev_reg_a <= prev_reg_b and reg_a > reg_b,
        19: (not is_first) and prev_reg_a >= prev_reg_b and reg_a < reg_b,
        20: (not is_first) and (prev_output is not None) and prev_output <= prev_reg_b and output > reg_b,
        21: (not is_first) and (prev_output is not None) and prev_output >= prev_reg_b and output < reg_b,
        22: (reg_a - reg_b) > emit_threshold,
        23: (not is_first) and (prev_reg_a - prev_reg_b) <= emit_threshold and (reg_a - reg_b) > emit_threshold,
        24: (output - reg_b) > emit_threshold,
        25: (not is_first) and (prev_output is not None) and (prev_output - prev_reg_b) <= emit_threshold and (output - reg_b) > emit_threshold,
    }[emit_mode]


def _simulate_interactive_two_register_emit_candidate(
    a_mode: int,
    b_mode: int,
    out_mode: int,
    emit_mode: int,
    trace: tuple[list[int], list[int]],
    *,
    const_a: int,
    const_b: int,
    emit_threshold: int,
) -> list[int] | None:
    inputs, _ = trace
    reg_a = 0
    reg_b = 0
    outputs: list[int] = []
    prev_output: int | None = None
    for idx, inp in enumerate(inputs):
        prev_reg_a = reg_a
        prev_reg_b = reg_b
        reg_a, reg_b = _step_two_registers(
            a_mode,
            b_mode,
            reg_a=reg_a,
            reg_b=reg_b,
            inp=inp,
            is_first=idx == 0,
            const_a=const_a,
            const_b=const_b,
        )
        out = _two_register_output(out_mode, reg_a=reg_a, reg_b=reg_b)
        if out is None:
            return None
        if _should_emit_two_register(
            emit_mode,
            inp=inp,
            output=out,
            prev_output=prev_output,
            prev_reg_a=prev_reg_a,
            prev_reg_b=prev_reg_b,
            reg_a=reg_a,
            reg_b=reg_b,
            emit_threshold=emit_threshold,
            is_first=idx == 0,
        ):
            outputs.append(out)
        prev_output = out
    return outputs


def _simulate_interactive_two_register_emit_latents(
    a_mode: int,
    b_mode: int,
    out_mode: int,
    emit_mode: int,
    trace: tuple[list[int], list[int]],
    *,
    const_a: int,
    const_b: int,
    emit_threshold: int,
) -> dict[str, list[int]] | None:
    inputs, _ = trace
    reg_a = 0
    reg_b = 0
    prev_output: int | None = None
    rollout = {
        "inputs": list(inputs),
        "reg_a": [],
        "reg_b": [],
        "out": [],
        "emit": [],
        "prev_out": [],
        "prev_reg_a": [],
        "prev_reg_b": [],
        "is_first": [],
    }
    for idx, inp in enumerate(inputs):
        prev_reg_a = reg_a
        prev_reg_b = reg_b
        prev_out_value = prev_output if prev_output is not None else 0
        reg_a, reg_b = _step_two_registers(
            a_mode,
            b_mode,
            reg_a=reg_a,
            reg_b=reg_b,
            inp=inp,
            is_first=idx == 0,
            const_a=const_a,
            const_b=const_b,
        )
        out = _two_register_output(out_mode, reg_a=reg_a, reg_b=reg_b)
        if out is None:
            return None
        emit = int(
            _should_emit_two_register(
                emit_mode,
                inp=inp,
                output=out,
                prev_output=prev_output,
                prev_reg_a=prev_reg_a,
                prev_reg_b=prev_reg_b,
                reg_a=reg_a,
                reg_b=reg_b,
                emit_threshold=emit_threshold,
                is_first=idx == 0,
            )
        )
        rollout["reg_a"].append(reg_a)
        rollout["reg_b"].append(reg_b)
        rollout["out"].append(out)
        rollout["emit"].append(emit)
        rollout["prev_out"].append(prev_out_value)
        rollout["prev_reg_a"].append(prev_reg_a)
        rollout["prev_reg_b"].append(prev_reg_b)
        rollout["is_first"].append(int(idx == 0))
        prev_output = out
    return rollout


def _exact_ambiguity_metadata(
    exact_structures: set[str],
    *,
    exact_candidate_count: int | None = None,
) -> dict[str, Any]:
    exact_alternatives = sorted(exact_structures)
    metadata: dict[str, Any] = {
        "ambiguity_count": len(exact_alternatives),
        "exact_alternatives": exact_alternatives[:8],
    }
    if exact_candidate_count is not None:
        metadata["exact_candidate_count"] = exact_candidate_count
    return metadata


def _dense_traces_from_rollouts(
    rollouts: list[dict[str, list[int]]],
    key: str,
) -> list[tuple[list[int], list[int]]]:
    return [(rollout["inputs"], rollout[key]) for rollout in rollouts]


def _state_update_from_rollouts(
    rollouts: list[dict[str, list[int]]],
    key: str,
) -> list[list[tuple[int, int]]]:
    traces = _state_update_traces(_dense_traces_from_rollouts(rollouts, key))
    if traces is None:
        raise ValueError(f"latent rollout for {key} is not a dense state update trace")
    return traces


def _latent_two_register_output_examples(
    rollouts: list[dict[str, list[int]]],
) -> list[tuple[tuple[float, ...], float]]:
    examples: list[tuple[tuple[float, ...], float]] = []
    for rollout in rollouts:
        for reg_a, reg_b, out in zip(
            rollout["reg_a"],
            rollout["reg_b"],
            rollout["out"],
            strict=True,
        ):
            examples.append(((float(reg_a), float(reg_b)), float(out)))
    return examples


def _latent_two_register_emit_examples(
    rollouts: list[dict[str, list[int]]],
) -> list[tuple[tuple[float, ...], float]]:
    examples: list[tuple[tuple[float, ...], float]] = []
    for rollout in rollouts:
        for values in zip(
            rollout["inputs"],
            rollout["reg_a"],
            rollout["reg_b"],
            rollout["out"],
            rollout["prev_out"],
            rollout["prev_reg_a"],
            rollout["prev_reg_b"],
            rollout["is_first"],
            rollout["emit"],
            strict=True,
        ):
            inp, reg_a, reg_b, out, prev_out, prev_reg_a, prev_reg_b, is_first, emit = values
            examples.append(
                (
                    (
                        float(inp),
                        float(reg_a),
                        float(reg_b),
                        float(out),
                        float(prev_out),
                        float(prev_reg_a),
                        float(prev_reg_b),
                        float(is_first),
                    ),
                    float(emit),
                )
            )
    return examples


def _latent_output_mode_loss(
    out_mode: int,
    examples: list[tuple[tuple[float, ...], float]],
) -> float:
    total_loss = 0.0
    for args, expected in examples:
        reg_a = int(round(args[0]))
        reg_b = int(round(args[1]))
        actual = _two_register_output(out_mode, reg_a=reg_a, reg_b=reg_b)
        if actual is None:
            return float("inf")
        total_loss += float((actual - expected) ** 2)
    return total_loss / max(len(examples), 1)


def _threshold_candidates_from_latent_emit_examples(
    examples: list[tuple[tuple[float, ...], float]],
    *,
    relation: str,
) -> list[int]:
    candidates = [0]
    seen = {0}
    for args, _ in examples:
        inp, reg_a, reg_b, output, prev_output, prev_reg_a, prev_reg_b, is_first = args
        if relation == "output":
            value = output
        elif relation == "reg_gap":
            value = reg_a - reg_b
        elif relation == "output_gap":
            value = output - reg_b
        else:
            raise ValueError(f"unknown latent emit threshold relation {relation}")
        center = int(round(value))
        for candidate in (center - 1, center, center + 1):
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    return candidates


def _latent_emit_mode_loss(
    emit_mode: int,
    examples: list[tuple[tuple[float, ...], float]],
    *,
    emit_threshold: int,
) -> float:
    total_loss = 0.0
    for args, expected in examples:
        inp, reg_a, reg_b, output, prev_output, prev_reg_a, prev_reg_b, is_first = args
        actual = int(
            _should_emit_two_register(
                emit_mode,
                inp=int(round(inp)),
                output=int(round(output)),
                prev_output=None if int(round(is_first)) == 1 else int(round(prev_output)),
                prev_reg_a=int(round(prev_reg_a)),
                prev_reg_b=int(round(prev_reg_b)),
                reg_a=int(round(reg_a)),
                reg_b=int(round(reg_b)),
                emit_threshold=emit_threshold,
                is_first=bool(int(round(is_first))),
            )
        )
        total_loss += float((actual - expected) ** 2)
    return total_loss / max(len(examples), 1)


def _solve_latent_output_examples(
    examples: list[tuple[tuple[float, ...], float]],
    *,
    steps: int,
    num_restarts: int,
    seed: int,
) -> dict[str, Any]:
    if not examples:
        return {
            "success": False,
            "loss": float("inf"),
            "structure": "latent_output_failed",
        }

    best_soft_loss = float("inf")
    best_candidate: tuple[float, int, str] | None = None
    exact_structures: set[str] = set()

    for restart in range(max(num_restarts, 1)):
        restart_seed = seed + restart * 977
        torch.manual_seed(restart_seed)
        prog = SoftLatentOutputProgram()
        soft_loss = train_latent_output_program(
            prog,
            examples,
            steps=steps,
            seed=restart_seed,
        )
        for out_mode in range(SoftLatentOutputProgram.NUM_OUTPUT_CANDIDATES):
            loss = _latent_output_mode_loss(out_mode, examples)
            candidate = (loss, out_mode, _latent_output_structure_name(out_mode))
            if loss < 1e-8:
                exact_structures.add(candidate[-1])
            if best_candidate is None or candidate < best_candidate:
                best_candidate = candidate
                best_soft_loss = soft_loss

    if best_candidate is None:
        return {
            "success": False,
            "loss": float("inf"),
            "structure": "latent_output_failed",
        }

    loss, out_mode, structure = best_candidate
    return {
        "success": loss < 1e-8,
        "loss": loss,
        "structure": structure,
        "metadata": {
            "out_mode": out_mode,
            "soft_loss": best_soft_loss,
            **_exact_ambiguity_metadata(exact_structures),
        },
    }


def _solve_latent_emit_examples(
    examples: list[tuple[tuple[float, ...], float]],
    *,
    steps: int,
    num_restarts: int,
    seed: int,
) -> dict[str, Any]:
    if not examples:
        return {
            "success": False,
            "loss": float("inf"),
            "structure": "latent_emit_failed",
        }

    best_soft_loss = float("inf")
    best_candidate: tuple[float, int, int, str] | None = None
    exact_structures: set[str] = set()

    for restart in range(max(num_restarts, 1)):
        restart_seed = seed + restart * 977
        torch.manual_seed(restart_seed)
        prog = SoftLatentEmitProgram()
        soft_loss = train_latent_emit_program(
            prog,
            examples,
            steps=steps,
            seed=restart_seed,
        )
        learned_threshold = float(prog.emit_threshold.item())
        output_thresholds = _threshold_candidates_from_latent_emit_examples(
            examples,
            relation="output",
        )
        reg_gap_thresholds = _threshold_candidates_from_latent_emit_examples(
            examples,
            relation="reg_gap",
        )
        output_gap_thresholds = _threshold_candidates_from_latent_emit_examples(
            examples,
            relation="output_gap",
        )
        for emit_mode in range(SoftLatentEmitProgram.NUM_EMIT_CANDIDATES):
            threshold_values = [0]
            if emit_mode in {11, 12, 13}:
                threshold_values = _round_candidates(learned_threshold) + output_thresholds
            elif emit_mode in {22, 23}:
                threshold_values = _round_candidates(learned_threshold) + reg_gap_thresholds
            elif emit_mode in {24, 25}:
                threshold_values = _round_candidates(learned_threshold) + output_gap_thresholds
            seen_thresholds: set[int] = set()
            for emit_threshold in threshold_values:
                if emit_threshold in seen_thresholds:
                    continue
                seen_thresholds.add(emit_threshold)
                loss = _latent_emit_mode_loss(
                    emit_mode,
                    examples,
                    emit_threshold=emit_threshold,
                )
                candidate = (
                    loss,
                    emit_mode,
                    emit_threshold,
                    _latent_emit_structure_name(emit_mode),
                )
                if loss < 1e-8:
                    exact_structures.add(candidate[-1])
                if best_candidate is None or candidate < best_candidate:
                    best_candidate = candidate
                    best_soft_loss = soft_loss

    if best_candidate is None:
        return {
            "success": False,
            "loss": float("inf"),
            "structure": "latent_emit_failed",
        }

    loss, emit_mode, emit_threshold, structure = best_candidate
    return {
        "success": loss < 1e-8,
        "loss": loss,
        "structure": structure,
        "metadata": {
            "emit_mode": emit_mode,
            "emit_threshold": emit_threshold,
            "soft_loss": best_soft_loss,
            **_exact_ambiguity_metadata(exact_structures),
        },
    }


def _dict_result_loss(result: dict[str, Any], *, default: float = 1e6) -> float:
    loss = result.get("loss")
    if isinstance(loss, int | float):
        return float(loss)
    return 0.0 if result.get("success") else default


def _dict_result_ambiguity(result: dict[str, Any]) -> int:
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        return 0
    ambiguity = metadata.get("ambiguity_count", 0)
    if isinstance(ambiguity, int):
        return ambiguity
    return 0


def _recursive_refine_two_register_emit_candidate(
    candidate: tuple[float, int, int, int, int, int, int, int, str],
    traces: list[tuple[list[int], list[int]]],
    *,
    steps: int,
    num_restarts: int,
    seed: int,
    core_cache: dict[tuple[int, int, int, int, int], dict[str, Any]] | None = None,
    include_core: bool = True,
) -> dict[str, Any]:
    (
        _loss,
        a_mode,
        b_mode,
        out_mode,
        emit_mode,
        const_a,
        const_b,
        emit_threshold,
        structure,
    ) = candidate
    rollouts: list[dict[str, list[int]]] = []
    for trace in traces:
        rollout = _simulate_interactive_two_register_emit_latents(
            a_mode,
            b_mode,
            out_mode,
            emit_mode,
            trace,
            const_a=const_a,
            const_b=const_b,
            emit_threshold=emit_threshold,
        )
        if rollout is None:
            return {
                "sort_key": (5, float("inf"), float("inf"), structure),
                "metadata": {
                    "structure": structure,
                    "error": "latent rollout failed during recursive refinement",
                },
            }
        rollouts.append(rollout)

    refine_steps = max(32, min(steps, 64))
    refine_restarts = 1

    core_key = (a_mode, b_mode, out_mode, const_a, const_b)
    latent_emit_examples = _latent_two_register_emit_examples(rollouts)
    emit_result = _solve_latent_emit_examples(
        latent_emit_examples,
        steps=refine_steps,
        num_restarts=refine_restarts,
        seed=seed + 37,
    )

    if not include_core:
        meaningful_key = (
            int(not emit_result.get("success")),
            _dict_result_loss(emit_result),
        )
        return {
            "meaningful_key": meaningful_key,
            "core_key": core_key,
            "metadata": {
                "structure": structure,
                "emit_only": True,
                "emit": {
                    "success": bool(emit_result.get("success")),
                    "structure": emit_result.get("structure"),
                    "loss": _dict_result_loss(emit_result),
                    "ambiguity_count": _dict_result_ambiguity(emit_result),
                },
            },
        }

    core_refinement = None if core_cache is None else core_cache.get(core_key)
    if core_refinement is None:
        reg_a_result = _solve_interactive(
            _state_update_from_rollouts(rollouts, "reg_a"),
            steps=refine_steps,
            num_restarts=refine_restarts,
            seed=seed + 11,
        )
        reg_b_result = _solve_interactive(
            _state_update_from_rollouts(rollouts, "reg_b"),
            steps=refine_steps,
            num_restarts=refine_restarts,
            seed=seed + 17,
        )
        dense_out_result = _solve_interactive_two_register(
            _state_update_from_rollouts(rollouts, "out"),
            steps=refine_steps,
            num_restarts=refine_restarts,
            seed=seed + 23,
        )
        latent_out_result = _solve_latent_output_examples(
            _latent_two_register_output_examples(rollouts),
            steps=refine_steps,
            num_restarts=refine_restarts,
            seed=seed + 29,
        )
        core_refinement = {
            "reg_a_result": reg_a_result,
            "reg_b_result": reg_b_result,
            "dense_out_result": dense_out_result,
            "latent_out_result": latent_out_result,
        }
        if core_cache is not None:
            core_cache[core_key] = core_refinement
    else:
        reg_a_result = core_refinement["reg_a_result"]
        reg_b_result = core_refinement["reg_b_result"]
        dense_out_result = core_refinement["dense_out_result"]
        latent_out_result = core_refinement["latent_out_result"]

    failure_count = (
        int(not reg_a_result.get("success"))
        + int(not reg_b_result.get("success"))
        + int(not dense_out_result.get("success"))
        + int(not latent_out_result.get("success"))
        + int(not emit_result.get("success"))
    )
    total_loss = (
        _dict_result_loss(reg_a_result)
        + _dict_result_loss(reg_b_result)
        + _dict_result_loss(dense_out_result)
        + _dict_result_loss(latent_out_result)
        + _dict_result_loss(emit_result)
    )
    total_ambiguity = (
        _dict_result_ambiguity(reg_a_result)
        + _dict_result_ambiguity(reg_b_result)
        + _dict_result_ambiguity(dense_out_result)
    )
    meaningful_key = (
        failure_count,
        total_loss,
        total_ambiguity,
        _dict_result_loss(emit_result),
        _dict_result_loss(latent_out_result),
    )
    return {
        "meaningful_key": meaningful_key,
        "metadata": {
            "structure": structure,
            "failure_count": failure_count,
            "total_loss": total_loss,
            "total_ambiguity": total_ambiguity,
            "reg_a": {
                "success": bool(reg_a_result.get("success")),
                "structure": reg_a_result.get("structure"),
                "loss": _dict_result_loss(reg_a_result),
                "ambiguity_count": _dict_result_ambiguity(reg_a_result),
            },
            "reg_b": {
                "success": bool(reg_b_result.get("success")),
                "structure": reg_b_result.get("structure"),
                "loss": _dict_result_loss(reg_b_result),
                "ambiguity_count": _dict_result_ambiguity(reg_b_result),
            },
            "dense_out": {
                "success": bool(dense_out_result.get("success")),
                "structure": dense_out_result.get("structure"),
                "loss": _dict_result_loss(dense_out_result),
                "ambiguity_count": _dict_result_ambiguity(dense_out_result),
            },
            "latent_out": {
                "success": bool(latent_out_result.get("success")),
                "structure": latent_out_result.get("structure"),
                "loss": _dict_result_loss(latent_out_result),
                "ambiguity_count": _dict_result_ambiguity(latent_out_result),
            },
            "emit": {
                "success": bool(emit_result.get("success")),
                "structure": emit_result.get("structure"),
                "loss": _dict_result_loss(emit_result),
                "ambiguity_count": _dict_result_ambiguity(emit_result),
            },
        },
    }


def _candidate_loss(
    mode: int,
    traces: list[list[tuple[int, int]]],
    *,
    const_a: int,
    const_b: int,
) -> float:
    total_loss = 0.0
    total_steps = 0
    for trace in traces:
        actual = _simulate_interactive_candidate(
            mode, trace, const_a=const_a, const_b=const_b
        )
        for pred, (_, expected) in zip(actual, trace, strict=True):
            total_loss += float((pred - expected) ** 2)
            total_steps += 1
    return total_loss / max(total_steps, 1)


def _render_interactive_code(mode: int, *, const_a: int, const_b: int) -> str:
    lines = ["fn main() -> i64 {"]
    if mode in {0, 1, 2, 4, 7, 8}:
        lines.append("    state: i64 = 0;")
    if mode in {7, 8}:
        lines.append("    started: i64 = 0;")
    lines.append("    while has_input() == 1 {")
    lines.append("        x := read_i64();")
    if mode == 0:
        lines.append("        state = state + x;")
        lines.append("        println_i64(state);")
    elif mode == 1:
        lines.append("        state = state - x;")
        lines.append("        println_i64(state);")
    elif mode == 2:
        lines.append("        state = state * x;")
        lines.append("        println_i64(state);")
    elif mode == 3:
        lines.append("        println_i64(x);")
    elif mode == 4:
        lines.append(f"        state = state + {const_a};")
        lines.append("        println_i64(state);")
    elif mode == 5:
        lines.append(f"        println_i64(x * {const_a});")
    elif mode == 6:
        lines.append(f"        println_i64({const_b});")
    elif mode == 7:
        lines.append("        if started == 0 {")
        lines.append("            state = x;")
        lines.append("            started = 1;")
        lines.append("        }")
        lines.append("        if x > state {")
        lines.append("            state = x;")
        lines.append("        }")
        lines.append("        println_i64(state);")
    elif mode == 8:
        lines.append("        if started == 0 {")
        lines.append("            state = x;")
        lines.append("            started = 1;")
        lines.append("        }")
        lines.append("        if x < state {")
        lines.append("            state = x;")
        lines.append("        }")
        lines.append("        println_i64(state);")
    else:
        raise ValueError(f"unknown interactive mode {mode}")
    lines.append("    }")
    lines.append("    return 0;")
    lines.append("}")
    return "\n".join(lines)


def _verify_interactive_code(code: str, traces: list[list[tuple[int, int]]]) -> bool:
    for trace in traces:
        result = interpret(
            code,
            input_data=[str(inp) for inp, _ in trace],
        )
        if not result.success:
            return False
        expected = [str(expected) for _, expected in trace]
        actual = result.output.splitlines()
        if actual != expected:
            return False
    return True


def _verify_interactive_stream_code(
    code: str,
    traces: list[tuple[list[int], list[int]]],
) -> bool:
    for inputs, expected_outputs in traces:
        result = interpret(
            code,
            input_data=[str(inp) for inp in inputs],
        )
        if not result.success:
            return False
        expected = [str(value) for value in expected_outputs]
        actual = result.output.splitlines()
        if actual != expected:
            return False
    return True


def _pair_mode_loss(
    mode: int,
    traces: list[list[tuple[int, int, int]]],
) -> float:
    total_loss = 0.0
    total_steps = 0
    for trace in traces:
        for a, b, expected in trace:
            actual = {
                0: a + b,
                1: a - b,
                2: b - a,
                3: a * b,
                4: max(a, b),
                5: min(a, b),
                6: abs(a - b),
                7: a,
                8: b,
            }[mode]
            total_loss += float((actual - expected) ** 2)
            total_steps += 1
    return total_loss / max(total_steps, 1)


def _render_interactive_pair_code(mode: int) -> str:
    lines = [
        "fn main() -> i64 {",
        "    buf: i64 = 0;",
        "    have_buf: i64 = 0;",
        "    while has_input() == 1 {",
        "        x := read_i64();",
        "        if have_buf == 0 {",
        "            buf = x;",
        "            have_buf = 1;",
        "        } else {",
    ]
    if mode == 0:
        lines.append("            println_i64(buf + x);")
    elif mode == 1:
        lines.append("            println_i64(buf - x);")
    elif mode == 2:
        lines.append("            println_i64(x - buf);")
    elif mode == 3:
        lines.append("            println_i64(buf * x);")
    elif mode == 4:
        lines.append("            result := buf;")
        lines.append("            if x > buf {")
        lines.append("                result = x;")
        lines.append("            }")
        lines.append("            println_i64(result);")
    elif mode == 5:
        lines.append("            result := buf;")
        lines.append("            if x < buf {")
        lines.append("                result = x;")
        lines.append("            }")
        lines.append("            println_i64(result);")
    elif mode == 6:
        lines.append("            result := buf - x;")
        lines.append("            if result < 0 {")
        lines.append("                result = 0 - result;")
        lines.append("            }")
        lines.append("            println_i64(result);")
    elif mode == 7:
        lines.append("            println_i64(buf);")
    elif mode == 8:
        lines.append("            println_i64(x);")
    else:
        raise ValueError(f"unknown pair interactive mode {mode}")
    lines.extend([
        "            have_buf = 0;",
        "        }",
        "    }",
        "    return 0;",
        "}",
    ])
    return "\n".join(lines)


def _sequence_loss(actual: list[int], expected: list[int]) -> float:
    total = 0.0
    for pred, target in zip(actual, expected, strict=False):
        total += float((pred - target) ** 2)
    total += 1e6 * abs(len(actual) - len(expected))
    return total / max(len(actual), len(expected), 1)


def _threshold_candidates_from_outputs(traces: list[tuple[list[int], list[int]]]) -> list[int]:
    candidates = [0]
    seen: set[int] = {0}
    for _, outputs in traces:
        for value in outputs:
            for candidate in (value - 1, value, value + 1):
                if candidate not in seen:
                    seen.add(candidate)
                    candidates.append(candidate)
    return candidates


def _threshold_candidates_from_two_register_relation(
    traces: list[tuple[list[int], list[int]]],
    *,
    a_mode: int,
    b_mode: int,
    out_mode: int,
    const_a: int,
    const_b: int,
    relation: str,
) -> list[int]:
    candidates = [0]
    seen: set[int] = {0}
    for inputs, _ in traces:
        reg_a = 0
        reg_b = 0
        for idx, inp in enumerate(inputs):
            reg_a, reg_b = _step_two_registers(
                a_mode,
                b_mode,
                reg_a=reg_a,
                reg_b=reg_b,
                inp=inp,
                is_first=idx == 0,
                const_a=const_a,
                const_b=const_b,
            )
            out = _two_register_output(out_mode, reg_a=reg_a, reg_b=reg_b)
            if out is None:
                return candidates
            if relation == "reg_gap":
                value = reg_a - reg_b
            elif relation == "output_gap":
                value = out - reg_b
            else:
                raise ValueError(f"unknown threshold relation {relation}")
            for candidate in (value - 1, value, value + 1):
                if candidate not in seen:
                    seen.add(candidate)
                    candidates.append(candidate)
    return candidates


def _render_interactive_state_emit_code(
    update_mode: int,
    emit_mode: int,
    *,
    const_a: int,
    const_b: int,
) -> str:
    lines = [
        "fn main() -> i64 {",
        "    state: i64 = 0;",
        "    started: i64 = 0;",
        "    while has_input() == 1 {",
        "        x := read_i64();",
        "        prev_state := state;",
        "        was_started := started;",
        "        started = 1;",
    ]

    if update_mode == 0:
        lines.append("        state = state + x;")
    elif update_mode == 1:
        lines.append("        state = state - x;")
    elif update_mode == 2:
        lines.append("        state = state * x;")
    elif update_mode == 3:
        lines.append("        state = x;")
    elif update_mode == 4:
        lines.append(f"        state = state + {const_a};")
    elif update_mode == 5:
        lines.append(f"        state = x * {const_a};")
    elif update_mode == 6:
        lines.append(f"        state = {const_b};")
    elif update_mode == 7:
        lines.append("        if was_started == 0 {")
        lines.append("            state = x;")
        lines.append("        }")
        lines.append("        if was_started == 1 {")
        lines.append("            if x > state {")
        lines.append("                state = x;")
        lines.append("            }")
        lines.append("        }")
    elif update_mode == 8:
        lines.append("        if was_started == 0 {")
        lines.append("            state = x;")
        lines.append("        }")
        lines.append("        if was_started == 1 {")
        lines.append("            if x < state {")
        lines.append("                state = x;")
        lines.append("            }")
        lines.append("        }")
    else:
        raise ValueError(f"unknown interactive update mode {update_mode}")

    if emit_mode == 0:
        lines.append("        println_i64(state);")
    elif emit_mode == 1:
        lines.append("        emit := 0;")
        lines.append("        if was_started == 0 { emit = 1; }")
        lines.append("        if state != prev_state { emit = 1; }")
        lines.append("        if emit == 1 { println_i64(state); }")
    elif emit_mode == 2:
        lines.append("        emit := 0;")
        lines.append("        if was_started == 0 { emit = 1; }")
        lines.append("        if state > prev_state { emit = 1; }")
        lines.append("        if emit == 1 { println_i64(state); }")
    elif emit_mode == 3:
        lines.append("        emit := 0;")
        lines.append("        if was_started == 0 { emit = 1; }")
        lines.append("        if state < prev_state { emit = 1; }")
        lines.append("        if emit == 1 { println_i64(state); }")
    elif emit_mode == 4:
        lines.append("        if x > 0 { println_i64(state); }")
    elif emit_mode == 5:
        lines.append("        if state > 0 { println_i64(state); }")
    elif emit_mode == 6:
        lines.append("        if was_started == 0 { println_i64(state); }")
    elif emit_mode == 7:
        pass
    else:
        raise ValueError(f"unknown interactive emit mode {emit_mode}")

    lines.extend([
        "    }",
        "    return 0;",
        "}",
    ])
    return "\n".join(lines)


def _render_interactive_two_register_code(
    a_mode: int,
    b_mode: int,
    out_mode: int,
    *,
    const_a: int,
    const_b: int,
) -> str:
    lines = [
        "fn main() -> i64 {",
        "    reg_a: i64 = 0;",
        "    reg_b: i64 = 0;",
        "    started: i64 = 0;",
        "    while has_input() == 1 {",
        "        x := read_i64();",
        "        was_started := started;",
        "        started = 1;",
    ]
    if a_mode == 0:
        lines.append("        reg_a = reg_a + x;")
    elif a_mode == 1:
        lines.append("        reg_a = reg_a - x;")
    elif a_mode == 2:
        lines.append("        reg_a = x;")
    elif a_mode == 3:
        lines.append("        if was_started == 0 {")
        lines.append("            reg_a = x;")
        lines.append("        }")
        lines.append("        if was_started == 1 {")
        lines.append("            if x > reg_a {")
        lines.append("                reg_a = x;")
        lines.append("            }")
        lines.append("        }")
    elif a_mode == 4:
        lines.append("        if was_started == 0 {")
        lines.append("            reg_a = x;")
        lines.append("        }")
        lines.append("        if was_started == 1 {")
        lines.append("            if x < reg_a {")
        lines.append("                reg_a = x;")
        lines.append("            }")
        lines.append("        }")
    elif a_mode == 5:
        lines.append(f"        reg_a = reg_a + {const_a};")
    elif a_mode == 6:
        lines.append("        tmp := reg_a + x;")
        lines.append("        if x > tmp { tmp = x; }")
        lines.append("        reg_a = tmp;")
    elif a_mode == 7:
        lines.append("        tmp := reg_a + x;")
        lines.append("        if x < tmp { tmp = x; }")
        lines.append("        reg_a = tmp;")
    else:
        raise ValueError(f"unknown two-register a mode {a_mode}")

    if b_mode == 0:
        pass
    elif b_mode == 1:
        lines.append(f"        reg_b = reg_b + {const_b};")
    elif b_mode == 2:
        lines.append("        reg_b = x;")
    elif b_mode == 3:
        lines.append("        reg_b = reg_b + x;")
    elif b_mode == 4:
        lines.append("        if was_started == 0 {")
        lines.append("            reg_b = reg_a;")
        lines.append("        }")
        lines.append("        if was_started == 1 {")
        lines.append("            if reg_a > reg_b {")
        lines.append("                reg_b = reg_a;")
        lines.append("            }")
        lines.append("        }")
    elif b_mode == 5:
        lines.append("        if was_started == 0 {")
        lines.append("            reg_b = reg_a;")
        lines.append("        }")
        lines.append("        if was_started == 1 {")
        lines.append("            if reg_a < reg_b {")
        lines.append("                reg_b = reg_a;")
        lines.append("            }")
        lines.append("        }")
    else:
        raise ValueError(f"unknown two-register b mode {b_mode}")

    if out_mode == 0:
        lines.append("        println_i64(reg_a);")
    elif out_mode == 1:
        lines.append("        println_i64(reg_b);")
    elif out_mode == 2:
        lines.append("        println_i64(reg_a + reg_b);")
    elif out_mode == 3:
        lines.append("        println_i64(reg_a - reg_b);")
    elif out_mode == 4:
        lines.append("        println_i64(reg_a / reg_b);")
    else:
        raise ValueError(f"unknown two-register output mode {out_mode}")

    lines.extend([
        "    }",
        "    return 0;",
        "}",
    ])
    return "\n".join(lines)


def _render_interactive_two_register_emit_code(
    a_mode: int,
    b_mode: int,
    out_mode: int,
    emit_mode: int,
    *,
    const_a: int,
    const_b: int,
    emit_threshold: int,
) -> str:
    lines = [
        "fn main() -> i64 {",
        "    reg_a: i64 = 0;",
        "    reg_b: i64 = 0;",
        "    started: i64 = 0;",
        "    prev_out: i64 = 0;",
        "    while has_input() == 1 {",
        "        x := read_i64();",
        "        old_reg_a := reg_a;",
        "        old_reg_b := reg_b;",
        "        was_started := started;",
        "        started = 1;",
    ]
    if a_mode == 0:
        lines.append("        reg_a = reg_a + x;")
    elif a_mode == 1:
        lines.append("        reg_a = reg_a - x;")
    elif a_mode == 2:
        lines.append("        reg_a = x;")
    elif a_mode == 3:
        lines.append("        if was_started == 0 {")
        lines.append("            reg_a = x;")
        lines.append("        }")
        lines.append("        if was_started == 1 {")
        lines.append("            if x > reg_a {")
        lines.append("                reg_a = x;")
        lines.append("            }")
        lines.append("        }")
    elif a_mode == 4:
        lines.append("        if was_started == 0 {")
        lines.append("            reg_a = x;")
        lines.append("        }")
        lines.append("        if was_started == 1 {")
        lines.append("            if x < reg_a {")
        lines.append("                reg_a = x;")
        lines.append("            }")
        lines.append("        }")
    elif a_mode == 5:
        lines.append(f"        reg_a = reg_a + {const_a};")
    elif a_mode == 6:
        lines.append("        tmp := reg_a + x;")
        lines.append("        if x > tmp { tmp = x; }")
        lines.append("        reg_a = tmp;")
    elif a_mode == 7:
        lines.append("        tmp := reg_a + x;")
        lines.append("        if x < tmp { tmp = x; }")
        lines.append("        reg_a = tmp;")
    else:
        raise ValueError(f"unknown two-register a mode {a_mode}")

    if b_mode == 0:
        pass
    elif b_mode == 1:
        lines.append(f"        reg_b = reg_b + {const_b};")
    elif b_mode == 2:
        lines.append("        reg_b = x;")
    elif b_mode == 3:
        lines.append("        reg_b = reg_b + x;")
    elif b_mode == 4:
        lines.append("        if was_started == 0 {")
        lines.append("            reg_b = reg_a;")
        lines.append("        }")
        lines.append("        if was_started == 1 {")
        lines.append("            if reg_a > reg_b {")
        lines.append("                reg_b = reg_a;")
        lines.append("            }")
        lines.append("        }")
    elif b_mode == 5:
        lines.append("        if was_started == 0 {")
        lines.append("            reg_b = reg_a;")
        lines.append("        }")
        lines.append("        if was_started == 1 {")
        lines.append("            if reg_a < reg_b {")
        lines.append("                reg_b = reg_a;")
        lines.append("            }")
        lines.append("        }")
    else:
        raise ValueError(f"unknown two-register b mode {b_mode}")

    if out_mode == 0:
        lines.append("        out := reg_a;")
    elif out_mode == 1:
        lines.append("        out := reg_b;")
    elif out_mode == 2:
        lines.append("        out := reg_a + reg_b;")
    elif out_mode == 3:
        lines.append("        out := reg_a - reg_b;")
    elif out_mode == 4:
        lines.append("        out := reg_a / reg_b;")
    else:
        raise ValueError(f"unknown two-register output mode {out_mode}")

    if emit_mode == 0:
        lines.append("        println_i64(out);")
    elif emit_mode == 1:
        lines.append("        if x > 0 { println_i64(out); }")
    elif emit_mode == 2:
        lines.append("        if x < 0 { println_i64(out); }")
    elif emit_mode == 3:
        lines.append("        if out > 0 { println_i64(out); }")
    elif emit_mode == 4:
        lines.append("        if reg_a > 0 { println_i64(out); }")
    elif emit_mode == 5:
        lines.append("        if reg_b > 0 { println_i64(out); }")
    elif emit_mode == 6:
        lines.append("        if was_started == 0 { println_i64(out); }")
    elif emit_mode == 7:
        pass
    elif emit_mode == 8:
        lines.append("        emit := 0;")
        lines.append("        if was_started == 0 { emit = 1; }")
        lines.append("        if out != prev_out { emit = 1; }")
        lines.append("        if emit == 1 { println_i64(out); }")
    elif emit_mode == 9:
        lines.append("        emit := 0;")
        lines.append("        if was_started == 0 { emit = 1; }")
        lines.append("        if out > prev_out { emit = 1; }")
        lines.append("        if emit == 1 { println_i64(out); }")
    elif emit_mode == 10:
        lines.append("        emit := 0;")
        lines.append("        if was_started == 0 { emit = 1; }")
        lines.append("        if out < prev_out { emit = 1; }")
        lines.append("        if emit == 1 { println_i64(out); }")
    elif emit_mode == 11:
        lines.append(f"        if out > {emit_threshold} {{ println_i64(out); }}")
    elif emit_mode == 12:
        lines.append("        emit := 0;")
        lines.append(f"        if was_started == 1 {{ if prev_out <= {emit_threshold} {{ if out > {emit_threshold} {{ emit = 1; }} }} }}")
        lines.append("        if emit == 1 { println_i64(out); }")
    elif emit_mode == 13:
        lines.append("        emit := 0;")
        lines.append(f"        if was_started == 1 {{ if prev_out >= {emit_threshold} {{ if out < {emit_threshold} {{ emit = 1; }} }} }}")
        lines.append("        if emit == 1 { println_i64(out); }")
    elif emit_mode == 14:
        lines.append("        if reg_a > reg_b { println_i64(out); }")
    elif emit_mode == 15:
        lines.append("        if reg_a < reg_b { println_i64(out); }")
    elif emit_mode == 16:
        lines.append("        if out > reg_b { println_i64(out); }")
    elif emit_mode == 17:
        lines.append("        if out < reg_b { println_i64(out); }")
    elif emit_mode == 18:
        lines.append("        emit := 0;")
        lines.append("        if was_started == 1 { if old_reg_a <= old_reg_b { if reg_a > reg_b { emit = 1; } } }")
        lines.append("        if emit == 1 { println_i64(out); }")
    elif emit_mode == 19:
        lines.append("        emit := 0;")
        lines.append("        if was_started == 1 { if old_reg_a >= old_reg_b { if reg_a < reg_b { emit = 1; } } }")
        lines.append("        if emit == 1 { println_i64(out); }")
    elif emit_mode == 20:
        lines.append("        emit := 0;")
        lines.append("        if was_started == 1 { if prev_out <= old_reg_b { if out > reg_b { emit = 1; } } }")
        lines.append("        if emit == 1 { println_i64(out); }")
    elif emit_mode == 21:
        lines.append("        emit := 0;")
        lines.append("        if was_started == 1 { if prev_out >= old_reg_b { if out < reg_b { emit = 1; } } }")
        lines.append("        if emit == 1 { println_i64(out); }")
    elif emit_mode == 22:
        lines.append(f"        if reg_a - reg_b > {emit_threshold} {{ println_i64(out); }}")
    elif emit_mode == 23:
        lines.append("        emit := 0;")
        lines.append(f"        if was_started == 1 {{ if old_reg_a - old_reg_b <= {emit_threshold} {{ if reg_a - reg_b > {emit_threshold} {{ emit = 1; }} }} }}")
        lines.append("        if emit == 1 { println_i64(out); }")
    elif emit_mode == 24:
        lines.append(f"        if out - reg_b > {emit_threshold} {{ println_i64(out); }}")
    elif emit_mode == 25:
        lines.append("        emit := 0;")
        lines.append(f"        if was_started == 1 {{ if prev_out - old_reg_b <= {emit_threshold} {{ if out - reg_b > {emit_threshold} {{ emit = 1; }} }} }}")
        lines.append("        if emit == 1 { println_i64(out); }")
    else:
        raise ValueError(f"unknown two-register emit mode {emit_mode}")

    lines.append("        prev_out = out;")
    lines.extend([
        "    }",
        "    return 0;",
        "}",
    ])
    return "\n".join(lines)


def _interactive_candidates(
    prog: SoftInteractiveProgram,
    traces: list[list[tuple[int, int]]],
) -> list[tuple[float, int, int, int, str]]:
    learned_a = float(prog.const_a.item())
    learned_b = float(prog.const_b.item())
    candidates: list[tuple[float, int, int, int, str]] = []
    for mode in range(SoftInteractiveProgram.NUM_CANDIDATES):
        const_a_values = [0]
        const_b_values = [0]
        if mode in {4, 5}:
            const_a_values = _round_candidates(learned_a) + _data_candidates_for_mode(mode, traces)
        if mode == 6:
            const_b_values = _round_candidates(learned_b) + _data_candidates_for_mode(mode, traces)
        seen_pairs: set[tuple[int, int]] = set()
        for const_a in const_a_values:
            for const_b in const_b_values:
                if (const_a, const_b) in seen_pairs:
                    continue
                seen_pairs.add((const_a, const_b))
                loss = _candidate_loss(mode, traces, const_a=const_a, const_b=const_b)
                candidates.append(
                    (
                        loss,
                        mode,
                        const_a,
                        const_b,
                        _interactive_structure_name(mode),
                    )
                )
    candidates.sort(key=lambda item: (item[0], item[1], abs(item[2]), abs(item[3])))
    return candidates


def _solve_interactive_pair(
    traces: list[list[tuple[int, int, int]]],
    original_traces: list[tuple[list[int], list[int]]],
    *,
    steps: int,
    num_restarts: int,
    seed: int,
) -> dict[str, Any]:
    if not traces:
        return {
            "supported": False,
            "success": False,
            "error": "interactive pair differentiable synthesis requires at least one grouped trace",
        }

    best_soft_loss = float("inf")
    best_mode = 0
    best_loss = float("inf")

    for restart in range(max(num_restarts, 1)):
        restart_seed = seed + restart * 977
        torch.manual_seed(restart_seed)
        prog = SoftInteractivePairProgram()
        soft_loss = train_interactive_pair_program(
            prog,
            traces,
            steps=steps,
            seed=restart_seed,
        )
        for mode in range(SoftInteractivePairProgram.NUM_CANDIDATES):
            loss = _pair_mode_loss(mode, traces)
            if loss < best_loss or (loss == best_loss and mode < best_mode):
                best_loss = loss
                best_mode = mode
                best_soft_loss = soft_loss

    structure = _interactive_pair_structure_name(best_mode)
    code = _render_interactive_pair_code(best_mode)
    verified = best_loss < 1e-8 and _verify_interactive_stream_code(code, original_traces)
    if not verified:
        return {
            "supported": True,
            "success": False,
            "loss": best_loss,
            "structure": structure,
            "metadata": {
                "mode": best_mode,
                "soft_loss": best_soft_loss,
            },
            "error": "interactive pair differentiable synthesis did not converge to a verified exact program",
        }

    return {
        "supported": True,
        "success": True,
        "code": code,
        "loss": best_loss,
        "structure": structure,
        "metadata": {
            "mode": best_mode,
            "soft_loss": best_soft_loss,
        },
        "error": None,
    }


def _solve_interactive_state_emit(
    traces: list[tuple[list[int], list[int]]],
    *,
    steps: int,
    num_restarts: int,
    seed: int,
) -> dict[str, Any]:
    if not traces:
        return {
            "supported": False,
            "success": False,
            "error": "interactive state-emit differentiable synthesis requires at least one trace",
        }

    best_soft_loss = float("inf")
    best_candidate: tuple[float, int, int, int, int, str] | None = None

    for restart in range(max(num_restarts, 1)):
        restart_seed = seed + restart * 977
        torch.manual_seed(restart_seed)
        prog = SoftInteractiveStateEmitProgram()
        soft_loss = train_interactive_state_emit_program(
            prog,
            traces,
            steps=steps,
            seed=restart_seed,
        )
        learned_a = float(prog.const_a.item())
        learned_b = float(prog.const_b.item())
        for update_mode in range(SoftInteractiveStateEmitProgram.NUM_UPDATE_CANDIDATES):
            const_a_values = [0]
            const_b_values = [0]
            if update_mode in {4, 5}:
                const_a_values = _round_candidates(learned_a)
            if update_mode == 6:
                const_b_values = _round_candidates(learned_b)
            seen_pairs: set[tuple[int, int]] = set()
            for const_a in const_a_values:
                for const_b in const_b_values:
                    if (const_a, const_b) in seen_pairs:
                        continue
                    seen_pairs.add((const_a, const_b))
                    for emit_mode in range(SoftInteractiveStateEmitProgram.NUM_EMIT_CANDIDATES):
                        loss = 0.0
                        for trace in traces:
                            actual = _simulate_interactive_state_emit_candidate(
                                update_mode,
                                emit_mode,
                                trace,
                                const_a=const_a,
                                const_b=const_b,
                            )
                            loss += _sequence_loss(actual, trace[1])
                        loss /= max(len(traces), 1)
                        candidate = (
                            loss,
                            update_mode,
                            emit_mode,
                            const_a,
                            const_b,
                            _interactive_state_emit_structure_name(update_mode, emit_mode),
                        )
                        if best_candidate is None or candidate < best_candidate:
                            best_candidate = candidate
                            best_soft_loss = soft_loss

    if best_candidate is None:
        return {
            "supported": True,
            "success": False,
            "error": "interactive state-emit differentiable synthesis failed to produce any candidate",
        }

    loss, update_mode, emit_mode, const_a, const_b, structure = best_candidate
    code = _render_interactive_state_emit_code(
        update_mode,
        emit_mode,
        const_a=const_a,
        const_b=const_b,
    )
    verified = loss < 1e-8 and _verify_interactive_stream_code(code, traces)
    if not verified:
        return {
            "supported": True,
            "success": False,
            "loss": loss,
            "structure": structure,
            "metadata": {
                "update_mode": update_mode,
                "emit_mode": emit_mode,
                "const_a": const_a,
                "const_b": const_b,
                "soft_loss": best_soft_loss,
            },
            "error": "interactive state-emit differentiable synthesis did not converge to a verified exact program",
        }

    return {
        "supported": True,
        "success": True,
        "code": code,
        "loss": loss,
        "structure": structure,
        "metadata": {
            "update_mode": update_mode,
            "emit_mode": emit_mode,
            "const_a": const_a,
            "const_b": const_b,
            "soft_loss": best_soft_loss,
        },
        "error": None,
    }


def _solve_interactive_two_register(
    traces: list[list[tuple[int, int]]],
    *,
    steps: int,
    num_restarts: int,
    seed: int,
) -> dict[str, Any]:
    if not traces:
        return {
            "supported": False,
            "success": False,
            "error": "interactive two-register differentiable synthesis requires at least one trace",
        }

    best_soft_loss = float("inf")
    best_candidate: tuple[float, int, int, int, int, int, str] | None = None
    exact_structures: set[str] = set()

    for restart in range(max(num_restarts, 1)):
        restart_seed = seed + restart * 977
        torch.manual_seed(restart_seed)
        prog = SoftInteractiveTwoRegisterProgram()
        soft_loss = train_interactive_two_register_program(
            prog,
            traces,
            steps=steps,
            seed=restart_seed,
        )
        learned_a = float(prog.const_a.item())
        learned_b = float(prog.const_b.item())
        for a_mode in range(SoftInteractiveTwoRegisterProgram.NUM_A_CANDIDATES):
            a_values = [0]
            if a_mode == 5:
                a_values = _round_candidates(learned_a)
            for b_mode in range(SoftInteractiveTwoRegisterProgram.NUM_B_CANDIDATES):
                b_values = [0]
                if b_mode == 1:
                    b_values = _round_candidates(learned_b)
                for const_a in a_values:
                    for const_b in b_values:
                        for out_mode in range(SoftInteractiveTwoRegisterProgram.NUM_OUTPUT_CANDIDATES):
                            loss = 0.0
                            valid = True
                            for trace in traces:
                                actual = _simulate_interactive_two_register_candidate(
                                    a_mode,
                                    b_mode,
                                    out_mode,
                                    trace,
                                    const_a=const_a,
                                    const_b=const_b,
                                )
                                if actual is None:
                                    valid = False
                                    break
                                for pred, (_, expected) in zip(actual, trace, strict=True):
                                    loss += float((pred - expected) ** 2)
                            if not valid:
                                continue
                            loss /= max(sum(len(trace) for trace in traces), 1)
                            candidate = (
                                loss,
                                a_mode,
                                b_mode,
                                out_mode,
                                const_a,
                                const_b,
                                _interactive_two_register_structure_name(
                                    a_mode,
                                    b_mode,
                                    out_mode,
                                ),
                            )
                            if loss < 1e-8:
                                exact_structures.add(candidate[-1])
                            if best_candidate is None or candidate < best_candidate:
                                best_candidate = candidate
                                best_soft_loss = soft_loss

    if best_candidate is None:
        return {
            "supported": True,
            "success": False,
            "error": "interactive two-register differentiable synthesis failed to produce any candidate",
        }

    loss, a_mode, b_mode, out_mode, const_a, const_b, structure = best_candidate
    code = _render_interactive_two_register_code(
        a_mode,
        b_mode,
        out_mode,
        const_a=const_a,
        const_b=const_b,
    )
    verified = loss < 1e-8 and _verify_interactive_code(code, traces)
    ambiguity_metadata = _exact_ambiguity_metadata(exact_structures)
    if not verified:
        return {
            "supported": True,
            "success": False,
            "loss": loss,
            "structure": structure,
            "metadata": {
                "a_mode": a_mode,
                "b_mode": b_mode,
                "out_mode": out_mode,
                "const_a": const_a,
                "const_b": const_b,
                "soft_loss": best_soft_loss,
                **ambiguity_metadata,
            },
            "error": "interactive two-register differentiable synthesis did not converge to a verified exact program",
        }

    return {
        "supported": True,
        "success": True,
        "code": code,
        "loss": loss,
        "structure": structure,
        "metadata": {
            "a_mode": a_mode,
            "b_mode": b_mode,
            "out_mode": out_mode,
            "const_a": const_a,
            "const_b": const_b,
            "soft_loss": best_soft_loss,
            **ambiguity_metadata,
        },
        "error": None,
    }


def _find_discriminating_trace(
    candidates: list[tuple[float, int, int, int, int, int, int, int, str]],
    traces: list[tuple[list[int], list[int]]],
) -> tuple[bool, str] | None:
    """Try to find a trace where two exact candidates disagree.

    Returns (resolved, description) when a discriminating input is found,
    or None if we cannot discriminate within the seed budget.
    """
    if len(candidates) < 2:
        return None

    # Build probe sequences: start with the actual training traces, then add
    # single-element perturbations and a few fixed probes.
    seed_inputs: list[list[int]] = []
    # Include full training input sequences first (they expose stateful differences)
    for inputs, _ in traces:
        if inputs and inputs not in seed_inputs:
            seed_inputs.append(list(inputs))
    # Single-element perturbations of values seen in traces
    seen_vals: set[int] = set()
    for inputs, _ in traces:
        for inp in inputs:
            for delta in range(-3, 4):
                v = inp + delta
                if v not in seen_vals:
                    seen_vals.add(v)
                    seed_inputs.append([v])
    # Also try a few fixed probes
    for v in [-5, -1, 0, 1, 2, 5, 10]:
        if v not in seen_vals:
            seen_vals.add(v)
            seed_inputs.append([v])

    dummy_expected: list[int] = []  # we only care about outputs, not matching expected
    for c1, c2 in zip(candidates[:4], candidates[1:5]):
        _, a1, b1, out1, emit1, ca1, cb1, et1, name1 = c1
        _, a2, b2, out2, emit2, ca2, cb2, et2, name2 = c2
        if name1 == name2:
            continue
        for probe_inputs in seed_inputs[:20]:
            probe_trace = (probe_inputs, dummy_expected)
            out_c1 = _simulate_interactive_two_register_emit_candidate(
                a1, b1, out1, emit1, probe_trace,
                const_a=ca1, const_b=cb1, emit_threshold=et1,
            )
            out_c2 = _simulate_interactive_two_register_emit_candidate(
                a2, b2, out2, emit2, probe_trace,
                const_a=ca2, const_b=cb2, emit_threshold=et2,
            )
            if out_c1 is not None and out_c2 is not None and out_c1 != out_c2:
                return True, (
                    f"candidates {name1} vs {name2} disagree on input {probe_inputs}: "
                    f"{out_c1} vs {out_c2}"
                )

    return None


def _solve_interactive_two_register_emit(
    traces: list[tuple[list[int], list[int]]],
    *,
    steps: int,
    num_restarts: int,
    seed: int,
) -> dict[str, Any]:
    if not traces:
        return {
            "supported": False,
            "success": False,
            "error": "interactive two-register sparse differentiable synthesis requires at least one trace",
        }

    best_soft_loss = float("inf")
    best_candidate: tuple[float, int, int, int, int, int, int, int, str] | None = None
    exact_structures: set[str] = set()
    exact_candidates: list[tuple[float, int, int, int, int, int, int, int, str]] = []
    exact_candidate_affinity: dict[tuple[int, int, int, int, int, int, int, str], float] = {}

    for restart in range(max(num_restarts, 1)):
        restart_seed = seed + restart * 977
        torch.manual_seed(restart_seed)
        prog = SoftInteractiveTwoRegisterEmitProgram()
        soft_loss = train_interactive_two_register_emit_program(
            prog,
            traces,
            steps=steps,
            seed=restart_seed,
        )
        learned_a = float(prog.const_a.item())
        learned_b = float(prog.const_b.item())
        learned_threshold = float(prog.emit_threshold.item())
        for a_mode in range(SoftInteractiveTwoRegisterEmitProgram.NUM_A_CANDIDATES):
            a_values = [0]
            if a_mode == 5:
                a_values = _round_candidates(learned_a)
            for b_mode in range(SoftInteractiveTwoRegisterEmitProgram.NUM_B_CANDIDATES):
                b_values = [0]
                if b_mode == 1:
                    b_values = _round_candidates(learned_b)
                for const_a in a_values:
                    for const_b in b_values:
                        for out_mode in range(SoftInteractiveTwoRegisterEmitProgram.NUM_OUTPUT_CANDIDATES):
                            reg_gap_thresholds = _threshold_candidates_from_two_register_relation(
                                traces,
                                a_mode=a_mode,
                                b_mode=b_mode,
                                out_mode=out_mode,
                                const_a=const_a,
                                const_b=const_b,
                                relation="reg_gap",
                            )
                            output_gap_thresholds = _threshold_candidates_from_two_register_relation(
                                traces,
                                a_mode=a_mode,
                                b_mode=b_mode,
                                out_mode=out_mode,
                                const_a=const_a,
                                const_b=const_b,
                                relation="output_gap",
                            )
                            for emit_mode in range(SoftInteractiveTwoRegisterEmitProgram.NUM_EMIT_CANDIDATES):
                                threshold_values = [0]
                                if emit_mode in {11, 12, 13}:
                                    threshold_values = (
                                        _round_candidates(learned_threshold)
                                        + _threshold_candidates_from_outputs(traces)
                                    )
                                elif emit_mode in {22, 23}:
                                    threshold_values = (
                                        _round_candidates(learned_threshold)
                                        + reg_gap_thresholds
                                    )
                                elif emit_mode in {24, 25}:
                                    threshold_values = (
                                        _round_candidates(learned_threshold)
                                        + output_gap_thresholds
                                    )
                                seen_thresholds: set[int] = set()
                                for emit_threshold in threshold_values:
                                    if emit_threshold in seen_thresholds:
                                        continue
                                    seen_thresholds.add(emit_threshold)
                                    loss = 0.0
                                    valid = True
                                    for trace in traces:
                                        actual = _simulate_interactive_two_register_emit_candidate(
                                            a_mode,
                                            b_mode,
                                            out_mode,
                                            emit_mode,
                                            trace,
                                            const_a=const_a,
                                            const_b=const_b,
                                            emit_threshold=emit_threshold,
                                        )
                                        if actual is None:
                                            valid = False
                                            break
                                        loss += _sequence_loss(actual, trace[1])
                                    if not valid:
                                        continue
                                    loss /= max(len(traces), 1)
                                    candidate = (
                                        loss,
                                        a_mode,
                                        b_mode,
                                        out_mode,
                                        emit_mode,
                                        const_a,
                                        const_b,
                                        emit_threshold,
                                        _interactive_two_register_emit_structure_name(
                                            a_mode,
                                            b_mode,
                                            out_mode,
                                            emit_mode,
                                        ),
                                    )
                                    if loss < 1e-8:
                                        exact_structures.add(candidate[-1])
                                        exact_candidates.append(candidate)
                                        candidate_key = _two_register_emit_candidate_key(candidate)
                                        affinity = _two_register_emit_candidate_affinity(
                                            prog,
                                            a_mode=a_mode,
                                            b_mode=b_mode,
                                            out_mode=out_mode,
                                            emit_mode=emit_mode,
                                            const_a=const_a,
                                            const_b=const_b,
                                            emit_threshold=emit_threshold,
                                        )
                                        prev_affinity = exact_candidate_affinity.get(candidate_key)
                                        if prev_affinity is None or affinity > prev_affinity:
                                            exact_candidate_affinity[candidate_key] = affinity
                                    if best_candidate is None or candidate < best_candidate:
                                        best_candidate = candidate
                                        best_soft_loss = soft_loss

    if best_candidate is None:
        return {
            "supported": True,
            "success": False,
            "error": "interactive two-register sparse differentiable synthesis failed to produce any candidate",
        }

    recursive_refinement_applied = False
    recursive_refinement_resolved = False
    recursive_refinement_winner = ""
    recursive_refinement_runs: list[dict[str, Any]] = []
    candidate_order: list[tuple[float, int, int, int, int, int, int, int, str]] = [best_candidate]

    if exact_candidates:
        deduped_exact_candidates: list[tuple[float, int, int, int, int, int, int, int, str]] = []
        seen_exact_candidates: set[tuple[int, int, int, int, int, int, int, str]] = set()
        for candidate in sorted(exact_candidates):
            candidate_key = _two_register_emit_candidate_key(candidate)
            if candidate_key in seen_exact_candidates:
                continue
            seen_exact_candidates.add(candidate_key)
            deduped_exact_candidates.append(candidate)
        candidate_order = deduped_exact_candidates
        if len(deduped_exact_candidates) > 1:
            recursive_refinement_applied = True
            emit_refinement_results: list[
                tuple[
                    tuple[int, float],
                    tuple[float, int, int, int, int, int, int, int, str],
                    dict[str, Any],
                    tuple[int, int, int, int, int],
                ]
            ] = []
            for idx, candidate in enumerate(deduped_exact_candidates[:8]):
                refinement = _recursive_refine_two_register_emit_candidate(
                    candidate,
                    traces,
                    steps=steps,
                    num_restarts=num_restarts,
                    seed=seed + idx * 131,
                    include_core=False,
                )
                emit_refinement_results.append(
                    (
                        refinement["meaningful_key"],
                        candidate,
                        refinement["metadata"],
                        refinement["core_key"],
                    )
                )
                emit_refinement_results.sort(key=lambda item: (item[0], item[1]))
            best_emit_key = emit_refinement_results[0][0]
            best_emit_results = [
                item
                for item in emit_refinement_results
                if item[0] == best_emit_key
            ]

            stage_two_results: list[
                tuple[
                    tuple[int, float, int, float, float],
                    tuple[float, int, int, int, int, int, int, int, str],
                    dict[str, Any],
                ]
            ] = []
            if len(best_emit_results) > 1 and len({item[3] for item in best_emit_results}) > 1:
                core_cache: dict[tuple[int, int, int, int, int], dict[str, Any]] = {}
                for idx, (_emit_key, candidate, _metadata, _core_key) in enumerate(best_emit_results):
                    refinement = _recursive_refine_two_register_emit_candidate(
                        candidate,
                        traces,
                        steps=steps,
                        num_restarts=num_restarts,
                        seed=seed + 1009 + idx * 131,
                        core_cache=core_cache,
                        include_core=True,
                    )
                    stage_two_results.append(
                        (
                            refinement["meaningful_key"],
                            candidate,
                            refinement["metadata"],
                        )
                    )
                stage_two_results.sort(
                    key=lambda item: (
                        item[0],
                        -exact_candidate_affinity.get(
                            _two_register_emit_candidate_key(item[1]),
                            float("-inf"),
                        ),
                        item[1],
                    )
                )
                refined_candidates = [item[1] for item in stage_two_results]
                recursive_refinement_runs = [
                    {
                        "structure": candidate[-1],
                        "sort_key": list(meaningful_key),
                        "soft_affinity": exact_candidate_affinity.get(
                            _two_register_emit_candidate_key(candidate),
                            float("-inf"),
                        ),
                        "details": metadata,
                    }
                    for meaningful_key, candidate, metadata in stage_two_results[:4]
                ]
                if len(stage_two_results) == 1:
                    recursive_refinement_resolved = True
                elif stage_two_results[0][0] != stage_two_results[1][0]:
                    recursive_refinement_resolved = True
            else:
                emit_refinement_results.sort(
                    key=lambda item: (
                        item[0],
                        -exact_candidate_affinity.get(
                            _two_register_emit_candidate_key(item[1]),
                            float("-inf"),
                        ),
                        item[1],
                    )
                )
                refined_candidates = [item[1] for item in emit_refinement_results]
                recursive_refinement_runs = [
                    {
                        "structure": candidate[-1],
                        "sort_key": list(meaningful_key),
                        "soft_affinity": exact_candidate_affinity.get(
                            _two_register_emit_candidate_key(candidate),
                            float("-inf"),
                        ),
                        "details": metadata,
                    }
                    for meaningful_key, candidate, metadata, _core_key in emit_refinement_results[:4]
                ]
                if len(best_emit_results) == 1:
                    recursive_refinement_resolved = True

            candidate_order = refined_candidates + [
                candidate
                for candidate in deduped_exact_candidates
                if candidate not in refined_candidates
            ]
            recursive_refinement_winner = candidate_order[0][-1]
        best_candidate = candidate_order[0]

    verified = False
    code = ""
    for candidate in candidate_order:
        (
            loss,
            a_mode,
            b_mode,
            out_mode,
            emit_mode,
            const_a,
            const_b,
            emit_threshold,
            structure,
        ) = candidate
        candidate_code = _render_interactive_two_register_emit_code(
            a_mode,
            b_mode,
            out_mode,
            emit_mode,
            const_a=const_a,
            const_b=const_b,
            emit_threshold=emit_threshold,
        )
        if loss < 1e-8 and _verify_interactive_stream_code(candidate_code, traces):
            best_candidate = candidate
            code = candidate_code
            verified = True
            break

    if not verified:
        (
            loss,
            a_mode,
            b_mode,
            out_mode,
            emit_mode,
            const_a,
            const_b,
            emit_threshold,
            structure,
        ) = best_candidate
        code = _render_interactive_two_register_emit_code(
            a_mode,
            b_mode,
            out_mode,
            emit_mode,
            const_a=const_a,
            const_b=const_b,
            emit_threshold=emit_threshold,
        )
        verified = loss < 1e-8 and _verify_interactive_stream_code(code, traces)

    exact_ambiguity_metadata = _exact_ambiguity_metadata(
        exact_structures,
        exact_candidate_count=len(exact_candidates),
    )

    # Active disambiguation: when multiple exact candidates exist, try to find a
    # discriminating input that separates them to help downstream diagnosis.
    discrimination_result = _find_discriminating_trace(
        candidate_order[:8] if exact_candidates else [],
        traces,
    )
    active_disambiguation_metadata: dict[str, Any] = {
        "active_disambiguation_attempted": bool(exact_candidates and len(candidate_order) > 1),
        "active_disambiguation_found": discrimination_result is not None,
        "active_disambiguation_description": discrimination_result[1] if discrimination_result else None,
    }

    if not verified:
        return {
            "supported": True,
            "success": False,
            "loss": loss,
            "structure": structure,
            "metadata": {
                "a_mode": a_mode,
                "b_mode": b_mode,
                "out_mode": out_mode,
                "emit_mode": emit_mode,
                "const_a": const_a,
                "const_b": const_b,
                "emit_threshold": emit_threshold,
                "soft_loss": best_soft_loss,
                "recursive_refinement_applied": recursive_refinement_applied,
                "recursive_refinement_resolved": recursive_refinement_resolved,
                "recursive_refinement_winner": recursive_refinement_winner,
                "recursive_refinement_candidates_considered": min(len(exact_candidates), 8),
                "recursive_refinement_runs": recursive_refinement_runs,
                **exact_ambiguity_metadata,
                **active_disambiguation_metadata,
            },
            "error": "interactive two-register sparse differentiable synthesis did not converge to a verified exact program",
        }

    return {
        "supported": True,
        "success": True,
        "code": code,
        "loss": loss,
        "structure": structure,
        "metadata": {
            "a_mode": a_mode,
            "b_mode": b_mode,
            "out_mode": out_mode,
            "emit_mode": emit_mode,
            "const_a": const_a,
            "const_b": const_b,
            "emit_threshold": emit_threshold,
            "soft_loss": best_soft_loss,
            "recursive_refinement_applied": recursive_refinement_applied,
            "recursive_refinement_resolved": recursive_refinement_resolved,
            "recursive_refinement_winner": recursive_refinement_winner,
            "recursive_refinement_candidates_considered": min(len(exact_candidates), 8),
            "recursive_refinement_runs": recursive_refinement_runs,
            **exact_ambiguity_metadata,
            **active_disambiguation_metadata,
        },
        "error": None,
    }


def _solve_interactive(
    traces: list[list[tuple[int, int]]],
    *,
    steps: int,
    num_restarts: int,
    seed: int,
) -> dict[str, Any]:
    if not traces:
        return {
            "supported": False,
            "success": False,
            "error": "interactive differentiable synthesis requires at least one trace",
        }

    best_soft_loss = float("inf")
    best_candidate: tuple[float, int, int, int, str] | None = None
    exact_structures: set[str] = set()

    for restart in range(max(num_restarts, 1)):
        restart_seed = seed + restart * 977
        torch.manual_seed(restart_seed)
        prog = SoftInteractiveProgram()
        soft_loss = train_interactive_program(
            prog,
            traces,
            steps=steps,
            seed=restart_seed,
        )
        candidates = _interactive_candidates(prog, traces)
        for candidate in candidates:
            if candidate[0] < 1e-8:
                exact_structures.add(candidate[-1])
        candidate = candidates[0]
        if candidate[0] < (best_candidate[0] if best_candidate is not None else float("inf")):
            best_candidate = candidate
            best_soft_loss = soft_loss

    if best_candidate is None:
        return {
            "supported": True,
            "success": False,
            "error": "interactive differentiable synthesis failed to produce any candidate",
        }

    loss, mode, const_a, const_b, structure = best_candidate
    code = _render_interactive_code(mode, const_a=const_a, const_b=const_b)
    verified = loss < 1e-8 and _verify_interactive_code(code, traces)
    ambiguity_metadata = _exact_ambiguity_metadata(exact_structures)
    if not verified:
        return {
            "supported": True,
            "success": False,
            "loss": loss,
            "structure": structure,
            "metadata": {
                "mode": mode,
                "const_a": const_a,
                "const_b": const_b,
                "soft_loss": best_soft_loss,
                **ambiguity_metadata,
            },
            "error": "interactive differentiable synthesis did not converge to a verified exact program",
        }

    return {
        "supported": True,
        "success": True,
        "code": code,
        "loss": loss,
        "structure": structure,
        "metadata": {
            "mode": mode,
            "const_a": const_a,
            "const_b": const_b,
            "soft_loss": best_soft_loss,
            **ambiguity_metadata,
        },
        "error": None,
    }


def _filter_condition_code(mode: int) -> str:
    return {
        0: "x > 0",
        1: "x < 0",
        2: "x == 0",
        3: "(x % 2) == 0",
        4: "(x % 2) != 0",
        5: "1 == 1",
        6: "1 == 0",
    }.get(mode, "1 == 0")


def _render_interactive_filter_code(mode: int) -> str:
    condition = _filter_condition_code(mode)
    return "\n".join([
        "fn main() -> i64 {",
        "    while has_input() == 1 {",
        "        x := read_i64();",
        f"        if {condition} {{",
        "            println_i64(x);",
        "        }",
        "    }",
        "    return 0;",
        "}",
    ])


def _filter_mode_loss(
    mode: int,
    emit_targets: list[list[int]],
    inputs: list[list[int]],
) -> float:
    total_loss = 0.0
    total_steps = 0
    for stream, targets in zip(inputs, emit_targets, strict=True):
        for inp, expected_emit in zip(stream, targets, strict=True):
            actual_emit = {
                0: int(inp > 0),
                1: int(inp < 0),
                2: int(inp == 0),
                3: int(abs(inp) % 2 == 0),
                4: int(abs(inp) % 2 != 0),
                5: 1,
                6: 0,
            }[mode]
            total_loss += float((actual_emit - expected_emit) ** 2)
            total_steps += 1
    return total_loss / max(total_steps, 1)


def _solve_interactive_filter(
    traces: list[tuple[list[int], list[int]]],
    *,
    steps: int,
    num_restarts: int,
    seed: int,
) -> dict[str, Any]:
    emit_targets = _derive_passthrough_emit_targets(traces)
    if emit_targets is None:
        return {
            "supported": True,
            "success": False,
            "error": "interactive differentiable filter currently supports passthrough subsequence traces only",
        }

    input_streams = [inputs for inputs, _ in traces]
    best_soft_loss = float("inf")
    best_mode = 6
    best_loss = float("inf")

    for restart in range(max(num_restarts, 1)):
        restart_seed = seed + restart * 977
        torch.manual_seed(restart_seed)
        prog = SoftInteractiveFilterProgram()
        soft_loss = train_interactive_filter_program(
            prog,
            input_streams,
            emit_targets,
            steps=steps,
            seed=restart_seed,
        )
        for mode in range(SoftInteractiveFilterProgram.NUM_CONDITIONS):
            loss = _filter_mode_loss(mode, emit_targets, input_streams)
            if loss < best_loss or (loss == best_loss and mode < best_mode):
                best_loss = loss
                best_mode = mode
                best_soft_loss = soft_loss

    structure = _interactive_filter_structure_name(best_mode)
    code = _render_interactive_filter_code(best_mode)
    verified = best_loss < 1e-8 and _verify_interactive_stream_code(code, traces)
    if not verified:
        return {
            "supported": True,
            "success": False,
            "loss": best_loss,
            "structure": structure,
            "metadata": {
                "mode": best_mode,
                "soft_loss": best_soft_loss,
            },
            "error": "interactive differentiable filter did not converge to a verified exact program",
        }

    return {
        "supported": True,
        "success": True,
        "code": code,
        "loss": best_loss,
        "structure": structure,
        "metadata": {
            "mode": best_mode,
            "soft_loss": best_soft_loss,
        },
        "error": None,
    }


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        mode = payload.get("mode", "scalar")
        steps = int(payload.get("steps", 300))
        num_restarts = int(payload.get("num_restarts", 1))
        seed = int(payload.get("seed", 42))
    except Exception as err:
        json.dump(
            {
                "supported": False,
                "success": False,
                "error": f"invalid request: {err}",
            },
            sys.stdout,
        )
        sys.stdout.write("\n")
        return 0

    try:
        if mode == "interactive":
            raw_traces = payload["interactive_traces"]
            traces = _coerce_interactive_traces(raw_traces)
            if traces is None:
                json.dump(
                    {
                        "supported": False,
                        "success": False,
                        "error": "interactive differentiable synthesis supports only integer stream traces",
                    },
                    sys.stdout,
                )
                sys.stdout.write("\n")
                return 0
            state_update = _state_update_traces(traces)
            pair_grouped = _pair_group_traces(traces)
            attempts: list[dict[str, Any]] = []
            if state_update is not None:
                attempts.append(
                    _solve_interactive(
                        state_update,
                        steps=steps,
                        num_restarts=num_restarts,
                        seed=seed,
                    )
                )
                attempts.append(
                    _solve_interactive_two_register(
                        state_update,
                        steps=steps,
                        num_restarts=num_restarts,
                        seed=seed,
                    )
                )
            if pair_grouped is not None:
                attempts.append(
                    _solve_interactive_pair(
                        pair_grouped,
                        traces,
                        steps=steps,
                        num_restarts=num_restarts,
                        seed=seed,
                    )
                )
            attempts.append(
                _solve_interactive_filter(
                    traces,
                    steps=steps,
                    num_restarts=num_restarts,
                    seed=seed,
                )
            )
            if state_update is None:
                attempts.append(
                    _solve_interactive_state_emit(
                        traces,
                        steps=steps,
                        num_restarts=num_restarts,
                        seed=seed,
                    )
                )
                attempts.append(
                    _solve_interactive_two_register_emit(
                        traces,
                        steps=steps,
                        num_restarts=num_restarts,
                        seed=seed,
                    )
                )
            chosen = next((attempt for attempt in attempts if attempt.get("success")), None)
            if chosen is None:
                errors = [attempt["error"] for attempt in attempts if attempt.get("error")]
                chosen = {
                    "supported": True,
                    "success": False,
                    "error": "; ".join(errors) if errors else "interactive differentiable synthesis failed",
                }
            json.dump(chosen, sys.stdout)
        else:
            signature = payload["signature"]
            function_name = payload["function_name"]
            raw_examples = payload["examples"]
            arg_names = _extract_arg_names(signature)
            examples = _coerce_examples(raw_examples)
            if examples is None:
                json.dump(
                    {
                        "supported": False,
                        "success": False,
                        "error": "only scalar numeric examples are supported",
                    },
                    sys.stdout,
                )
                sys.stdout.write("\n")
                return 0
            result = gradient_solve(
                arg_names,
                examples,
                function_name=function_name,
                steps=steps,
                num_restarts=num_restarts,
                seed=seed,
            )
            json.dump(
                {
                    "supported": True,
                    "success": bool(result.success),
                    "code": result.code,
                    "loss": float(result.loss),
                    "structure": result.structure,
                    "metadata": result.metadata,
                    "error": None,
                },
                sys.stdout,
            )
        sys.stdout.write("\n")
    except Exception as err:
        json.dump(
            {
                "supported": True,
                "success": False,
                "error": f"gradient solve failed: {err}",
            },
            sys.stdout,
        )
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
