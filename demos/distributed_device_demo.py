#!/usr/bin/env python3
"""Distributed device dispatch demo for nCPU.

Shows:
1. discovered execution backends
2. per-core device assignment
3. scheduler placement with device affinity / backend requirements
4. rebalance behavior
5. a tiny parallel execution summary

Run:
    PYTHONPATH=. python demos/distributed_device_demo.py
"""

from __future__ import annotations

from ncpu.distributed import (
    DistributedNCPU,
    DistributedScheduler,
    ProcessDescriptor,
    SchedulingPolicy,
)
from ncpu.differentiable.execution import FixedProgram, Instruction, OPCODES


def _add_program() -> FixedProgram:
    return FixedProgram([
        Instruction(OPCODES["ADD"], dst=2, src1=0, src2=1),
        Instruction(OPCODES["HALT"]),
    ])


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    print_section("1. Discovered execution devices")
    devices = DistributedNCPU.discover_devices()
    for dev in devices:
        print(f"- {dev.name:8s} kind={dev.kind} available={dev.available}")

    print_section("2. Core assignment with auto dispatch")
    dcpu = DistributedNCPU(num_cores=4, core_config=None, devices=["auto"], device_strategy="round_robin")
    for row in dcpu.get_device_assignment_report():
        print(
            f"core {row['core_id']}: requested={row['requested_device']} "
            f"assigned={row['assigned_device']} reason={row['reason']}"
        )

    print_section("3. Scheduler placement with device constraints")
    scheduler = DistributedScheduler(
        num_cores=4,
        policy=SchedulingPolicy.LOAD_BALANCED,
        core_devices=[dcpu.get_device_map()[i] for i in range(dcpu.num_cores)],
    )
    scheduler.submit_batch([
        ProcessDescriptor(pid=1, program=_add_program(), device_affinity="cpu", priority=5),
        ProcessDescriptor(pid=2, program=_add_program(), required_backend="cpu", priority=2),
    ])
    assignments = scheduler.schedule()
    for core_id, procs in assignments.items():
        if not procs:
            continue
        print(f"core {core_id} ({scheduler.core_devices[core_id]}): {[p.pid for p in procs]}")

    print_section("4. Rebalance example")
    dcpu.rebalance_devices(["cpu"], strategy="mirror")
    for row in dcpu.get_device_assignment_report():
        print(
            f"core {row['core_id']}: requested={row['requested_device']} "
            f"assigned={row['assigned_device']} reason={row['reason']}"
        )

    print_section("5. Tiny parallel execution summary")
    result = dcpu.execute_parallel(
        {0: _add_program(), 1: _add_program()},
        inputs={0: {0: 3.0, 1: 5.0}, 1: {0: 10.0, 1: 7.0}},
    )
    for core_id, core_result in result.core_results.items():
        print(
            f"core {core_id} on {result.device_assignments[core_id]} -> "
            f"R2={core_result.registers[2].item():.1f} steps={core_result.steps_executed}"
        )

    print()
    print("Demo complete.")


if __name__ == "__main__":
    main()
