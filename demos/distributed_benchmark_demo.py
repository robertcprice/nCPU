#!/usr/bin/env python3
"""Distributed scheduling/dispatch benchmark demo for nCPU.

Compares scheduler policies and shows how schedule-and-run integrates with the
controller.

Run:
    PYTHONPATH=. python demos/distributed_benchmark_demo.py
"""

from __future__ import annotations

from ncpu.distributed import (
    DistributedNCPU,
    DistributedScheduler,
    ProcessDescriptor,
    SchedulingPolicy,
)
from ncpu.differentiable.execution import FixedProgram, Instruction, OPCODES


def _program(opcode_name: str) -> FixedProgram:
    return FixedProgram([
        Instruction(OPCODES[opcode_name], dst=7, src1=0, src2=1),
        Instruction(OPCODES["HALT"]),
    ])


def make_processes() -> list[ProcessDescriptor]:
    return [
        ProcessDescriptor(pid=0, program=_program("ADD"), inputs={0: 1.0, 1: 2.0}, priority=2),
        ProcessDescriptor(pid=1, program=_program("MUL"), inputs={0: 3.0, 1: 4.0}, priority=5),
        ProcessDescriptor(pid=2, program=_program("SUB"), inputs={0: 10.0, 1: 3.0}, priority=1),
        ProcessDescriptor(pid=3, program=_program("ADD"), inputs={0: 7.0, 1: 9.0}, required_backend="cpu"),
    ]


def print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def run_policy_demo(policy: SchedulingPolicy) -> None:
    dcpu = DistributedNCPU(num_cores=4, devices=["cpu"], device_strategy="mirror")
    scheduler = DistributedScheduler(
        num_cores=4,
        policy=policy,
        core_devices=[dcpu.get_device_map()[i] for i in range(dcpu.num_cores)],
    )
    result = dcpu.execute_scheduled(make_processes(), scheduler=scheduler)

    print_header(f"Policy: {policy.value}")
    print("Device map:", result.device_assignments)
    print("Core -> pids:")
    for core_id, pids in result.core_to_pids.items():
        if pids:
            print(f"  core {core_id}: {pids}")
    print("Process results:")
    for pid, exec_result in sorted(result.process_results.items()):
        print(
            f"  pid {pid}: core={result.process_to_core[pid]} "
            f"R7={exec_result.registers[7].item():.1f} steps={exec_result.steps_executed}"
        )


def main() -> None:
    print("nCPU Distributed Benchmark Demo")
    print("Compares scheduler policies using the integrated schedule-and-run helper.")

    for policy in (
        SchedulingPolicy.ROUND_ROBIN,
        SchedulingPolicy.LOAD_BALANCED,
        SchedulingPolicy.AFFINITY,
    ):
        run_policy_demo(policy)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
