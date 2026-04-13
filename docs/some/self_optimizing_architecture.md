# Self-Optimizing Architecture in nCPU

## Purpose

This file explains how SOME fits into the larger nCPU repository.

The repo now has three closely related but distinct layers:

1. **GPU-native machine**
- Rust Metal kernel
- GPU OS / microkernel
- ELF loader
- BusyBox / Alpine runtime
- GPU-native debugging toolkit

2. **Execution-grounded optimization**
- benchmark runners
- verifiers
- trajectory capture
- async buffered inference API

3. **Internalized self-optimization**
- latent action / halt / memory / descriptor / patch heads
- task-local fast weights
- segmented decode path
- controller bundles and training pipeline

## Relationship To The Rest Of The Repo

SOME is not supposed to replace the GPU OS and debugging stack. It uses them.

- The GPU-native execution substrate gives the project a place to execute and inspect code.
- The debugging toolkit gives unusually rich state visibility.
- The benchmark and verifier layers convert that execution into training and evaluation data.
- The hidden controller and latent heads try to internalize more of that loop into the model side.

## Why The Architecture Is Split

The repo needed a benchmarkable path before the fully internal path existed. That is why there is both:

- an external verifier bridge
- and a growing internal hidden-controller path

The external bridge was necessary to get real benchmark evidence first. The current work is about steadily replacing more of that bridge with latent control and weight-side adaptation.

## Current Publishable Story

The most defensible public story is:

- nCPU already demonstrates a GPU-native machine, OS, compiler, and debugger stack
- SOME adds an execution-grounded hidden controller for code and reasoning
- SOME is moving from wrapper-loop behavior toward internal adaptation via latent heads, fast weights, segmented decode, and recurrent memory
- there is benchmark evidence that this direction already helps on real coding tasks

That is a stronger and more honest story than pretending the final architecture is already complete.
