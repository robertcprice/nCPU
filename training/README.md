# Training Directory

This directory is for local training artifacts produced by the SOME controller pipeline:

- prepared JSONL datasets
- bundle templates
- controller checkpoints
- proof-run outputs

These artifact trees are intentionally gitignored. The repository should track:

- the code that generates them
- the documentation that explains them
- concise summary results that are worth citing

The current tracked entry points are under `ncpu/self_optimizing/`, especially:

- `prepare_internal_training_data.py`
- `controller_training.py`
- `train_internal_controller.py`
- `run_real_memory_proof.py`
