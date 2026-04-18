## nSynth

This is the Rust synthesis workspace that used to live under `mog_synth/`.

For now:

- the folder name is `nsynth/` so the hub reads cleanly
- the Cargo crate and compiled binary are still named `mog_synth`
- code outside this folder should resolve paths through `nsynth/target/.../mog_synth`

That keeps the workspace organized now without forcing a crate rename in the middle of active development.
