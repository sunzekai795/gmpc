# Implementation Notes

This repository is intentionally split into two layers:

1) A lightweight orchestration layer (`tgp_synth`) that only depends on standard Python packages + ffmpeg.
2) Heavy ML backends (Orpheus-TTS, LatentSync, optional OpenCLIP/FAISS) that are integrated via pluggable backends.

## Mapping to the paper

- Text stage: `tgp_synth.text_annotation` produces `annotations/sessions.jsonl`.
- Audio stage: `tgp_synth.tts_orpheus` generates per-turn wav files and `audio_manifest.jsonl`.
- Video stage:
  - Video-RAG: `tgp_synth.video_rag` builds a CLIP index over 30s segments.
  - Reference selection + synthesis: `tgp_synth.video_synthesis` selects one reference segment per session, then calls the configured video backend.

## Backends

### TTS

- `orpheus.backend=subprocess` runs `scripts/orpheus_tts_infer.py` inside the user-provided Orpheus python environment.
- `orpheus.backend=dummy` is a lightweight sine-wave generator used for smoke tests.

### Video

- `latentsync.backend=subprocess` runs `python -m scripts.inference ...` inside the user-provided LatentSync repo.
- `latentsync.backend=dummy` loops the reference video and muxes the synthesized audio, allowing end-to-end validation without GPUs.

## Why we keep a dummy backend

LatentSync and Orpheus are both heavy and environment-sensitive. A deterministic dummy backend allows you to:

- validate JSON schema & filesystem layout
- validate orchestration logic and manifests
- debug prompts and intermediate artifacts

Then you can switch to real backends by editing `user_provided/toolchain.yaml` and `configs/run.yaml`.
