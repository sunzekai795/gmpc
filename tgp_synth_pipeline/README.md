# TGP Synth

## Setup

```bash
cd tgp_synth_pipeline
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

System deps:

```bash
# Ubuntu
sudo apt-get update && sudo apt-get install -y ffmpeg
```

## User-provided integration

1) Implement your LLM caller:

- Edit `user_provided/llm_caller.py` and implement `generate(prompt: str) -> str`.

2) Point to your Orpheus-TTS + LatentSync installations:

- Edit `user_provided/toolchain.yaml`.

## Run

```bash
tgp-synth run-all \
  --config configs/run.yaml
```

Outputs are written to `paths.output_dir` in the config.
