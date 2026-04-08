"""Orpheus TTS inference helper.

This script is intended to be executed inside a dedicated Orpheus environment.
It reads a JSON payload and writes a 24kHz mono WAV.

Usage:
  python scripts/orpheus_tts_infer.py --payload /path/to/payload.json

Payload schema (JSON object):
  {
    "model_name": "canopylabs/orpheus-tts-0.1-finetune-prod",
    "voice": "tara",
    "prompt": "<text with optional paralinguistic tags>",
    "out_wav": "/abs/or/rel/path.wav",
    "temperature": 0.4,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "max_tokens": 2000,
    "stop_token_id": 128258
  }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))

    from orpheus_tts import OrpheusModel

    model = OrpheusModel(model_name=payload["model_name"], max_model_len=2048)
    generator = model.generate_speech(
        prompt=payload["prompt"],
        voice=payload["voice"],
        temperature=float(payload.get("temperature", 0.4)),
        top_p=float(payload.get("top_p", 0.9)),
        repetition_penalty=float(payload.get("repetition_penalty", 1.1)),
        max_tokens=int(payload.get("max_tokens", 2000)),
        stop_token_ids=[int(payload.get("stop_token_id", 128258))],
    )

    out_path = Path(payload["out_wav"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import wave

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        for chunk in generator:
            wf.writeframes(chunk)


if __name__ == "__main__":
    main()
