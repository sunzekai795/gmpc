from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import OrpheusConfig
from ..utils.ffmpeg import resample_audio
from ..utils.subprocess_utils import CommandError, run_command
from .base import TTSSynthesisRequest, TTSSynthesisResult, TTSBackend


@dataclass
class _OrpheusPayload:
    model_name: str
    voice: str
    prompt: str
    out_wav: str
    temperature: float
    top_p: float
    repetition_penalty: float
    max_tokens: int
    stop_token_id: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "voice": self.voice,
            "prompt": self.prompt,
            "out_wav": self.out_wav,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "max_tokens": self.max_tokens,
            "stop_token_id": self.stop_token_id,
        }


def _build_payload(cfg: OrpheusConfig, prompt: str, out_wav_24k: Path) -> _OrpheusPayload:
    return _OrpheusPayload(
        model_name=cfg.model_name,
        voice=cfg.voice,
        prompt=prompt,
        out_wav=str(out_wav_24k),
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        repetition_penalty=cfg.repetition_penalty,
        max_tokens=cfg.max_tokens,
        stop_token_id=cfg.stop_token_id,
    )


class OrpheusSubprocessBackend:
    name = "orpheus_subprocess"

    def __init__(self, python_exe: Path, helper_script: Path, cfg: OrpheusConfig):
        self._python_exe = python_exe
        self._helper_script = helper_script
        self._cfg = cfg

    def synthesize(self, request: TTSSynthesisRequest, out_wav: Path) -> TTSSynthesisResult:
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        out_wav_24k = out_wav.with_suffix(".raw24k.wav")
        payload = _build_payload(self._cfg, prompt=request.text, out_wav_24k=out_wav_24k)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(payload.to_dict(), f, ensure_ascii=False)
            payload_path = Path(f.name)

        try:
            run_command([self._python_exe, self._helper_script, "--payload", payload_path])
        except CommandError as e:
            raise RuntimeError(f"Orpheus subprocess failed for {request.session_id}/{request.turn_id}") from e
        finally:
            payload_path.unlink(missing_ok=True)

        resample_audio(out_wav_24k, out_wav, self._cfg.out_sample_rate)
        return TTSSynthesisResult(
            session_id=request.session_id,
            turn_id=request.turn_id,
            wav_path=out_wav,
            sample_rate=self._cfg.out_sample_rate,
        )


class OrpheusInProcessBackend:
    name = "orpheus_inprocess"

    def __init__(self, cfg: OrpheusConfig):
        self._cfg = cfg

    def synthesize(self, request: TTSSynthesisRequest, out_wav: Path) -> TTSSynthesisResult:
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        try:
            from orpheus_tts import OrpheusModel
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "orpheus_tts is not importable in the current environment. "
                "Install orpheus-speech, or switch to backend=subprocess."
            ) from e

        model = OrpheusModel(model_name=self._cfg.model_name, max_model_len=2048)
        generator = model.generate_speech(
            prompt=request.text,
            voice=self._cfg.voice,
            temperature=float(self._cfg.temperature),
            top_p=float(self._cfg.top_p),
            repetition_penalty=float(self._cfg.repetition_penalty),
            max_tokens=int(self._cfg.max_tokens),
            stop_token_ids=[int(self._cfg.stop_token_id)],
        )

        import wave

        out_wav_24k = out_wav.with_suffix(".raw24k.wav")
        with wave.open(str(out_wav_24k), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            for chunk in generator:
                wf.writeframes(chunk)

        resample_audio(out_wav_24k, out_wav, self._cfg.out_sample_rate)
        return TTSSynthesisResult(
            session_id=request.session_id,
            turn_id=request.turn_id,
            wav_path=out_wav,
            sample_rate=self._cfg.out_sample_rate,
        )


class DummyTTSBackend:
    """A lightweight TTS backend for local pipeline validation.

    It generates a deterministic sine-wave-like audio with a duration proportional
    to the number of characters.

    This backend is intentionally self-contained and requires no ML dependencies.
    """

    name = "dummy_tts"

    def __init__(self, sample_rate: int = 44100, base_seconds: float = 1.0, char_rate: float = 22.0):
        self._sr = sample_rate
        self._base = base_seconds
        self._char_rate = char_rate

    def synthesize(self, request: TTSSynthesisRequest, out_wav: Path) -> TTSSynthesisResult:
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        seconds = self._base + len(request.text) / max(self._char_rate, 1.0)
        n = int(seconds * self._sr)

        import numpy as np
        import wave

        t = np.arange(n, dtype=np.float32) / float(self._sr)
        freq = 220.0 + 30.0 * math.sin(request.turn_id)
        y = 0.1 * np.sin(2 * math.pi * freq * t)
        pcm = (y * 32767.0).astype(np.int16)

        with wave.open(str(out_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sr)
            wf.writeframes(pcm.tobytes())

        return TTSSynthesisResult(
            session_id=request.session_id,
            turn_id=request.turn_id,
            wav_path=out_wav,
            sample_rate=self._sr,
        )


def build_orpheus_backend(cfg: OrpheusConfig) -> TTSBackend:
    """Factory for TTS backends."""

    if cfg.backend == "dummy":
        return DummyTTSBackend(sample_rate=cfg.out_sample_rate)

    if cfg.backend == "inprocess":
        return OrpheusInProcessBackend(cfg)

    if cfg.backend == "subprocess":
        if cfg.python is None:
            raise ValueError("orpheus.backend=subprocess requires orpheus.python")

        helper = Path(__file__).resolve().parents[3] / "scripts" / "orpheus_tts_infer.py"
        if not helper.exists():
            raise FileNotFoundError(f"Orpheus helper script missing: {helper}")
        return OrpheusSubprocessBackend(cfg.python, helper, cfg)

    raise NotImplementedError(f"Unsupported Orpheus backend: {cfg.backend}")
