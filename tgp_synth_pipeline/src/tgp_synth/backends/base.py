from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol


@dataclass(frozen=True)
class TTSSynthesisRequest:
    """A single utterance synthesis request."""

    session_id: str
    turn_id: int
    speaker: str
    text: str
    emotion: str
    tts_guidance: str = ""


@dataclass(frozen=True)
class TTSSynthesisResult:
    session_id: str
    turn_id: int
    wav_path: Path
    sample_rate: int


class TTSBackend(Protocol):
    """Backend interface for generating waveform audio."""

    name: str

    def synthesize(self, request: TTSSynthesisRequest, out_wav: Path) -> TTSSynthesisResult:  # pragma: no cover
        ...


@dataclass(frozen=True)
class VideoSynthesisRequest:
    """A single utterance video synthesis request."""

    session_id: str
    turn_id: int
    reference_video: Path
    audio_wav: Path


@dataclass(frozen=True)
class VideoSynthesisResult:
    session_id: str
    turn_id: int
    video_path: Path


class VideoBackend(Protocol):
    """Backend interface for generating an mp4 for one utterance."""

    name: str

    def synthesize(self, request: VideoSynthesisRequest, out_mp4: Path) -> VideoSynthesisResult:  # pragma: no cover
        ...
