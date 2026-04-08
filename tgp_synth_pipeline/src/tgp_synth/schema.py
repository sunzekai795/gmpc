from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Speaker = Literal["client", "therapist", "other"]


class Turn(BaseModel):
    """A single dialogue turn."""

    session_id: str
    turn_id: int
    speaker: Speaker
    language: str = "zh"
    text: str

    extra: Dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "Turn":
        base_keys = ["session_id", "turn_id", "speaker", "language", "text"]
        base = {k: obj.get(k) for k in base_keys}
        extra = {k: v for k, v in obj.items() if k not in base_keys}
        return Turn(**base, extra=extra)

    def to_json(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "speaker": self.speaker,
            "language": self.language,
            "text": self.text,
            **self.extra,
        }


class TurnAnnotation(BaseModel):
    """LLM-produced annotation for a turn."""

    text_with_paralinguistic: str
    labels: Dict[str, Any] = Field(default_factory=dict)


class AnnotatedTurn(BaseModel):
    """Wrapper containing original turn and its annotation."""

    turn: Dict[str, Any]
    annotation: Dict[str, Any]


class SessionAnnotations(BaseModel):
    """The core intermediate artifact of TGP."""

    session_id: str
    language: str
    session_labels: Dict[str, Any] = Field(default_factory=dict)
    turns: List[Dict[str, Any]]


class AudioManifestRow(BaseModel):
    session_id: str
    turn_id: int
    wav_path: Path
    sample_rate: int
    duration_s: float
    tts_backend: str
    prompt_hash: str


class ReferenceSelectionRow(BaseModel):
    session_id: str
    query: str
    segment_path: Path
    score: float


class VideoManifestRow(BaseModel):
    session_id: str
    turn_id: int
    reference_video: Path
    audio_wav: Path
    out_video: Path
    backend: str


class VideoRagHit(BaseModel):
    segment_id: str
    segment_path: Path
    score: float
    best_frame_path: Optional[Path] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
