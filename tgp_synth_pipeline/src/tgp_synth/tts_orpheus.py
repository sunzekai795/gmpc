from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from tqdm import tqdm

from .backends.base import TTSSynthesisRequest
from .backends.orpheus import build_orpheus_backend
from .config import OrpheusConfig
from .io import read_jsonl, write_jsonl
from .llm import LLMClient
from .prompts import EMOTION_REASONING_TTS_TEMPLATE
from .utils.hashing import sha1_text


_PARALINGUISTIC_TAGS = [
    "<laugh>",
    "<chuckle>",
    "<sigh>",
    "<cough>",
    "<sniffle>",
    "<groan>",
    "<yawn>",
    "<gasp>",
]


def _detect_existing_tag(text: str) -> bool:
    return any(tag in text for tag in _PARALINGUISTIC_TAGS)


def _pick_paralinguistic_tag(emotion: str, guidance: str) -> str:
    guidance_l = (guidance or "").lower()

    if any(k in guidance_l for k in ["sigh", "叹气"]):
        return "<sigh>"
    if any(k in guidance_l for k in ["laugh", "笑", "chuckle"]):
        return "<chuckle>"
    if any(k in guidance_l for k in ["sniff", "抽泣", "哽咽"]):
        return "<sniffle>"
    if any(k in guidance_l for k in ["cough", "咳"]):
        return "<cough>"
    if any(k in guidance_l for k in ["yawn", "哈欠"]):
        return "<yawn>"

    e = (emotion or "").strip().lower()
    if e in {"anxiety", "fear", "nervous"}:
        return "<sigh>"
    if e in {"sad", "depressed"}:
        return "<sniffle>"
    if e in {"happy", "relieved"}:
        return "<chuckle>"
    if e in {"anger", "annoyed"}:
        return "<groan>"
    return ""


def _wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        sr = wf.getframerate()
    return float(frames) / float(sr)


@dataclass
class AudioManifestRow:
    session_id: str
    turn_id: int
    speaker: str
    emotion: str
    text: str
    tts_guidance: str
    tts_backend: str
    wav_path: str
    duration_s: float
    sample_rate: int
    prompt_hash: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "speaker": self.speaker,
            "emotion": self.emotion,
            "text": self.text,
            "tts_guidance": self.tts_guidance,
            "tts_backend": self.tts_backend,
            "wav_path": self.wav_path,
            "duration_s": self.duration_s,
            "sample_rate": self.sample_rate,
            "prompt_hash": self.prompt_hash,
        }


def _build_tts_guidance(
    llm: Optional[LLMClient],
    speaker: str,
    emotion: str,
    topic: str,
    raw_text: str,
    enabled: bool,
) -> str:
    if llm is None or not enabled:
        return ""
    return llm.complete(
        EMOTION_REASONING_TTS_TEMPLATE.format(
            speaker=speaker,
            emotion=emotion,
            topic=topic,
            text=raw_text,
        )
    ).strip()


def synthesize_from_annotations(
    annotations_jsonl: Path,
    output_dir: Path,
    cfg: OrpheusConfig,
    llm: Optional[LLMClient] = None,
) -> Path:
    """Generate WAV files for every annotated turn.

    The output structure is:
      output_dir/audio/<session_id>/<turn_id>.wav

    Additionally, a manifest jsonl is produced:
      output_dir/audio_manifest.jsonl
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    manifest_path = output_dir / "audio_manifest.jsonl"

    backend = build_orpheus_backend(cfg)

    rows: list[dict] = []
    for sess in tqdm(read_jsonl(annotations_jsonl), desc="audio"):
        session_id = sess["session_id"]
        topic = (sess.get("session_labels", {}) or {}).get("topic", "")

        for item in sess.get("turns", []):
            turn = item.get("turn", {})
            ann = item.get("annotation", {})

            turn_id = int(turn.get("turn_id"))
            speaker = str(turn.get("speaker", "client"))
            raw_text = str(turn.get("text", ""))

            # Prefer LLM-inserted paralinguistic text; otherwise use raw.
            text = str(ann.get("text_with_paralinguistic") or raw_text)
            emotion = str((ann.get("labels", {}) or {}).get("emotion", "Neutral"))

            tts_guidance = _build_tts_guidance(
                llm=llm,
                speaker=speaker,
                emotion=emotion,
                topic=topic,
                raw_text=raw_text,
                enabled=cfg.use_emotion_reasoning,
            )

            if cfg.insert_paralinguistic_if_missing and not _detect_existing_tag(text):
                tag = _pick_paralinguistic_tag(emotion, tts_guidance)
                if tag:
                    text = f"{tag} {text}"

            out_wav = audio_dir / session_id / f"{turn_id:04d}.wav"
            if cfg.skip_existing and out_wav.exists():
                dur = _wav_duration_s(out_wav)
                rows.append(
                    AudioManifestRow(
                        session_id=session_id,
                        turn_id=turn_id,
                        speaker=speaker,
                        emotion=emotion,
                        text=text,
                        tts_guidance=tts_guidance,
                        tts_backend=backend.name,
                        wav_path=str(out_wav),
                        duration_s=dur,
                        sample_rate=cfg.out_sample_rate,
                        prompt_hash=sha1_text(text),
                    ).to_json()
                )
                continue

            req = TTSSynthesisRequest(
                session_id=session_id,
                turn_id=turn_id,
                speaker=speaker,
                text=text,
                emotion=emotion,
                tts_guidance=tts_guidance,
            )
            result = backend.synthesize(req, out_wav)
            dur = _wav_duration_s(result.wav_path)

            rows.append(
                AudioManifestRow(
                    session_id=session_id,
                    turn_id=turn_id,
                    speaker=speaker,
                    emotion=emotion,
                    text=text,
                    tts_guidance=tts_guidance,
                    tts_backend=backend.name,
                    wav_path=str(result.wav_path),
                    duration_s=dur,
                    sample_rate=result.sample_rate,
                    prompt_hash=sha1_text(text),
                ).to_json()
            )

    write_jsonl(manifest_path, rows)
    return manifest_path


__all__ = [
    "synthesize_from_annotations",
]
