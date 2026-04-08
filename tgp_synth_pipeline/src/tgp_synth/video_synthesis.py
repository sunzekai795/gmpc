from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .backends.base import VideoSynthesisRequest
from .backends.latentsync import build_video_backend
from .config import LatentSyncConfig, VideoRagConfig
from .io import read_jsonl, write_jsonl
from .llm import LLMClient
from .prompts import VIDEO_RAG_QUERY_TEMPLATE
from .utils.ffmpeg import concat_videos, crop_square_and_scale, make_synthetic_reference_video
from .utils.hashing import sha1_text
from .video_rag import load_video_rag_db, query_video_rag


@dataclass
class ReferenceSelection:
    session_id: str
    query: str
    segment_path: str
    score: float

    def to_json(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "segment_path": self.segment_path,
            "score": self.score,
        }


@dataclass
class VideoManifestRow:
    session_id: str
    turn_id: int
    reference_video: str
    audio_wav: str
    out_video: str
    backend: str
    query_hash: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "reference_video": self.reference_video,
            "audio_wav": self.audio_wav,
            "out_video": self.out_video,
            "backend": self.backend,
            "query_hash": self.query_hash,
        }


def _build_query(llm: LLMClient, speaker: str, emotion: str, topic: str, text: str) -> str:
    prompt = VIDEO_RAG_QUERY_TEMPLATE.format(speaker=speaker, emotion=emotion, topic=topic, text=text)
    return llm.complete(prompt).strip()


def _select_reference_segment(
    output_dir: Path,
    cfg: VideoRagConfig,
    llm: Optional[LLMClient],
    session: dict,
    override_ref_video: Optional[Path],
) -> ReferenceSelection:
    session_id = session["session_id"]

    if override_ref_video is not None:
        return ReferenceSelection(session_id=session_id, query="override", segment_path=str(override_ref_video), score=1.0)

    if not cfg.enabled:
        raise ValueError("video_rag.enabled=false and no --reference_video provided")

    if llm is None:
        raise ValueError("LLM is required for generating Video-RAG queries")

    db_dir = output_dir / "video_rag_db"
    if not db_dir.exists():
        raise FileNotFoundError(f"Video-RAG db not found: {db_dir}")

    session_labels = session.get("session_labels", {}) or {}
    topic = str(session_labels.get("topic", ""))

    first = (session.get("turns") or [])[0]
    turn = first.get("turn", {})
    ann = first.get("annotation", {})

    speaker = str(turn.get("speaker", "client"))
    emotion = str((ann.get("labels", {}) or {}).get("emotion", "Neutral"))
    text = str(turn.get("text", ""))

    query = _build_query(llm, speaker=speaker, emotion=emotion, topic=topic, text=text)

    db = load_video_rag_db(db_dir)
    hits = query_video_rag(db, cfg, query=query, top_k=max(cfg.top_k, 1))
    if not hits:
        raise RuntimeError("Video-RAG returned empty results")

    top = hits[0]
    return ReferenceSelection(
        session_id=session_id,
        query=query,
        segment_path=str(top["segment_path"]),
        score=float(top["score"]),
    )


def synthesize_video_from_annotations(
    annotations_jsonl: Path,
    output_dir: Path,
    video_rag_cfg: VideoRagConfig,
    latentsync_cfg: LatentSyncConfig,
    llm: Optional[LLMClient] = None,
    reference_video: Optional[Path] = None,
) -> Path:
    """Generate per-turn videos and concat them per session."""

    output_dir.mkdir(parents=True, exist_ok=True)

    backend = build_video_backend(latentsync_cfg)

    ref_sel_rows: List[dict] = []
    video_rows: List[dict] = []

    sessions = list(read_jsonl(annotations_jsonl))
    for sess in tqdm(sessions, desc="video"):
        session_id = sess["session_id"]

        selection = _select_reference_segment(
            output_dir=output_dir,
            cfg=video_rag_cfg,
            llm=llm,
            session=sess,
            override_ref_video=reference_video,
        )
        ref_sel_rows.append(selection.to_json())

        ref_video_raw = Path(selection.segment_path)
        if not ref_video_raw.exists():
            raise FileNotFoundError(f"Reference video does not exist: {ref_video_raw}")

        ref_video = output_dir / "ref_videos" / f"{session_id}.mp4"
        crop_square_and_scale(ref_video_raw, ref_video, resolution=latentsync_cfg.resolution)

        per_turn: List[Path] = []
        for item in sess.get("turns", []):
            turn = item.get("turn", {})
            turn_id = int(turn.get("turn_id"))

            audio_wav = output_dir / "audio" / session_id / f"{turn_id:04d}.wav"
            if not audio_wav.exists():
                raise FileNotFoundError(f"Audio missing: {audio_wav}")

            out_mp4 = output_dir / "video" / session_id / f"{turn_id:04d}.mp4"
            if latentsync_cfg.skip_existing and out_mp4.exists():
                per_turn.append(out_mp4)
                video_rows.append(
                    VideoManifestRow(
                        session_id=session_id,
                        turn_id=turn_id,
                        reference_video=str(ref_video),
                        audio_wav=str(audio_wav),
                        out_video=str(out_mp4),
                        backend=backend.name,
                        query_hash=sha1_text(selection.query),
                    ).to_json()
                )
                continue

            req = VideoSynthesisRequest(
                session_id=session_id,
                turn_id=turn_id,
                reference_video=ref_video,
                audio_wav=audio_wav,
            )
            backend.synthesize(req, out_mp4)
            per_turn.append(out_mp4)

            video_rows.append(
                VideoManifestRow(
                    session_id=session_id,
                    turn_id=turn_id,
                    reference_video=str(ref_video),
                    audio_wav=str(audio_wav),
                    out_video=str(out_mp4),
                    backend=backend.name,
                    query_hash=sha1_text(selection.query),
                ).to_json()
            )

        if per_turn:
            out_session = output_dir / "video_sessions" / f"{session_id}.mp4"
            concat_videos(per_turn, out_session)

    write_jsonl(output_dir / "reference_selection.jsonl", ref_sel_rows)
    write_jsonl(output_dir / "video_manifest.jsonl", video_rows)
    return output_dir / "video_manifest.jsonl"


def ensure_reference_video(output_dir: Path, resolution: int = 512) -> Path:
    """Create a synthetic reference video for dummy backend smoke tests."""

    ref = output_dir / "_synthetic_ref" / "ref.mp4"
    if not ref.exists():
        make_synthetic_reference_video(ref, resolution=resolution)
    return ref


__all__ = [
    "synthesize_video_from_annotations",
    "ensure_reference_video",
]
