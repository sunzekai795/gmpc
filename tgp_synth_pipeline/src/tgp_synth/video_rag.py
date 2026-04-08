from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .config import VideoRagConfig
from .utils.ffmpeg import cut_video_segment, extract_keyframes
from .utils.hashing import sha1_file, sha1_text


def _ffprobe_duration_s(video: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr}")
    return float(proc.stdout.strip())


def _normalize(v: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v / denom


@dataclass
class SegmentRecord:
    segment_id: str
    segment_path: Path
    source_video: Path
    start_s: float
    duration_s: float
    best_frame_path: Path
    embedding_type: str

    def to_json(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "segment_path": str(self.segment_path),
            "source_video": str(self.source_video),
            "start_s": self.start_s,
            "duration_s": self.duration_s,
            "best_frame_path": str(self.best_frame_path),
            "embedding_type": self.embedding_type,
        }


@dataclass
class VideoRagDB:
    db_dir: Path
    embeddings: np.ndarray
    records: List[SegmentRecord]
    index: Optional[object] = None

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[SegmentRecord, float]]:
        q = query_vec.astype(np.float32)

        if self.index is not None:
            import numpy as _np

            scores, idxs = self.index.search(_np.expand_dims(q, 0), top_k)
            idxs = idxs[0].tolist()
            scores = scores[0].tolist()
            return [(self.records[i], float(s)) for i, s in zip(idxs, scores)]

        sims = (self.embeddings @ q).tolist()
        idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        return [(self.records[i], float(sims[i])) for i in idxs]


class _OpenCLIP:
    def __init__(self, cfg: VideoRagConfig):
        try:
            import open_clip
            import torch
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Video-RAG requires extra deps: pip install -e '.[rag]' ") from e

        self._torch = torch
        model, _, preprocess = open_clip.create_model_and_transforms(cfg.clip_model, pretrained=cfg.clip_pretrained)
        model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(self._device)

        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(cfg.clip_model)

    @property
    def embedding_dim(self) -> int:
        return int(self._model.text_projection.shape[1])  # type: ignore[attr-defined]

    def encode_images(self, image_paths: Sequence[Path]) -> np.ndarray:
        from PIL import Image

        batch = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            batch.append(self._preprocess(img))
        x = self._torch.stack(batch).to(self._device)
        with self._torch.no_grad():
            feats = self._model.encode_image(x)
        feats = feats.detach().cpu().numpy().astype(np.float32)
        return _normalize(feats)

    def encode_text(self, text: str) -> np.ndarray:
        tokens = self._tokenizer([text]).to(self._device)
        with self._torch.no_grad():
            feats = self._model.encode_text(tokens)
        feats = feats.detach().cpu().numpy().astype(np.float32)
        return _normalize(feats)[0]


class VideoRagBuilder:
    def __init__(self, cfg: VideoRagConfig):
        self._cfg = cfg
        self._clip = _OpenCLIP(cfg)

    def iter_video_files(self, videos_dir: Path) -> List[Path]:
        exts = {".mp4", ".mov", ".mkv", ".webm"}
        files = [p for p in videos_dir.glob("**/*") if p.is_file() and p.suffix.lower() in exts]
        return sorted(files)

    def build(self, videos_dir: Path, output_dir: Path) -> Path:
        db_dir = output_dir / "video_rag_db"
        segments_dir = db_dir / "segments"
        frames_dir = db_dir / "frames"
        segments_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)

        records: List[SegmentRecord] = []
        embeddings: List[np.ndarray] = []

        video_files = self.iter_video_files(videos_dir)
        for video in tqdm(video_files, desc="video-rag"):
            dur = _ffprobe_duration_s(video)
            n_seg = max(1, math.ceil(dur / self._cfg.segment_seconds))

            for idx in range(n_seg):
                start = idx * self._cfg.segment_seconds
                seg_id = f"{video.stem}__{idx:04d}__{sha1_text(str(video))[:8]}"
                out_seg = segments_dir / f"{seg_id}.mp4"

                cut_video_segment(video, out_seg, start_s=start, duration_s=self._cfg.segment_seconds)

                out_frames = frames_dir / seg_id
                extract_keyframes(out_seg, out_frames, fps=self._cfg.keyframe_fps)
                frame_paths = sorted(out_frames.glob("*.jpg"))
                if not frame_paths:
                    continue

                frame_paths = frame_paths[: self._cfg.max_keyframes_per_segment]
                feats = self._clip.encode_images(frame_paths)
                seg_feat = _normalize(feats.mean(axis=0))

                embeddings.append(seg_feat)
                records.append(
                    SegmentRecord(
                        segment_id=seg_id,
                        segment_path=out_seg,
                        source_video=video,
                        start_s=float(start),
                        duration_s=float(self._cfg.segment_seconds),
                        best_frame_path=frame_paths[0],
                        embedding_type=f"openclip:{self._cfg.clip_model}/{self._cfg.clip_pretrained}",
                    )
                )

        if not embeddings:
            raise RuntimeError(f"No segments built from {videos_dir}")

        emb = np.stack(embeddings).astype(np.float32)

        (db_dir / "config.json").write_text(json.dumps(self._cfg.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(db_dir / "embeddings.npy", emb)
        (db_dir / "segments.jsonl").write_text(
            "\n".join(json.dumps(r.to_json(), ensure_ascii=False) for r in records) + "\n",
            encoding="utf-8",
        )

        try:
            import faiss

            index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)
            faiss.write_index(index, str(db_dir / "index.faiss"))
        except Exception:
            pass

        return db_dir


def build_video_rag_db(videos_dir: Path, output_dir: Path, cfg: VideoRagConfig) -> Path:
    return VideoRagBuilder(cfg).build(videos_dir=videos_dir, output_dir=output_dir)


def load_video_rag_db(db_dir: Path) -> VideoRagDB:
    emb = np.load(db_dir / "embeddings.npy").astype(np.float32)

    records: List[SegmentRecord] = []
    for line in (db_dir / "segments.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        records.append(
            SegmentRecord(
                segment_id=obj["segment_id"],
                segment_path=Path(obj["segment_path"]),
                source_video=Path(obj["source_video"]),
                start_s=float(obj["start_s"]),
                duration_s=float(obj["duration_s"]),
                best_frame_path=Path(obj["best_frame_path"]),
                embedding_type=obj.get("embedding_type", "openclip"),
            )
        )

    index = None
    idx_path = db_dir / "index.faiss"
    if idx_path.exists():
        try:
            import faiss

            index = faiss.read_index(str(idx_path))
        except Exception:
            index = None

    return VideoRagDB(db_dir=db_dir, embeddings=emb, records=records, index=index)


def query_video_rag(db: VideoRagDB, cfg: VideoRagConfig, query: str, top_k: int) -> List[dict]:
    clip = _OpenCLIP(cfg)
    q = clip.encode_text(query)
    hits = db.search(q, top_k=top_k)
    return [
        {
            "segment_id": rec.segment_id,
            "segment_path": str(rec.segment_path),
            "score": float(score),
            "best_frame_path": str(rec.best_frame_path),
            "metadata": {"source_video": str(rec.source_video), "start_s": rec.start_s},
        }
        for rec, score in hits
    ]
