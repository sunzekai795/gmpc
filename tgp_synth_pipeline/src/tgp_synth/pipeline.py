from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import PipelineConfig
from .llm import LLMClient, load_llm_callable
from .text_annotation import run_annotate_text
from .tts_orpheus import synthesize_from_annotations
from .video_rag import build_video_rag_db
from .video_synthesis import ensure_reference_video, synthesize_video_from_annotations


@dataclass
class PipelineArtifacts:
    annotations_jsonl: Path
    audio_manifest_jsonl: Path
    video_manifest_jsonl: Optional[Path] = None


def build_llm_client(cfg: PipelineConfig, llm_override: Optional[str] = None) -> LLMClient:
    spec = llm_override or cfg.llm.callable
    return LLMClient(
        generate=load_llm_callable(spec),
        max_retries=cfg.llm.max_retries,
        timeout_s=cfg.llm.timeout_s,
    )


def run_all(cfg: PipelineConfig, llm_override: Optional[str], videos_dir: Optional[Path]) -> PipelineArtifacts:
    """Run full TGP pipeline."""

    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)

    llm = build_llm_client(cfg, llm_override)

    annotations = run_annotate_text(
        input_turns_jsonl=cfg.paths.input_turns_jsonl,
        output_dir=cfg.paths.output_dir,
        llm=llm,
        cfg=cfg.annotation,
    )

    audio_manifest = synthesize_from_annotations(
        annotations_jsonl=annotations,
        output_dir=cfg.paths.output_dir,
        cfg=cfg.orpheus,
        llm=llm,
    )

    if cfg.video_rag.enabled:
        if videos_dir is None:
            raise ValueError("video_rag.enabled=true requires --videos_dir")
        build_video_rag_db(videos_dir=videos_dir, output_dir=cfg.paths.output_dir, cfg=cfg.video_rag)

        video_manifest = synthesize_video_from_annotations(
            annotations_jsonl=annotations,
            output_dir=cfg.paths.output_dir,
            video_rag_cfg=cfg.video_rag,
            latentsync_cfg=cfg.latentsync,
            llm=llm,
        )
    else:
        ref = ensure_reference_video(cfg.paths.output_dir, resolution=cfg.latentsync.resolution)
        video_manifest = synthesize_video_from_annotations(
            annotations_jsonl=annotations,
            output_dir=cfg.paths.output_dir,
            video_rag_cfg=cfg.video_rag,
            latentsync_cfg=cfg.latentsync,
            llm=llm,
            reference_video=ref,
        )

    return PipelineArtifacts(annotations_jsonl=annotations, audio_manifest_jsonl=audio_manifest, video_manifest_jsonl=video_manifest)
