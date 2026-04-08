from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .config import load_config
from .llm import LLMClient
from .pipeline import PipelineArtifacts, build_llm_client, run_all
from .text_annotation import run_annotate_text
from .tts_orpheus import synthesize_from_annotations
from .utils.logging import setup_logging
from .video_rag import build_video_rag_db
from .video_synthesis import ensure_reference_video, synthesize_video_from_annotations

app = typer.Typer(add_completion=False, help="TGP (text -> audio -> video) synthesis pipeline")


@app.callback()
def _main(
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level")
) -> None:
    setup_logging(log_level)


def _load(cfg_path: Path):
    cfg = load_config(cfg_path)
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg


@app.command("print-config")
def print_config(config: Path = typer.Option(..., "--config", exists=True)):
    cfg = _load(config)
    typer.echo(json.dumps(cfg.model_dump(mode="json"), ensure_ascii=False, indent=2))


@app.command("annotate-text")
def annotate_text(
    config: Path = typer.Option(..., "--config", exists=True),
    llm: Optional[str] = typer.Option(None, "--llm", help='Override LLM callable spec: "module:function" or "file.py:function"'),
):
    cfg = _load(config)
    llm_client = build_llm_client(cfg, llm)
    out = run_annotate_text(cfg.paths.input_turns_jsonl, cfg.paths.output_dir, llm_client, cfg.annotation)
    typer.echo(str(out))


@app.command("synthesize-audio")
def synthesize_audio(
    config: Path = typer.Option(..., "--config", exists=True),
    annotations: Optional[Path] = typer.Option(None, "--annotations"),
    llm: Optional[str] = typer.Option(None, "--llm", help="LLM callable used for emotion reasoning (optional)"),
):
    cfg = _load(config)
    ann = annotations or (cfg.paths.output_dir / "annotations" / "sessions.jsonl")

    llm_client: Optional[LLMClient] = None
    if cfg.orpheus.use_emotion_reasoning:
        llm_client = build_llm_client(cfg, llm)

    out = synthesize_from_annotations(ann, cfg.paths.output_dir, cfg.orpheus, llm=llm_client)
    typer.echo(str(out))


@app.command("build-video-rag")
def build_video_rag(
    config: Path = typer.Option(..., "--config", exists=True),
    videos_dir: Path = typer.Option(..., "--videos_dir", exists=True),
):
    cfg = _load(config)
    out = build_video_rag_db(videos_dir=videos_dir, output_dir=cfg.paths.output_dir, cfg=cfg.video_rag)
    typer.echo(str(out))


@app.command("synthesize-video")
def synthesize_video(
    config: Path = typer.Option(..., "--config", exists=True),
    annotations: Optional[Path] = typer.Option(None, "--annotations"),
    llm: Optional[str] = typer.Option(None, "--llm", help="LLM callable used for RAG query generation"),
    reference_video: Optional[Path] = typer.Option(None, "--reference_video", help="Skip RAG and use a fixed reference video"),
):
    cfg = _load(config)
    ann = annotations or (cfg.paths.output_dir / "annotations" / "sessions.jsonl")

    llm_client: Optional[LLMClient] = None
    if reference_video is None and cfg.video_rag.enabled:
        llm_client = build_llm_client(cfg, llm)

    if reference_video is None and not cfg.video_rag.enabled:
        reference_video = ensure_reference_video(cfg.paths.output_dir, resolution=cfg.latentsync.resolution)

    out = synthesize_video_from_annotations(
        annotations_jsonl=ann,
        output_dir=cfg.paths.output_dir,
        video_rag_cfg=cfg.video_rag,
        latentsync_cfg=cfg.latentsync,
        llm=llm_client,
        reference_video=reference_video,
    )
    typer.echo(str(out))


@app.command("run-all")
def run_all_cmd(
    config: Path = typer.Option(..., "--config", exists=True),
    llm: Optional[str] = typer.Option(None, "--llm"),
    videos_dir: Optional[Path] = typer.Option(None, "--videos_dir"),
):
    cfg = _load(config)
    artifacts: PipelineArtifacts = run_all(cfg, llm_override=llm, videos_dir=videos_dir)
    typer.echo(json.dumps(artifacts.__dict__, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    app()
