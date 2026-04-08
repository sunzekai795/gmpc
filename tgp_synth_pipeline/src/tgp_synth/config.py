from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field

from .utils.paths import resolve_under


class ToolchainConfig(BaseModel):
    """User-owned configuration.

    This file is intended to be edited by end users and should contain machine-
    specific paths, e.g. separate conda envs for Orpheus/LatentSync.
    """

    llm_callable: Optional[str] = Field(
        None,
        description='LLM callable spec ("module:function" or "/abs/path.py:function")',
    )
    orpheus_python: Optional[Path] = Field(None, description="Python executable for Orpheus environment")
    latentsync_repo_dir: Optional[Path] = Field(None, description="Path to LatentSync git repo")
    latentsync_python: Optional[Path] = Field(None, description="Python executable for LatentSync environment")


class PathsConfig(BaseModel):
    input_turns_jsonl: Path = Field(..., description="jsonl, one turn per line")
    output_dir: Path = Field(..., description="output root")

    toolchain_yaml: Optional[Path] = Field(
        None,
        description="Optional: a user-owned toolchain yaml that stores machine-specific paths",
    )


class LLMConfig(BaseModel):
    callable: str = Field(..., description='LLM function spec: "module:function" or "/path/file.py:function"')
    max_retries: int = 2
    timeout_s: Optional[float] = None


class AnnotationConfig(BaseModel):
    self_consistency_votes: int = Field(3, description="Number of samples for self-consistency voting")
    language_default: str = "zh"


class OrpheusConfig(BaseModel):
    backend: Literal["subprocess", "inprocess", "dummy"] = "subprocess"

    python: Optional[Path] = Field(
        None,
        description="backend=subprocess: python executable for Orpheus environment",
    )

    model_name: str = "canopylabs/orpheus-tts-0.1-finetune-prod"
    voice: str = "tara"

    use_emotion_reasoning: bool = True
    insert_paralinguistic_if_missing: bool = True

    temperature: float = 0.4
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    max_tokens: int = 2000
    stop_token_id: int = 128258

    out_sample_rate: int = 44100
    skip_existing: bool = True


class VideoRagConfig(BaseModel):
    enabled: bool = True
    segment_seconds: int = 30
    keyframe_fps: float = 1.0
    max_keyframes_per_segment: int = 32

    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"

    top_k: int = 5


class LatentSyncConfig(BaseModel):
    backend: Literal["subprocess", "dummy"] = "subprocess"

    repo_dir: Optional[Path] = Field(None, description="LatentSync repo directory")
    python: Optional[Path] = Field(None, description="Python executable for LatentSync environment")

    unet_config_path: str = "configs/unet/stage2_512.yaml"
    inference_ckpt_path: str = "checkpoints/latentsync_unet.pt"
    inference_steps: int = 20
    guidance_scale: float = 1.5
    enable_deepcache: bool = True

    resolution: int = 512
    temp_root: Path = Path("latentsync_tmp")

    skip_existing: bool = True


class PipelineConfig(BaseModel):
    paths: PathsConfig
    llm: LLMConfig
    annotation: AnnotationConfig = AnnotationConfig()
    orpheus: OrpheusConfig = OrpheusConfig()
    video_rag: VideoRagConfig = VideoRagConfig()
    latentsync: LatentSyncConfig = LatentSyncConfig()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge override into base (override wins)."""

    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(config_path: Path) -> PipelineConfig:
    """Load config and resolve all paths.

    Relative paths are resolved against the directory of the config file.

    If paths.toolchain_yaml is provided, its content will be merged into the
    config with the following mapping:

    - toolchain.llm_callable -> llm.callable
    - toolchain.orpheus_python -> orpheus.python
    - toolchain.latentsync_repo_dir -> latentsync.repo_dir
    - toolchain.latentsync_python -> latentsync.python
    """

    raw = _load_yaml(config_path)
    cfg = PipelineConfig.model_validate(raw)

    base = config_path.parent.resolve()
    cfg.paths.input_turns_jsonl = resolve_under(base, cfg.paths.input_turns_jsonl)  # type: ignore
    cfg.paths.output_dir = resolve_under(base, cfg.paths.output_dir)  # type: ignore
    cfg.paths.toolchain_yaml = resolve_under(base, cfg.paths.toolchain_yaml)  # type: ignore

    cfg.orpheus.python = resolve_under(base, cfg.orpheus.python)  # type: ignore
    cfg.latentsync.repo_dir = resolve_under(base, cfg.latentsync.repo_dir)  # type: ignore
    cfg.latentsync.python = resolve_under(base, cfg.latentsync.python)  # type: ignore
    cfg.latentsync.temp_root = resolve_under(cfg.paths.output_dir, cfg.latentsync.temp_root)  # type: ignore

    if cfg.paths.toolchain_yaml is not None and cfg.paths.toolchain_yaml.exists():
        tool = ToolchainConfig.model_validate(_load_yaml(cfg.paths.toolchain_yaml))
        if tool.llm_callable:
            cfg.llm.callable = tool.llm_callable
        if tool.orpheus_python:
            cfg.orpheus.python = resolve_under(cfg.paths.toolchain_yaml.parent, tool.orpheus_python)
        if tool.latentsync_repo_dir:
            cfg.latentsync.repo_dir = resolve_under(cfg.paths.toolchain_yaml.parent, tool.latentsync_repo_dir)
        if tool.latentsync_python:
            cfg.latentsync.python = resolve_under(cfg.paths.toolchain_yaml.parent, tool.latentsync_python)

    return cfg
