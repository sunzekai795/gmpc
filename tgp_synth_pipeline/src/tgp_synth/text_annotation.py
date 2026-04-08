from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .config import AnnotationConfig
from .io import load_turns_grouped, write_jsonl
from .llm import LLMClient
from .prompts import (
    ANNOTATION_SYSTEM,
    ANNOTATION_USER_TEMPLATE,
    PARALINGUISTIC_TAGS,
    format_dialog_for_prompt,
)
from .schema import SessionAnnotations, Turn
from .utils.json_utils import JsonExtractError, extract_json_object


class AnnotationError(RuntimeError):
    pass


DEFAULT_EMOTION_VOCAB = [
    "Neutral",
    "Anxiety",
    "Happy",
    "Sad",
    "Anger",
    "Fear",
    "Surprise",
    "Disgust",
    "Shame",
    "Guilt",
]

DEFAULT_STRATEGY_VOCAB = [
    "Question",
    "Reflection of feelings",
    "Reassurance",
    "Summarization",
    "Interpretation",
    "Encouragement",
    "Empathy",
    "Psychoeducation",
    "Unknown",
]


def _normalize_choice(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _coerce_speaker(x: Any) -> str:
    s = str(x or "").strip().lower()
    if s in {"client", "c"}:
        return "client"
    if s in {"therapist", "t", "counselor"}:
        return "therapist"
    return "other"


def _agreement_score(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    score = 0

    a_s = a.get("session_labels", {}) or {}
    b_s = b.get("session_labels", {}) or {}
    for k in ["topic", "psychotherapy", "stage"]:
        if _normalize_choice(a_s.get(k)) and _normalize_choice(a_s.get(k)) == _normalize_choice(b_s.get(k)):
            score += 2

    a_turns = {int(t.get("turn_id")): t for t in a.get("turns", []) if t.get("turn_id") is not None}
    b_turns = {int(t.get("turn_id")): t for t in b.get("turns", []) if t.get("turn_id") is not None}

    for tid, at in a_turns.items():
        bt = b_turns.get(tid)
        if not bt:
            continue
        if _normalize_choice(at.get("emotion")) == _normalize_choice(bt.get("emotion")):
            score += 1
        if _normalize_choice(at.get("therapist_strategy")) == _normalize_choice(bt.get("therapist_strategy")):
            score += 1

    return score


def _pick_best_by_agreement(candidates: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
    if len(candidates) == 1:
        return 0, candidates[0]

    scores = []
    for i, c in enumerate(candidates):
        s = 0
        for j, other in enumerate(candidates):
            if i == j:
                continue
            s += _agreement_score(c, other)
        scores.append(s)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return best_idx, candidates[best_idx]


def _validate_annotation(turns: List[Turn], obj: Dict[str, Any]) -> None:
    if "session_labels" not in obj or "turns" not in obj:
        raise AnnotationError("Missing required keys: session_labels/turns")

    ann_turns = obj.get("turns")
    if not isinstance(ann_turns, list):
        raise AnnotationError("turns must be a list")

    expected = {t.turn_id for t in turns}
    got = set()
    for t in ann_turns:
        if not isinstance(t, dict) or "turn_id" not in t:
            raise AnnotationError("each turn must be an object containing turn_id")
        got.add(int(t["turn_id"]))

    if expected != got:
        raise AnnotationError(f"turn_id mismatch: expected={sorted(expected)}, got={sorted(got)}")


@dataclass
class AnnotationCandidate:
    idx: int
    obj: Dict[str, Any]


def annotate_one_session(llm: LLMClient, turns: List[Turn], cfg: AnnotationConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dialog = format_dialog_for_prompt(turns)
    prompt = ANNOTATION_USER_TEMPLATE.format(
        dialog=dialog,
        emotion_vocab=", ".join(DEFAULT_EMOTION_VOCAB),
        strategy_vocab=", ".join(DEFAULT_STRATEGY_VOCAB),
        paralinguistic_tags=", ".join(PARALINGUISTIC_TAGS),
    )

    candidates: List[Dict[str, Any]] = []
    errors: List[str] = []

    for _ in range(max(cfg.self_consistency_votes, 1)):
        try:
            resp = llm.complete(ANNOTATION_SYSTEM + "\n\n" + prompt)
            obj = extract_json_object(resp)
            _validate_annotation(turns, obj)
            candidates.append(obj)
        except (AnnotationError, JsonExtractError, ValueError) as e:
            errors.append(str(e))
            continue

    if not candidates:
        raise AnnotationError(f"LLM failed to produce a valid annotation. errors={errors[:3]}")

    best_idx, best = _pick_best_by_agreement(candidates)

    meta = {
        "self_consistency_votes": cfg.self_consistency_votes,
        "valid_candidates": len(candidates),
        "invalid_candidates": len(errors),
        "selected_index": best_idx,
    }
    return best, meta


def _normalize_to_session_annotations(session_id: str, turns: List[Turn], ann: Dict[str, Any], meta: Dict[str, Any]) -> SessionAnnotations:
    ann_turns_map = {int(t["turn_id"]): t for t in ann.get("turns", [])}

    turns_out = []
    for t in turns:
        a = ann_turns_map[t.turn_id]
        turns_out.append(
            {
                "turn": t.to_json(),
                "annotation": {
                    "text_with_paralinguistic": a.get("text_with_paralinguistic", t.text),
                    "labels": {
                        "emotion": a.get("emotion"),
                        "therapist_strategy": a.get("therapist_strategy"),
                    },
                },
            }
        )

    sess = SessionAnnotations(
        session_id=session_id,
        language=(turns[0].language if turns else "zh"),
        session_labels=ann.get("session_labels", {}),
        turns=turns_out,
    )

    payload = sess.model_dump()
    payload["_meta"] = meta
    return SessionAnnotations.model_validate(payload)


def run_annotate_text(input_turns_jsonl: Path, output_dir: Path, llm: LLMClient, cfg: AnnotationConfig) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions = load_turns_grouped(input_turns_jsonl)

    out_path = output_dir / "annotations" / "sessions.jsonl"
    rows = []

    for session_id, turns in tqdm(sessions.items(), desc="annotate"):
        ann, meta = annotate_one_session(llm, turns, cfg)
        sess = _normalize_to_session_annotations(session_id, turns, ann, meta)
        rows.append(sess.model_dump())

    write_jsonl(out_path, rows)
    return out_path
