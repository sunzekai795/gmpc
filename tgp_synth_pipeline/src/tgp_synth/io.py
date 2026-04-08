from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from .schema import Turn


class JsonlIOError(RuntimeError):
    pass


def read_jsonl(path: Path) -> Iterator[dict]:
    """Stream jsonl rows."""

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise JsonlIOError(f"Invalid JSON at {path}:{line_no}: {e}") from e


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write rows to jsonl (overwrite)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class SessionTurns:
    session_id: str
    turns: List[Turn]


def load_turns_grouped(path: Path, strict_order: bool = True) -> Dict[str, List[Turn]]:
    """Load and group turns by session_id.

    If strict_order is True, ensures turn_id is unique within a session.
    """

    sessions: Dict[str, List[Turn]] = defaultdict(list)
    for obj in read_jsonl(path):
        turn = Turn.from_json(obj)
        sessions[turn.session_id].append(turn)

    for sid in list(sessions.keys()):
        turns = sessions[sid]
        turns.sort(key=lambda t: t.turn_id)
        if strict_order:
            ids = [t.turn_id for t in turns]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate turn_id in session {sid}: {ids}")

    return dict(sessions)


def iter_sessions(path: Path) -> Iterator[SessionTurns]:
    for sid, turns in load_turns_grouped(path).items():
        yield SessionTurns(session_id=sid, turns=turns)
