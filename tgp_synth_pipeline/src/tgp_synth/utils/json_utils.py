from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


class JsonExtractError(ValueError):
    pass


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def _try_load(snippet: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(snippet)
        if isinstance(obj, dict):
            return obj
        return None
    except Exception:
        return None


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from an LLM response.

    Common patterns:
    - Pure JSON
    - Markdown fenced code block
    - JSON preceded/followed by commentary
    """

    text = text.strip()

    obj = _try_load(text)
    if obj is not None:
        return obj

    for m in _FENCE_RE.finditer(text):
        inner = m.group(1).strip()
        obj = _try_load(inner)
        if obj is not None:
            return obj

    m = _JSON_OBJ_RE.search(text)
    if not m:
        raise JsonExtractError("No JSON object found in response")

    snippet = m.group(0)
    obj = _try_load(snippet)
    if obj is None:
        raise JsonExtractError("Invalid JSON extracted")
    return obj
