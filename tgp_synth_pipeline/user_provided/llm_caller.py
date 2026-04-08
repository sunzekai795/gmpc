"""User-owned LLM caller.

You must implement `generate(prompt: str) -> str`.
The pipeline will call it repeatedly for:
- text annotation (expects JSON output)
- optional emotion reasoning (short text output)
- optional Video-RAG query generation (short query)

Keep this file in your own fork/workspace; do not commit secrets.
"""

from __future__ import annotations


def generate(prompt: str) -> str:  # pragma: no cover
    raise NotImplementedError(
        "Implement your LLM caller here. The function must return a string response."
    )
