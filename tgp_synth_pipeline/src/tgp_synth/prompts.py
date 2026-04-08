from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .schema import Turn


PARALINGUISTIC_TAGS = [
    "<laugh>",
    "<chuckle>",
    "<sigh>",
    "<cough>",
    "<sniffle>",
    "<groan>",
    "<yawn>",
    "<gasp>",
]


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


def format_dialog_for_prompt(turns: List[Turn]) -> str:
    lines: List[str] = []
    for t in turns:
        speaker = "Client" if t.speaker == "client" else "Therapist" if t.speaker == "therapist" else t.speaker
        lines.append(f"[{t.turn_id}][{speaker}] {t.text}")
    return "\n".join(lines)


ANNOTATION_SYSTEM_ZH = "你是一个严谨的心理咨询数据标注助手。你必须严格按要求输出 JSON，不能输出任何多余文字。"


ANNOTATION_USER_TEMPLATE_ZH = """给定一段多轮心理咨询对话（可能是中文或英文），请你为该对话生成结构化标注，并在每句可朗读文本中插入旁语言标签（paralinguistic tags），以便后续 TTS 合成时更有情绪表达。

需要生成的 8 类标注：
1) Background：对话背景/来访者身份与求助原因（session级）
2) Topic：本次咨询的主要心理主题（session级）
3) Psychotherapy：咨询师采用的心理治疗方法（session级，如 CBT/SFBT/精神分析等；不确定可填 "Unknown"）
4) Stage：咨询进程阶段（session级，如 建立关系/问题澄清/目标设定/干预/总结 等；不确定可填 "Unknown"）
5) Guide：对咨询师说话的指导原则（session级，短句即可；不确定可填 "Unknown"）
6) Reasoning：简短解释：为什么你给出了以上 session 级判断（session级）
7) ClientEmotion：逐 turn 标注说话者当下情绪（turn级；优先使用 {emotion_vocab} 中的一个；不确定用 "Neutral"）
8) TherapistStrategy：仅对 Therapist turn 标注该句主要策略（turn级；优先使用 {strategy_vocab} 中的一个；不确定用 "Unknown"；对非 Therapist turn 填 null）

旁语言标签要求：
- 仅使用以下标签集合：{paralinguistic_tags}
- 标签应自然地插入到文本内部，数量适度（通常 0-2 个）；必须与情绪一致。
- 不要改变原句语义，不要引入新事实。

输出格式（必须是严格 JSON，可被 json.loads 解析）：
{{
  "session_labels": {{
    "background": "...",
    "topic": "...",
    "psychotherapy": "...",
    "stage": "...",
    "guide": "...",
    "reasoning": "..."
  }},
  "turns": [
    {{
      "turn_id": 0,
      "speaker": "client|therapist|other",
      "emotion": "...",
      "therapist_strategy": "... or null",
      "text_with_paralinguistic": "..."
    }}
  ]
}}

对话如下：
{dialog}
"""


EMOTION_REASONING_TTS_TEMPLATE = """你是一个情绪推理代理。给定心理咨询对话的标签与当前一句话内容，你需要给出一段“给 TTS 模型的合成指导”，帮助它更好表达情绪。

要求：
- 用中文输出一段简短指导（1-3 句）
- 重点关注：情绪强度、语速、停顿、气息、语调
- 不要编造未出现的事实

输入：
- speaker: {speaker}
- emotion: {emotion}
- topic: {topic}
- text: {text}

输出：
"""


VIDEO_RAG_QUERY_TEMPLATE = """你是一个视频检索提示词生成器。我们有一批心理咨询场景的对话视频片段，视频里包含自然的身体语言和表情。

请基于输入信息，生成一个用于检索的短 query（中文或英文均可），用于找到“身体语言/情感表达/语境最匹配”的参考视频片段。

要求：
- query 长度控制在 15-40 个词/字
- 需要体现：说话者身份、情绪、咨询主题/语境、希望的身体语言风格（如紧张、叹气、安抚、点头、停顿等）
- 只输出 query 本身，不要输出解释

输入：
- speaker: {speaker}
- emotion: {emotion}
- topic: {topic}
- text: {text}

输出：
"""


@dataclass
class PromptBuilder:
    emotion_vocab: List[str] = None  # type: ignore[assignment]
    strategy_vocab: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.emotion_vocab is None:
            self.emotion_vocab = list(DEFAULT_EMOTION_VOCAB)
        if self.strategy_vocab is None:
            self.strategy_vocab = list(DEFAULT_STRATEGY_VOCAB)

    def build_annotation_prompt(self, turns: List[Turn]) -> str:
        return ANNOTATION_USER_TEMPLATE_ZH.format(
            dialog=format_dialog_for_prompt(turns),
            emotion_vocab=", ".join(self.emotion_vocab),
            strategy_vocab=", ".join(self.strategy_vocab),
            paralinguistic_tags=", ".join(PARALINGUISTIC_TAGS),
        )


ANNOTATION_SYSTEM = ANNOTATION_SYSTEM_ZH
ANNOTATION_USER_TEMPLATE = ANNOTATION_USER_TEMPLATE_ZH
