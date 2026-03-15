from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def generate_adhd_speaking_pack(
    *,
    llm,
    learner_name: str,
    english_level: str,
    topic: str,
    scenario: str,
    minutes: int,
    ld_profile: Optional[Dict[str, Any]] = None,
) -> str:
    """
    生成 ADHD 友好的口语训练包（输出为英文，结构强、短句、低负担）。
    返回 Markdown 文本，前端直接 linebreaksbr 渲染。
    """
    minutes = max(3, min(30, _safe_int(minutes, 8)))
    topic = (topic or "").strip() or "Daily life"
    scenario = (scenario or "").strip() or "School / Work"
    english_level = (english_level or "").strip() or "A2-B1"

    ld_confirmed: List[str] = []
    if isinstance(ld_profile, dict):
        ld_confirmed = list(ld_profile.get("confirmed") or [])

    system = (
        "你是英语口语教练，专门为 ADHD 学习者设计训练内容。\n"
        "要求：\n"
        "- 输出必须是英文。\n"
        "- 结构清晰、分段短、每段句子尽量短。\n"
        "- 不要长篇解释，不要输出多余免责声明。\n"
        "- 使用 Markdown，标题用 '##' 或 '###'。\n"
        "- 内容必须包含：30秒启动、1分钟自我介绍、90秒核心话题、迷你对话、复述模板、卡住救场句。\n"
        "- 给出可替换词/可选句，但不要超过 8 条。\n"
        "- 语法词汇难度适配学习者水平。\n"
        "- 全文控制在约 400-900 英文单词。\n"
    )

    user = (
        f"Learner name: {learner_name}\n"
        f"English level: {english_level}\n"
        f"Topic: {topic}\n"
        f"Scenario: {scenario}\n"
        f"Time budget (minutes): {minutes}\n"
        f"LD confirmed: {json.dumps(ld_confirmed)}\n"
        "请生成今天的 ADHD-friendly speaking practice pack。"
    )

    return llm.chat(system=system, user=user, temperature=1.0).strip()


def speaking_coach_reply(
    *,
    llm,
    practice_pack_markdown: str,
    learner_msg: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    口语陪练聊天回复：短反馈 + 下一步提示。
    输出英文，便于直接用于口语练习。
    """
    learner_msg = (learner_msg or "").strip()
    if not learner_msg:
        return "Please type something you want to practice."

    # 仅把最近几轮带入，避免 prompt 过长
    tail = (history or [])[-6:]
    hist_text = "\n".join([f"{h.get('role','')}: {h.get('content','')}" for h in tail if h])

    system = (
        "You are a friendly English speaking coach for an ADHD learner.\n"
        "Rules:\n"
        "- Output must be in English.\n"
        "- Keep it short and actionable.\n"
        "- Use this structure:\n"
        "  1) Micro feedback (1-3 bullet points)\n"
        "  2) Better version (1 short paragraph)\n"
        "  3) Next prompt (one question)\n"
        "- Prefer simple words and short sentences.\n"
        "- Do not be harsh. Do not add medical advice.\n"
    )

    user = (
        "Practice pack (reference):\n"
        f"{practice_pack_markdown}\n\n"
        "Recent chat (optional):\n"
        f"{hist_text}\n\n"
        "Learner message:\n"
        f"{learner_msg}\n"
    )

    return llm.chat(system=system, user=user, temperature=0.9).strip()

