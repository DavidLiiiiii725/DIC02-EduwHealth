from __future__ import annotations

from typing import Any, Dict, List, Optional


def _build_daily_summary_payload(stats: Dict[str, Any]) -> str:
    """
    Render a compact, JSON-like text block summarising learner performance.

    This is kept as plain text so we can pass it directly into the LLM prompt.
    """
    lines: List[str] = []
    lines.append("performance_summary = {")
    for k, v in stats.items():
        lines.append(f"  {k!r}: {v!r},")
    lines.append("}")
    return "\n".join(lines)


def _build_cognitive_payload(cognitive: Optional[Dict[str, Any]]) -> str:
    """
    Render a small block describing current cognitive / affective state.

    This is optional – if no snapshot is provided we simply return a note
    that no real-time signals are available.
    """
    if not cognitive:
        return "current_cognitive_state = null  # no real-time signals provided"

    lines: List[str] = []
    lines.append("current_cognitive_state = {")
    for k, v in cognitive.items():
        lines.append(f"  {k!r}: {v!r},")
    lines.append("}")
    return "\n".join(lines)


def generate_study_plan(
    stats: Dict[str, Any],
    llm,
    *,
    cognitive_snapshot: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Given quantitative stats about recent learning behaviour (and
    optionally a real-time cognitive snapshot), ask the LLM to propose a
    concrete study plan for *today*.

    The returned string is markdown in **English only** so the frontend
    can render it directly.
    """
    summary_block = _build_daily_summary_payload(stats)
    cognitive_block = _build_cognitive_payload(cognitive_snapshot)

    system = (
        "You are an expert learning coach who designs daily study plans.\n"
        "All output MUST be in clear, simple English suitable for a student.\n"
        "You receive quantitative data about the learner's recent performance\n"
        "(reading time, scores, hint usage, etc.) *and* an optional snapshot\n"
        "of their current cognitive state (working memory load, motivation,\n"
        "fatigue, affect).\n"
        "You must:\n"
        "1) Infer their strengths and weaknesses.\n"
        "2) Propose a realistic plan *for today only*.\n"
        "3) Explain briefly WHY you chose this plan.\n"
        "4) Calibrate difficulty to avoid overload but still push growth,\n"
        "   taking the current cognitive state into account when provided.\n"
        "Keep the whole answer under 500 words."
    )

    user = f"""
Here are today's inputs about the learner. Keys are simple English names:

{summary_block}

Here is an optional snapshot of the learner's *current* cognitive state
(all values are normalised between 0 and 1 where applicable):

{cognitive_block}

Design a concrete daily study plan for **today**.

Constraints:
- The total study time today should be approximately `total_minutes_budget` minutes.
- Put the plan in sections with explicit time blocks, e.g. "Block 1 — 20 minutes: ...".
- Use bullet points, headings, and short sentences.
- Include a short section called **Why this plan** that explains the reasoning using the numbers.
- If there is very little data, still propose a gentle starter plan and say that data is limited.
"""
    return llm.chat(system=system, user=user, temperature=0.5)


def study_plan_chat_reply(
    stats: Dict[str, Any],
    current_plan_markdown: str,
    user_question: str,
    cognitive_snapshot: Optional[Dict[str, Any]],
    llm,
) -> str:
    """
    Follow-up chat turn: the learner asks about or wants to adjust today's plan.

    We give the LLM the original stats plus the previously generated plan.
    """
    summary_block = _build_daily_summary_payload(stats)
    cognitive_block = _build_cognitive_payload(cognitive_snapshot)

    system = (
        "You are a study-planning assistant having a short conversation\n"
        "with a learner about *today's* study plan.\n"
        "You must always reply in clear, friendly English.\n"
        "Keep answers concise (under 250 words) and practically focused.\n"
        "You may adjust the plan, but stay within the approximate time budget."
    )

    user = f"""
Here is the numeric summary of the learner:

{summary_block}

Here is an optional snapshot of their *current* cognitive state:

{cognitive_block}

Here is today's current plan (markdown):

{current_plan_markdown}

The learner now says:
\"\"\"{user_question}\"\"\"

Respond to the learner, referencing specific parts of the plan where helpful.
If you adjust the plan, clearly state the change.
"""
    return llm.chat(system=system, user=user, temperature=0.6)

