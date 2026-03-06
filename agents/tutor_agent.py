# agents/tutor_agent.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  Tutor Agent (Extended)
#
# Added:
#   - WM-load-aware density control
#   - ADHD attention anchor injection
#   - Fatigue-aware brevity mode
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any, Dict


def tutor_agent(state: Dict[str, Any], llm) -> Dict[str, Any]:
    rag       = state.get("rag_context", "")
    int_flags = state.get("intervention_flags", {})
    ld_profile = state.get("ld_profile", {})

    all_ld = set(
        ld_profile.get("confirmed", []) +
        ld_profile.get("suspected", [])
    )

    # ── Build adaptive instructions based on cognitive state ──────
    adaptive_instructions = []

    if int_flags.get("wm_overload"):
        adaptive_instructions.append(
            "IMPORTANT: Working memory is overloaded. "
            "Give MAXIMUM 3 bullet points. No nested lists. "
            "Bold the single most important term. One concept only."
        )
    if int_flags.get("fatigue_high"):
        adaptive_instructions.append(
            "IMPORTANT: The student is fatigued. "
            "Be extremely brief (under 80 words). "
            "End with: 'Want to take a 5-minute break before continuing?'"
        )
    if "adhd" in all_ld:
        adaptive_instructions.append(
            "Use ⚡ before any key point so the student knows where to focus. "
            "Use short sentences. Add a transition word before every new idea "
            "(First / Then / Finally / Most importantly)."
        )
    if int_flags.get("affect_negative"):
        adaptive_instructions.append(
            "The student is upset. Acknowledge this with one short empathetic sentence "
            "BEFORE giving any content."
        )

    adaptive_block = "\n".join(adaptive_instructions)
    if adaptive_block:
        adaptive_block = "\n[ADAPTIVE INSTRUCTIONS]\n" + adaptive_block + "\n"

    user_prompt = f"""
You MUST use the following retrieved knowledge as your primary grounding.
If the knowledge is insufficient, say what is missing and ask one clarifying question.
{adaptive_block}
{rag}

Student question:
{state["user_input"]}
""".strip()

    response = llm.chat(
        system="You are an academic tutor. Be precise, structured, and grounded in retrieved context.",
        user=user_prompt,
        temperature=0.4,
    )

    return {"tutor_response": response}
