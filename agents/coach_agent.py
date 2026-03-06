# agents/coach_agent.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  Coach Agent (Extended)
#
# Added:
#   - Proactive activation mode (before motivational collapse)
#   - Expectancy repair framing (Steel 2007)
#   - Attribution-aware language (Bandura 1993)
#   - Adapts tone based on cognitive state flags
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any, Dict


def coach_agent(state: Dict[str, Any], llm) -> Dict[str, Any]:
    rag              = state.get("rag_context", "")
    cog_state        = state.get("cognitive_state", {})
    int_flags        = state.get("intervention_flags", {})
    traj_flags       = state.get("trajectory_flags", {})
    ld_profile       = state.get("ld_profile", {})
    good_strategies  = state.get("successful_strategies", [])

    motivation = cog_state.get("motivation_level", 0.70)
    affect     = cog_state.get("affect_valence",  0.10)

    # ── Select coaching mode ──────────────────────────────────────

    # Crisis: affect very negative
    if int_flags.get("affect_negative"):
        mode = "affective_support"
    # Proactive: motivation declining trend (before it hits the floor)
    elif traj_flags.get("motivation_declining") and motivation > 0.30:
        mode = "proactive_encouragement"
    # Active low motivation
    elif int_flags.get("motivation_low"):
        mode = "expectancy_repair"
    # Default motivational support
    else:
        mode = "standard_sdt"

    # ── Build system prompt based on mode ────────────────────────

    strategy_hint = ""
    if good_strategies:
        strategy_hint = (
            f"\nNote: This learner has responded well to: {', '.join(good_strategies[:3])}. "
            "Use a similar approach if relevant."
        )

    if mode == "affective_support":
        system = (
            "You are an empathetic coach. The student is in distress right now. "
            "Do NOT push learning content. Acknowledge their feelings first. "
            "Be warm, brief, and human. Ask one gentle open question."
            + strategy_hint
        )
    elif mode == "proactive_encouragement":
        system = (
            "You are a motivational coach. The student's engagement is starting to drop — "
            "catch it early. Give a brief, energising message that: "
            "(1) notices their effort so far, "
            "(2) reframes the remaining work as small and doable, "
            "(3) reminds them of one thing they already did well today."
            + strategy_hint
        )
    elif mode == "expectancy_repair":
        system = (
            "You are a motivational coach. The student's motivation is low. "
            "Apply expectancy repair (Steel, 2007): "
            "(1) give a micro-task they can complete in under 3 minutes, "
            "(2) predict success explicitly ('I think you can do this'), "
            "(3) use process praise not ability praise ('you worked hard', not 'you're smart'). "
            "Do not give the full task. Give only the micro-task."
            + strategy_hint
        )
    else:  # standard SDT
        system = (
            "You are an empathetic motivational coach grounded in Self-Determination Theory. "
            "Support the student's autonomy (give choices), competence (calibrate difficulty), "
            "and relatedness (warm tone). Avoid medical claims."
            + strategy_hint
        )

    user_prompt = f"""
{"Retrieved context:\n" + rag if rag else ""}

Student message:
{state["user_input"]}
""".strip()

    response = llm.chat(system=system, user=user_prompt, temperature=0.7)

    return {
        "coach_response": response,
        "_coach_mode": mode,
    }
