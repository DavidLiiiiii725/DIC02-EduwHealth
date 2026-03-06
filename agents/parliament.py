# agents/parliament.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  Parliament Node (Orchestrator Selector)
#
# Replaces the old naive concat with a priority-based agent selector.
# Each agent gets an activation_score = activation_probability × priority_weight.
# The highest-scoring agent's response becomes final_response.
# Ties resolved by static priority from config.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from config import AGENT_PRIORITY, AGENT_TIE_THRESHOLD


def parliament_node(state: Dict[str, Any]) -> Dict[str, Any]:
    int_flags  = state.get("intervention_flags",  {})
    traj_flags = state.get("trajectory_flags",    {})
    cog_state  = state.get("cognitive_state",     {})
    ld_profile = state.get("ld_profile",          {})

    # ── Compute activation probability per agent ──────────────────
    scores: Dict[str, float] = {}

    # Tutor: always available, score based on inverse WM load
    wm = cog_state.get("working_memory_load", 0.30)
    scores["tutor"] = AGENT_PRIORITY["tutor"] * (1.0 - 0.5 * wm)

    # Coach: higher score when motivation is low or declining
    motivation = cog_state.get("motivation_level", 0.70)
    mot_prob = _sigmoid_activation(motivation, threshold=0.40, invert=True)
    if traj_flags.get("motivation_declining"):
        mot_prob = min(1.0, mot_prob + 0.30)   # proactive boost
    if int_flags.get("affect_negative"):
        mot_prob = min(1.0, mot_prob + 0.20)
    scores["coach"] = AGENT_PRIORITY["coach"] * mot_prob

    # LD Specialist: activates when LD profile is present AND state warrants it
    all_ld = set(
        ld_profile.get("confirmed", []) +
        ld_profile.get("suspected", [])
    )
    ld_response = state.get("ld_specialist_response", "")
    if all_ld and ld_response:
        ld_trigger = (
            int_flags.get("wm_overload") or
            int_flags.get("motivation_low") or
            _is_writing_task(state.get("user_input", ""))
        )
        ld_prob = 0.85 if ld_trigger else 0.40
        scores["ld_specialist"] = AGENT_PRIORITY["ld_specialist"] * ld_prob
    else:
        scores["ld_specialist"] = 0.0

    # Meta-Cognition: only fires in calm moments (gate checked before node runs)
    metacog_response = state.get("metacog_response", "")
    scores["meta_cognition"] = (
        AGENT_PRIORITY["meta_cognition"] * 0.80 if metacog_response else 0.0
    )

    # Critic: always runs safety check in the background but rarely wins primary slot
    # (Critic output is appended as a safety note, not the main response)
    scores["critic"] = AGENT_PRIORITY["critic"] * 0.30

    # ── Select winner ─────────────────────────────────────────────
    ranked: List[Tuple[str, float]] = sorted(
        scores.items(), key=lambda x: x[1], reverse=True
    )
    winner_name, winner_score = ranked[0][0], ranked[0][1]
    runner_name, runner_score = ranked[1][0], ranked[1][1]

    # Tie-break: if scores are too close, static priority wins (already encoded in AGENT_PRIORITY)
    if abs(winner_score - runner_score) < AGENT_TIE_THRESHOLD:
        # Both are essentially tied; choose the one with higher base priority
        winner_name = (
            winner_name
            if AGENT_PRIORITY.get(winner_name, 0) >= AGENT_PRIORITY.get(runner_name, 0)
            else runner_name
        )

    # ── Assemble final response ───────────────────────────────────
    response_map = {
        "tutor":          state.get("tutor_response",          ""),
        "coach":          state.get("coach_response",          ""),
        "ld_specialist":  state.get("ld_specialist_response",  ""),
        "meta_cognition": state.get("metacog_response",        ""),
        "critic":         state.get("critic_response",         ""),
    }

    primary = response_map.get(winner_name, "")

    # Always append safety note if Critic flagged something urgent
    critic_text = state.get("critic_response", "")
    safety_suffix = ""
    if critic_text and _critic_flagged(critic_text):
        safety_suffix = f"\n\n---\n⚠️ Safety note: {critic_text}"

    final = (primary + safety_suffix).strip()

    return {
        "final_response": final,
        "active_agent":   winner_name,
        "_agent_scores":  scores,    # debug metadata
    }


# ── Helpers ───────────────────────────────────────────────────────

def _sigmoid_activation(value: float, threshold: float, invert: bool = False) -> float:
    """Maps a value to activation probability around a threshold."""
    import math
    k = 10.0   # steepness
    x = threshold - value if invert else value - threshold
    return 1.0 / (1.0 + math.exp(-k * x))


def _is_writing_task(text: str) -> bool:
    keywords = ["write", "essay", "paragraph", "draft", "writing", "composition",
                "文章", "写作", "段落", "论文"]
    tl = text.lower()
    return any(k in tl for k in keywords)


def _critic_flagged(critic_text: str) -> bool:
    keywords = ["risk", "self-harm", "crisis", "escalate", "flag", "concern"]
    tl = critic_text.lower()
    return any(k in tl for k in keywords)
