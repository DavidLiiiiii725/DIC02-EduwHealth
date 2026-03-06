# agents/state.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  Extended LangGraph State
# ─────────────────────────────────────────────────────────────────
from typing import Any, Dict, List, NotRequired, TypedDict


class TutorState(TypedDict):
    # ── Core input ────────────────────────────────────────────────
    user_input:  str
    learner_id:  NotRequired[str]       # injected by main.py; defaults to "default"
    turn_number: NotRequired[int]

    # ── RAG ───────────────────────────────────────────────────────
    rag_context:    NotRequired[str]
    rag_evidence:   NotRequired[Dict[str, Any]]
    rag_semantic:   NotRequired[List[str]]
    rag_structured: NotRequired[List[str]]

    # ── Affect (original) ─────────────────────────────────────────
    emotion: NotRequired[Dict[str, float]]

    # ── Cognitive State Machine ───────────────────────────────────
    cognitive_state:        NotRequired[Dict[str, float]]   # current 4-dim vector
    intervention_flags:     NotRequired[Dict[str, bool]]    # current-state flags
    trajectory_flags:       NotRequired[Dict[str, bool]]    # proactive trend flags
    cognitive_signals:      NotRequired[Dict[str, float]]   # raw signals from extractor

    # ── Learner Profile (snapshot passed into agents) ─────────────
    ld_profile:             NotRequired[Dict[str, Any]]
    scaffold_density:       NotRequired[str]    # "high" | "medium" | "low"
    successful_strategies:  NotRequired[List[str]]

    # ── Agent responses ───────────────────────────────────────────
    tutor_response:        NotRequired[str]
    coach_response:        NotRequired[str]
    critic_response:       NotRequired[str]
    ld_specialist_response: NotRequired[str]
    metacog_response:      NotRequired[str]

    # ── Risk ──────────────────────────────────────────────────────
    risk_score:   NotRequired[float]
    risk_level:   NotRequired[str]
    risk_reasons: NotRequired[Dict[str, float]]

    # ── Output ────────────────────────────────────────────────────
    final_response:  NotRequired[str]
    active_agent:    NotRequired[str]    # which agent generated the final response
    escalation:      NotRequired[str]
