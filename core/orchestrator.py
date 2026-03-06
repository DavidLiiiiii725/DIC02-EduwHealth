# core/orchestrator.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  TutorOrchestrator (Extended)
#
# Changes vs 1.0:
#   - Accepts learner_id for persistent profile tracking
#   - Passes learner_id into LangGraph state
#   - Post-session: updates LearnerProfile baseline + scaffold fade
#   - Returns richer output including active_agent + cognitive_state
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from agents.graph import build_graph
from safety.escalation import HumanEscalation
from memory.vector_store import VectorStore
from memory.knowledge_graph import KnowledgeGraph
from memory.hybrid_memory import HybridMemory
from memory.learner_model import LearnerModelStore

from config import RISK_THRESHOLD, KB_STORE_DIR


class TutorOrchestrator:
    def __init__(
        self,
        kb_store_dir: str = KB_STORE_DIR,
        learner_id:   str = "default",
    ):
        # ── Load Vector KB ────────────────────────────────────────
        kb_store   = Path(kb_store_dir)
        index_path = kb_store / "vector.index"
        texts_path = kb_store / "vector_texts.jsonl"

        if not index_path.exists() or not texts_path.exists():
            raise RuntimeError(
                "Vector KB not built yet.\n"
                f"Expected: {index_path}  and  {texts_path}\n"
                "Run build_vector_kb.py first."
            )

        vs     = VectorStore.load(str(kb_store))
        kg     = KnowledgeGraph()
        memory = HybridMemory(kg, vs)

        # ── Build LangGraph ───────────────────────────────────────
        self.app          = build_graph(memory)
        self.hem          = HumanEscalation()
        self.learner_id   = learner_id
        self.learner_store = LearnerModelStore()

        # Session stats for post-session profile update
        self._session_turns    = 0
        self._session_successes = 0

    # ── Public API ────────────────────────────────────────────────

    def handle(self, user_input: str) -> Dict[str, Any]:
        """Process one user turn. Returns full response dict."""
        self._session_turns += 1

        initial_state = {
            "user_input":  user_input,
            "learner_id":  self.learner_id,
        }

        state = self.app.invoke(initial_state)

        risk       = state.get("risk_score", 0.0)
        escalation = self.hem.check(risk) if risk > RISK_THRESHOLD else "OK"

        # Rough success signal: if risk is low and no avoidance, count as success
        if risk < 0.3 and state.get("cognitive_state", {}).get("motivation_level", 0.5) > 0.4:
            self._session_successes += 1

        return {
            # Primary output
            "response":       state.get("final_response",  ""),
            "active_agent":   state.get("active_agent",    "unknown"),
            # State snapshots
            "cognitive_state":    state.get("cognitive_state",    {}),
            "intervention_flags": state.get("intervention_flags", {}),
            "trajectory_flags":   state.get("trajectory_flags",   {}),
            "emotion":            state.get("emotion",            {}),
            # Risk
            "risk":        risk,
            "risk_level":  state.get("risk_level",  "low"),
            "escalation":  escalation,
            # Debug
            "rag_context": state.get("rag_context", ""),
            "agent_scores": state.get("_agent_scores", {}),
        }

    def end_session(self, session_attention_min: float = 15.0) -> None:
        """
        Call at end of session to update persistent learner profile.
        Pass actual session duration in minutes.
        """
        if self._session_turns == 0:
            return

        success_rate = self._session_successes / self._session_turns

        profile = self.learner_store.load(self.learner_id)
        profile.update_baseline_after_session(
            success_rate=success_rate,
            avg_response_latency_sec=profile.baseline.task_initiation_latency_sec,
            session_attention_min=session_attention_min,
        )
        profile.update_fade_index(success_rate)
        profile.total_turns += self._session_turns
        self.learner_store.save(profile)

        # Reset for next session
        self._session_turns     = 0
        self._session_successes = 0

    def set_learner_ld_profile(
        self,
        confirmed: list = None,
        suspected: list = None,
        severity:  dict = None,
    ) -> None:
        """
        Convenience method: update the learner's LD profile manually
        (e.g., from an intake form or clinician input).
        """
        profile = self.learner_store.load(self.learner_id)
        if confirmed:
            profile.ld_profile.confirmed = confirmed
        if suspected:
            profile.ld_profile.suspected = suspected
        if severity:
            profile.ld_profile.severity.update(severity)
        self.learner_store.save(profile)
