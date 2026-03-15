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
import re

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
        self.app           = build_graph(memory)
        self.hem           = HumanEscalation()
        self.learner_id    = learner_id
        self.vector_store  = vs
        self.learner_store = LearnerModelStore()

        # Session stats for post-session profile update
        self._session_turns     = 0
        self._session_successes = 0

    # ── Public API ────────────────────────────────────────────────

    def add_paragraph_to_memory(self, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a paragraph to the in-memory vector store so it can be
        retrieved by subsequent RAG calls during this process lifetime.

        This does NOT persist back to disk; it is session/process scoped.
        """
        if not text:
            return
        clean = text.strip()
        if not clean:
            return

        base_meta: Dict[str, Any] = {
            "kb_domain": "user_paragraph",
            "learner_id": self.learner_id,
        }
        if meta:
            base_meta.update(meta)

        # One-paragraph batch for now
        self.vector_store.add([clean], [base_meta])

    def _format_for_adhd_chat(self, raw: str) -> str:
        """
        Post-process the LangGraph 'final_response' into a concise,
        ADHD‑friendly English reply for the AI Chat Hub.

        - Strips safety / ethics notes.
        - Keeps the main LLM answer (trimmed if extremely long).
        - Compresses self‑reflection into a short interactive block when present.
        - Keeps a single short encouragement sentence when present.
        """
        if not raw:
            return ""

        text = raw.strip()

        # Drop any safety note section (everything from the first safety marker onward)
        safety_split = re.split(r"(?:---\s*⚠️?|\u26a0️?\s*Safety note:)", text, maxsplit=1)
        if safety_split:
            text = safety_split[0].strip()

        # Separate main explanation from explicit self-reflection section if present
        main_text = text
        self_block = ""
        m_self = re.search(r"Self-Reflection Prompt:?(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        if m_self:
            main_text = text[: m_self.start()].strip()
            self_block = m_self.group(1).strip()

        encouragement_line: Optional[str] = None

        # Try to capture an existing encouragement sentence to reuse briefly.
        m_enc = re.search(r"Encouragement Sentence:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
        if m_enc:
            raw_enc = m_enc.group(1).strip()
            # Take only the first sentence to keep it short.
            m_first = re.match(r"(.+?[.!?])(\s|$)", raw_enc)
            encouragement_line = (m_first.group(1) if m_first else raw_enc).strip()

        # If the answer is long, trim it hard so it can be read at a glance.
        max_main_chars = 400
        main_out = main_text
        if len(main_out) > max_main_chars:
            main_out = main_out[: max_main_chars].rstrip()
            # Try not to cut mid‑sentence
            last_dot = main_out.rfind(".")
            if last_dot > max_main_chars * 0.6:
                main_out = main_out[: last_dot + 1]
            main_out += "\n\n(Answer truncated to stay concise.)"

        # Compress self‑reflection into a short interactive prompt, reusing user‑specific wording if possible.
        self_lines = []
        if self_block:
            bullets = re.findall(r"\"(.+?)\"", self_block) or re.findall(r"\d+\.\s*\"?(.+)", self_block)
            learned = bullets[0].strip() if len(bullets) >= 1 else ""
            unclear = bullets[1].strip() if len(bullets) >= 2 else ""
            action = bullets[2].strip() if len(bullets) >= 3 else ""
            def _shorten(s: str, limit: int = 120) -> str:
                if len(s) <= limit:
                    return s
                cut = s[:limit].rstrip()
                last_dot = cut.rfind(".")
                if last_dot > limit * 0.5:
                    cut = cut[: last_dot + 1]
                return cut + " ..."

            self_lines.append("Quick self-reflection:")
            if learned:
                self_lines.append(f"- I learned: {_shorten(learned)}")
            if unclear:
                self_lines.append(f"- Still unclear: {_shorten(unclear)}")
            if action:
                self_lines.append(f"- Next step: {_shorten(action)}")

        parts = [main_out] if main_out else []
        if self_lines:
            parts.append("\n".join(self_lines))
        if encouragement_line:
            parts.append(f"Encouragement: {encouragement_line}")

        # Fallback: if everything was stripped somehow, at least return original text.
        final = "\n\n".join(p for p in parts if p.strip())
        return final if final else text

    def handle(self, user_input: str, *, hub_mode: bool = False) -> Dict[str, Any]:
        """Process one user turn. Returns full response dict."""
        self._session_turns += 1

        initial_state = {
            "user_input":  user_input,
            "learner_id":  self.learner_id,
            "hub_mode":    hub_mode,
        }

        state = self.app.invoke(initial_state)

        risk       = state.get("risk_score", 0.0)
        escalation = self.hem.check(risk) if risk > RISK_THRESHOLD else "OK"

        # Rough success signal: if risk is low and no avoidance, count as success
        if risk < 0.3 and state.get("cognitive_state", {}).get("motivation_level", 0.5) > 0.4:
            self._session_successes += 1

        formatted_response = self._format_for_adhd_chat(state.get("final_response", ""))

        return {
            # Primary output
            "response":       formatted_response,
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
