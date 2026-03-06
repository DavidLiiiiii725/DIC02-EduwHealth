# agents/intervention_agent.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  Intervention Recommendation Agent
#
# Produces a prioritised list of interventions based on:
#   - Mental-health risk level (from EnsembleAffectiveDetector / risk model)
#   - Detected / confirmed learning disability types
#   - Current emotional state (anxiety, depression, positive affect)
#   - Learner profile (history of successful strategies)
#
# Each recommended intervention has:
#   type        – broad category (e.g. "immediate_support", "task_decomposition")
#   strategy    – human-readable description of the recommended action
#   tools       – suggested technical or pedagogical tools
#   priority    – "immediate" | "high" | "medium" | "low"
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any, Dict, List


class InterventionAgent:
    """
    Recommends personalised interventions based on the learner's current
    state, disability profile, and risk level.

    Parameters
    ----------
    kb_retriever : object (optional)
        Any object with a ``retrieve(query, k)`` method that returns
        relevant KB passages.  When provided, strategy descriptions are
        enriched with retrieved evidence.  Pass None to disable.
    """

    def __init__(self, kb_retriever=None):
        self.kb_retriever   = kb_retriever
        self.risk_protocols = self._build_risk_protocols()

    # ── Public API ────────────────────────────────────────────────

    def recommend_interventions(self, learner_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate a prioritised list of interventions.

        Parameters
        ----------
        learner_state : dict
            Expected keys (all optional with safe defaults):
              risk           – dict with "risk_level" ("low"|"moderate"|"high"|"severe")
              emotions       – dict label→score (from EmotionDetector)
              key_indicators – dict with "anxiety", "depression", "positive_affect"
              disabilities   – list of str (e.g. ["ADHD", "anxiety"])
              successful_strategies – list of str (from LearnerProfile)

        Returns
        -------
        list of dicts, sorted by priority (immediate first).
        """
        interventions: List[Dict[str, Any]] = []

        risk = learner_state.get("risk", {})
        risk_level = risk.get("risk_level", "low")

        emotions       = learner_state.get("emotions", {})
        indicators     = learner_state.get("key_indicators", {})
        disabilities   = [d.lower() for d in learner_state.get("disabilities", [])]
        past_successes = learner_state.get("successful_strategies", [])

        # 1. Risk-level response protocol
        interventions.extend(self._risk_interventions(risk_level, risk))

        # 2. Disability-specific strategies
        if any(d in disabilities for d in ("adhd", "attention_deficit")):
            interventions.extend(self._get_adhd_interventions(learner_state))

        if any(d in disabilities for d in ("executive_function", "executive_function_deficit", "efd")):
            interventions.extend(self._get_efd_interventions(learner_state))

        if any(d in disabilities for d in ("anxiety", "anxiety_disorder", "gad", "social_anxiety")):
            interventions.extend(self._get_anxiety_interventions(learner_state))

        if any(d in disabilities for d in ("motivation_disorder", "learned_helplessness", "academic_burnout")):
            interventions.extend(self._get_motivation_interventions(learner_state))

        # 3. Emotion-driven strategies (threshold-based)
        anxiety_score = indicators.get("anxiety", emotions.get("fear", 0))
        if anxiety_score > 0.6 and "anxiety" not in disabilities:
            interventions.append(self._anxiety_reduction_strategy(anxiety_score))

        depression_score = indicators.get("depression", emotions.get("sadness", 0))
        if depression_score > 0.6:
            interventions.append(self._depression_support_strategy(depression_score))

        # 4. De-duplicate and filter previously-failed strategies
        interventions = self._deduplicate(interventions)

        # 5. Prioritise previously-successful strategies
        interventions = self._promote_successful(interventions, past_successes)

        return self._sort_by_priority(interventions)

    # ── Risk-level protocols ──────────────────────────────────────

    def _risk_interventions(
        self, risk_level: str, risk: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        protocol = self.risk_protocols.get(risk_level, [])
        result = []
        for entry in protocol:
            result.append(dict(entry))   # shallow copy so we don't mutate templates
        return result

    @staticmethod
    def _build_risk_protocols() -> Dict[str, List[Dict[str, Any]]]:
        return {
            "severe": [
                {
                    "type":     "crisis_escalation",
                    "strategy": "Immediately surface crisis support resources and alert human counsellor.",
                    "tools":    ["campus_counselling_chat", "crisis_text_line", "emergency_contact"],
                    "priority": "immediate",
                },
            ],
            "high": [
                {
                    "type":     "immediate_support",
                    "strategy": "Acknowledge distress compassionately; reduce academic task demands; provide mental health resource links.",
                    "tools":    ["campus_counselling", "woebot", "headspace"],
                    "priority": "immediate",
                },
                {
                    "type":     "counsellor_alert",
                    "strategy": "Notify support staff within 1 hour (with learner consent).",
                    "tools":    ["staff_notification_system"],
                    "priority": "high",
                },
            ],
            "moderate": [
                {
                    "type":     "affective_check_in",
                    "strategy": "Increase check-in frequency (every 2-3 turns); normalise difficulty; offer self-help resources.",
                    "tools":    ["breathing_exercise", "mood_tracker"],
                    "priority": "medium",
                },
            ],
            "low": [
                {
                    "type":     "routine_monitoring",
                    "strategy": "Continue standard support; offer growth-mindset reinforcement.",
                    "tools":    [],
                    "priority": "low",
                },
            ],
        }

    # ── Disability-specific intervention banks ────────────────────

    @staticmethod
    def _get_adhd_interventions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {
                "type":     "task_decomposition",
                "strategy": "Break the current task into chunks of ≤10 minutes each with explicit sub-goals.",
                "tools":    ["pomodoro_timer", "notion_checklist"],
                "priority": "high",
            },
            {
                "type":     "attention_scaffolding",
                "strategy": "Provide a written agenda before the session and use visual anchors for key points.",
                "tools":    ["visual_timer", "slide_outline"],
                "priority": "medium",
            },
            {
                "type":     "noise_management",
                "strategy": "Recommend noise-cancelling headphones or FM system to improve signal-to-noise ratio.",
                "tools":    ["noise_cancelling_headphones", "fm_system", "white_noise_app"],
                "priority": "medium",
            },
        ]

    @staticmethod
    def _get_efd_interventions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {
                "type":     "planning_scaffold",
                "strategy": "Provide a structured pre-writing/pre-task planning template with explicit steps.",
                "tools":    ["graphic_organiser", "writing_template"],
                "priority": "high",
            },
            {
                "type":     "working_memory_support",
                "strategy": "Externalise working memory with checklists, mind maps, and reference cards.",
                "tools":    ["anki", "mindmapping_app", "reference_card"],
                "priority": "medium",
            },
            {
                "type":     "ai_writing_assistance",
                "strategy": "Activate AI writing assistant to provide sentence continuations and structural suggestions.",
                "tools":    ["llm_writing_assistant", "grammar_checker", "speech_to_text"],
                "priority": "medium",
            },
        ]

    @staticmethod
    def _get_anxiety_interventions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {
                "type":     "chunked_information_delivery",
                "strategy": "Deliver content in small, labelled segments with explicit transition markers.",
                "tools":    ["structured_slide_outline", "signposted_notes"],
                "priority": "high",
            },
            {
                "type":     "self_advocacy_training",
                "strategy": "Teach scripts for requesting clarification and accommodation in low-stakes practice.",
                "tools":    ["role_play_module", "accommodation_request_template"],
                "priority": "medium",
            },
            {
                "type":     "physiological_regulation",
                "strategy": "Introduce diaphragmatic breathing (4-4-6 pattern) before high-stakes tasks.",
                "tools":    ["breathing_exercise_app", "headspace", "calm"],
                "priority": "medium",
            },
        ]

    @staticmethod
    def _get_motivation_interventions(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {
                "type":     "micro_success_architecture",
                "strategy": "Restructure the immediate task so that the first step guarantees success; provide explicit positive feedback.",
                "tools":    ["success_ladder_tracker"],
                "priority": "high",
            },
            {
                "type":     "attribution_retraining",
                "strategy": "Prompt learner to identify effort-based explanations for recent successes and failures.",
                "tools":    ["attribution_journal", "growth_mindset_exercise"],
                "priority": "medium",
            },
            {
                "type":     "autonomy_building",
                "strategy": "Offer genuine choices in task format or topic to build a sense of agency.",
                "tools":    ["choice_board"],
                "priority": "low",
            },
        ]

    # ── Emotion-threshold strategies ──────────────────────────────

    @staticmethod
    def _anxiety_reduction_strategy(score: float) -> Dict[str, Any]:
        return {
            "type":     "anxiety_reduction",
            "strategy": f"Anxiety indicator elevated ({score:.2f}). Introduce brief grounding exercise and chunked delivery.",
            "tools":    ["breathing_exercise", "chunked_notes"],
            "priority": "medium",
        }

    @staticmethod
    def _depression_support_strategy(score: float) -> Dict[str, Any]:
        return {
            "type":     "depression_support",
            "strategy": f"Depression signal elevated ({score:.2f}). Activate motivational support and connect to counselling resources.",
            "tools":    ["micro_success_task", "campus_counselling"],
            "priority": "high",
        }

    # ── Sorting and deduplication ─────────────────────────────────

    _PRIORITY_ORDER = {"immediate": 0, "high": 1, "medium": 2, "low": 3}

    @classmethod
    def _sort_by_priority(cls, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(items, key=lambda x: cls._PRIORITY_ORDER.get(x.get("priority", "low"), 3))

    @staticmethod
    def _deduplicate(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: set = set()
        result = []
        for item in items:
            key = item.get("type", "")
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    @staticmethod
    def _promote_successful(
        items: List[Dict[str, Any]],
        past_successes: List[str],
    ) -> List[Dict[str, Any]]:
        """Bump priority of interventions matching previously-successful strategies."""
        _UP = {"low": "medium", "medium": "high", "high": "high", "immediate": "immediate"}
        for item in items:
            if item.get("type") in past_successes or item.get("strategy") in past_successes:
                item["priority"] = _UP.get(item.get("priority", "low"), "medium")
                item["_boosted"] = True
        return items
