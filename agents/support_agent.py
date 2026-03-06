# agents/support_agent.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  Real-time Adaptive Support Agent
#
# Provides conversational support adapted to the learner's disability
# profile and current affective state:
#
#   - ADHD / EFD     → task decomposition dialogue
#   - Anxiety        → anxiety de-escalation and chunked information
#   - Motivational   → micro-success encouragement and attribution work
#   - General        → empathic acknowledgement + strategy suggestion
#
# The agent uses the LLMClient for natural-language generation while
# selecting an appropriate system prompt and structured template based
# on the learner's profile.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any, Dict, List


# ── System prompt templates ───────────────────────────────────────

_SYSTEM_ADHD = (
    "You are an educational support specialist trained in ADHD interventions. "
    "Your role is to help the learner break overwhelming tasks into tiny, manageable steps. "
    "Be warm, direct, and energetic. Use short sentences. Number each step clearly. "
    "Never present more than three steps at once. End every message with one concrete next action."
)

_SYSTEM_ANXIETY = (
    "You are a supportive educational counsellor trained in anxiety management. "
    "Your tone is calm, gentle, and reassuring. "
    "Normalise the learner's experience of difficulty. "
    "Use clear signposting ('First … Then … Finally …') to reduce uncertainty. "
    "Do not rush the learner. Offer breathing or grounding if distress is high. "
    "End every message with a simple, non-threatening next step."
)

_SYSTEM_MOTIVATION = (
    "You are a motivational coach for students experiencing academic burnout or learned helplessness. "
    "Your role is to help the learner find one tiny success they can achieve right now. "
    "Be encouraging without being hollow — celebrate effort and strategy, not just outcome. "
    "Help the learner reframe failures as strategy mismatches rather than personal flaws. "
    "End every message with the smallest possible next step."
)

_SYSTEM_GENERAL = (
    "You are a warm and knowledgeable educational support specialist. "
    "Acknowledge the learner's feelings before giving advice. "
    "Suggest one practical strategy and explain why it might help. "
    "Be concise and supportive. End with an open question to invite the learner to respond."
)

# ── User-facing prompt templates ──────────────────────────────────

_PROMPT_TASK_DECOMPOSE = """\
The learner is working on the following task and feels stuck or overwhelmed:

Task: {user_input}

Retrieved context (if any):
{rag_context}

Please help by:
1. Acknowledging that this feels like a lot.
2. Breaking the task into exactly 3 small steps (numbered).
3. Asking the learner to confirm they are ready to try step 1.
"""

_PROMPT_ANXIETY_SUPPORT = """\
The learner has expressed anxiety or distress about the following:

Situation: {user_input}

Retrieved context (if any):
{rag_context}

Please respond by:
1. Validating their feelings in one or two sentences.
2. Offering one brief grounding or breathing exercise (describe it simply).
3. Breaking the most important information into three clearly signposted parts.
4. Ending with a gentle, low-stakes question to help them re-engage.
"""

_PROMPT_MOTIVATION_SUPPORT = """\
The learner appears to be struggling with motivation or feels they cannot succeed:

Situation: {user_input}

Retrieved context (if any):
{rag_context}

Please respond by:
1. Acknowledging the difficulty without dismissing it.
2. Identifying the ONE smallest step that would count as a real success.
3. Framing any recent difficulty as a strategy issue, not a personal failing.
4. Ending with a specific, achievable invitation: "Would you like to try just [micro-task]?"
"""

_PROMPT_STRATEGY_SUGGESTION = """\
Learner message: {user_input}

Retrieved context (if any):
{rag_context}

Please:
1. Briefly acknowledge what the learner is experiencing.
2. Suggest one concrete learning strategy suited to the situation.
3. Explain clearly why this strategy might help.
4. End with an open question.
"""


class SupportAgent:
    """
    Adaptive real-time support agent.

    Parameters
    ----------
    llm : LLMClient
        The LLM client to use for natural-language generation.
    """

    def __init__(self, llm):
        self.llm = llm

    # ── Public API ────────────────────────────────────────────────

    def respond(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a support response adapted to the learner's state.

        Reads from ``state``:
          user_input, rag_context, ld_profile, intervention_flags,
          key_indicators (from EnsembleAffectiveDetector), disabilities

        Returns a dict with ``support_response`` and ``_support_mode``.
        """
        user_input = state.get("user_input", "")
        rag        = state.get("rag_context", "")

        ld_profile    = state.get("ld_profile", {})
        confirmed     = ld_profile.get("confirmed", [])
        suspected     = ld_profile.get("suspected", [])
        all_ld        = {d.lower() for d in (confirmed + suspected)}
        # also accept disabilities list injected directly
        extra_ld      = {d.lower() for d in state.get("disabilities", [])}
        all_ld       |= extra_ld

        int_flags     = state.get("intervention_flags", {})
        indicators    = state.get("key_indicators", {})
        risk_level    = state.get("risk_level", state.get("risk", {}).get("risk_level", "low"))

        system, prompt, mode = self._select_mode(
            all_ld=all_ld,
            int_flags=int_flags,
            indicators=indicators,
            risk_level=risk_level,
            user_input=user_input,
            rag=rag,
        )

        response = self.llm.chat(system=system, user=prompt, temperature=0.45)

        return {
            "support_response": response,
            "_support_mode":    mode,
        }

    # ── Mode selection ────────────────────────────────────────────

    def _select_mode(
        self,
        all_ld: set,
        int_flags: Dict[str, Any],
        indicators: Dict[str, float],
        risk_level: str,
        user_input: str,
        rag: str,
    ):
        rag_ctx = f"Retrieved context:\n{rag}" if rag else "(none)"

        # High / severe risk → anxiety-support mode regardless of LD
        if risk_level in ("high", "severe"):
            return (
                _SYSTEM_ANXIETY,
                _PROMPT_ANXIETY_SUPPORT.format(user_input=user_input, rag_context=rag_ctx),
                "crisis_support",
            )

        # ADHD / EFD with working-memory overload → task decomposition
        if (
            any(d in all_ld for d in ("adhd", "executive_function", "executive_function_deficit", "efd"))
            and (int_flags.get("wm_overload") or indicators.get("anxiety", 0) < 0.5)
        ):
            return (
                _SYSTEM_ADHD,
                _PROMPT_TASK_DECOMPOSE.format(user_input=user_input, rag_context=rag_ctx),
                "task_decomposition",
            )

        # Anxiety disorder or high anxiety indicator
        if (
            any(d in all_ld for d in ("anxiety", "anxiety_disorder", "gad", "social_anxiety"))
            or indicators.get("anxiety", 0) > 0.6
        ):
            return (
                _SYSTEM_ANXIETY,
                _PROMPT_ANXIETY_SUPPORT.format(user_input=user_input, rag_context=rag_ctx),
                "anxiety_support",
            )

        # Motivational disorder / learned helplessness
        if (
            any(d in all_ld for d in ("motivation_disorder", "learned_helplessness", "academic_burnout"))
            or int_flags.get("motivation_low")
        ):
            return (
                _SYSTEM_MOTIVATION,
                _PROMPT_MOTIVATION_SUPPORT.format(user_input=user_input, rag_context=rag_ctx),
                "motivation_support",
            )

        # Default: general strategy suggestion
        return (
            _SYSTEM_GENERAL,
            _PROMPT_STRATEGY_SUGGESTION.format(user_input=user_input, rag_context=rag_ctx),
            "general_support",
        )
