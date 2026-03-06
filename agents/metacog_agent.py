# agents/metacog_agent.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  Meta-Cognition Agent
#
# Activates ONLY in low-cognitive-demand moments (after task completion,
# WM load < 0.40, affect positive). Inserts structured reflection prompts
# calibrated to the learner's LD profile.
#
# Goal: build the learner's self-regulatory capacity over time so they
# need less external scaffolding.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any, Dict


# ── Reflection prompt templates ───────────────────────────────────

# ADHD: Click-or-Clunk protocol
_CLICK_OR_CLUNK = """\
Quick check-in (takes 30 seconds):

⚡ CLICK or CLUNK?
  → Did what we just covered make sense to you? (CLICK = yes / CLUNK = no)

If CLICK: Can you say it back in ONE sentence, in your own words?
If CLUNK: What's the first word or idea that lost you?

No right or wrong answer — just honest.
"""

# Motivational Disorder: Attribution retraining
_ATTRIBUTION_RETRAINING = """\
Reflection moment:

You just completed a task. Before we move on — quick question:

  What made this task go the way it did?

Think about: Was it the topic? The way I explained it? How much sleep you got?
Your effort today? Something else?

(I'm asking because understanding the real cause helps us do better next time.
It's almost never "I'm just bad at this.")
"""

# Executive Function: Process awareness
_PROCESS_AWARENESS = """\
One-minute reflection:

Looking back at what you just did:
  1. Did you have a plan before you started? (yes / no / sort of)
  2. At what point did it feel hardest? (beginning / middle / end)
  3. What would you do differently next time?

Short answers are fine. This helps build your planning muscle over time.
"""

# General: Basic comprehension monitoring
_GENERAL_METACOG = """\
Quick self-check:

  • What was the main thing you just learned?
  • Is there anything that still feels fuzzy?
  • What would help you remember this?

One sentence per question is enough.
"""


def metacog_agent(state: Dict[str, Any], llm) -> Dict[str, Any]:
    """
    LangGraph-compatible node.
    Only generates a response when the moment is right (checked by Orchestrator).
    If activated, selects the appropriate reflection prompt type.
    """
    ld_profile  = state.get("ld_profile", {})
    cog_state   = state.get("cognitive_state", {})
    user_input  = state.get("user_input", "")

    confirmed = set(ld_profile.get("confirmed", []))
    suspected = set(ld_profile.get("suspected", []))
    all_ld    = confirmed | suspected

    # Select prompt based on primary LD
    if "adhd" in all_ld:
        reflection_prompt = _CLICK_OR_CLUNK
        prompt_type = "click_or_clunk"
    elif "motivation_disorder" in all_ld:
        reflection_prompt = _ATTRIBUTION_RETRAINING
        prompt_type = "attribution_retraining"
    elif "executive_function" in all_ld:
        reflection_prompt = _PROCESS_AWARENESS
        prompt_type = "process_awareness"
    else:
        reflection_prompt = _GENERAL_METACOG
        prompt_type = "general_metacog"

    system = (
        "You are a metacognitive coach for a student with learning differences. "
        "Present the reflection prompt below in a warm, non-judgmental tone. "
        "Do not add extra questions. Keep the message exactly as structured. "
        "After the prompt, add one short encouragement sentence."
    )

    response = llm.chat(system=system, user=reflection_prompt, temperature=0.5)

    return {
        "metacog_response": response,
        "_metacog_prompt_type": prompt_type,
    }


def should_activate_metacog(state: Dict[str, Any]) -> bool:
    """
    Gate function: metacog agent only fires when cognitive conditions are right.
    Called by the Orchestrator before adding metacog to the priority queue.
    """
    cog   = state.get("cognitive_state", {})
    flags = state.get("intervention_flags", {})

    wm_load      = cog.get("working_memory_load", 1.0)
    affect       = cog.get("affect_valence", 0.0)
    motivation   = cog.get("motivation_level", 0.0)

    # Only fire when: low WM load, positive affect, not in crisis
    return (
        wm_load   < 0.40 and
        affect    > 0.00 and
        motivation > 0.35 and
        not flags.get("affect_negative") and
        not flags.get("fatigue_high")
    )
