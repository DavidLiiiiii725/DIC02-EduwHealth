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
    Generates a concise, ADHD-friendly reflection that directly uses
    the retrieved passage / notes in ``rag_context``.
    """
    ld_profile  = state.get("ld_profile", {})
    rag_context = state.get("rag_context", "") or ""

    confirmed = set(ld_profile.get("confirmed", []))
    suspected = set(ld_profile.get("suspected", []))
    all_ld    = confirmed | suspected

    if "adhd" in all_ld:
        prompt_type = "adhd_reflection"
    else:
        prompt_type = "general_reflection"

    system = (
        "You are a metacognitive coach for a student with learning differences. "
        "Write very short, concrete reflections that can be read at a glance. "
        "Always ground your answer in the retrieved passage and notes. "
        "Do not add extra sections or disclaimers."
    )

    user_prompt = f"""You are given retrieved context from the learner's notes and current passage (rag_context).

rag_context:
{rag_context}

Using ONLY this context, write a concise reflection in English following EXACTLY this structure:

Main Learning Reflection Prompt:
- I've come to understand <one clear concept or skill from the passage> and how it applies in real life situations such as <one short, specific example>.

Fuzzy Areas Check:
- One part that still feels unclear to me is <name ONE specific idea, step, or detail from the passage>.

Encouragement:
- <ONE short, kind sentence that encourages continued effort.>

Strict rules:
- Replace every <...> with concrete content grounded in rag_context.
- Do NOT output placeholders like [insert ...], '(Student fills in ...)', or '...'.
- Keep each bullet to at most two short sentences.
- Keep the whole answer brief enough to read at a glance.
"""

    response = llm.chat(system=system, user=user_prompt, temperature=0.4)

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
