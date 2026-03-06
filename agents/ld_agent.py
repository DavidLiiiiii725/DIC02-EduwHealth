# agents/ld_agent.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  LD Specialist Agent
#
# Routes to disability-specific intervention protocols:
#   - Writing + Executive Function → 3-phase scaffold (Plan/Generate/Revise)
#   - ADHD + Listening/Reading    → chunked delivery + anchors
#   - Motivational Disorder       → micro-task + expectancy repair
#   - Anxiety + Auditory          → chunked info + transition markers
#
# Scaffold density is driven by ef_severity × fade_index from LearnerModel.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any, Dict


# ── Scaffold templates ────────────────────────────────────────────

_SCAFFOLD_HIGH = """\
[WRITING SCAFFOLD — Full Support]

STEP 1 — PLAN (complete before writing anything)
  • What is the ONE main point you want to make? (one sentence)
  • Who is your reader? What do they already know?
  • List 2-3 pieces of evidence or examples you will use.
  • What do you want the reader to think/feel/do after reading?

STEP 2 — GENERATE (write only this paragraph first)
  Start your paragraph with this frame:
    "The main idea I want to discuss is ______."
  Then add ONE piece of evidence: "For example, ______."
  Close with: "This shows that ______."

STEP 3 — REVISE (only after you finish Step 2)
  Ask yourself: Does the first sentence tell the reader exactly what this paragraph is about?
  If not, rewrite just that sentence. Do not change anything else yet.

{rag_hint}

Student task:
{user_input}
"""

_SCAFFOLD_MEDIUM = """\
[WRITING SCAFFOLD — Guided Framework]

Before writing, answer these questions briefly:
  1. What is your main argument?
  2. What is your strongest piece of evidence?
  3. How will you conclude?

Then write your response. Use this structure:
  → Topic sentence  → Evidence  → Analysis  → Closing link

{rag_hint}

Student task:
{user_input}
"""

_SCAFFOLD_LOW = """\
[WRITING SCAFFOLD — Light Prompt]

Take 2 minutes to plan before writing:
  - What is the core idea?
  - What supports it?

Then write your response.

{rag_hint}

Student task:
{user_input}
"""

_ADHD_CHUNK_TEMPLATE = """\
[ATTENTION SUPPORT MODE]
⚡ KEY POINT: {key_point}

Here is what you need to know — one piece at a time:

{rag_hint}

Your question:
{user_input}

(I will give you the answer in 2-3 short steps. After each step, let me know if it clicked.)
"""

_MOTIVATION_REPAIR_TEMPLATE = """\
[MICRO-TASK MODE]
You don't need to solve everything right now.

Here is ONE small step (should take about 2 minutes):
  → {micro_task}

That's it. Just that one step.

{rag_hint}

Context:
{user_input}
"""

_ANXIETY_CHUNK_TEMPLATE = """\
[STRUCTURED LISTENING / READING MODE]

I'll break this into clear parts. Each part has a signal word so you can follow along:

**First** — the main idea
**Then** — the key detail
**Finally** — what this means for you

{rag_hint}

Your question:
{user_input}
"""


# ── Agent function ────────────────────────────────────────────────

def ld_specialist_agent(state: Dict[str, Any], llm) -> Dict[str, Any]:
    """
    LangGraph-compatible agent node.
    Selects the intervention protocol based on LD profile + task type.
    """
    user_input       = state.get("user_input", "")
    rag              = state.get("rag_context", "")
    ld_profile       = state.get("ld_profile", {})
    scaffold_density = state.get("scaffold_density", "medium")
    cog_state        = state.get("cognitive_state", {})
    int_flags        = state.get("intervention_flags", {})

    confirmed  = ld_profile.get("confirmed",  [])
    suspected  = ld_profile.get("suspected",  [])
    all_ld     = set(confirmed + suspected)

    rag_hint = f"Retrieved context:\n{rag}" if rag else ""

    # ── Route to protocol ─────────────────────────────────────────

    if _is_writing_task(user_input) and "executive_function" in all_ld:
        prompt = _writing_scaffold(user_input, rag_hint, scaffold_density)
        protocol = f"writing_scaffold_{scaffold_density}"

    elif ("adhd" in all_ld) and int_flags.get("wm_overload"):
        key_point = _extract_key_point(user_input)
        prompt = _ADHD_CHUNK_TEMPLATE.format(
            key_point=key_point,
            rag_hint=rag_hint,
            user_input=user_input,
        )
        protocol = "adhd_chunk_delivery"

    elif "motivation_disorder" in all_ld or int_flags.get("motivation_low"):
        micro = _generate_micro_task(user_input)
        prompt = _MOTIVATION_REPAIR_TEMPLATE.format(
            micro_task=micro,
            rag_hint=rag_hint,
            user_input=user_input,
        )
        protocol = "motivation_micro_task"

    elif "anxiety" in all_ld:
        prompt = _ANXIETY_CHUNK_TEMPLATE.format(
            rag_hint=rag_hint,
            user_input=user_input,
        )
        protocol = "anxiety_structured_chunks"

    else:
        # Generic LD support: reduce cognitive load
        prompt = _generic_ld_prompt(user_input, rag_hint, int_flags)
        protocol = "generic_load_reduction"

    system = (
        "You are an educational specialist trained in learning disabilities. "
        "Follow the scaffold or structure in the prompt exactly. "
        "Be warm, precise, and concise. Do not overwhelm the student."
    )

    response = llm.chat(system=system, user=prompt, temperature=0.45)

    return {
        "ld_specialist_response": response,
        "_ld_protocol_used": protocol,   # debug metadata
    }


# ── Helper functions ──────────────────────────────────────────────

def _is_writing_task(text: str) -> bool:
    keywords = [
        "write", "essay", "paragraph", "draft", "composition",
        "writing", "argue", "explain in writing", "written",
        "文章", "写作", "段落", "论文",
    ]
    tl = text.lower()
    return any(k in tl for k in keywords)


def _writing_scaffold(user_input: str, rag_hint: str, density: str) -> str:
    if density == "high":
        return _SCAFFOLD_HIGH.format(rag_hint=rag_hint, user_input=user_input)
    if density == "medium":
        return _SCAFFOLD_MEDIUM.format(rag_hint=rag_hint, user_input=user_input)
    return _SCAFFOLD_LOW.format(rag_hint=rag_hint, user_input=user_input)


def _extract_key_point(text: str) -> str:
    """Very lightweight: take the first sentence as the anchor point."""
    sentences = text.replace("?", ".").replace("!", ".").split(".")
    first = next((s.strip() for s in sentences if len(s.strip()) > 8), text[:80])
    return first


def _generate_micro_task(text: str) -> str:
    """Generate a minimal first step from the user's request."""
    tl = text.lower()
    if "write" in tl or "essay" in tl:
        return "Write ONE sentence that states your main idea. Just one sentence."
    if "read" in tl or "understand" in tl:
        return "Read only the first paragraph. Then tell me ONE thing you noticed."
    if "solve" in tl or "problem" in tl or "math" in tl:
        return "Write down what information you are given. Don't solve yet."
    return "Write down what you already know about this topic. Just 2-3 words is fine."


def _generic_ld_prompt(user_input: str, rag_hint: str, flags: Dict) -> str:
    density_note = ""
    if flags.get("wm_overload"):
        density_note = "[Note: Keep your answer to 3 bullet points maximum.]\n"
    return f"""{density_note}
{rag_hint}

Student:
{user_input}
""".strip()
