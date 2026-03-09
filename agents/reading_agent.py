# agents/reading_agent.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  IELTS Reading Agent
#
# Responsibilities:
#   1. Parse a raw IELTS reading passage (article + questions)
#      into structured sections and question objects.
#   2. Deliver sections one at a time with guided prompts
#      adapted to the learner's LD profile (ADHD, anxiety, …).
#   3. Evaluate answers and generate adaptive hints.
#   4. Self-optimise the study strategy based on performance and
#      cognitive state, logging recommendations for the learner.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# ── Parsing helpers ───────────────────────────────────────────────

def _split_passage_and_questions(raw: str):
    """Return (passage_text, questions_raw) by heuristic splitting.

    Strategy:
    • Look for a "Questions" / "Questions 1–N" marker or a line that
      starts with "1." or "1 " followed by a question pattern.
    • Everything before that marker is the passage; everything after
      is the questions block.
    """
    # Common IELTS markers that signal the start of the question section
    q_marker = re.search(
        r'(?m)^(Questions?\s+\d|Question\s+\d|\*\*Questions|'
        r'QUESTIONS|Questions and answers|Reading comprehension questions)',
        raw,
    )
    if q_marker:
        passage_text = raw[: q_marker.start()].strip()
        questions_raw = raw[q_marker.start() :].strip()
        return passage_text, questions_raw

    # Fallback: find the first standalone "1." line
    first_q = re.search(r'(?m)^\s*1[\.\)]\s+\S', raw)
    if first_q:
        passage_text = raw[: first_q.start()].strip()
        questions_raw = raw[first_q.start() :].strip()
        return passage_text, questions_raw

    # No questions found – treat everything as passage
    return raw.strip(), ''


def _extract_questions(questions_raw: str) -> List[str]:
    """Parse individual question strings from the questions block."""
    if not questions_raw:
        return []

    # Split on lines that start with a number followed by . or )
    parts = re.split(r'(?m)(?=^\s*\d+[\.\)]\s)', questions_raw)
    questions = []
    for part in parts:
        cleaned = re.sub(r'^\s*\d+[\.\)]\s*', '', part.strip())
        # Skip header lines like "Questions 1–5" or "Questions 1-13"
        if re.match(r'^Questions?\s+[\d–\-]+', cleaned, re.IGNORECASE):
            continue
        if cleaned:
            questions.append(cleaned.strip())
    return questions


def _split_into_sections(passage_text: str, num_sections: int = 3) -> List[dict]:
    """Split the passage into roughly equal sections.

    If the text contains explicit heading markers (e.g. bold Paragraph A/B/C
    or labelled sections), split on those.  Otherwise divide evenly by
    paragraphs.
    """
    # Try explicit heading patterns (Paragraph A, Section 1, etc.)
    heading_split = re.split(
        r'(?m)((?:Paragraph|Section|Part)\s+[A-Z\d]+\b[^\n]*)',
        passage_text,
    )
    if len(heading_split) > 3:
        sections = []
        for i in range(1, len(heading_split), 2):
            heading = heading_split[i].strip()
            body = heading_split[i + 1].strip() if i + 1 < len(heading_split) else ''
            if body:
                sections.append({'heading': heading, 'body': body})
        if sections:
            return sections

    # Fallback: split by paragraphs then group
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', passage_text) if p.strip()]
    if not paragraphs:
        return [{'heading': 'Full Passage', 'body': passage_text}]

    chunk_size = max(1, len(paragraphs) // num_sections)
    sections = []
    for i in range(num_sections):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_sections - 1 else len(paragraphs)
        body = '\n\n'.join(paragraphs[start:end])
        if body:
            sections.append({'heading': f'Section {i + 1}', 'body': body})
    return sections


def _assign_questions_to_sections(
    questions: List[str], sections: List[dict]
) -> List[List[int]]:
    """Assign question indices to sections evenly.

    Returns a list of lists: assignments[section_idx] = [q_idx, …]
    """
    if not questions or not sections:
        return [[] for _ in sections]

    # Simple strategy: distribute questions evenly across sections,
    # last section gets any remainder
    assignments: List[List[int]] = [[] for _ in sections]
    per_section = len(questions) // len(sections)
    remainder = len(questions) % len(sections)

    q_idx = 0
    for s_idx, _ in enumerate(sections):
        count = per_section + (1 if s_idx < remainder else 0)
        for _ in range(count):
            if q_idx < len(questions):
                assignments[s_idx].append(q_idx)
                q_idx += 1
    return assignments


def parse_passage(raw_text: str, num_sections: int = 3):
    """Full parse pipeline.

    Returns:
        {
          "sections": [{"heading": str, "body": str, "question_indices": [int]}],
          "questions": [str],
        }
    """
    passage_text, questions_raw = _split_passage_and_questions(raw_text)
    questions = _extract_questions(questions_raw)
    sections = _split_into_sections(passage_text, num_sections=num_sections)
    assignments = _assign_questions_to_sections(questions, sections)

    for i, sec in enumerate(sections):
        sec['question_indices'] = assignments[i]

    return {'sections': sections, 'questions': questions}


# ── Guidance generation (LLM-free fallback) ──────────────────────

_ADHD_TIPS = [
    "⚡ Read the heading first – it tells you what the section is about.",
    "⚡ Underline any name, date, or number you see.",
    "⚡ Read one sentence at a time.  Stop and think after each one.",
]

_GENERAL_TIPS = [
    "Skim the section once quickly, then read carefully.",
    "Pay attention to topic sentences (usually the first sentence of each paragraph).",
    "Look for keywords that match the question wording.",
]


def _build_section_intro(
    section: dict,
    section_num: int,
    total_sections: int,
    ld_profile: dict,
    attempt_score: Optional[float],
) -> str:
    """Generate a short guiding message to display before a section."""
    all_ld = set(
        ld_profile.get('confirmed', []) + ld_profile.get('suspected', [])
    )

    lines = []
    lines.append(f"📖 Section {section_num} of {total_sections}: **{section['heading']}**")

    if attempt_score is not None and attempt_score < 0.5:
        lines.append(
            "\n💡 You found the previous section tricky – take your time here."
        )

    if 'adhd' in all_ld:
        lines.append('\n' + _ADHD_TIPS[section_num % len(_ADHD_TIPS)])
    else:
        lines.append('\n' + _GENERAL_TIPS[section_num % len(_GENERAL_TIPS)])

    lines.append(
        "\nRead the passage below, then answer the questions at the bottom."
    )
    return '\n'.join(lines)


def _build_hint(
    question_text: str,
    section_body: str,
    ld_profile: dict,
    hints_used: int,
) -> str:
    """Generate a hint for the given question, escalating with more hints_used."""
    all_ld = set(
        ld_profile.get('confirmed', []) + ld_profile.get('suspected', [])
    )

    # Extract a candidate key sentence from the section
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', section_body) if len(s.strip()) > 30]

    # Find sentence most similar to question by naive keyword overlap
    q_words = set(re.findall(r'\b\w{4,}\b', question_text.lower()))
    best = max(sentences, key=lambda s: len(q_words & set(re.findall(r'\b\w{4,}\b', s.lower()))), default='')

    hint_lines = []

    if 'adhd' in all_ld:
        hint_lines.append("⚡ Focus: re-read the section carefully.")

    if hints_used == 0:
        hint_lines.append("💡 **Hint 1**: Look for keywords from the question in the text.")
    elif hints_used == 1:
        q_kw = [w for w in q_words if len(w) > 4][:4]
        if q_kw:
            hint_lines.append(
                f"💡 **Hint 2**: Try searching for: {', '.join(q_kw)}"
            )
        else:
            hint_lines.append("💡 **Hint 2**: Re-read the section and think about the main idea.")
    else:
        if best:
            preview = best[:120] + ('…' if len(best) > 120 else '')
            hint_lines.append(
                f"💡 **Hint 3**: The answer is near this sentence:\n\n> _{preview}_"
            )
        else:
            hint_lines.append("💡 **Hint 3**: Try re-reading the entire section one more time.")

    return '\n'.join(hint_lines)


def _evaluate_answer(
    user_answer: str,
    question_text: str,
    section_body: str,
) -> dict:
    """Simple heuristic answer evaluation.

    Returns {'correct': bool, 'score': float, 'feedback': str}
    """
    if not user_answer.strip():
        return {'correct': False, 'score': 0.0, 'feedback': 'Please write an answer before submitting.'}

    # Keyword overlap between answer and section
    answer_words = set(re.findall(r'\b\w{4,}\b', user_answer.lower()))
    section_words = set(re.findall(r'\b\w{4,}\b', section_body.lower()))
    q_words = set(re.findall(r'\b\w{4,}\b', question_text.lower()))

    # The answer should reference content from the section
    overlap_section = len(answer_words & section_words)
    overlap_question = len(answer_words & q_words)

    score = min(1.0, (overlap_section * 0.06 + overlap_question * 0.15))

    if score >= 0.5:
        feedback = "✅ Good answer! Your response uses relevant content from the passage."
    elif score >= 0.25:
        feedback = (
            "🟡 Partially correct. Try to include more details directly from the passage."
        )
    else:
        feedback = (
            "❌ Your answer may be off-track.  Re-read the section and look for specific details."
        )

    return {'correct': score >= 0.5, 'score': round(score, 2), 'feedback': feedback}


def _build_strategy(attempt_data: dict, ld_profile: dict) -> str:
    """Generate a personalised study strategy based on attempt history."""
    all_ld = set(
        ld_profile.get('confirmed', []) + ld_profile.get('suspected', [])
    )
    answers = attempt_data.get('answers', {})
    hints_used = attempt_data.get('hints_used', 0)

    total = len(answers)
    correct = sum(1 for v in answers.values() if isinstance(v, dict) and v.get('correct'))

    lines = ['### 📊 Your Learning Strategy Summary\n']

    if total > 0:
        pct = int(correct / total * 100)
        lines.append(f"**Score so far:** {correct}/{total} ({pct}%)\n")

    if hints_used > 2:
        lines.append(
            "💡 **Strategy tip:** You requested several hints.  "
            "Try the SQ3R method next time:\n"
            "1. **Survey** – skim headings and bold words\n"
            "2. **Question** – turn headings into questions\n"
            "3. **Read** – read to answer your questions\n"
            "4. **Recite** – close the text and recall\n"
            "5. **Review** – check your recall against the text\n"
        )
    elif total > 0 and pct < 50:
        lines.append(
            "💡 **Strategy tip:** Focus on the **topic sentence** of each paragraph "
            "(usually the first sentence).  IELTS questions almost always refer back to these.\n"
        )
    else:
        lines.append(
            "💡 **Strategy tip:** You are doing well!  "
            "For even better results, practise **skimming** (fast read for gist) "
            "before **scanning** (targeted search for specific information).\n"
        )

    if 'adhd' in all_ld:
        lines.append(
            "\n⚡ **ADHD tip:** Use a pencil/finger to track the line you are reading. "
            "Take a 2-minute movement break between sections."
        )
    if 'anxiety' in all_ld:
        lines.append(
            "\n🌀 **Anxiety tip:** Remember – IELTS questions test comprehension, not "
            "prior knowledge.  All answers are *in the text*."
        )

    return '\n'.join(lines)


# ── Main agent entry point ────────────────────────────────────────

def reading_agent_guide_section(
    section: dict,
    section_num: int,
    total_sections: int,
    ld_profile: dict,
    attempt_score: Optional[float] = None,
) -> str:
    """Return the intro guidance text for displaying a new section."""
    return _build_section_intro(section, section_num, total_sections, ld_profile, attempt_score)


def reading_agent_hint(
    question_text: str,
    section_body: str,
    ld_profile: dict,
    hints_used: int,
) -> str:
    """Return a hint for the current question."""
    return _build_hint(question_text, section_body, ld_profile, hints_used)


def reading_agent_evaluate(
    user_answer: str,
    question_text: str,
    section_body: str,
) -> dict:
    """Evaluate a learner's answer."""
    return _evaluate_answer(user_answer, question_text, section_body)


def reading_agent_strategy(attempt_data: dict, ld_profile: dict) -> str:
    """Generate a personalised learning strategy."""
    return _build_strategy(attempt_data, ld_profile)
