# tests/test_reading_agent.py
# ─────────────────────────────────────────────────────────────────
# Unit tests for agents/reading_agent.py
#
# Tests focus on:
#  1. detect_question_pages() — various dash types, "and"/"to" connectors,
#     singular "Question N", and the numbered-line fallback heuristic.
#  2. _extract_question_groups() — group header parsing with en-dash, em-dash,
#     "and" connectors, and bare numbered lines.
# ─────────────────────────────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from agents.reading_agent import (
    _Q_HEADER_RE,
    _Q_NUMBERED_LINE_RE,
    _Q_NUMBERED_LINE_THRESHOLD,
    _GROUP_HEADER_RE,
    _extract_question_groups,
)


# ── Helpers ───────────────────────────────────────────────────────

def _make_fake_page(text: str):
    """Return a minimal fake 'page' object whose get_text() returns ``text``."""
    class _FakePage:
        def get_text(self, *args, **kwargs):
            return text
    return _FakePage()


def _detect_question_pages_from_texts(page_texts):
    """Run the same logic as detect_question_pages() on a list of strings.

    This avoids needing a real PDF/fitz document in the tests.
    """
    question_page_indices = set()
    for page_num, text in enumerate(page_texts):
        if _Q_HEADER_RE.search(text):
            question_page_indices.add(page_num)
            continue
        if len(_Q_NUMBERED_LINE_RE.findall(text)) >= _Q_NUMBERED_LINE_THRESHOLD:
            question_page_indices.add(page_num)
    return question_page_indices


# ── Test: question page detection ────────────────────────────────

class TestDetectQuestionPages(unittest.TestCase):
    """Verify _Q_HEADER_RE and _Q_NUMBERED_LINE_RE match the right pages."""

    # ---- primary signal: explicit header ----

    def test_hyphen_range(self):
        """'Questions 1-3' (ASCII hyphen) should be detected."""
        texts = ["Questions 1-3\nSome question text here."]
        self.assertIn(0, _detect_question_pages_from_texts(texts))

    def test_en_dash_range(self):
        """'Questions 4\u20139' (en-dash U+2013) should be detected."""
        texts = ["Questions 4\u20139\nAnother question."]
        self.assertIn(0, _detect_question_pages_from_texts(texts))

    def test_em_dash_range(self):
        """'Questions 25\u201426' (em-dash U+2014) should be detected."""
        texts = ["Questions 25\u201426\nYet another question."]
        self.assertIn(0, _detect_question_pages_from_texts(texts))

    def test_and_connector(self):
        """'Questions 10 and 11' should be detected."""
        texts = ["Questions 10 and 11\nDo something."]
        self.assertIn(0, _detect_question_pages_from_texts(texts))

    def test_to_connector(self):
        """'Questions 1 to 3' should be detected."""
        texts = ["Questions 1 to 3\nDescribe …"]
        self.assertIn(0, _detect_question_pages_from_texts(texts))

    def test_singular_question(self):
        """'Question 1' (singular, no range) should be detected."""
        texts = ["Question 1\nWhat is …?"]
        self.assertIn(0, _detect_question_pages_from_texts(texts))

    def test_case_insensitive(self):
        """Lowercase 'questions 12-14' should be detected."""
        texts = ["questions 12-14\nSome text."]
        self.assertIn(0, _detect_question_pages_from_texts(texts))

    def test_passage_page_not_detected(self):
        """A normal passage paragraph page should NOT be detected as a question page."""
        passage_text = (
            "A  The rapid expansion of cities across the developing world has put\n"
            "enormous pressure on infrastructure. Governments have struggled to\n"
            "keep pace with the demand for clean water and sanitation.\n"
            "B  Urban planners are increasingly turning to data-driven models …\n"
        )
        self.assertNotIn(0, _detect_question_pages_from_texts([passage_text]))

    def test_multiple_pages_only_question_detected(self):
        """Only the question page index is returned when both page types exist."""
        passage = "A  The city grew rapidly in the 20th century.\nB  Industry followed.\n"
        questions = "Questions 1-5\n1. What caused the growth?\n2. Where was industry?"
        result = _detect_question_pages_from_texts([passage, questions])
        self.assertEqual(result, {1})

    # ---- fallback signal: numbered lines ----

    def test_fallback_numbered_lines(self):
        """Page with >= 3 bare numbered lines (no explicit header) is detected."""
        text = (
            "25 The track was originally built for steam engines.\n"
            "26 Bingham found out about the ruins from a local farmer.\n"
            "27 The restoration project was completed ahead of schedule.\n"
        )
        self.assertIn(0, _detect_question_pages_from_texts([text]))

    def test_fallback_below_threshold_not_detected(self):
        """Fewer than _Q_NUMBERED_LINE_THRESHOLD numbered lines should not trigger."""
        # Two numbered lines but no explicit header.
        text = (
            "25 The track was originally built for steam engines.\n"
            "26 Bingham found out about the ruins from a local farmer.\n"
        )
        if _Q_NUMBERED_LINE_THRESHOLD > 2:
            self.assertNotIn(0, _detect_question_pages_from_texts([text]))


# ── Test: question group extraction ──────────────────────────────

class TestExtractQuestionGroups(unittest.TestCase):
    """Verify _extract_question_groups handles all header variants."""

    def test_hyphen_group_header(self):
        raw = "Questions 1-3\n1. What does the author mean?\n2. Choose the correct option.\n3. Describe the process.\n"
        groups = _extract_question_groups(raw)
        self.assertEqual(len(groups), 3)
        self.assertTrue(all(g['group_label'].startswith('Questions 1') for g in groups))

    def test_en_dash_group_header(self):
        raw = "Questions 4\u20139\n4. Complete the table.\n5. True, False, or Not Given.\n"
        groups = _extract_question_groups(raw)
        labels = {g['group_label'] for g in groups}
        # At least one item has the group label
        self.assertTrue(any('4' in lbl for lbl in labels))
        self.assertEqual(len(groups), 2)

    def test_and_connector_group_header(self):
        raw = "Questions 10 and 11\n10. Match the headings.\n11. Identify the main idea.\n"
        groups = _extract_question_groups(raw)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all('10 and 11' in g['group_label'] for g in groups))

    def test_multiple_groups(self):
        raw = (
            "Questions 1-3\n"
            "1. First question.\n"
            "2. Second question.\n"
            "3. Third question.\n"
            "Questions 4\u20139\n"
            "4. Fourth question.\n"
            "5. Fifth question.\n"
        )
        groups = _extract_question_groups(raw)
        self.assertEqual(len(groups), 5)
        q1_label = next(g['group_label'] for g in groups if g['order'] == 1)
        q4_label = next(g['group_label'] for g in groups if g['order'] == 4)
        self.assertIn('1', q1_label)
        self.assertIn('4', q4_label)

    def test_bare_numbered_lines(self):
        """Questions without trailing '.' or ')' (bare numbered lines) are extracted."""
        raw = (
            "Questions 25-26\n"
            "25 The track was originally built for steam engines.\n"
            "26 Bingham found out about the ruins from a local farmer.\n"
        )
        groups = _extract_question_groups(raw)
        orders = {g['order'] for g in groups}
        self.assertIn(25, orders)
        self.assertIn(26, orders)

    def test_empty_raw_returns_empty_list(self):
        self.assertEqual(_extract_question_groups(''), [])

    def test_question_text_preserved(self):
        raw = "Questions 1-2\n1. What is the main argument of the passage?\n2. How does the author support this?\n"
        groups = _extract_question_groups(raw)
        texts = [g['text'] for g in groups]
        self.assertTrue(any('main argument' in t for t in texts))
        self.assertTrue(any('author support' in t for t in texts))


# ── Test: _GROUP_HEADER_RE ────────────────────────────────────────

class TestGroupHeaderRE(unittest.TestCase):
    """Verify _GROUP_HEADER_RE matches all expected header forms."""

    def _match(self, s):
        return bool(_GROUP_HEADER_RE.match(s))

    def test_hyphen(self):
        self.assertTrue(self._match("Questions 1-3"))

    def test_en_dash(self):
        self.assertTrue(self._match("Questions 4\u20139"))

    def test_em_dash(self):
        self.assertTrue(self._match("Questions 25\u201426"))

    def test_and_connector(self):
        self.assertTrue(self._match("Questions 10 and 11"))

    def test_to_connector(self):
        self.assertTrue(self._match("Questions 1 to 3"))

    def test_singular(self):
        self.assertTrue(self._match("Question 1"))

    def test_passage_text_no_match(self):
        self.assertFalse(self._match("A  The city expanded rapidly."))


if __name__ == '__main__':
    unittest.main()
