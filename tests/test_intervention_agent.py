# tests/test_intervention_agent.py
# ─────────────────────────────────────────────────────────────────
# Unit tests for agents/intervention_agent.py
# ─────────────────────────────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from agents.intervention_agent import InterventionAgent


class TestInterventionAgentBasic(unittest.TestCase):
    """Basic output-shape tests — no LLM or KB required."""

    def setUp(self):
        self.agent = InterventionAgent(kb_retriever=None)

    def _base_state(self, **kwargs):
        state = {
            "risk":           {"risk_level": "low"},
            "emotions":       {"joy": 0.8, "sadness": 0.1, "fear": 0.05},
            "key_indicators": {"anxiety": 0.1, "depression": 0.1, "positive_affect": 0.7},
            "disabilities":   [],
        }
        state.update(kwargs)
        return state

    def test_returns_list(self):
        result = self.agent.recommend_interventions(self._base_state())
        self.assertIsInstance(result, list)

    def test_each_item_has_required_keys(self):
        result = self.agent.recommend_interventions(self._base_state())
        for item in result:
            for key in ("type", "strategy", "priority"):
                self.assertIn(key, item)

    def test_low_risk_no_immediate(self):
        result = self.agent.recommend_interventions(self._base_state())
        priorities = [i["priority"] for i in result]
        self.assertNotIn("immediate", priorities)

    def test_severe_risk_triggers_immediate(self):
        state  = self._base_state(risk={"risk_level": "severe"})
        result = self.agent.recommend_interventions(state)
        priorities = [i["priority"] for i in result]
        self.assertIn("immediate", priorities)

    def test_high_risk_triggers_immediate(self):
        state  = self._base_state(risk={"risk_level": "high"})
        result = self.agent.recommend_interventions(state)
        priorities = [i["priority"] for i in result]
        self.assertIn("immediate", priorities)

    def test_sorted_by_priority(self):
        state  = self._base_state(
            risk={"risk_level": "high"},
            disabilities=["ADHD", "anxiety"],
        )
        result = self.agent.recommend_interventions(state)
        order  = {"immediate": 0, "high": 1, "medium": 2, "low": 3}
        vals   = [order[i["priority"]] for i in result]
        self.assertEqual(vals, sorted(vals))


class TestInterventionAgentDisabilities(unittest.TestCase):
    """Tests that disability-specific interventions are included."""

    def setUp(self):
        self.agent = InterventionAgent(kb_retriever=None)

    def _state(self, disabilities, risk_level="low"):
        return {
            "risk":           {"risk_level": risk_level},
            "emotions":       {},
            "key_indicators": {"anxiety": 0.2, "depression": 0.2},
            "disabilities":   disabilities,
        }

    def test_adhd_includes_task_decomposition(self):
        result = self.agent.recommend_interventions(self._state(["ADHD"]))
        types  = [i["type"] for i in result]
        self.assertIn("task_decomposition", types)

    def test_efd_includes_planning_scaffold(self):
        result = self.agent.recommend_interventions(self._state(["executive_function_deficit"]))
        types  = [i["type"] for i in result]
        self.assertIn("planning_scaffold", types)

    def test_anxiety_includes_chunked_delivery(self):
        result = self.agent.recommend_interventions(self._state(["anxiety_disorder"]))
        types  = [i["type"] for i in result]
        self.assertIn("chunked_information_delivery", types)

    def test_motivation_includes_micro_success(self):
        result = self.agent.recommend_interventions(self._state(["learned_helplessness"]))
        types  = [i["type"] for i in result]
        self.assertIn("micro_success_architecture", types)

    def test_no_duplicate_types(self):
        result = self.agent.recommend_interventions(
            self._state(["ADHD", "anxiety_disorder", "executive_function_deficit"])
        )
        types  = [i["type"] for i in result]
        self.assertEqual(len(types), len(set(types)))


class TestInterventionAgentEmotionThresholds(unittest.TestCase):
    """Tests for emotion-score-driven interventions."""

    def setUp(self):
        self.agent = InterventionAgent(kb_retriever=None)

    def test_high_anxiety_triggers_strategy(self):
        state = {
            "risk":           {"risk_level": "low"},
            "emotions":       {"fear": 0.8},
            "key_indicators": {"anxiety": 0.75, "depression": 0.1},
            "disabilities":   [],
        }
        result = self.agent.recommend_interventions(state)
        types  = [i["type"] for i in result]
        self.assertIn("anxiety_reduction", types)

    def test_high_depression_triggers_support(self):
        state = {
            "risk":           {"risk_level": "low"},
            "emotions":       {"sadness": 0.8},
            "key_indicators": {"anxiety": 0.1, "depression": 0.75},
            "disabilities":   [],
        }
        result = self.agent.recommend_interventions(state)
        types  = [i["type"] for i in result]
        self.assertIn("depression_support", types)


class TestInterventionAgentPastSuccesses(unittest.TestCase):
    """Tests that previously-successful strategies are boosted."""

    def setUp(self):
        self.agent = InterventionAgent(kb_retriever=None)

    def test_boosted_strategy_not_downgraded(self):
        state = {
            "risk":                 {"risk_level": "low"},
            "emotions":             {},
            "key_indicators":       {"anxiety": 0.2, "depression": 0.2},
            "disabilities":         ["ADHD"],
            "successful_strategies": ["task_decomposition"],
        }
        result = self.agent.recommend_interventions(state)
        td = next((i for i in result if i["type"] == "task_decomposition"), None)
        self.assertIsNotNone(td)
        # boosted from medium/high to at least high
        self.assertIn(td["priority"], ("immediate", "high"))


if __name__ == "__main__":
    unittest.main()
