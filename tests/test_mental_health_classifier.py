# tests/test_mental_health_classifier.py
# ─────────────────────────────────────────────────────────────────
# Unit tests for affect/mental_health_classifier.py
# ─────────────────────────────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch, MagicMock

from affect.mental_health_classifier import MentalHealthRiskDetector


class TestMentalHealthRiskDetectorHeuristic(unittest.TestCase):
    """
    Test the heuristic fallback path (no model loaded).
    Patches _load_model so that no network/GPU access is required.
    """

    def _make_detector(self):
        with patch.object(MentalHealthRiskDetector, "_load_model", lambda self: None):
            det = MentalHealthRiskDetector.__new__(MentalHealthRiskDetector)
            det.model_name = "test-model"
            det._model = None
            return det

    def test_severe_keywords(self):
        det    = self._make_detector()
        result = det.assess_risk("I want to kill myself and end my life")
        self.assertEqual(result["risk_level"], "severe")
        self.assertGreater(result["score"], 0.8)
        self.assertEqual(result["method"], "heuristic")

    def test_high_keywords(self):
        det    = self._make_detector()
        result = det.assess_risk("I feel completely hopeless and worthless")
        self.assertEqual(result["risk_level"], "high")
        self.assertGreater(result["score"], 0.6)

    def test_moderate_keywords(self):
        det    = self._make_detector()
        result = det.assess_risk("I am so overwhelmed and anxious about everything")
        self.assertEqual(result["risk_level"], "moderate")
        self.assertGreater(result["score"], 0.3)

    def test_low_risk_text(self):
        det    = self._make_detector()
        result = det.assess_risk("What is the difference between supervised and unsupervised learning?")
        self.assertEqual(result["risk_level"], "low")
        self.assertLess(result["score"], 0.3)

    def test_result_keys(self):
        det    = self._make_detector()
        result = det.assess_risk("I feel a bit stressed")
        for key in ("risk_level", "score", "confidence", "raw_scores", "method"):
            self.assertIn(key, result)

    def test_score_range(self):
        det = self._make_detector()
        for text in ["I'm fine", "I feel hopeless", "I want to die"]:
            result = det.assess_risk(text)
            self.assertGreaterEqual(result["score"], 0.0)
            self.assertLessEqual(result["score"], 1.0)


class TestMentalHealthRiskDetectorAggregation(unittest.TestCase):
    """Tests for the model-score aggregation logic."""

    def test_suicidal_label(self):
        raw = {"suicidal": 0.9, "normal": 0.1}
        score = MentalHealthRiskDetector._aggregate_model_score(raw)
        self.assertGreater(score, 0.7)

    def test_safe_label_dominates(self):
        raw = {"normal": 0.95, "label_1": 0.05}
        score = MentalHealthRiskDetector._aggregate_model_score(raw)
        self.assertLess(score, 0.2)

    def test_empty_raw(self):
        score = MentalHealthRiskDetector._aggregate_model_score({})
        self.assertAlmostEqual(score, 0.3)

    def test_score_to_level_boundaries(self):
        from config import RISK_THRESHOLDS
        det = MentalHealthRiskDetector.__new__(MentalHealthRiskDetector)
        self.assertEqual(det._score_to_level(0.0),  "low")
        self.assertEqual(det._score_to_level(0.29), "low")
        self.assertEqual(det._score_to_level(0.3),  "moderate")
        self.assertEqual(det._score_to_level(0.59), "moderate")
        self.assertEqual(det._score_to_level(0.6),  "high")
        self.assertEqual(det._score_to_level(0.79), "high")
        self.assertEqual(det._score_to_level(0.8),  "severe")
        self.assertEqual(det._score_to_level(1.0),  "severe")


class TestMentalHealthRiskDetectorModelPath(unittest.TestCase):
    """Test graceful fallback when model loading raises an exception."""

    def test_falls_back_to_heuristic_on_import_error(self):
        with patch("affect.mental_health_classifier.MentalHealthRiskDetector._load_model") as mock_load:
            mock_load.side_effect = Exception("no network")
            det = MentalHealthRiskDetector.__new__(MentalHealthRiskDetector)
            det.model_name = "test"
            det._model = None
            result = det.assess_risk("I feel sad")
            # Should still return a valid result via heuristic
            self.assertIn("risk_level", result)


if __name__ == "__main__":
    unittest.main()
