# tests/test_ensemble_detector.py
# ─────────────────────────────────────────────────────────────────
# Unit tests for affect/ensemble_detector.py
#
# All heavy models are mocked so these tests run offline and without
# the transformers / torch packages installed.
# ─────────────────────────────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Inject lightweight stubs for optional heavy dependencies so the modules
# can be imported even if transformers / torch are not installed.
# ---------------------------------------------------------------------------
_transformers_stub = MagicMock()
_transformers_stub.pipeline = MagicMock(return_value=MagicMock())
sys.modules.setdefault("transformers", _transformers_stub)
sys.modules.setdefault("torch",        MagicMock())

# Import with stubs in place.
from affect.mental_health_classifier import MentalHealthRiskDetector  # noqa: E402
# Prevent any real model loading in tests.
MentalHealthRiskDetector._load_model = lambda self: setattr(self, "_model", None)

from affect.emotion_model  import EmotionDetector         # noqa: E402
from affect.ensemble_detector import EnsembleAffectiveDetector  # noqa: E402


def _make_det(emotion_scores, risk_result):
    """Return an EnsembleAffectiveDetector with fully-mocked sub-detectors."""
    det = object.__new__(EnsembleAffectiveDetector)

    ed_mock = MagicMock()
    ed_mock.detect.return_value = emotion_scores
    det.emotion_detector = ed_mock

    rd_mock = MagicMock()
    rd_mock.assess_risk.return_value = risk_result
    det.risk_detector = rd_mock

    return det


class TestEnsembleDetectorInit(unittest.TestCase):
    """Test that EnsembleAffectiveDetector stores both sub-detectors."""

    def test_has_emotion_and_risk_detectors(self):
        det = _make_det({}, {"risk_level": "low", "score": 0.1, "confidence": 0.9, "raw_scores": {}})
        self.assertTrue(hasattr(det, "emotion_detector"))
        self.assertTrue(hasattr(det, "risk_detector"))


class TestEnsembleDetectorComprehensiveAnalysis(unittest.TestCase):
    """Test the comprehensive_analysis output structure."""

    def test_output_keys_present(self):
        emotions = {"joy": 0.6, "sadness": 0.1, "fear": 0.05, "anger": 0.05, "surprise": 0.2}
        risk     = {"risk_level": "low", "score": 0.1, "confidence": 0.8, "raw_scores": {}}
        det      = _make_det(emotions, risk)

        result = det.comprehensive_analysis("I am doing well today")
        for key in ("emotions", "risk", "key_indicators", "intervention_priority", "summary"):
            self.assertIn(key, result)

    def test_key_indicators_range(self):
        emotions = {"joy": 0.1, "sadness": 0.6, "fear": 0.5, "anger": 0.1, "surprise": 0.05}
        risk     = {
            "risk_level": "high",
            "score": 0.7,
            "confidence": 0.85,
            "raw_scores": {"depression": 0.6, "anxiety": 0.55},
        }
        det    = _make_det(emotions, risk)
        result = det.comprehensive_analysis("I feel hopeless")

        for k in ("anxiety", "depression", "positive_affect"):
            self.assertGreaterEqual(result["key_indicators"][k], 0.0)
            self.assertLessEqual(result["key_indicators"][k],    1.0)

    def test_severe_risk_immediate_priority(self):
        emotions = {"fear": 0.9, "sadness": 0.8, "joy": 0.0, "anger": 0.0, "surprise": 0.0}
        risk     = {
            "risk_level": "severe",
            "score": 0.95,
            "confidence": 0.9,
            "raw_scores": {"suicidal": 0.95},
        }
        det    = _make_det(emotions, risk)
        result = det.comprehensive_analysis("I want to end it all")
        self.assertEqual(result["intervention_priority"], "immediate")

    def test_low_risk_low_priority(self):
        emotions = {"joy": 0.8, "sadness": 0.05, "fear": 0.05, "anger": 0.05, "surprise": 0.05}
        risk     = {
            "risk_level": "low",
            "score": 0.05,
            "confidence": 0.9,
            "raw_scores": {"normal": 0.95},
        }
        det    = _make_det(emotions, risk)
        result = det.comprehensive_analysis("Explain gradient descent please")
        self.assertEqual(result["intervention_priority"], "low")

    def test_summary_is_string(self):
        emotions = {"joy": 0.5, "sadness": 0.3, "fear": 0.1, "anger": 0.0, "surprise": 0.1}
        risk     = {"risk_level": "moderate", "score": 0.45, "confidence": 0.7, "raw_scores": {}}
        det      = _make_det(emotions, risk)
        result   = det.comprehensive_analysis("I am struggling a bit")
        self.assertIsInstance(result["summary"], str)
        self.assertGreater(len(result["summary"]), 0)


class TestEnsembleDetectorHelpers(unittest.TestCase):
    """Unit tests for static helper methods."""

    def test_intervention_priority_severe(self):
        p = EnsembleAffectiveDetector._intervention_priority(
            "severe", {"anxiety": 0.1, "depression": 0.1, "positive_affect": 0.5}
        )
        self.assertEqual(p, "immediate")

    def test_intervention_priority_moderate_high_anxiety(self):
        p = EnsembleAffectiveDetector._intervention_priority(
            "moderate", {"anxiety": 0.75, "depression": 0.2, "positive_affect": 0.3}
        )
        self.assertEqual(p, "high")

    def test_intervention_priority_low_no_signal(self):
        p = EnsembleAffectiveDetector._intervention_priority(
            "low", {"anxiety": 0.1, "depression": 0.1, "positive_affect": 0.8}
        )
        self.assertEqual(p, "low")

    def test_extract_key_indicators_clamp(self):
        emotions = {"fear": 1.0, "sadness": 1.0, "joy": 1.0, "surprise": 1.0}
        risk     = {"raw_scores": {"anxiety": 1.0, "depression": 1.0}}
        ind = EnsembleAffectiveDetector._extract_key_indicators(emotions, risk)
        for v in ind.values():
            self.assertLessEqual(v, 1.0)
            self.assertGreaterEqual(v, 0.0)


if __name__ == "__main__":
    unittest.main()
