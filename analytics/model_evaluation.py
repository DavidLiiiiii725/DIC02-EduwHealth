# analytics/model_evaluation.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  Model Evaluation Utilities
#
# Provides tools for evaluating and comparing:
#   - The emotion detection model (EmotionDetector)
#   - The mental-health risk detector (MentalHealthRiskDetector)
#   - The combined risk pipeline (human-weighted vs ML-based)
#
# Designed to run offline against labelled test datasets.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class ModelEvaluator:
    """
    Evaluate emotion-detection and risk-assessment models.

    All evaluation methods accept a ``test_data`` list of dicts with
    consistent keys so that datasets can be swapped without code changes.

    Parameters
    ----------
    emotion_detector : EmotionDetector | None
        Pre-instantiated emotion detector.  If None, the evaluator
        imports and instantiates it lazily on first use.
    risk_detector : MentalHealthRiskDetector | None
        Pre-instantiated risk detector.  Same lazy-loading behaviour.
    """

    def __init__(
        self,
        emotion_detector=None,
        risk_detector=None,
    ):
        self._emotion_detector = emotion_detector
        self._risk_detector    = risk_detector

    # ── Lazy loaders ──────────────────────────────────────────────

    @property
    def emotion_detector(self):
        if self._emotion_detector is None:
            from affect.emotion_model import EmotionDetector
            self._emotion_detector = EmotionDetector()
        return self._emotion_detector

    @property
    def risk_detector(self):
        if self._risk_detector is None:
            from affect.mental_health_classifier import MentalHealthRiskDetector
            self._risk_detector = MentalHealthRiskDetector()
        return self._risk_detector

    # ── Emotion model evaluation ──────────────────────────────────

    def evaluate_emotion_model(
        self,
        test_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate the emotion detection model against labelled data.

        Parameters
        ----------
        test_data : list of dicts
            Each dict must contain:
              "text"           – input string
              "expected_label" – the dominant expected emotion label

        Returns
        -------
        {
            "accuracy":    float,
            "n_correct":   int,
            "n_total":     int,
            "per_label":   Dict[label, {"correct": int, "total": int}],
            "errors":      list of {"text", "expected", "predicted"},
        }
        """
        n_correct = 0
        per_label: Dict[str, Dict[str, int]] = {}
        errors: List[Dict[str, str]] = []

        for item in test_data:
            text     = item["text"]
            expected = item["expected_label"].lower()

            scores    = self.emotion_detector.detect(text)
            predicted = max(scores, key=scores.get).lower()

            # Per-label tracking
            if expected not in per_label:
                per_label[expected] = {"correct": 0, "total": 0}
            per_label[expected]["total"] += 1

            if predicted == expected:
                n_correct += 1
                per_label[expected]["correct"] += 1
            else:
                errors.append({
                    "text":      text,
                    "expected":  expected,
                    "predicted": predicted,
                })

        n_total  = len(test_data)
        accuracy = n_correct / n_total if n_total else 0.0

        return {
            "accuracy":  round(accuracy, 4),
            "n_correct": n_correct,
            "n_total":   n_total,
            "per_label": per_label,
            "errors":    errors,
        }

    # ── Risk model evaluation ─────────────────────────────────────

    def evaluate_risk_model(
        self,
        test_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate the risk detector with special emphasis on false negatives
        (high/severe cases incorrectly classified as low/moderate).

        Parameters
        ----------
        test_data : list of dicts
            Each dict must contain:
              "text"               – input string
              "expected_risk_level" – "low" | "moderate" | "high" | "severe"

        Returns
        -------
        {
            "accuracy":          float,
            "fn_rate":           float,   # false-negative rate for high/severe
            "fp_rate":           float,   # false-positive rate (low flagged as high)
            "n_total":           int,
            "confusion":         Dict,
            "high_risk_missed":  list of missed high/severe cases,
        }
        """
        _HIGH = {"high", "severe"}
        _LOW  = {"low", "moderate"}

        n_correct = 0
        confusion: Dict[str, Dict[str, int]] = {}
        high_risk_missed: List[Dict[str, str]] = []

        tp = fp = tn = fn = 0

        for item in test_data:
            text     = item["text"]
            expected = item["expected_risk_level"].lower()

            result    = self.risk_detector.assess_risk(text)
            predicted = result["risk_level"]

            # Accuracy
            if predicted == expected:
                n_correct += 1

            # Confusion matrix
            if expected not in confusion:
                confusion[expected] = {}
            confusion[expected][predicted] = confusion[expected].get(predicted, 0) + 1

            # Binary high-risk detection metrics
            is_truly_high  = expected   in _HIGH
            is_pred_high   = predicted  in _HIGH

            if is_truly_high and is_pred_high:
                tp += 1
            elif is_truly_high and not is_pred_high:
                fn += 1
                high_risk_missed.append({
                    "text":      text,
                    "expected":  expected,
                    "predicted": predicted,
                    "score":     result.get("score", 0),
                })
            elif not is_truly_high and is_pred_high:
                fp += 1
            else:
                tn += 1

        n_total  = len(test_data)
        accuracy = n_correct / n_total if n_total else 0.0
        fn_rate  = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fp_rate  = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return {
            "accuracy":         round(accuracy, 4),
            "fn_rate":          round(fn_rate,  4),
            "fp_rate":          round(fp_rate,  4),
            "n_total":          n_total,
            "confusion":        confusion,
            "high_risk_missed": high_risk_missed,
        }

    # ── Model comparison ──────────────────────────────────────────

    def compare_models(
        self,
        human_weighted_results: List[Dict[str, Any]],
        ml_model_results:       List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compare predictions from the legacy human-weighted pipeline and
        the new ML-based risk detector.

        Parameters
        ----------
        human_weighted_results : list of dicts
            Each dict: {"text": str, "level": str, "score": float}
        ml_model_results : list of dicts
            Each dict: {"text": str, "level": str, "score": float}

        Returns
        -------
        {
            "agreement_rate":         float,   # fraction where both agree on level
            "high_risk_only_human":   int,     # cases flagged only by human weighting
            "high_risk_only_ml":      int,     # cases flagged only by ML model
            "avg_score_delta":        float,   # mean(|score_human - score_ml|)
            "disagreements":          list,
        }
        """
        if len(human_weighted_results) != len(ml_model_results):
            raise ValueError(
                "human_weighted_results and ml_model_results must have the same length."
            )

        _HIGH = {"high", "severe"}
        agrees = 0
        only_human = 0
        only_ml    = 0
        score_deltas: List[float] = []
        disagreements: List[Dict[str, Any]] = []

        for hw, ml in zip(human_weighted_results, ml_model_results):
            hw_level = hw.get("level", "low")
            ml_level = ml.get("level", "low")
            hw_score = hw.get("score", 0.0)
            ml_score = ml.get("score", 0.0)

            score_deltas.append(abs(hw_score - ml_score))

            if hw_level == ml_level:
                agrees += 1
            else:
                disagreements.append({
                    "text":     hw.get("text", ""),
                    "human":    hw_level,
                    "ml":       ml_level,
                    "hw_score": hw_score,
                    "ml_score": ml_score,
                })
                if hw_level in _HIGH and ml_level not in _HIGH:
                    only_human += 1
                elif ml_level in _HIGH and hw_level not in _HIGH:
                    only_ml += 1

        n = len(human_weighted_results)
        return {
            "agreement_rate":       round(agrees / n, 4) if n else 0.0,
            "high_risk_only_human": only_human,
            "high_risk_only_ml":    only_ml,
            "avg_score_delta":      round(sum(score_deltas) / n, 4) if n else 0.0,
            "disagreements":        disagreements,
        }
