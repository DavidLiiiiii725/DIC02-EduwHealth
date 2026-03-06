# affect/ensemble_detector.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  Ensemble Affective Detector
#
# Fuses:
#   1. EmotionDetector  (fine-grained emotion labels)
#   2. MentalHealthRiskDetector  (psychological risk level)
#
# Produces a comprehensive analysis that downstream agents and the
# InterventionAgent can act on.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Any, Dict, List

from affect.emotion_model           import EmotionDetector
from affect.mental_health_classifier import MentalHealthRiskDetector


class EnsembleAffectiveDetector:
    """
    Fuses emotion detection and mental-health risk assessment into a
    single comprehensive analysis object.

    Attributes
    ----------
    emotion_detector : EmotionDetector
        Detects fine-grained emotions (joy, sadness, fear, anger, …).
    risk_detector : MentalHealthRiskDetector
        Classifies psychological risk level (low → severe).
    """

    def __init__(self):
        print("[APU] Initialising EnsembleAffectiveDetector …")
        self.emotion_detector = EmotionDetector()
        self.risk_detector    = MentalHealthRiskDetector()
        print("[APU] EnsembleAffectiveDetector ready.")

    # ── Public API ────────────────────────────────────────────────

    def comprehensive_analysis(self, text: str) -> Dict[str, Any]:
        """
        Run both detectors and return a fused analysis dict.

        Returns
        -------
        {
            "emotions":              Dict[str, float],
            "risk":                  Dict[str, Any],   # from MentalHealthRiskDetector
            "key_indicators": {
                "anxiety":           float,
                "depression":        float,
                "positive_affect":   float,
            },
            "intervention_priority": str,   # "immediate" | "high" | "medium" | "low"
            "summary":               str,
        }
        """
        emotions = self.emotion_detector.detect(text)
        risk     = self.risk_detector.assess_risk(text)

        indicators  = self._extract_key_indicators(emotions, risk)
        priority    = self._intervention_priority(risk["risk_level"], indicators)
        summary     = self._build_summary(risk["risk_level"], emotions, indicators)

        return {
            "emotions":              emotions,
            "risk":                  risk,
            "key_indicators":        indicators,
            "intervention_priority": priority,
            "summary":               summary,
        }

    # ── Internals ─────────────────────────────────────────────────

    @staticmethod
    def _extract_key_indicators(
        emotions: Dict[str, float],
        risk: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Distil the fused signal into three clinically meaningful scores.

        anxiety        – weighted blend of fear + model anxiety signal
        depression     – weighted blend of sadness + model depression
        positive_affect – joy + surprise (positive flavour)
        """
        fear    = emotions.get("fear",    0.0)
        sadness = emotions.get("sadness", 0.0)
        joy     = emotions.get("joy",     0.0)
        surprise = emotions.get("surprise", 0.0)

        model_raw = risk.get("raw_scores", {})
        lc_raw    = {k.lower(): v for k, v in model_raw.items()}

        model_anxiety    = lc_raw.get("anxiety",    0.0)
        model_depression = lc_raw.get("depression", 0.0)

        anxiety         = min(1.0, 0.6 * fear    + 0.4 * model_anxiety)
        depression      = min(1.0, 0.6 * sadness + 0.4 * model_depression)
        positive_affect = min(1.0, 0.7 * joy     + 0.3 * surprise)

        return {
            "anxiety":        round(anxiety,        4),
            "depression":     round(depression,     4),
            "positive_affect": round(positive_affect, 4),
        }

    @staticmethod
    def _intervention_priority(
        risk_level: str,
        indicators: Dict[str, float],
    ) -> str:
        if risk_level in ("severe", "high"):
            return "immediate"
        if risk_level == "moderate":
            if indicators.get("anxiety", 0) > 0.6 or indicators.get("depression", 0) > 0.6:
                return "high"
            return "medium"
        # low risk but noticeable anxiety/depression still warrants attention
        if indicators.get("anxiety", 0) > 0.5 or indicators.get("depression", 0) > 0.5:
            return "medium"
        return "low"

    @staticmethod
    def _build_summary(
        risk_level: str,
        emotions: Dict[str, float],
        indicators: Dict[str, float],
    ) -> str:
        dominant = max(emotions, key=emotions.get) if emotions else "unknown"
        anxiety  = indicators.get("anxiety", 0)
        depress  = indicators.get("depression", 0)
        pos      = indicators.get("positive_affect", 0)

        parts: List[str] = [
            f"Risk level: {risk_level}.",
            f"Dominant emotion: {dominant}.",
        ]
        if anxiety > 0.5:
            parts.append(f"Elevated anxiety ({anxiety:.2f}).")
        if depress > 0.5:
            parts.append(f"Elevated depression signal ({depress:.2f}).")
        if pos > 0.5:
            parts.append(f"Positive affect present ({pos:.2f}).")

        return " ".join(parts)
