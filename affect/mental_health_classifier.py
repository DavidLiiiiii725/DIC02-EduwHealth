# affect/mental_health_classifier.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  Mental Health Risk Detector
#
# Uses a pre-trained mental health classification model (or a
# rule-based fallback when the model is not available) to assess
# the psychological risk level expressed in a learner's text.
#
# Risk levels: low | moderate | high | severe
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

from typing import Dict, Any

from config import MENTAL_HEALTH_MODEL, RISK_THRESHOLDS


class MentalHealthRiskDetector:
    """
    Uses a pre-trained mental health classification model to assess
    risk level from free-text input.

    Falls back to a keyword-and-heuristic approach if the model
    cannot be loaded (e.g., offline / CPU-only environments).

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier used for inference.
    _model : pipeline | None
        Loaded transformers pipeline, or None if unavailable.
    """

    def __init__(self, model_name: str = MENTAL_HEALTH_MODEL):
        self.model_name = model_name
        self._model = None
        self._load_model()

    # ── Model loading ─────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Attempt to load the transformers pipeline.
        On failure, fall back to the heuristic scorer.
        """
        try:
            from transformers import pipeline  # type: ignore
            print(f"[APU] Loading mental health model: {self.model_name} …")
            self._model = pipeline(
                "text-classification",
                model=self.model_name,
                return_all_scores=True,
                truncation=True,
                max_length=512,
            )
            print("[APU] Mental health model loaded.")
        except Exception as exc:  # noqa: BLE001
            print(f"[APU] Mental health model unavailable ({exc}). "
                  "Using heuristic fallback.")
            self._model = None

    # ── Public API ────────────────────────────────────────────────

    def assess_risk(self, text: str) -> Dict[str, Any]:
        """
        Assess the mental-health risk level of the given text.

        Parameters
        ----------
        text : str
            Free-form user message.

        Returns
        -------
        dict with keys:
            risk_level  – "low" | "moderate" | "high" | "severe"
            score       – float 0..1 (overall risk probability)
            confidence  – float 0..1 (model certainty)
            raw_scores  – dict of label → score from the model
            method      – "model" | "heuristic"
        """
        if self._model is not None:
            return self._model_assess(text)
        return self._heuristic_assess(text)

    # ── Model-based assessment ────────────────────────────────────

    def _model_assess(self, text: str) -> Dict[str, Any]:
        try:
            results = self._model(text)[0]
            raw = {r["label"]: r["score"] for r in results}

            # Aggregate into a single risk score.  Different models use
            # different label sets; we map them conservatively.
            score = self._aggregate_model_score(raw)
            level = self._score_to_level(score)

            # Confidence = max single-label score
            confidence = max(raw.values()) if raw else 0.0

            return {
                "risk_level": level,
                "score": round(score, 4),
                "confidence": round(confidence, 4),
                "raw_scores": raw,
                "method": "model",
            }
        except Exception as exc:  # noqa: BLE001
            print(f"[APU] Model inference failed ({exc}); falling back to heuristic.")
            return self._heuristic_assess(text)

    @staticmethod
    def _aggregate_model_score(raw: Dict[str, float]) -> float:
        """
        Map model label scores to a unified 0..1 risk score.

        Supports several common mental-health model label formats:
          - "depression" / "anxiety" / "suicidal" / "stress"  (multi-label)
          - "LABEL_0" (not-depressed) / "LABEL_1" (depressed)
          - "POSITIVE" / "NEGATIVE"
        """
        lc = {k.lower(): v for k, v in raw.items()}

        # Direct risk labels
        risk_keys = [
            "suicidal", "severe", "depression", "anxiety",
            "crisis", "distress", "label_1", "positive",
        ]
        safe_keys = ["normal", "not depressed", "label_0", "negative", "low"]

        risk_sum = sum(lc.get(k, 0.0) for k in risk_keys)
        safe_sum = sum(lc.get(k, 0.0) for k in safe_keys)

        if risk_sum + safe_sum > 0:
            return min(1.0, risk_sum / (risk_sum + safe_sum))

        # Fallback: treat as unknown → moderate-low
        return 0.3

    # ── Heuristic fallback ────────────────────────────────────────

    @staticmethod
    def _heuristic_assess(text: str) -> Dict[str, Any]:
        """
        Lightweight keyword heuristic used when the model is unavailable.
        """
        tl = text.lower()

        severe_kw = [
            "suicide", "kill myself", "end my life", "want to die",
            "no reason to live", "self-harm", "self harm", "cut myself",
        ]
        high_kw = [
            "hopeless", "worthless", "can't go on", "cannot cope",
            "everything is falling apart", "no one cares",
            "i give up", "i quit everything",
        ]
        moderate_kw = [
            "overwhelmed", "anxious", "depressed", "exhausted",
            "stressed", "can't sleep", "not eating", "falling behind",
            "nobody understands", "feel like a failure",
        ]

        if any(k in tl for k in severe_kw):
            score, level = 0.92, "severe"
        elif any(k in tl for k in high_kw):
            score, level = 0.72, "high"
        elif any(k in tl for k in moderate_kw):
            score, level = 0.48, "moderate"
        else:
            score, level = 0.12, "low"

        return {
            "risk_level": level,
            "score": score,
            "confidence": 0.6,   # heuristic is less certain
            "raw_scores": {},
            "method": "heuristic",
        }

    # ── Helper ────────────────────────────────────────────────────

    @staticmethod
    def _score_to_level(score: float) -> str:
        for level, (lo, hi) in RISK_THRESHOLDS.items():
            if lo <= score < hi:
                return level
        return "severe"
