# analytics/risk_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

from analystics.feature_extractor import FeatureExtractorLLM, ExtractedFeatures


@dataclass
class RiskResult:
    score: float       # 0..1
    level: str         # low/medium/high
    reasons: Dict[str, float]


class RiskModelLLM:
    """
    Risk model using LLM-extracted features (no manual keyword lists).
    """

    def __init__(self, feature_extractor: FeatureExtractorLLM):
        self.fx = feature_extractor

    def predict(self, state: Dict[str, Any]) -> RiskResult:
        feats = self.fx.extract(state)
        score, reasons = self._score(feats)
        level = self._level(score, feats)
        return RiskResult(score=score, level=level, reasons=reasons)

    def _score(self, f: ExtractedFeatures) -> Tuple[float, Dict[str, float]]:
        # Strongest signal: self-harm risk
        # Then: hopelessness + urgency + overwhelm/panic + (sadness+fear) + impairment + intensity
        score = 0.0
        score += 0.78 * f.self_harm_risk
        score += 0.22 * f.hopelessness
        score += 0.18 * f.urgency
        score += 0.14 * f.overwhelm
        score += 0.14 * f.panic
        score += 0.18 * (0.55 * f.sadness + 0.45 * f.fear)
        score += 0.10 * f.functional_impairment
        score += 0.08 * f.intensity

        # small uncertainty bump if no RAG context exists (optional)
        if f.rag_empty > 0.5:
            score += 0.02

        # clamp to [0,1]
        score = max(0.0, min(1.0, score))

        reasons = f.to_dict()
        return score, reasons

    def _level(self, score: float, f: ExtractedFeatures) -> str:
        # hard rule: if self_harm_risk high, escalate
        if f.self_harm_risk >= 0.65:
            return "high"
        if score >= 0.75:
            return "high"
        if score >= 0.45:
            return "medium"
        return "low"
