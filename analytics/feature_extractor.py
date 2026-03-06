# analytics/feature_extractor.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  Feature Extractor (Extended)
#
# Adds cognitive state signals on top of the original risk features:
#   wm_load_estimate      – working memory load proxy
#   motivation_estimate   – motivation / engagement level
#   affect_estimate       – valence (-1..1)
#   fatigue_estimate      – session fatigue proxy
#   negative_attribution  – "I'm bad at this" style attribution
#   topic_shift           – sudden topic change (ADHD drift signal)
#   task_avoidance        – explicit avoidance language
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExtractedFeatures:
    # ── Original risk features ────────────────────────────────────
    sadness:               float = 0.0
    fear:                  float = 0.0
    anger:                 float = 0.0
    joy:                   float = 0.0
    self_harm_risk:        float = 0.0
    hopelessness:          float = 0.0
    overwhelm:             float = 0.0
    panic:                 float = 0.0
    functional_impairment: float = 0.0
    urgency:               float = 0.0
    intensity:             float = 0.0
    negation_or_denial:    float = 0.0
    rag_empty:             float = 0.0
    rag_len_norm:          float = 0.0
    user_len_norm:         float = 0.0

    # ── NEW: Cognitive state signals ──────────────────────────────
    wm_load_estimate:     float = 0.30   # 0..1  working memory load
    motivation_estimate:  float = 0.70   # 0..1  engagement / motivation
    affect_estimate:      float = 0.10   # -1..1 negative ↔ positive
    fatigue_estimate:     float = 0.10   # 0..1  cognitive fatigue
    negative_attribution: float = 0.0    # 0..1  "I can't do this" fixed mindset
    topic_shift:          float = 0.0    # 0..1  abrupt topic change (ADHD drift)
    task_avoidance:       float = 0.0    # 0..1  avoidance language

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()

    def cognitive_signals(self) -> Dict[str, float]:
        return {
            "wm_load_estimate":     self.wm_load_estimate,
            "motivation_estimate":  self.motivation_estimate,
            "affect_estimate":      self.affect_estimate,
            "fatigue_estimate":     self.fatigue_estimate,
            "negative_attribution": self.negative_attribution,
            "topic_shift":          self.topic_shift,
            "task_avoidance":       self.task_avoidance,
        }


class FeatureExtractorLLM:
    """
    Single LLM call that extracts BOTH risk features AND cognitive state signals.
    Backward compatible: existing RiskModelLLM still works unchanged.
    """

    # Extended schema sent to the LLM
    _SCHEMA = {
        # ── original ──
        "sadness":               "float 0..1",
        "fear":                  "float 0..1",
        "anger":                 "float 0..1",
        "joy":                   "float 0..1",
        "self_harm_risk":        "float 0..1 – signals of self-harm or suicidal ideation",
        "hopelessness":          "float 0..1",
        "overwhelm":             "float 0..1",
        "panic":                 "float 0..1",
        "functional_impairment": "float 0..1 – sleep/eat/focus disruption",
        "urgency":               "float 0..1 – urgency to seek support",
        "intensity":             "float 0..1 – emotional intensity",
        "negation_or_denial":    "float 0..1 – explicitly denies self-harm intent",
        # ── NEW cognitive signals ──
        "wm_load_estimate":      "float 0..1 – how cognitively overloaded the message feels (short fragmented sentences, confusion, errors → high)",
        "motivation_estimate":   "float 0..1 – how engaged or willing the learner seems (1=very motivated)",
        "affect_estimate":       "float -1..1 – overall valence: -1=very negative, 0=neutral, +1=very positive",
        "fatigue_estimate":      "float 0..1 – signs of tiredness or reduced effort (vague answers, short replies)",
        "negative_attribution":  "float 0..1 – fixed-mindset or self-defeating attribution ('I'm stupid', 'I always fail')",
        "topic_shift":           "float 0..1 – abrupt unexplained change of topic (ADHD attention drift signal)",
        "task_avoidance":        "float 0..1 – explicit avoidance language ('I don't want to', 'can we skip this')",
    }

    def __init__(self, llm_client, *, max_retries: int = 2):
        self.llm = llm_client
        self.max_retries = max_retries

    def extract(self, state: Dict[str, Any]) -> ExtractedFeatures:
        user_input  = (state.get("user_input")  or "").strip()
        rag_context = (state.get("rag_context") or "").strip()

        # Pre-compute non-lexical meta
        user_len_norm = self._length_norm(user_input)
        rag_len_norm  = self._length_norm(rag_context)
        rag_empty     = 1.0 if not rag_context else 0.0

        system = (
            "You are a strict feature extractor for an educational tutor safety and adaptation system.\n"
            "Task: Given a student's message, output ONLY valid JSON with numeric fields in the specified ranges.\n"
            "Do not output any other text. Do not diagnose; only estimate indicators.\n"
        )
        user_prompt = (
            "Return JSON with the following keys:\n"
            f"{json.dumps(self._SCHEMA, ensure_ascii=False, indent=2)}\n\n"
            "Student message:\n"
            f"{user_input}\n"
        )

        raw  = self._call_with_retries(system, user_prompt)
        data = self._safe_json_load(raw)

        c01  = self._clamp01
        c11  = self._clamp11

        feats = ExtractedFeatures(
            # original
            sadness               = c01(data.get("sadness", 0.0)),
            fear                  = c01(data.get("fear", 0.0)),
            anger                 = c01(data.get("anger", 0.0)),
            joy                   = c01(data.get("joy", 0.0)),
            self_harm_risk        = c01(data.get("self_harm_risk", 0.0)),
            hopelessness          = c01(data.get("hopelessness", 0.0)),
            overwhelm             = c01(data.get("overwhelm", 0.0)),
            panic                 = c01(data.get("panic", 0.0)),
            functional_impairment = c01(data.get("functional_impairment", 0.0)),
            urgency               = c01(data.get("urgency", 0.0)),
            intensity             = c01(data.get("intensity", 0.0)),
            negation_or_denial    = c01(data.get("negation_or_denial", 0.0)),
            rag_empty             = rag_empty,
            rag_len_norm          = rag_len_norm,
            user_len_norm         = user_len_norm,
            # NEW cognitive signals
            wm_load_estimate      = c01(data.get("wm_load_estimate",     0.30)),
            motivation_estimate   = c01(data.get("motivation_estimate",  0.70)),
            affect_estimate       = c11(data.get("affect_estimate",      0.10)),
            fatigue_estimate      = c01(data.get("fatigue_estimate",     0.10)),
            negative_attribution  = c01(data.get("negative_attribution", 0.0)),
            topic_shift           = c01(data.get("topic_shift",          0.0)),
            task_avoidance        = c01(data.get("task_avoidance",       0.0)),
        )

        # Safety tempering: strong denial reduces self-harm signal
        if feats.self_harm_risk > 0.6 and feats.negation_or_denial > 0.7:
            feats.self_harm_risk = max(0.2, feats.self_harm_risk * 0.5)

        return feats

    # ── Private helpers ───────────────────────────────────────────

    def _call_with_retries(self, system: str, user: str) -> str:
        last = ""
        for _ in range(self.max_retries + 1):
            out = self.llm.chat(system=system, user=user, temperature=0.0)
            last = out.strip()
            if last.startswith("{") and last.rstrip().endswith("}"):
                return last
            user = "IMPORTANT: Output ONLY JSON. No markdown, no explanation.\n" + user
        return last

    @staticmethod
    def _safe_json_load(text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(text[s:e+1])
            except Exception:
                pass
        return {}

    @staticmethod
    def _clamp01(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            return 0.0
        return 0.0 if v != v else max(0.0, min(1.0, v))

    @staticmethod
    def _clamp11(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            return 0.0
        return 0.0 if v != v else max(-1.0, min(1.0, v))

    @staticmethod
    def _length_norm(text: str) -> float:
        n = max(0, len(text or ""))
        return float(min(1.0, math.log1p(n) / math.log1p(2000)))
