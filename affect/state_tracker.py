# affect/state_tracker.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  Affective State Tracker (Extended)
#
# Tracks per-session:
#   - emotion score history
#   - risk-level history (from MentalHealthRiskDetector)
#   - detected / flagged learning disability types
#   - executed intervention records
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class EmotionalState:
    """
    Tracks emotional and risk state throughout a learning session.

    Extended attributes
    -------------------
    risk_history : list of dicts
        Each entry: {"timestamp", "risk_level", "risk_score",
                     "emotions", "context"}.
    disability_flags : set of str
        Learning disability types observed/flagged this session
        (e.g. "ADHD", "anxiety", "executive_function_deficit").
    intervention_log : list of dicts
        Each entry: {"timestamp", "type", "strategy", "triggered_by"}.
    """

    def __init__(self):
        self.history:          List[Dict[str, Any]] = []
        self.risk_history:     List[Dict[str, Any]] = []
        self.disability_flags: set                  = set()
        self.intervention_log: List[Dict[str, Any]] = []

    # ── Emotion tracking ──────────────────────────────────────────

    def update(self, emotion_scores: Dict[str, float]) -> None:
        """Append a new emotion-score snapshot."""
        self.history.append(emotion_scores)

    def is_distressed(self) -> bool:
        if not self.history:
            return False
        last = self.history[-1]
        return last.get("sadness", 0) > 0.4 or last.get("fear", 0) > 0.4

    # ── Risk tracking ─────────────────────────────────────────────

    def update_risk(
        self,
        risk_level: str,
        risk_score: float,
        emotions: Optional[Dict[str, float]] = None,
        context: str = "",
    ) -> None:
        """
        Record a new risk assessment snapshot.

        Parameters
        ----------
        risk_level  : "low" | "moderate" | "high" | "severe"
        risk_score  : float 0..1
        emotions    : latest emotion scores (optional, for richer history)
        context     : brief description of what triggered the assessment
        """
        self.risk_history.append({
            "timestamp":  time.time(),
            "risk_level": risk_level,
            "risk_score": risk_score,
            "emotions":   emotions or {},
            "context":    context,
        })

    def current_risk_level(self) -> str:
        """Return the most recent risk level, or "low" if no history."""
        if not self.risk_history:
            return "low"
        return self.risk_history[-1]["risk_level"]

    def is_high_risk(self) -> bool:
        return self.current_risk_level() in ("high", "severe")

    # ── Learning disability flags ─────────────────────────────────

    def flag_disability(self, disability_type: str) -> None:
        """Mark a learning disability as observed this session."""
        self.disability_flags.add(disability_type.lower())

    def has_disability(self, disability_type: str) -> bool:
        return disability_type.lower() in self.disability_flags

    def get_disability_flags(self) -> List[str]:
        return sorted(self.disability_flags)

    # ── Intervention records ──────────────────────────────────────

    def log_intervention(
        self,
        intervention_type: str,
        strategy: str,
        triggered_by: str = "",
    ) -> None:
        """
        Record that an intervention was executed.

        Parameters
        ----------
        intervention_type : e.g. "task_decomposition", "anxiety_reduction"
        strategy          : human-readable description
        triggered_by      : reason / signal that triggered it
        """
        self.intervention_log.append({
            "timestamp":         time.time(),
            "type":              intervention_type,
            "strategy":          strategy,
            "triggered_by":      triggered_by,
        })

    def get_intervention_log(self) -> List[Dict[str, Any]]:
        return list(self.intervention_log)
