# core/cognitive_state.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  Cognitive State Machine (CSM)
#
# Maintains a continuous 4-dimensional state vector per session:
#   working_memory_load  [0, 1]
#   motivation_level     [0, 1]
#   affect_valence      [-1, 1]
#   cognitive_fatigue    [0, 1]
#
# Uses EWMA smoothing so a single odd message doesn't flip the state.
# Exposes trajectory analysis for proactive intervention.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from config import (
    CSM_ALPHA,
    CSM_WM_OVERLOAD_THRESHOLD,
    CSM_MOTIVATION_LOW_THRESHOLD,
    CSM_FATIGUE_HIGH_THRESHOLD,
    CSM_AFFECT_NEG_THRESHOLD,
    CSM_TRAJECTORY_WINDOW,
    CSM_MOTIVATION_SLOPE_TRIGGER,
)


@dataclass
class CognitiveStateVector:
    working_memory_load: float = 0.30   # start at a neutral-low baseline
    motivation_level:    float = 0.70
    affect_valence:      float = 0.10
    cognitive_fatigue:   float = 0.10

    def clamp(self) -> "CognitiveStateVector":
        self.working_memory_load = max(0.0, min(1.0, self.working_memory_load))
        self.motivation_level    = max(0.0, min(1.0, self.motivation_level))
        self.affect_valence      = max(-1.0, min(1.0, self.affect_valence))
        self.cognitive_fatigue   = max(0.0, min(1.0, self.cognitive_fatigue))
        return self

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class CognitiveStateSnapshot:
    """One recorded point in the session history."""
    turn: int
    timestamp: float
    state: CognitiveStateVector
    signals: Dict[str, Any] = field(default_factory=dict)


class CognitiveStateMachine:
    """
    Stateful per-session cognitive state tracker.

    Usage:
        csm = CognitiveStateMachine()
        updated_state = csm.update(signals, turn=1)
        flags = csm.get_intervention_flags()
        trajectory = csm.get_trajectory_flags()
    """

    def __init__(self, alpha: float = CSM_ALPHA):
        self.alpha = alpha
        self.current = CognitiveStateVector()
        self.history: List[CognitiveStateSnapshot] = []
        self._session_start = time.time()
        self._baseline_attention_minutes: float = 15.0  # default; updated from LearnerModel

    # ── Public API ────────────────────────────────────────────────

    def set_baseline(self, attention_minutes: float) -> None:
        """Inject per-learner attention baseline from the LearnerModel."""
        self._baseline_attention_minutes = max(1.0, attention_minutes)

    def update(self, signals: Dict[str, float], turn: int) -> CognitiveStateVector:
        """
        Compute new state estimate from LLM-extracted signals, apply EWMA.

        Expected signal keys (all optional, default 0.0):
            wm_load_estimate      – 0..1 from FeatureExtractor
            motivation_estimate   – 0..1
            affect_estimate       – -1..1
            error_rate            – 0..1  (fraction of recent turns with errors/confusion)
        """
        # ── 1. Point estimate from signals ──────────────────────
        session_minutes = (time.time() - self._session_start) / 60.0
        fatigue_from_time = min(1.0, session_minutes / self._baseline_attention_minutes)

        point = CognitiveStateVector(
            working_memory_load = self._get(signals, "wm_load_estimate",   0.30),
            motivation_level    = self._get(signals, "motivation_estimate", 0.70),
            affect_valence      = self._get(signals, "affect_estimate",     0.10),
            cognitive_fatigue   = max(
                fatigue_from_time,
                self._get(signals, "fatigue_estimate", 0.10)
            ),
        ).clamp()

        # ── 2. EWMA smoothing ────────────────────────────────────
        prev = self.current
        self.current = CognitiveStateVector(
            working_memory_load = self.alpha * point.working_memory_load + (1 - self.alpha) * prev.working_memory_load,
            motivation_level    = self.alpha * point.motivation_level    + (1 - self.alpha) * prev.motivation_level,
            affect_valence      = self.alpha * point.affect_valence      + (1 - self.alpha) * prev.affect_valence,
            cognitive_fatigue   = self.alpha * point.cognitive_fatigue   + (1 - self.alpha) * prev.cognitive_fatigue,
        ).clamp()

        # ── 3. Record history ────────────────────────────────────
        self.history.append(CognitiveStateSnapshot(
            turn=turn,
            timestamp=time.time(),
            state=CognitiveStateVector(**asdict(self.current)),
            signals=signals,
        ))

        return self.current

    def get_intervention_flags(self) -> Dict[str, bool]:
        """Current-state flags consumed by the Orchestrator."""
        s = self.current
        return {
            "wm_overload":        s.working_memory_load > CSM_WM_OVERLOAD_THRESHOLD,
            "motivation_low":     s.motivation_level    < CSM_MOTIVATION_LOW_THRESHOLD,
            "affect_negative":    s.affect_valence      < CSM_AFFECT_NEG_THRESHOLD,
            "fatigue_high":       s.cognitive_fatigue   > CSM_FATIGUE_HIGH_THRESHOLD,
        }

    def get_trajectory_flags(self) -> Dict[str, bool]:
        """
        Slope-based proactive flags: look at last N turns.
        Returns True when a deteriorating trend is detected BEFORE threshold breach.
        """
        window = self.history[-CSM_TRAJECTORY_WINDOW:]
        if len(window) < 2:
            return {"motivation_declining": False, "wm_climbing": False}

        mot_vals = [s.state.motivation_level    for s in window]
        wm_vals  = [s.state.working_memory_load for s in window]

        mot_slope = self._slope(mot_vals)
        wm_slope  = self._slope(wm_vals)

        return {
            "motivation_declining": mot_slope < CSM_MOTIVATION_SLOPE_TRIGGER,
            "wm_climbing":          wm_slope  > 0.08,
        }

    def state_dict(self) -> Dict[str, float]:
        return self.current.to_dict()

    # ── Private helpers ───────────────────────────────────────────

    @staticmethod
    def _get(d: Dict, key: str, default: float) -> float:
        try:
            return float(d.get(key, default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _slope(values: List[float]) -> float:
        """Simple linear regression slope over a list."""
        n = len(values)
        if n < 2:
            return 0.0
        xs = list(range(n))
        x_mean = sum(xs) / n
        y_mean = sum(values) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
        den = sum((x - x_mean) ** 2 for x in xs)
        return num / den if den != 0 else 0.0
