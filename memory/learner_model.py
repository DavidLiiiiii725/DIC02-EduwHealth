# memory/learner_model.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  Persistent Learner Model (Cognitive Fingerprint)
#
# Stores and updates a per-learner profile including:
#   - LD profile (type + severity)
#   - Cognitive baseline estimates
#   - Intervention history (what worked / what failed)
#   - Metacognitive development log
#   - Session trajectory
#
# Profiles are persisted as JSON files in LEARNER_MODEL_PATH.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from config import LEARNER_MODEL_PATH, SCAFFOLD_MINIMAL_MAX, SCAFFOLD_MODERATE_MAX


# ── Data classes ──────────────────────────────────────────────────

@dataclass
class LDProfile:
    """Learning disability characterisation for one learner."""
    confirmed:  List[str] = field(default_factory=list)   # e.g. ["executive_function", "adhd"]
    suspected:  List[str] = field(default_factory=list)
    severity:   Dict[str, float] = field(default_factory=dict)  # ld_type -> 0..1

    def ef_severity(self) -> float:
        return self.severity.get("executive_function", 0.0)

    def has(self, ld_type: str) -> bool:
        return ld_type in self.confirmed or ld_type in self.suspected


@dataclass
class CognitiveBaseline:
    """Estimated cognitive parameters for this learner."""
    working_memory_span:         float = 5.0    # estimated units (neurotypical ~7)
    avg_session_attention_min:   float = 15.0   # minutes before fatigue
    task_initiation_latency_sec: float = 20.0   # seconds before starting a task
    frustration_threshold:       float = 0.55   # motivation level below which avoidance kicks in
    # running estimates: updated each session via Bayesian nudge
    recent_success_rate:         float = 0.50   # fraction of task attempts succeeded recently


@dataclass
class InterventionRecord:
    strategy: str
    outcome: str       # "success" | "failure" | "neutral"
    context: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class MetacogEntry:
    turn: int
    prompt_given: str
    learner_response_quality: float   # 0..1: how reflective/accurate the self-report was
    timestamp: float = field(default_factory=time.time)


@dataclass
class LearnerProfile:
    learner_id: str
    ld_profile:            LDProfile        = field(default_factory=LDProfile)
    baseline:              CognitiveBaseline = field(default_factory=CognitiveBaseline)
    intervention_history:  List[InterventionRecord] = field(default_factory=list)
    metacog_log:           List[MetacogEntry]        = field(default_factory=list)
    session_count:         int   = 0
    total_turns:           int   = 0
    scaffold_fade_index:   float = 1.0   # 1.0 = full scaffold; decreases as competence grows
    created_at:            float = field(default_factory=time.time)
    updated_at:            float = field(default_factory=time.time)

    # ── Scaffold density ──────────────────────────────────────────

    def scaffold_density(self) -> str:
        """
        Returns 'high', 'medium', or 'low' based on ef_severity and fade_index.
        """
        ef = self.ld_profile.ef_severity()
        effective = ef * self.scaffold_fade_index
        if effective > SCAFFOLD_MODERATE_MAX:
            return "high"
        if effective > SCAFFOLD_MINIMAL_MAX:
            return "medium"
        return "low"

    def update_fade_index(self, session_success_rate: float) -> None:
        """
        Reduce scaffold density when the learner is succeeding.
        Fade is slow (max 5% per session) and never below 0.20.
        """
        if session_success_rate > 0.70:
            self.scaffold_fade_index = max(0.20, self.scaffold_fade_index - 0.05)
        elif session_success_rate < 0.40:
            # Regression: gently increase density
            self.scaffold_fade_index = min(1.0, self.scaffold_fade_index + 0.03)

    # ── Intervention logging ──────────────────────────────────────

    def log_intervention(self, strategy: str, outcome: str, context: str = "") -> None:
        self.intervention_history.append(
            InterventionRecord(strategy=strategy, outcome=outcome, context=context)
        )
        self.updated_at = time.time()

    def successful_strategies(self) -> List[str]:
        return [r.strategy for r in self.intervention_history if r.outcome == "success"]

    def failed_strategies(self) -> List[str]:
        return [r.strategy for r in self.intervention_history if r.outcome == "failure"]

    # ── Metacognition logging ─────────────────────────────────────

    def log_metacog(self, turn: int, prompt: str, quality: float) -> None:
        self.metacog_log.append(
            MetacogEntry(turn=turn, prompt_given=prompt, learner_response_quality=quality)
        )
        self.updated_at = time.time()

    def metacog_development_score(self) -> float:
        """Average quality of last 5 metacog entries. 0 if no entries."""
        recent = self.metacog_log[-5:]
        if not recent:
            return 0.0
        return sum(e.learner_response_quality for e in recent) / len(recent)

    # ── Baseline Bayesian update ──────────────────────────────────

    def update_baseline_after_session(
        self,
        success_rate: float,
        avg_response_latency_sec: float,
        session_attention_min: float,
    ) -> None:
        b = self.baseline
        # Gentle nudge: 10% weight on new observation
        b.recent_success_rate         = 0.9 * b.recent_success_rate + 0.1 * success_rate
        b.task_initiation_latency_sec = 0.9 * b.task_initiation_latency_sec + 0.1 * avg_response_latency_sec
        b.avg_session_attention_min   = 0.9 * b.avg_session_attention_min   + 0.1 * session_attention_min
        self.session_count += 1
        self.updated_at = time.time()


# ── Persistence ───────────────────────────────────────────────────

class LearnerModelStore:
    """
    Loads and saves LearnerProfile objects as JSON files.
    Each learner has one file: <LEARNER_MODEL_PATH>/<learner_id>.json
    """

    def __init__(self, path: str = LEARNER_MODEL_PATH):
        self.path = path
        os.makedirs(path, exist_ok=True)

    def _filepath(self, learner_id: str) -> str:
        safe = learner_id.replace("/", "_").replace("\\", "_")
        return os.path.join(self.path, f"{safe}.json")

    def load(self, learner_id: str) -> LearnerProfile:
        fp = self._filepath(learner_id)
        if not os.path.exists(fp):
            return LearnerProfile(learner_id=learner_id)
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self._deserialize(data)

    def save(self, profile: LearnerProfile) -> None:
        fp = self._filepath(profile.learner_id)
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(self._serialize(profile), f, ensure_ascii=False, indent=2)

    # ── Serialization helpers ─────────────────────────────────────

    @staticmethod
    def _serialize(p: LearnerProfile) -> Dict[str, Any]:
        d = asdict(p)
        return d

    @staticmethod
    def _deserialize(d: Dict[str, Any]) -> LearnerProfile:
        ld = LDProfile(**d.get("ld_profile", {}))
        bl = CognitiveBaseline(**d.get("baseline", {}))
        ih = [InterventionRecord(**r) for r in d.get("intervention_history", [])]
        ml = [MetacogEntry(**e)       for e in d.get("metacog_log", [])]
        return LearnerProfile(
            learner_id=d["learner_id"],
            ld_profile=ld,
            baseline=bl,
            intervention_history=ih,
            metacog_log=ml,
            session_count=d.get("session_count", 0),
            total_turns=d.get("total_turns", 0),
            scaffold_fade_index=d.get("scaffold_fade_index", 1.0),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
        )
