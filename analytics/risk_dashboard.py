# analytics/risk_dashboard.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  Risk Monitoring Dashboard
#
# Provides summary statistics over the risk history stored in each
# learner's JSON profile.  All data is read from the LearnerModelStore
# so that no additional database is required.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import LEARNER_MODEL_PATH


class RiskDashboard:
    """
    Read-only analytics layer over persisted learner profiles.

    Learner profiles are expected to contain a ``risk_history`` list
    (added by the extended state tracker / session manager).  Profiles
    that pre-date the risk history extension are handled gracefully.

    Parameters
    ----------
    profile_dir : str
        Directory that holds the learner JSON profiles.
        Defaults to the ``LEARNER_MODEL_PATH`` from config.
    """

    def __init__(self, profile_dir: str = LEARNER_MODEL_PATH):
        self.profile_dir = Path(profile_dir)

    # ── Public API ────────────────────────────────────────────────

    def get_learner_risk_timeline(self, learner_id: str) -> List[Dict[str, Any]]:
        """
        Return the full risk-assessment history for one learner.

        Each entry:
            timestamp   – Unix epoch float
            risk_level  – "low" | "moderate" | "high" | "severe"
            risk_score  – float 0..1
            emotions    – dict label→score (may be empty for old entries)
            context     – str description of the triggering event

        Returns an empty list if the learner has no history.
        """
        profile = self._load_profile(learner_id)
        return profile.get("risk_history", [])

    def get_intervention_effectiveness(self, learner_id: str) -> Dict[str, Any]:
        """
        Summarise the effectiveness of past interventions for one learner.

        Returns
        -------
        {
            "total":    int,
            "by_outcome": {"success": int, "failure": int, "neutral": int},
            "top_strategies": [str, …],   # up to 5 most-successful strategies
            "recent":   list of last 10 intervention records,
        }
        """
        profile = self._load_profile(learner_id)
        history = profile.get("intervention_history", [])

        by_outcome: Dict[str, int] = {"success": 0, "failure": 0, "neutral": 0}
        for rec in history:
            outcome = rec.get("outcome", "neutral")
            by_outcome[outcome] = by_outcome.get(outcome, 0) + 1

        top_strategies = [
            r["strategy"] for r in history if r.get("outcome") == "success"
        ]
        # Most frequently successful strategies first
        seen: Dict[str, int] = {}
        for s in top_strategies:
            seen[s] = seen.get(s, 0) + 1
        ranked = sorted(seen, key=lambda x: seen[x], reverse=True)[:5]

        return {
            "total":         len(history),
            "by_outcome":    by_outcome,
            "top_strategies": ranked,
            "recent":        history[-10:],
        }

    def get_high_risk_learners(self, lookback_hours: float = 24.0) -> List[Dict[str, Any]]:
        """
        Return a list of learners who had a high or severe risk event
        within the last ``lookback_hours`` hours.

        Each entry:
            learner_id  – str
            risk_level  – most recent high/severe level
            risk_score  – corresponding score
            timestamp   – epoch of the event
        """
        cutoff = time.time() - lookback_hours * 3600
        results = []

        for fp in self.profile_dir.glob("*.json"):
            try:
                with open(fp, encoding="utf-8") as f:
                    profile = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            learner_id = profile.get("learner_id", fp.stem)
            risk_hist  = profile.get("risk_history", [])

            for entry in reversed(risk_hist):
                ts = entry.get("timestamp", 0)
                if ts < cutoff:
                    break  # history is ordered oldest-first
                level = entry.get("risk_level", "low")
                if level in ("high", "severe"):
                    results.append({
                        "learner_id":  learner_id,
                        "risk_level":  level,
                        "risk_score":  entry.get("risk_score", 0.0),
                        "timestamp":   ts,
                    })
                    break  # only the most-recent event per learner

        return sorted(results, key=lambda x: x["timestamp"], reverse=True)

    def get_risk_summary(self, learner_id: str) -> Dict[str, Any]:
        """
        Return aggregate risk statistics for one learner.

        Returns
        -------
        {
            "total_assessments":  int,
            "level_counts":       {"low": int, "moderate": int, "high": int, "severe": int},
            "avg_score":          float,
            "latest_level":       str,
            "latest_score":       float,
            "latest_timestamp":   float | None,
        }
        """
        history = self.get_learner_risk_timeline(learner_id)

        counts: Dict[str, int] = {"low": 0, "moderate": 0, "high": 0, "severe": 0}
        scores: List[float] = []

        for entry in history:
            lvl = entry.get("risk_level", "low")
            counts[lvl] = counts.get(lvl, 0) + 1
            scores.append(entry.get("risk_score", 0.0))

        latest = history[-1] if history else {}

        return {
            "total_assessments": len(history),
            "level_counts":      counts,
            "avg_score":         round(sum(scores) / len(scores), 4) if scores else 0.0,
            "latest_level":      latest.get("risk_level", "low"),
            "latest_score":      latest.get("risk_score", 0.0),
            "latest_timestamp":  latest.get("timestamp"),
        }

    # ── Internal helpers ──────────────────────────────────────────

    def _load_profile(self, learner_id: str) -> Dict[str, Any]:
        safe = learner_id.replace("/", "_").replace("\\", "_")
        fp   = self.profile_dir / f"{safe}.json"
        if not fp.exists():
            return {"learner_id": learner_id}
        try:
            with open(fp, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"learner_id": learner_id}
