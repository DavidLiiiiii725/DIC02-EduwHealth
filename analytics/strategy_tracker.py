# analytics/strategy_tracker.py
# ─────────────────────────────────────────────────────────────────
# EduwHealth 2.0  –  Strategy Performance Tracker
#
# Utilities for recording and querying anonymised strategy
# performance data used by the StrategyOptimizerAgent.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def record_experiment_completion(
    attempt_id: int,
    score: float,
) -> bool:
    """Mark a ReadingStrategyExperiment as completed with its final score.

    Called at the end of a reading attempt (when the learner finishes the
    passage) so that the optimizer can aggregate real performance data.

    Args:
        attempt_id: PK of the ``ReadingAttempt`` that just completed.
        score:      Final score (0–1 float) for this attempt.

    Returns:
        True if the experiment record was found and updated; False otherwise.
    """
    try:
        from tutor.models import ReadingStrategyExperiment
        updated = ReadingStrategyExperiment.objects.filter(
            attempt_id=attempt_id, completed=False
        ).update(score=score, completed=True)
        return updated > 0
    except Exception as exc:
        logger.warning("record_experiment_completion failed: %s", exc)
        return False


def get_best_variant_for_ld(ld_type: str) -> Optional[str]:
    """Return the best-performing strategy variant for a given LD type.

    Args:
        ld_type: e.g. ``"adhd"``, ``"anxiety"``, ``"general"``.

    Returns:
        Variant name string, or ``None`` if no data is available.
    """
    try:
        from tutor.models import StrategyPerformance
        best = (
            StrategyPerformance.objects
            .filter(ld_profile_type=ld_type, total_attempts__gte=1)
            .order_by('-avg_score')
            .values_list('strategy_variant', flat=True)
            .first()
        )
        return best
    except Exception as exc:
        logger.warning("get_best_variant_for_ld failed: %s", exc)
        return None


def get_performance_summary() -> List[Dict[str, Any]]:
    """Return a sorted summary of all strategy performance records.

    Suitable for rendering in an admin dashboard.

    Returns:
        List of dicts, each describing one strategy variant's performance.
    """
    try:
        from tutor.models import StrategyPerformance
        records = StrategyPerformance.objects.all().order_by('-avg_score')
        return [
            {
                'strategy_variant': r.strategy_variant,
                'ld_profile_type':  r.ld_profile_type,
                'total_attempts':   r.total_attempts,
                'avg_score':        round(r.avg_score, 4),
                'avg_hints_used':   round(r.avg_hints_used, 2),
                'last_updated':     r.last_updated.isoformat(),
            }
            for r in records
        ]
    except Exception as exc:
        logger.warning("get_performance_summary failed: %s", exc)
        return []


def create_experiment(
    learner_id: int,
    attempt_id: int,
    variant: str,
) -> Optional[int]:
    """Create a new ReadingStrategyExperiment record.

    Args:
        learner_id: PK of the ``LearnerProfile``.
        attempt_id: PK of the ``ReadingAttempt``.
        variant:    Strategy variant name (e.g. ``"focus_v1"``).

    Returns:
        PK of the created experiment, or ``None`` on error.
    """
    try:
        from tutor.models import ReadingStrategyExperiment
        exp = ReadingStrategyExperiment.objects.create(
            learner_id=learner_id,
            attempt_id=attempt_id,
            strategy_variant=variant,
        )
        return exp.pk
    except Exception as exc:
        logger.warning("create_experiment failed: %s", exc)
        return None
