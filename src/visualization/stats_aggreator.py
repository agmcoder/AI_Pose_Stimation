# src/visualization/stats_aggregator.py
from dataclasses import dataclass, field
from ..core.interfaces import IExerciseCounter


@dataclass
class ExerciseSummary:
    total: int = 0
    by_id: dict[int, int] = field(default_factory=dict)


class StatsAggregator:
    """Query view sobre IExerciseCounter — solo formatea datos para el renderer."""

    def compute(self, counter: IExerciseCounter) -> dict[str, ExerciseSummary]:
        return {
            ex: ExerciseSummary(
                total=counter.total(ex),
                by_id=counter.by_id(ex),
            )
            for ex in counter.all_exercises()
        }
