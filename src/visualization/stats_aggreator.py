# src/visualization/stats_aggregator.py
from dataclasses import dataclass, field
from collections import defaultdict
from ..core.types import Person


@dataclass
class ExerciseSummary:
    """Totales + desglose por ID para un ejercicio."""
    total: int = 0
    by_id: dict[int, int] = field(default_factory=dict)


class StatsAggregator:
    """
    Single Responsibility: agrega conteos de ejercicios de todas las personas.
    Los datos individuales siguen viviendo en Person.exercises — aquí sólo
    se computa la vista agregada bajo demanda.
    """

    def compute(self, persons: list[Person]) -> dict[str, ExerciseSummary]:
        """
        Devuelve un dict {exercise_name: ExerciseSummary} calculado
        a partir del estado actual de cada Person.
        """
        summaries: dict[str, ExerciseSummary] = defaultdict(ExerciseSummary)

        for person in persons:
            for ex_name, state in person.exercises.items():
                summary = summaries[ex_name]
                summary.by_id[person.track_id] = state.rep_count
                summary.total = sum(summary.by_id.values())

        return dict(summaries)
