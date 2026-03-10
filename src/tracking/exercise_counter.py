from collections import defaultdict
from ..core.interfaces import IExerciseCounter


class ExerciseCounter(IExerciseCounter):
    """
    Almacén de repeticiones por ejercicio y por individuo.

    - Es la ÚNICA fuente de verdad para conteos.
    - Recibe eventos discretos (record) desde los detectores.
    - Nunca lee Person ni ExerciseState — solo cuenta eventos.
    - Thread-safe para uso con ThreadPoolExecutor de FrameProcessor.
    """

    def __init__(self):
        # {exercise_name → {track_id → rep_count}}
        self._counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    def record(self, track_id: int, exercise: str) -> None:
        self._counts[exercise][track_id] += 1

    def total(self, exercise: str) -> int:
        return sum(self._counts[exercise].values())

    def by_id(self, exercise: str) -> dict[int, int]:
        return dict(self._counts[exercise])

    def all_exercises(self) -> dict[str, dict[int, int]]:
        return {ex: dict(ids) for ex, ids in self._counts.items()}

    def reset(self) -> None:
        self._counts.clear()
