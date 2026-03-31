# src/pipeline/frame_processor.py
from concurrent.futures import ThreadPoolExecutor
from ..core.types import Person
from ..core.interfaces import IExerciseCounter
from ..exercises.registry import ExerciseRegistry


class FrameProcessor:
    """
    Procesa ejercicios de múltiples personas en paralelo.
    Dependency Inversion: depende de IExerciseDetector e IExerciseCounter,
    no de clases concretas.
    """

    def __init__(self, exercise_cfg: dict, num_workers: int, counter: IExerciseCounter):
        self.registry  = ExerciseRegistry(exercise_cfg)
        self.executor  = ThreadPoolExecutor(max_workers=num_workers)
        self._counter  = counter

    def process(self, persons: list[Person]) -> list[Person]:
        # Batch-capable detectors pre-compute in a single GPU call
        for detector in self.registry.active_detectors.values():
            if hasattr(detector, "prepare_batch"):
                detector.prepare_batch(persons)

        # For 0-1 persons the ThreadPoolExecutor overhead exceeds the benefit;
        # process sequentially to avoid unnecessary synchronization cost.
        if len(persons) <= 1:
            return [self._process_person(p) for p in persons]
        futures = {
            self.executor.submit(self._process_person, p): p
            for p in persons
        }
        return [fut.result() for fut in futures]

    def set_active_exercises(self, names: set[str]) -> None:
        """Delega al registry el filtrado de ejercicios activos."""
        self.registry.set_active(names)

    def available_exercises(self) -> list[str]:
        """Ejercicios disponibles para la checklist."""
        return self.registry.available_names

    def active_exercise_names(self) -> set[str]:
        """Ejercicios actualmente activos."""
        return set(self.registry.active_detectors.keys())

    def _process_person(self, person: Person) -> Person:
        for ex_name, detector in self.registry.active_detectors.items():
            person.exercises[ex_name] = detector.update(person, self._counter)
        return person
