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
        futures = {
            self.executor.submit(self._process_person, p): p
            for p in persons
        }
        return [fut.result() for fut in futures]

    def _process_person(self, person: Person) -> Person:
        for ex_name, detector in self.registry.active_detectors.items():
            person.exercises[ex_name] = detector.update(person, self._counter)
        return person
