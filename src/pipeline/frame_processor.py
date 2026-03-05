from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from ..core.types import Person
from ..exercises.registry import ExerciseRegistry

class FrameProcessor:
    """
    Procesa ejercicios de múltiples personas en paralelo.
    Dependency Inversion: depende de IExerciseDetector, no de clases concretas.
    """
    def __init__(self, exercise_cfg: dict, num_workers: int = 4):
        self.registry = ExerciseRegistry(exercise_cfg)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def process(self, persons: list[Person]) -> list[Person]:
        futures = {
            self.executor.submit(self._process_person, p): p
            for p in persons
        }
        updated = []
        for fut, person in futures.items():
            person = fut.result()
            updated.append(person)
        return updated

    def _process_person(self, person: Person) -> Person:
        for ex_name, detector in self.registry.active_detectors.items():
            person.exercises[ex_name] = detector.update(person)
        return person
