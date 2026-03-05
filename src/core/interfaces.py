from abc import ABC, abstractmethod
from typing import List
from .types import Frame, Person, ExerciseState


class IDetector(ABC):
    """Single Responsibility: sólo detecta poses."""

    @abstractmethod
    def detect(self, frame: Frame) -> List[Person]:
        ...


class ITracker(ABC):
    """Open/Closed: extensible con otros trackers sin modificar clientes."""

    @abstractmethod
    def update(self, persons: List[Person]) -> List[Person]:
        ...


class IExerciseDetector(ABC):
    """Liskov: todos los ejercicios son intercambiables."""

    @abstractmethod
    def update(self, person: Person) -> ExerciseState:
        ...


class IRenderer(ABC):
    """Interface Segregation: sólo responsable de visualización."""

    @abstractmethod
    def render(self, frame: Frame, persons: List[Person]) -> Frame:
        ...
