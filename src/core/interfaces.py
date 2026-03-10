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


class IExerciseCounter(ABC):              # ← PRIMERO: no depende de nadie
    """
    SRP: única responsabilidad — almacenar y consultar repeticiones.
    Event sink: recibe eventos de rep completada y acumula.
    """

    @abstractmethod
    def record(self, track_id: int, exercise: str) -> None:
        ...

    @abstractmethod
    def total(self, exercise: str) -> int:
        ...

    @abstractmethod
    def by_id(self, exercise: str) -> dict[int, int]:
        ...

    @abstractmethod
    def all_exercises(self) -> dict[str, dict[int, int]]:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...


class IExerciseDetector(ABC):             # ← DESPUÉS: ya conoce IExerciseCounter
    """Liskov: todos los ejercicios son intercambiables."""

    @abstractmethod
    def update(self, person: Person, counter: IExerciseCounter) -> ExerciseState:
        ...


class IRenderer(ABC):
    """Interface Segregation: sólo responsable de visualización."""

    @abstractmethod
    def render(self, frame: Frame, persons: List[Person]) -> Frame:
        ...
