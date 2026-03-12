from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

from .types import Frame, Person, ExerciseState

if TYPE_CHECKING:
    from .types import FrameRecord, RepRecord


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


class IDataCollector(ABC):
    """
    Interface Segregation: sólo expone ingesta de datos, no detalles de almacenamiento.
    Dependency Inversion: el main loop depende de esta abstracción, no de CSV/JSON/DB.
    """

    @abstractmethod
    def on_frame(self, record: FrameRecord) -> None:
        """Persiste un registro de un frame procesado."""
        ...

    @abstractmethod
    def on_rep(self, record: RepRecord) -> None:
        """Persiste un registro de una repetición completada."""
        ...

    @abstractmethod
    def flush(self) -> None:
        """Fuerza la escritura de datos en buffer al destino."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Libera recursos y realiza flush final."""
        ...
