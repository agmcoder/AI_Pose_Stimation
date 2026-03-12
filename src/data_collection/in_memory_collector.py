from src.core.interfaces import IDataCollector
from src.core.types import FrameRecord, RepRecord


class InMemoryCollector(IDataCollector):
    """
    Almacena records en listas Python.

    Útil para:
      - Tests unitarios (inspeccionar qué se recolectó).
      - Análisis en sesión (dashboard, alertas en tiempo real).
    """

    def __init__(self):
        self.frames: list[FrameRecord] = []
        self.reps: list[RepRecord] = []

    def on_frame(self, record: FrameRecord) -> None:
        self.frames.append(record)

    def on_rep(self, record: RepRecord) -> None:
        self.reps.append(record)

    def flush(self) -> None:
        pass  # in-memory, nada que flushear

    def close(self) -> None:
        pass  # no hay recursos que liberar

    def clear(self) -> None:
        """Limpia los datos acumulados."""
        self.frames.clear()
        self.reps.clear()
