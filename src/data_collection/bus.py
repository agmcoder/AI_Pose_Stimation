from loguru import logger
from src.core.interfaces import IDataCollector
from src.core.types import FrameRecord, RepRecord


class DataCollectionBus(IDataCollector):
    """
    Composite + Observer: al recibir un evento, lo reenvía a todos los
    collectors registrados.

    - OCP: añadir un nuevo backend = añadirlo a la lista, sin tocar esta clase.
    - SRP: esta clase sólo enruta, no persiste.
    """

    def __init__(self, collectors: list[IDataCollector] | None = None):
        self._collectors: list[IDataCollector] = collectors or []

    def add(self, collector: IDataCollector) -> None:
        self._collectors.append(collector)

    def on_frame(self, record: FrameRecord) -> None:
        for collector in self._collectors:
            try:
                collector.on_frame(record)
            except Exception as e:
                logger.warning(f"DataCollector error on_frame: {e}")

    def on_rep(self, record: RepRecord) -> None:
        for collector in self._collectors:
            try:
                collector.on_rep(record)
            except Exception as e:
                logger.warning(f"DataCollector error on_rep: {e}")

    def flush(self) -> None:
        for collector in self._collectors:
            try:
                collector.flush()
            except Exception as e:
                logger.warning(f"DataCollector error flush: {e}")

    def close(self) -> None:
        for collector in self._collectors:
            try:
                collector.close()
            except Exception as e:
                logger.warning(f"DataCollector error close: {e}")
