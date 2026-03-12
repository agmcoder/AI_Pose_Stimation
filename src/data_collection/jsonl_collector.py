import json
from pathlib import Path
from loguru import logger
from src.core.interfaces import IDataCollector
from src.core.types import FrameRecord, RepRecord


class JsonlCollector(IDataCollector):
    """
    Persiste FrameRecords y RepRecords como JSON Lines (.jsonl).

    Cada línea es un objeto JSON independiente con un campo `_type`
    que indica si es un "frame" o un "rep".

    Ideal para: análisis exploratorio, streaming, y carga con pandas.
    """

    def __init__(self, output_dir: Path, filename_prefix: str = "session"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._filepath = self._output_dir / f"{filename_prefix}.jsonl"
        self._file = None

    def _ensure_open(self) -> None:
        if self._file is None:
            self._file = open(self._filepath, "w", encoding="utf-8")

    def on_frame(self, record: FrameRecord) -> None:
        self._ensure_open()
        entry = {"_type": "frame", **record.to_dict()}
        self._file.write(json.dumps(entry) + "\n")

    def on_rep(self, record: RepRecord) -> None:
        self._ensure_open()
        entry = {"_type": "rep", **record.to_dict()}
        self._file.write(json.dumps(entry) + "\n")

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            logger.info(f"📄 JSONL cerrado: {self._filepath}")
