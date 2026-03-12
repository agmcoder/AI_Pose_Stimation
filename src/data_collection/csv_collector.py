import csv
from pathlib import Path
from loguru import logger
from src.core.interfaces import IDataCollector
from src.core.types import FrameRecord, RepRecord


class CsvCollector(IDataCollector):
    """
    Persiste FrameRecords en un CSV plano.

    - Los keypoints se aplanan a 51 columnas (x0,y0,c0,...,x16,y16,c16).
    - Los ejercicios NO se incluyen en el CSV (usar JsonlCollector para eso).
    - El fichero se crea en el primer write con cabecera automática.

    Ideal para: cargar con pandas/numpy para ML offline.
    """

    def __init__(self, output_dir: Path, filename_prefix: str = "frames"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._filepath = self._output_dir / f"{filename_prefix}.csv"
        self._file = None
        self._writer = None
        self._header_written = False

    def _ensure_open(self) -> None:
        if self._file is None:
            self._file = open(self._filepath, "w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)

    def on_frame(self, record: FrameRecord) -> None:
        self._ensure_open()
        if not self._header_written:
            self._writer.writerow(FrameRecord.csv_header())
            self._header_written = True
        self._writer.writerow(record.to_csv_row())

    def on_rep(self, record: RepRecord) -> None:
        # CSV collector sólo persiste frames; reps van al JSON/DB.
        pass

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            logger.info(f"📄 CSV cerrado: {self._filepath}")
