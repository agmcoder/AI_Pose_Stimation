from datetime import datetime
from pathlib import Path
from loguru import logger
from src.core.interfaces import IDataCollector
from src.data_collection.csv_collector import CsvCollector
from src.data_collection.jsonl_collector import JsonlCollector


_BACKEND_REGISTRY: dict[str, type] = {
    "csv": CsvCollector,
    "jsonl": JsonlCollector,
}


def create_collectors(cfg: dict) -> tuple[list[IDataCollector], str]:
    """
    Factory: lee la config de data_collection y construye los collectors.

    Returns (collectors, session_id).

    Ejemplo de config (en app.yml):
        data_collection:
          enabled: true
          output_dir: "data/sessions"
          backends:
            - type: csv
              filename_prefix: "frames"
            - type: jsonl
              filename_prefix: "session"

    OCP: registrar un nuevo backend = añadir una entrada a _BACKEND_REGISTRY.
    """
    session_ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if not cfg.get("enabled", False):
        logger.info("📊 Data collection deshabilitado")
        return [], session_ts

    output_dir  = Path(cfg.get("output_dir", "data/sessions")) / session_ts
    collectors: list[IDataCollector] = []

    for backend_cfg in cfg.get("backends", []):
        backend_type = backend_cfg.get("type", "")
        prefix = backend_cfg.get("filename_prefix", backend_type)

        if backend_type not in _BACKEND_REGISTRY:
            logger.warning(f"Backend desconocido: '{backend_type}', ignorando")
            continue

        collector = _BACKEND_REGISTRY[backend_type](
            output_dir=output_dir,
            filename_prefix=prefix,
        )
        collectors.append(collector)
        logger.info(f"📊 Data collector registrado: {backend_type} → {output_dir}/{prefix}.*")

    return collectors, session_ts
