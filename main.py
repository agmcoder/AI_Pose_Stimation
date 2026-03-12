import time
import cv2
import yaml
from loguru import logger
from src.pipeline.device_selector import select_device
from src.models.yolo_pose import YoloPoseDetector
from src.pipeline.frame_processor import FrameProcessor
from src.pipeline.person_registry import PersonRegistry
from src.tracking.exercise_counter import ExerciseCounter
from src.visualization.dashboard_renderer import DashboardRenderer
from src.data_collection.factory import create_collectors
from src.data_collection.bus import DataCollectionBus
from src.data_collection.record_builder import build_frame_records


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def create_window(name: str, width: int, height: int) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, width, height)


def main():
    # ── Configuración ─────────────────────────────────────────────────────────
    app_cfg       = load_config("config/app.yml")["app"]
    pipe_cfg      = load_config("config/app.yml")["pipeline"]
    model_cfg     = load_config("config/model.yml")["model"]
    ex_cfg        = load_config("config/exercises.yml")["exercises"]
    win_cfg       = app_cfg["window"]
    dashboard_cfg = app_cfg["dashboard"]

    # ── Pipeline ──────────────────────────────────────────────────────────────
    device    = select_device(model_cfg["device"])
    counter   = ExerciseCounter()
    detector  = YoloPoseDetector(model_cfg, device)
    processor = FrameProcessor(ex_cfg, pipe_cfg["num_workers"], counter)
    registry  = PersonRegistry()
    renderer  = DashboardRenderer(win_cfg, dashboard_cfg, counter)

    # ── Data Collection ───────────────────────────────────────────────────────
    collectors = create_collectors(app_cfg.get("data_collection", {}))
    data_bus   = DataCollectionBus(collectors)

    # ── Ventana ───────────────────────────────────────────────────────────────
    if app_cfg["display_window"]:
        create_window(win_cfg["name"], win_cfg["initial_width"], win_cfg["initial_height"])

    # ── Stream ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(app_cfg["source"])
    logger.info("🎥 Iniciando stream — pulsa Q para salir")
    frame_number = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        persons = detector.detect(frame)      # Person[] frescos, sin historial
        persons = registry.merge(persons)     # restaurar estado acumulado por track_id
        persons = processor.process(persons)  # detectar ejercicios + emitir eventos
        registry.persist(persons)             # guardar estado post-procesamiento

        # ── Data Collection (non-invasive, after all processing) ──────────
        timestamp = time.time()
        for record in build_frame_records(persons, frame_number, timestamp):
            data_bus.on_frame(record)
        frame_number += 1

        if app_cfg["display_window"]:
            rect = cv2.getWindowImageRect(win_cfg["name"])
            raw_w, raw_h = rect[2], rect[3]

            canvas_w = max(raw_w, win_cfg["min_width"])  if raw_w > 0 else win_cfg["initial_width"]
            canvas_h = max(raw_h, win_cfg["min_height"]) if raw_h > 0 else win_cfg["initial_height"]

            output = renderer.render(frame, persons, display_size=(canvas_w, canvas_h))
            cv2.imshow(win_cfg["name"], output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    data_bus.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

