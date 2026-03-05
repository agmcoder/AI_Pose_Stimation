import cv2
import yaml
import platform
from loguru import logger
from src.pipeline.device_selector import select_device
from src.models.yolo_pose import YoloPoseDetector
from src.pipeline.frame_processor import FrameProcessor
from src.visualization.dashboard_renderer import DashboardRenderer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def create_window(name: str, width: int, height: int) -> None:
    flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    cv2.namedWindow(name, flags)
    cv2.resizeWindow(name, width, height)


def main():
    app_cfg   = load_config("config/app.yml")["app"]
    pipe_cfg  = load_config("config/app.yml")["pipeline"]
    model_cfg = load_config("config/model.yml")["model"]
    ex_cfg    = load_config("config/exercises.yml")["exercises"]

    win_cfg       = app_cfg["window"]
    dashboard_cfg = app_cfg["dashboard"]

    device    = select_device(model_cfg["device"])
    detector  = YoloPoseDetector(model_cfg, device)
    processor = FrameProcessor(ex_cfg, pipe_cfg["num_workers"])
    renderer  = DashboardRenderer(win_cfg, dashboard_cfg)

    cv2.namedWindow(win_cfg["name"], cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_cfg["name"], win_cfg["initial_width"], win_cfg["initial_height"])

    cap = cv2.VideoCapture(app_cfg["source"])
    logger.info("🎥 Iniciando stream — pulsa Q para salir")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        persons = detector.detect(frame)
        persons = processor.process(persons)

        if app_cfg["display_window"]:
            rect = cv2.getWindowImageRect(win_cfg["name"])
            raw_w, raw_h = rect[2], rect[3]

            canvas_w = max(raw_w, win_cfg["min_width"])  if raw_w > 0 else win_cfg["initial_width"]
            canvas_h = max(raw_h, win_cfg["min_height"]) if raw_h > 0 else win_cfg["initial_height"]

            output = renderer.render(frame, persons, display_size=(canvas_w, canvas_h))
            cv2.imshow(win_cfg["name"], output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break




if __name__ == "__main__":
    main()
