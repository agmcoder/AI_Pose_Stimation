import cv2
import yaml
from loguru import logger
from src.pipeline.device_selector import select_device
from src.models.yolo_pose import YoloPoseDetector
from src.pipeline.frame_processor import FrameProcessor
from src.visualization.renderer import Renderer

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    app_cfg  = load_config("config/app.yml")["app"]
    pipe_cfg = load_config("config/app.yml")["pipeline"]
    model_cfg = load_config("config/model.yml")["model"]
    ex_cfg    = load_config("config/exercises.yml")["exercises"]

    device   = select_device(model_cfg["device"])
    detector = YoloPoseDetector(model_cfg, device)
    processor = FrameProcessor(ex_cfg, pipe_cfg["num_workers"])
    renderer  = Renderer()

    cap = cv2.VideoCapture(app_cfg["source"])
    logger.info("🎥 Iniciando stream — pulsa Q para salir")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        persons  = detector.detect(frame)          # detección + tracking
        persons  = processor.process(persons)      # ejercicios en paralelo
        output   = renderer.render(frame, persons) # visualización

        if app_cfg["display_window"]:
            cv2.imshow("Pose Tracker", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
