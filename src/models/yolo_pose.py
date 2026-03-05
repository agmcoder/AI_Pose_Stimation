from ultralytics import YOLO
from src.core.interfaces import IDetector   # ← absoluto
from src.core.types import Person, Keypoints, Frame
import numpy as np

class YoloPoseDetector(IDetector):
    def __init__(self, cfg: dict, device: str):
        self.model = YOLO(cfg["weights"])
        self.conf  = cfg["confidence"]
        self.iou   = cfg["iou_threshold"]
        self.device = device
        self.half  = cfg.get("half_precision", False) and device in ("cuda", "mps")

    def detect(self, frame: Frame) -> list[Person]:
        results = self.model.track(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            half=self.half,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        return self._parse(results)

    def _parse(self, results) -> list[Person]:
        persons = []
        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue
            for i, (box, tid) in enumerate(
                zip(r.boxes.xyxy, r.boxes.id.int())
            ):
                kps = r.keypoints[i]
                persons.append(Person(
                    track_id=int(tid),
                    bbox=tuple(box.cpu().numpy()),
                    keypoints=Keypoints(
                        coords=kps.xy[0].cpu().numpy(),
                        scores=kps.conf[0].cpu().numpy(),
                    )
                ))
        return persons
