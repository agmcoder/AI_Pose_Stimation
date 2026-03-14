import cv2
import numpy as np
from ..core.types import Person

SKELETON = [
    (5,6),(5,7),(7,9),(6,8),(8,10),   # brazos
    (5,11),(6,12),(11,12),             # torso
    (11,13),(13,15),(12,14),(14,16),   # piernas
]
COLORS = [(np.random.randint(50,255), np.random.randint(50,255),
           np.random.randint(50,255)) for _ in range(100)]

class Renderer:
    def render(self, frame, persons: list[Person]):
        # Draw directly on the frame — no copy, no addWeighted blend.
        # cap.read() returns a new buffer each call so in-place mutation is safe.
        for p in persons:
            color = COLORS[p.track_id % 100]
            self._draw_skeleton(frame, p, color)
            self._draw_info(frame, p, color)
        return frame

    def _draw_skeleton(self, frame, person: Person, color):
        if person.keypoints is None:
            return
        kps = person.keypoints
        for a, b in SKELETON:
            if kps.scores[a] > 0.4 and kps.scores[b] > 0.4:
                cv2.line(frame, tuple(kps.coords[a].astype(int)),
                         tuple(kps.coords[b].astype(int)), color, 2)
        for i, (pt, sc) in enumerate(zip(kps.coords, kps.scores)):
            if sc > 0.4:
                cv2.circle(frame, tuple(pt.astype(int)), 4, color, -1)

    def _draw_info(self, frame, person: Person, color):
        x1, y1, _, _ = [int(v) for v in person.bbox]
        cv2.putText(frame, f"ID:{person.track_id}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for i, (name, state) in enumerate(person.exercises.items()):
            # Ángulos de rodilla y cadera si están disponibles
            knee = state.angles.get("knee", state.angle)
            hip  = state.angles.get("hip", 0.0)
            text = f"{name}: {state.rep_count} reps | K:{knee:.0f}° H:{hip:.0f}°"
            cv2.putText(frame, text,
                        (x1, y1 - 28 - i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Feedback en rojo si hay problema de técnica
        if any(s.feedback for s in person.exercises.values()):
            fb = " | ".join(s.feedback for s in person.exercises.values() if s.feedback)
            cv2.putText(frame, fb,
                        (x1, y1 - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
