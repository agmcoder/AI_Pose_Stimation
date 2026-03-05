# src/visualization/dashboard_renderer.py
import cv2
import numpy as np
from ..core.interfaces import IRenderer
from ..core.types import Frame, Person
from .renderer import Renderer
from .stats_aggreator import StatsAggregator, ExerciseSummary

# ── Constantes de diseño del panel ──────────────────────────────────────────
PANEL_WIDTH   = 320          # px del panel lateral izquierdo
BG_COLOR      = (20, 20, 20) # fondo casi negro
HEADER_COLOR  = (255, 200, 0) # amarillo título
TEXT_COLOR    = (230, 230, 230)
ACCENT_COLOR  = (100, 220, 100)
ID_COLOR      = (160, 160, 160)
FONT          = cv2.FONT_HERSHEY_SIMPLEX


class DashboardRenderer(IRenderer):
    """
    Vista contenedora: panel de estadísticas (izquierda) + video con tracking (derecha).

    Open/Closed: extiende IRenderer sin modificar Renderer ni FrameProcessor.
    Composite Pattern: delega el render del video al Renderer original.
    """

    def __init__(self):
        self._video_renderer = Renderer()          # delega skeleton + info boxes
        self._aggregator     = StatsAggregator()   # SRP: sólo agrega stats

    # ── IRenderer contract ───────────────────────────────────────────────────
    def render(self, frame: Frame, persons: list[Person]) -> Frame:
        # 1. Render del video con tracking (esqueletos, IDs, ángulos)
        video_frame = self._video_renderer.render(frame, persons)

        # 2. Calcular estadísticas agregadas
        summaries = self._aggregator.compute(persons)

        # 3. Construir panel lateral
        panel = self._build_panel(frame.shape[0], summaries, len(persons))

        # 4. Concatenar horizontalmente: panel | video
        return np.hstack([panel, video_frame])

    # ── Panel builder ────────────────────────────────────────────────────────
    def _build_panel(
        self,
        height: int,
        summaries: dict[str, ExerciseSummary],
        num_persons: int,
    ) -> np.ndarray:
        panel = np.full((height, PANEL_WIDTH, 3), BG_COLOR, dtype=np.uint8)

        # Línea separadora vertical
        cv2.line(panel, (PANEL_WIDTH - 1, 0), (PANEL_WIDTH - 1, height),
                 (60, 60, 60), 2)

        y = 30
        # ── Título ──────────────────────────────────────────────────────────
        cv2.putText(panel, "CLASS STATS", (12, y),
                    FONT, 0.7, HEADER_COLOR, 2, cv2.LINE_AA)
        y += 8
        cv2.line(panel, (12, y), (PANEL_WIDTH - 12, y), HEADER_COLOR, 1)
        y += 22

        # ── Personas trackeadas ──────────────────────────────────────────────
        cv2.putText(panel, f"Tracked persons: {num_persons}", (12, y),
                    FONT, 0.55, ACCENT_COLOR, 1, cv2.LINE_AA)
        y += 28
        cv2.line(panel, (12, y), (PANEL_WIDTH - 12, y), (50, 50, 50), 1)
        y += 18

        # ── Ejercicios ───────────────────────────────────────────────────────
        if not summaries:
            cv2.putText(panel, "No exercise data yet", (12, y),
                        FONT, 0.45, ID_COLOR, 1, cv2.LINE_AA)
        else:
            for ex_name, summary in summaries.items():
                # Nombre del ejercicio + total
                label = f"{ex_name.upper()}:  {summary.total}"
                cv2.putText(panel, label, (12, y),
                            FONT, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)
                y += 22

                # Desglose por ID (una línea compacta)
                if summary.by_id:
                    parts = [f"id{tid}:{reps}" for tid, reps
                             in sorted(summary.by_id.items())]
                    # Partir en líneas si hay muchos IDs
                    line, lines = "", []
                    for part in parts:
                        if len(line) + len(part) + 2 > 34:
                            lines.append(line.rstrip(", "))
                            line = part + ", "
                        else:
                            line += part + ", "
                    lines.append(line.rstrip(", "))

                    for ln in lines:
                        cv2.putText(panel, f"  ({ln})", (12, y),
                                    FONT, 0.42, ID_COLOR, 1, cv2.LINE_AA)
                        y += 18

                y += 6
                cv2.line(panel, (12, y), (PANEL_WIDTH - 12, y),
                         (45, 45, 45), 1)
                y += 14

                # Guard: no salir del panel
                if y > height - 40:
                    break

        # ── Footer ───────────────────────────────────────────────────────────
        footer = "AI Fitness Tracker"
        cv2.putText(panel, footer, (12, height - 12),
                    FONT, 0.4, (80, 80, 80), 1, cv2.LINE_AA)

        return panel
