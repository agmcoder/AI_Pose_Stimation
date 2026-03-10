import cv2
import numpy as np
from ..core.interfaces import IExerciseCounter, IRenderer
from ..core.types import Frame, Person
from .renderer import Renderer
from src.visualization.stats_aggreator import StatsAggregator, ExerciseSummary

# ── Tema visual (estilo puro, no configuración operacional) ──────────────────
_BG_COLOR     = (20,  20,  20)
_HEADER_COLOR = (255, 200,   0)
_TEXT_COLOR   = (230, 230, 230)
_ACCENT_COLOR = (100, 220, 100)
_ID_COLOR     = (160, 160, 160)
_SEP_COLOR    = ( 60,  60,  60)
_LINE_COLOR   = ( 45,  45,  45)
_FOOTER_COLOR = ( 80,  80,  80)
_FONT         = cv2.FONT_HERSHEY_SIMPLEX


class DashboardRenderer(IRenderer):
    """
    Vista contenedora: panel fijo (izquierda) + video escalable (derecha).

    - Panel: anchura y fuentes FIJAS siempre, se reconstruye a la altura actual.
    - Video: escala con aspect-ratio en el área restante (letterbox negro).
    - Tamaño mínimo: se aplica en composición, no en la ventana del SO.

    Open/Closed  : extiende IRenderer sin modificar Renderer ni FrameProcessor.
    Composite    : delega skeleton/info al Renderer original.
    DIP          : recibe configuración inyectada desde YAML.
    """

    def __init__(self, window_cfg: dict, dashboard_cfg: dict, counter: IExerciseCounter):
        self._video_renderer = Renderer()
        self._aggregator     = StatsAggregator()
        self._counter        = counter              # ← esta línea falta en tu fichero

        self._min_w        = window_cfg["min_width"]
        self._min_h        = window_cfg["min_height"]
        self._footer_text  = window_cfg["name"]

        self._panel_width  = dashboard_cfg["panel_width"]
        self._panel_title  = dashboard_cfg["panel_title"]
        self._ids_per_line = dashboard_cfg["ids_per_line_limit"]


    # ── IRenderer contract ───────────────────────────────────────────────────
    def render(
        self,
        frame: Frame,
        persons: list[Person],
        display_size: tuple[int, int] | None = None,
    ) -> Frame:
        """
        display_size = (win_w, win_h) ya con mínimos aplicados desde main.
        Si es None usa la resolución nativa del frame (tests / grabación).
        """
        video_frame = self._video_renderer.render(frame, persons)
        summaries   = self._aggregator.compute(self._counter)

        if display_size is None:
            panel = self._build_panel(video_frame.shape[0], summaries, len(persons))
            return np.hstack([panel, video_frame])

        canvas_w, canvas_h = display_size
        pw           = self._panel_width
        video_area_w = canvas_w - pw

        # 1. Panel: anchura y fuentes FIJAS, altura = canvas_h
        panel = self._build_panel(canvas_h, summaries, len(persons))

        # 2. Video: escala aspect-ratio dentro del área disponible
        h, w   = video_frame.shape[:2]
        scale  = min(video_area_w / w, canvas_h / h)
        new_w  = int(w * scale)
        new_h  = int(h * scale)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        video_scaled = cv2.resize(video_frame, (new_w, new_h), interpolation=interp)

        # 3. Centrar video en su área con letterbox negro
        video_canvas = np.zeros((canvas_h, video_area_w, 3), dtype=np.uint8)
        y_off = (canvas_h - new_h) // 2
        x_off = (video_area_w - new_w) // 2
        video_canvas[y_off:y_off + new_h, x_off:x_off + new_w] = video_scaled

        return np.hstack([panel, video_canvas])

    # ── Panel builder ────────────────────────────────────────────────────────
    def _build_panel(
        self,
        height: int,
        summaries: dict[str, ExerciseSummary],
        num_persons: int,
    ) -> np.ndarray:
        pw    = self._panel_width
        panel = np.full((height, pw, 3), _BG_COLOR, dtype=np.uint8)

        cv2.line(panel, (pw - 1, 0), (pw - 1, height), _SEP_COLOR, 2)

        y = 30
        # Título
        cv2.putText(panel, self._panel_title, (12, y),
                    _FONT, 0.7, _HEADER_COLOR, 2, cv2.LINE_AA)
        y += 8
        cv2.line(panel, (12, y), (pw - 12, y), _HEADER_COLOR, 1)
        y += 22

        # Personas trackeadas
        cv2.putText(panel, f"Tracked persons: {num_persons}", (12, y),
                    _FONT, 0.55, _ACCENT_COLOR, 1, cv2.LINE_AA)
        y += 28
        cv2.line(panel, (12, y), (pw - 12, y), (50, 50, 50), 1)
        y += 18

        # Ejercicios
        if not summaries:
            cv2.putText(panel, "No exercise data yet", (12, y),
                        _FONT, 0.45, _ID_COLOR, 1, cv2.LINE_AA)
        else:
            for ex_name, summary in summaries.items():
                cv2.putText(panel, f"{ex_name.upper()}:  {summary.total}", (12, y),
                            _FONT, 0.6, _TEXT_COLOR, 1, cv2.LINE_AA)
                y += 22

                if summary.by_id:
                    parts = [f"id{tid}:{reps}"
                             for tid, reps in sorted(summary.by_id.items())]
                    line, lines = "", []
                    for part in parts:
                        if len(line) + len(part) + 2 > self._ids_per_line:
                            lines.append(line.rstrip(", "))
                            line = part + ", "
                        else:
                            line += part + ", "
                    lines.append(line.rstrip(", "))
                    for ln in lines:
                        cv2.putText(panel, f"  ({ln})", (12, y),
                                    _FONT, 0.42, _ID_COLOR, 1, cv2.LINE_AA)
                        y += 18

                y += 6
                cv2.line(panel, (12, y), (pw - 12, y), _LINE_COLOR, 1)
                y += 14

                if y > height - 40:   # guard: no salir del panel
                    break

        # Footer
        cv2.putText(panel, self._footer_text, (12, height - 12),
                    _FONT, 0.4, _FOOTER_COLOR, 1, cv2.LINE_AA)

        return panel   # ← aquí, siempre dentro de _build_panel
