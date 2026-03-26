"""
src/presentation/adapters.py

Infrastructure adapters: convert low-level types to Qt types.

FrameToQImageAdapter
  - Converts an OpenCV BGR numpy array into a QPixmap usable by Qt widgets
  - Isolates the BGR→RGB colour swap and Qt API details from the rest of the code
  - Stateless: pure static conversion, no side effects
"""
from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage, QPixmap


class FrameToQImageAdapter:
    """
    Adapter: np.ndarray (BGR, uint8) → QPixmap.

    The adapter is responsible for:
      1. Colour-channel flip (BGR → RGB required by QImage)
      2. Ensuring C-contiguous memory layout
      3. QImage construction with correct stride
      4. Conversion to QPixmap for efficient display

    Dependency Inversion: callers depend on this small adapter,
    not on scattered QImage construction spread across widgets.
    """

    @staticmethod
    def convert(frame: np.ndarray) -> QPixmap:
        """
        Convert *frame* (H×W×3, BGR uint8) to a QPixmap.
        Returns a valid QPixmap or a null QPixmap if *frame* is invalid.
        """
        if frame is None or frame.size == 0:
            return QPixmap()

        # Format_BGR888 (Qt 6 / Qt 5.14+) reads channels in B-G-R order,
        # so no channel swap is needed — eliminates a full-frame copy.
        bgr = np.ascontiguousarray(frame)
        h, w, ch = bgr.shape

        qimg = QImage(
            bgr.data,
            w,
            h,
            ch * w,
            QImage.Format.Format_BGR888,
        )
        # QImage references external buffer; copy before numpy GC
        return QPixmap.fromImage(qimg.copy())
