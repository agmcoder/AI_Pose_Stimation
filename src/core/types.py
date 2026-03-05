from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np

# Type alias — un frame es simplemente un array HxWxC uint8
Frame = np.ndarray


@dataclass
class Keypoints:
    coords: np.ndarray  # shape (17, 2)
    scores: np.ndarray  # shape (17,)


@dataclass
class ExerciseState:
    name: str
    rep_count: int = 0
    phase: str = "up"
    angle: float = 0.0
    feedback: str = ""


@dataclass
class Person:
    track_id: int
    bbox: Tuple[float, float, float, float]
    keypoints: Optional[Keypoints] = None
    exercises: Dict[str, ExerciseState] = field(default_factory=dict)
