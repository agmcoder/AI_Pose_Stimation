from .squat import SquatDetector
from ..core.interfaces import IExerciseDetector

class ExerciseRegistry:
    """Factory Pattern — añadir ejercicios sin tocar el pipeline."""
    _REGISTRY = {"squat": SquatDetector}

    def __init__(self, cfg: dict):
        self.active_detectors: dict[str, IExerciseDetector] = {}
        for name, ex_cfg in cfg.items():
            if ex_cfg.get("enabled") and name in self._REGISTRY:
                self.active_detectors[name] = self._REGISTRY[name](ex_cfg)

    @classmethod
    def register(cls, name: str, detector_cls):
        """Abierto a extensión: nuevo ejercicio = una línea."""
        cls._REGISTRY[name] = detector_cls
