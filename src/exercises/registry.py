from .squat import SquatDetector
from ..core.interfaces import IExerciseDetector


class ExerciseRegistry:
    """Factory Pattern — añadir ejercicios sin tocar el pipeline."""
    _REGISTRY = {"squat": SquatDetector}

    def __init__(self, cfg: dict):
        self._all_detectors: dict[str, IExerciseDetector] = {}
        for name, ex_cfg in cfg.items():
            if ex_cfg.get("enabled") and name in self._REGISTRY:
                self._all_detectors[name] = self._REGISTRY[name](ex_cfg)

        self._active_names: set[str] = set(self._all_detectors.keys())

    @property
    def active_detectors(self) -> dict[str, IExerciseDetector]:
        """Solo los detectores activos en runtime."""
        return {n: d for n, d in self._all_detectors.items() if n in self._active_names}

    @property
    def available_names(self) -> list[str]:
        """Todos los ejercicios disponibles (cargados desde config)."""
        return sorted(self._all_detectors.keys())

    def is_active(self, name: str) -> bool:
        return name in self._active_names

    def set_active(self, names: set[str]) -> None:
        """Actualiza el conjunto de ejercicios activos."""
        self._active_names = names & set(self._all_detectors.keys())

    @classmethod
    def register(cls, name: str, detector_cls):
        """Abierto a extensión: nuevo ejercicio = una línea."""
        cls._REGISTRY[name] = detector_cls
