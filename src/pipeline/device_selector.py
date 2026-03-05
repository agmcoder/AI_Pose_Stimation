import torch
from loguru import logger

def select_device(preference: str = "auto") -> str:
    """
    Prioridad: CUDA > MPS (Apple Silicon) > CPU
    Dependency Inversion: consumidores dependen de esta abstracción.
    """
    if preference != "auto":
        return preference

    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"✅ NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("✅ Apple Silicon MPS activado")
    else:
        device = "cpu"
        logger.warning("⚠️  CPU mode — rendimiento limitado")
    return device
