import torch
from loguru import logger

def select_device(preference: str = "auto") -> str:
    """
    Prioridad: CUDA > MPS (Apple Silicon) > CPU
    Dependency Inversion: consumidores dependen de esta abstracción.
    """
    # 1. Gestión de preferencia manual
    if preference != "auto":
        if preference == "cpu" and (
            torch.cuda.is_available() or torch.backends.mps.is_available()
        ):
            logger.warning(
                "⚠️ CPU forzado pero GPU disponible — usando GPU obligatoriamente"
            )
        else:
            dev = torch.device(preference)
            logger.info(f"🚀 Usando dispositivo forzado: {dev}")
            return dev

    # 2. Prioridad 1: NVIDIA GPU (CUDA)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"✅ NVIDIA GPU detectada: {device_name}")
        return torch.device("cuda")

    # 3. Prioridad 2: Apple Silicon (MPS)
    if torch.backends.mps.is_available():
        try:
            # Test silencioso: intentamos una operación pequeña
            test_tensor = torch.ones(1, device="mps")
            logger.info("✅ Apple Silicon MPS activado")
            return torch.device("mps")
        except Exception as e:
            logger.warning(f"⚠️ MPS disponible pero falló el test: {e}. Usando CPU.")
    
    # 4. Fallback final: CPU
    logger.warning("⚠️ Modo CPU — Rendimiento limitado")
    return torch.device("cpu")
