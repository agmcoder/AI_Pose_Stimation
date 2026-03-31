import tensorflow as tf
from loguru import logger


def configure_tf_device(preference: str = "auto") -> str:
    """
    Configura TensorFlow para usar GPU obligatoriamente si está disponible.
    Habilita memory growth para coexistir con PyTorch en la misma GPU.

    Retorna string de device TF ("/GPU:0" o "/CPU:0").
    """
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass  # Ya configurado en esta sesión

        if preference == "cpu":
            logger.warning(
                "⚠️ TF: CPU forzado pero GPU disponible — usando GPU obligatoriamente"
            )

        logger.info(f"✅ TensorFlow usando GPU: {gpus[0].name}")
        return "/GPU:0"

    # Sin GPU disponible
    if preference == "gpu":
        logger.error("❌ TF: GPU solicitada pero no disponible")

    logger.warning("⚠️ TensorFlow en modo CPU — Rendimiento limitado")
    return "/CPU:0"
