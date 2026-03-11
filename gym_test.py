import cv2
import torch
import logging
from ultralytics import solutions

# Configuración de logs para ver qué pasa en la terminal
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def run_gym():
    # 1. Detectar dispositivo (NVIDIA RTX 2060)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🚀 Usando dispositivo: {device}")

    # 2. Cargar video (0 para webcam o ruta a un mp4)
    video_path = "/home/alejandro/Downloads/DDAA_MARISTAS.mp4" # <--- CAMBIA ESTO o pon 0 para usar tu cámara
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error("❌ No se pudo abrir el video. Revisa la ruta.")
        return

    # 3. Obtener propiedades del video
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, 
                                          cv2.CAP_PROP_FRAME_HEIGHT, 
                                          cv2.CAP_PROP_FPS))

    # 4. Configurar el grabador
    video_writer = cv2.VideoWriter("resultado_gym.avi", 
                                   cv2.VideoWriter_fourcc(*"mp4v"), 
                                   fps, (w, h))

    # 5. Inicializar el contador de Gimnasio (Pushups por defecto)
    # kpts=[6, 8, 10] -> Hombro, Codo, Muñeca (para brazos)
    gym = solutions.AIGym(
        up_angle = 145.0,  # Ángulo mínimo para considerar "arriba"
        down_angle = 100.0, # Ángulo máximo para considerar "abajo"
        show=True,
        kpts=[11, 13, 15], # caderas, rodillas, tobillos
        model="yolo26x-pose.pt", 
        #tracker = "bot_sort.yaml",
        conf = 0.65,
        iou = 0.7,
        verbose = True,
        device = device,
        show_conf = True,
        show_labels = False
    )

    logger.info("🎥 Procesando... presiona 'q' en la ventana para salir.")

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break

        # Procesar el frame con la IA
        # Ultralytics maneja el movimiento a la GPU internamente si el modelo se cargó ahí
        results = gym(im0)

        # Escribir el video
        video_writer.write(results.plot_im)

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    logger.info("✅ Proceso finalizado. Video guardado como 'resultado_gym.avi'")

if __name__ == "__main__":
    run_gym()