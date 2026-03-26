import json
import pandas as pd
import numpy as np
import os

def normalize_keypoints(kp_list):
    """
    Aplica normalización biomecánica:
    1. Centra el esqueleto poniendo el origen (0,0) en el punto medio de las caderas.
    2. Escala el esqueleto dividiendo por la longitud del torso.
    """
    kp = np.array(kp_list) # Convertir a matriz (17 puntos, 2 ejes)

    # Identificadores de puntos YOLO/COCO
    # Hombros: 5, 6 | Caderas: 11, 12
    
    # 1. Calcular el centro de la cadera (punto de referencia central)
    hip_center = (kp[11] + kp[12]) / 2
    
    # Traslación: Restar el centro a todos los puntos
    kp_centered = kp - hip_center
    
    # 2. Calcular factor de escala (Longitud del torso)
    shoulder_center = (kp[5] + kp[6]) / 2
    torso_size = np.linalg.norm(shoulder_center - hip_center)
    
    # Evitar división por cero
    if torso_size == 0:
        torso_size = 1.0
        
    # Escalado
    kp_normalized = kp_centered / torso_size
    
    # Aplanar a una lista de 34 valores (x0, y0, x1, y1...)
    return kp_normalized.flatten().tolist()

def generate_training_csv(positive_file, negative_file, output_csv):
    """
    Lee los JSONL, normaliza y guarda en CSV.
    """
    dataset = []
    
    # Definir los archivos y sus etiquetas (1 para Squat, 0 para Negativo)
    sources = [
        (positive_file, 1),
        (negative_file, 0)
    ]
    
    print("--- Iniciando limpieza y normalización ---")
    
    for file_path, label in sources:
        if not os.path.exists(file_path):
            print(f"⚠️ Alerta: No se encontró el archivo {file_path}")
            continue
            
        count = 0
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Solo nos interesa la columna 'keypoints_xy'
                    if 'keypoints_xy' in data:
                        raw_kp = data['keypoints_xy']
                        # Normalizar puntos
                        normalized_kp = normalize_keypoints(raw_kp)
                        # Añadir fila: Puntos + Etiqueta
                        dataset.append(normalized_kp + [label])
                        count += 1
                except:
                    continue
        print(f"✅ Procesados {count} frames de: {file_path}")

    # Crear nombres de columnas para el CSV (x0, y0, ..., x16, y16, target)
    columns = []
    for i in range(17):
        columns.extend([f'x{i}', f'y{i}'])
    columns.append('target')

    # Convertir a DataFrame y guardar
    df = pd.DataFrame(dataset, columns=columns)
    df.to_csv(output_csv, index=False)
    
    print("-" * 40)
    print(f"📊 Dataset final guardado en: {output_csv}")
    print(f"📈 Total de muestras: {len(df)}")
    print(df['target'].value_counts()) # Muestra cuántos hay de cada tipo

# --- CONFIGURACIÓN DE RUTAS ---
# Asegúrate de que estos nombres coincidan con tus archivos
generate_training_csv(
    positive_file='../../../data/sessions/2026-03-26_18-45-34/squat.jsonl', 
    negative_file='../../../data/sessions/2026-03-26_18-47-36/squat_negative.jsonl', 
    output_csv='dataset_squats.csv'
)