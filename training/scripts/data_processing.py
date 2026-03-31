import json
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from src.utils.normalization import normalize_keypoints


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
    # Asegurar que el directorio de salida existe
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("-" * 40)
    print(f"📊 Dataset final guardado en: {output_csv}")
    print(f"📈 Total de muestras: {len(df)}")
    print(df['target'].value_counts()) # Muestra cuántos hay de cada tipo


if __name__ == "__main__":
    # --- CONFIGURACIÓN DE RUTAS ---
    # Asegúrate de que estos nombres coincidan con tus archivos
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))

    generate_training_csv(
        positive_file=os.path.join(project_root, 'data/sessions/2026-03-26_18-45-34/squat.jsonl'),
        negative_file=os.path.join(project_root, 'data/sessions/2026-03-26_18-47-36/squat_negative.jsonl'),
        output_csv=os.path.join(script_dir, '../data/dataset_squats.csv')
    )