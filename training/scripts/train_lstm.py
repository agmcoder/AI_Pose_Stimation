import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# 1. CARGAR DATOS

# Obtener la ruta del directorio donde está este script
base_path = os.path.dirname(os.path.abspath(__file__))
# Unir con el nombre del archivo (asumiendo que están en la misma carpeta)
csv_path = os.path.join(base_path, '../data/dataset_squats.csv')

# Cargar datos
df = pd.read_csv(csv_path)
X = df.drop('target', axis=1).values
y = df['target'].values

# 2. CREAR SECUENCIAS (Ventanas de tiempo)
# Necesitamos transformar los datos de [muestras, 34] a [muestras, 30, 34]
def create_sequences(data, labels, window_size=30):
    X_seq, y_seq = [], []
    # Usamos un paso (step) de 1 para tener más datos de entrenamiento
    for i in range(len(data) - window_size):
        X_seq.append(data[i:i+window_size])
        y_seq.append(labels[i+window_size]) # La etiqueta del final de la ventana
    return np.array(X_seq), np.array(y_seq)

WINDOW_SIZE = 30
X_windows, y_windows = create_sequences(X, y, WINDOW_SIZE)

# Dividir en entrenamiento y prueba (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X_windows, y_windows, test_size=0.2, random_state=42)

# 3. DEFINIR MODELO LSTM
model = Sequential([
    # Input shape: (30 frames, 34 coordenadas)
    LSTM(64, input_shape=(WINDOW_SIZE, 34), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid') # Salida binaria: 0 o 1
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. ENTRENAR
print("--- Iniciando entrenamiento del cerebro LSTM ---")
history = model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=32, 
    validation_data=(X_test, y_test),
    verbose=1
)

# 5. GUARDAR MODELO
model.save('../../models/lstm/squat_model.h5')
print("✅ Modelo guardado como '../../models/lstm/squat_model.h5'")

# Evaluación rápida
loss, accuracy = model.evaluate(X_test, y_test)
print(f"📈 Precisión final en test: {accuracy*100:.2f}%")