# Notas de Análisis del Proyecto

## Propósito

Este documento sirve como registro de observaciones, análisis y hallazgos durante el desarrollo del sistema de análisis de ejercicios con pose estimation. Su objetivo es capturar información útil para la documentación final, análisis de rendimiento y lecciones aprendidas.

---

## 1. Contexto del Proyecto

**Fecha de creación del registro**: 2026-04-14  
**Estado actual**: Desarrollo en progreso  
**Rama activa**: `feature/data_processing_exercise_model_training`

### Objetivos principales
- Sistema modular para análisis de ejercicios físicos mediante pose estimation
- Evaluación en tiempo real con feedback visual
- Recolección de datos para entrenamiento de modelos
- Detección de fases de ejercicio usando LSTM

### Tecnologías clave
- **Pose detection**: YOLOv8 pose (Ultralytics)
- **Interfaz**: PySide6 (Qt para Python)
- **ML para ejercicios**: TensorFlow/Keras (LSTM), scikit-learn
- **Procesamiento**: OpenCV, NumPy
- **Arquitectura**: Python con inyección de dependencias, interfaces explícitas

---

## 2. Entrenamiento de Modelos LSTM

### 2.1. Primer Entrenamiento - Squats

**Fecha**: 2026-04-14  
**Dataset**: `training/data/merged_squats.csv` (combinación de CSVs limpios)  
**Ejecución**:
```bash
python training/scripts/train_lstm.py \
  --input_csv training/data/merged_squats.csv \
  --window_size 50 \
  --batch_size 16 \
  --test_size 0.15 \
  --epochs 40 \
  --output_model models/lstm/squat_small_dataset.h5
```

#### 2.1.1. Características del Dataset
- **Total de frames**: 2,149
- **Frames positivos (squat)**: 1,312 (61.1%)
- **Frames negativos (none)**: 837 (38.9%)
- **Características seleccionadas**: 75 columnas (ángulos articulares + derivadas)
- **Valores NaN**: 721 (rellenados con 0)
- **Split entrenamiento/prueba**: 85%/15%
- **Ventanas temporales**: 2,099 ventanas de 50 frames

**Nota**: La proporción positivo/negativo sugiere un dataset desbalanceado que puede afectar métricas.

#### 2.1.2. Arquitectura del Modelo
```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ lstm (LSTM)                          │ (None, 50, 64)              │          35,840 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 50, 64)              │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_1 (LSTM)                        │ (None, 32)                  │          12,416 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 32)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 16)                  │             528 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 1)                   │              17 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
Total params: 48,801 (190.63 KB)
```

#### 2.1.3. Resultados del Entrenamiento

**Métricas finales (epoch 40)**:
- **Pérdida (loss)**: 1.3873
- **Precisión (accuracy)**: 63.17%
- **Precisión (precision)**: 45.56%
- **Recall**: 37.96%
- **F1-score**: 41.41%

**Análisis de sobreajuste**:
- **Entrenamiento**: accuracy ~94.4%, loss ~0.133
- **Validación**: accuracy ~63.2%, loss ~1.387
- **Divergencia**: 31.2% en accuracy, factor 10x en loss

**Observaciones por época**:
1. **Epochs 1-10**: Validación oscila entre 34-69% accuracy, sin estabilidad
2. **Epochs 10-20**: Mejora gradual en entrenamiento (78% → 87%), validación estancada ~55-65%
3. **Epochs 20-40**: Entrenamiento continúa mejorando (87% → 94%), validación fluctuante (52-65%)

#### 2.1.4. Problemas Identificados
1. **Sobreajuste severo**: La divergencia entrenamiento/validación indica overfitting
2. **Dataset pequeño**: 2,149 frames → 2,099 ventanas → solo 315 ventanas de prueba
3. **Desbalance de clases**: 61% squat vs 39% none
4. **Arquitectura compleja**: 48K parámetros para dataset pequeño
5. **Falta de regularización**: Dropout 0.2 puede ser insuficiente

#### 2.1.5. Recomendaciones para próximos entrenamientos
1. **Simplificar arquitectura**:
   - Reducir unidades LSTM (ej: 32→16)
   - Aumentar dropout (0.3-0.5)
   - Considerar single LSTM layer
2. **Aumentar dataset**:
   - Más sesiones de grabación
   - Data augmentation temporal
3. **Balancear clases**:
   - Subsampling de clase mayoritaria
   - Pesos de clase en loss function
4. **Añadir early stopping**:
   - Detener cuando val_loss aumente consistentemente
5. **Probando configuración**:
   - `--window_size 30` (menor ventana)
   - `--batch_size 8` (batch más pequeño)
   - `--test_size 0.2` (más datos de validación)

### 2.2. Implementación de Mejoras de Regularización

**Fecha**: 2026-04-14  
**Script modificado**: `training/scripts/train_lstm.py`  
**Objetivo**: Reducir overfitting identificado en el primer entrenamiento mediante técnicas de regularización y configuración flexible.

#### Cambios Implementados

1. **Nuevos parámetros de línea de comandos**:
   - `--lstm_units1`, `--lstm_units2`: Control de tamaño de capas LSTM
   - `--dropout_rate`: Tasa de dropout ajustable (default: 0.3)
   - `--l2_reg`: Regularización L2 para kernels recurrentes (default: 0.001)
   - `--early_stopping`: Habilitar early stopping con `--early_stopping_patience`
   - `--reduce_lr`: Reducción de learning rate con `--reduce_lr_patience` y `--reduce_lr_factor`
   - `--class_weight`: Aplicar pesos automáticos para clases desbalanceadas

2. **Arquitectura del modelo**:
   - Regularización L2 aplicada a kernels recurrentes de capas LSTM
   - Dropout después de cada capa LSTM
   - Configuración flexible: una o dos capas LSTM según `--lstm_units2`
   - Capa densa intermedia opcional con regularización L2

3. **Callbacks de entrenamiento**:
   - **EarlyStopping**: Monitoriza `val_loss`, restaura mejores pesos
   - **ReduceLROnPlateau**: Reduce LR cuando val_loss se estanca
   - Ambos callbacks configurables vía parámetros

4. **Balance de clases**:
   - Cálculo automático de pesos con `sklearn.utils.class_weight.compute_class_weight`
   - Los pesos se aplican en la función de pérdida durante el entrenamiento

5. **Mejoras de código**:
   - Numeración clara de secciones en el flujo (1-9)
   - Mensajes informativos de progreso
   - Guardado del scaler junto al modelo para inferencia consistente

#### Configuración Recomendada para Dataset Pequeño (~2,100 frames)
```bash
python training/scripts/train_lstm.py \
  --input_csv training/data/merged_squats.csv \
  --window_size 30 \
  --batch_size 8 \
  --test_size 0.2 \
  --epochs 100 \
  --lstm_units1 16 \
  --lstm_units2 0 \          # Una sola capa LSTM
  --dropout_rate 0.4 \
  --l2_reg 0.01 \
  --early_stopping \
  --early_stopping_patience 15 \
  --reduce_lr \
  --reduce_lr_patience 10 \
  --reduce_lr_factor 0.5 \
  --class_weight \
  --output_model models/lstm/squat_regularized.h5
```

#### Próximos Pasos
- Ejecutar entrenamiento con nueva configuración para evaluar reducción de overfitting
- Monitorear divergencia entre accuracy de entrenamiento y validación
- Considerar data augmentation temporal para aumentar dataset
- Evaluar modelo en tiempo real mediante `config/exercises.yml`

### 2.3. Ajuste de hiperparámetros para mejorar recall

**Fecha**: 2026-04-14  
**Contexto**: Tras evaluación con métricas (loss: 0.6537, accuracy: 66.03%, precision: 50.79%, recall: 29.63%, F1: 37.43%), se identificó necesidad de mejorar recall sin sacrificar precision.

**Cambios en defaults de `train_lstm.py`**:
- `--window_size`: 30 → 40 (mayor contexto temporal)
- `--l2_reg`: 0.01 → 0.001 (regularización más suave)
- `--early_stopping_patience`: 15 → 20 (más paciencia)
- `--reduce_lr_factor`: 0.5 → 0.2 (reducción LR más agresiva)
- `--dense_units`: 8 → 16 (más capacidad en capa densa)

**Configuración recomendada v3**:
```bash
python training/scripts/train_lstm.py \
  --input_csv training/data/merged_squats.csv \
  --window_size 40 \
  --batch_size 16 \
  --test_size 0.15 \
  --epochs 100 \
  --lstm_units1 32 \
  --lstm_units2 16 \          # Dos capas LSTM para patrones complejos
  --dropout_rate 0.3 \
  --dense_units 16 \
  --l2_reg 0.001 \
  --early_stopping \
  --early_stopping_patience 20 \
  --reduce_lr \
  --reduce_lr_patience 10 \
  --reduce_lr_factor 0.2 \
  --class_weight \
  --output_model models/lstm/squat_v3.h5
```

**Objetivos esperados**:
- Recall > 40% manteniendo precision > 45%
- F1-score > 40%
- Divergencia entrenamiento/validación < 15%

---

## 3. Pipeline de Datos

### 3.1. Flujo Actual
1. **Grabación**: `main.py` → `DataCollectionBus` → CSV/JSONL
2. **Limpieza**: Scripts en `training/scripts/` (normalización, filtrado)
3. **Unificación**: `merge_csvs.py` combina *cleaned.csv
4. **Entrenamiento**: `train_lstm.py` con selección automática de características

### 3.2. Problemas Comunes
- **Paths relativos**: Scripts asumen diferentes directorios base (corregido en `train_lstm.py`)
- **NaN values**: 721 valores NaN en dataset actual (¿origen?)
- **Orden temporal**: Importancia de ordenar por `frame_number`
- **Consistencia columnas**: Todos los CSV deben tener mismo esquema

### 3.3. Scripts Disponibles
- `merge_csvs.py`: Unión de CSVs con opción `--add_source_file`
- `clean_training_csv.py`: (presumible) limpieza y normalización
- `train_lstm.py`: Entrenamiento con parámetros configurables

---

## 4. Arquitectura y Código

### 4.1. Principios Diseño
- **Inversión de dependencias**: Interfaces en `src/core/`, implementaciones concretas en módulos específicos
- **Responsabilidad única**: Cada clase tiene propósito claro
- **Separación capas**: Pipeline → Evaluación → Tracking → Presentación
- **Testabilidad**: Contracts explícitos, inyección de dependencias

### 4.2. Módulos Clave
1. **`src/core/`**: Tipos base, evaluadores, normalización
2. **`src/exercises/`**: Lógica específica por ejercicio
3. **`src/pipeline/`**: Procesamiento frames, registro personas
4. **`src/data_collection/`**: Persistencia datos (CSV, JSONL)
5. **`src/models/`**: Wrappers YOLO, acceso modelos externos

### 4.3. Patrones Identificados
- **Factory Pattern**: `ExerciseRegistry` crea evaluadores basados en configuración
- **Strategy Pattern**: Evaluadores (angle-based vs LSTM) intercambiables
- **Observer Pattern**: `DataCollectionBus` notifica múltiples backends
- **Builder Pattern**: `RecordBuilder` para estructuras de datos complejas

---

## 5. Problemas y Soluciones

### 5.1. Resueltos
1. **Path resolution en train_lstm.py**  
   **Problema**: Script unía rutas con directorio del script en lugar de cwd  
   **Solución**: Usar `pathlib.Path().resolve()` para rutas relativas al project root

2. **Importación de pandas**  
   **Problema**: ModuleNotFoundError al ejecutar scripts sin entorno activado  
   **Solución**: Añadir try-except con mensaje claro en `merge_csvs.py`

3. **Unificación CSVs inconsistentes**  
   **Problema**: Múltiples CSVs con mismo esquema pero diferentes sesiones  
   **Solución**: Script `merge_csvs.py` con validación y opción `--add_source_file`

### 5.2. Pendientes
1. **Overfitting LSTM**: Arquitectura demasiado compleja para dataset pequeño
2. **NaN values**: Origen de valores faltantes en datos
3. **GPU CUDA**: Error `CUDA_ERROR_UNKNOWN` aunque CUDA parece disponible
4. **Balance dataset**: Proporción squat/none desbalanceada

---

## 6. Rendimiento y Optimización

### 6.1. Tiempos de Ejecución
- **Entrenamiento LSTM**: ~2-3 segundos por época (CPU)
- **Inferencia YOLO**: (por medir)
- **Pipeline completo**: (por medir)

### 6.2. Uso de Recursos
- **CPU**: TensorFlow usando instrucciones AVX2/FMA
- **GPU**: CUDA no inicializa correctamente (`CUDA_ERROR_UNKNOWN`)
- **Memoria**: Dataset ~2K frames × 75 features → tamaño manejable

### 6.3. Optimizaciones Potenciales
1. **Quantization**: Modelos LSTM para inferencia más rápida
2. **Batch processing**: Procesamiento por lotes en pipeline
3. **Caching**: Resultados intermedios reutilizables
4. **Thread pooling**: Mejor uso de CPU para procesamiento concurrente

---

## 7. Pruebas y Validación

### 7.1. Tests Existentes
- **Unitarios**: En `tests/` para lógica de dominio
- **Integración**: (limitados) para pipeline componentes
- **End-to-end**: Ejecución completa con `main.py`

### 7.2. Métricas de Calidad
- **Cobertura código**: (por medir)
- **Deuda técnica**: Baja (arquitectura limpia, código bien estructurado)
- **Mantenibilidad**: Alta (separación de responsabilidades, naming semántico)

### 7.3. Validación del Modelo
- **Offline**: Métricas (accuracy, precision, recall, F1)
- **Online**: Ejecución en tiempo real con `main.py`
- **Cross-validation**: (pendiente) K-fold para dataset pequeño

---

## 8. Lecciones Aprendidas

### 8.1. Machine Learning
1. **Dataset size matters**: 2K frames es muy pequeño para LSTM complejo
2. **Regularization crucial**: Dropout, early stopping, weight decay necesarios
3. **Class balance**: Desbalance afecta métricas (precision vs recall trade-off)
4. **Temporal windows**: Window_size 50 puede ser excesivo para movimientos rápidos

### 8.2. Ingeniería Software
1. **Path handling**: Siempre usar `pathlib` sobre `os.path`
2. **Error handling**: Mensajes claros y recovery cuando posible
3. **Configuration management**: YAML + CLI args proporciona flexibilidad
4. **Modular design**: Facilita testing y mantenimiento

### 8.3. Proceso Desarrollo
1. **Documentación incremental**: Este documento como ejemplo
2. **Version control**: Commits atómicos con mensajes semánticos
3. **Iterative improvement**: Corrección bugs → optimización → nuevas features
4. **Tooling adecuado**: Claude Code para navegación y refactor

---

## 9. Próximos Pasos

### 9.1. Corto Plazo (1-2 semanas)
1. **Mejorar modelo LSTM**:
   - Simplificar arquitectura
   - Añadir early stopping
   - Probando data augmentation
2. **Aumentar dataset**:
   - Más sesiones grabación
   - Balancear clases
3. **Integrar modelo**: Usar en `config/exercises.yml` con `evaluator_type: lstm`

### 9.2. Medio Plazo (1 mes)
1. **Más ejercicios**: Extender a jumping jacks, kicks, etc.
2. **Mejorar UI**: Feedback visual más detallado
3. **Optimización**: Profiling y optimización bottlenecks
4. **Testing**: Más tests de integración y e2e

### 9.3. Largo Plazo (2+ meses)
1. **Modelos avanzados**: Transformers, atención temporal
2. **Personalización**: Modelos adaptativos por usuario
3. **Plataforma cloud**: Backend para tracking progreso
4. **Mobile app**: Versión iOS/Android con cámara integrada

---

## 10. Registro de Cambios

| Fecha | Cambio | Responsable |
|-------|--------|-------------|
| 2026-04-14 | Primer entrenamiento LSTM, overfitting severo | Sistema |
| 2026-04-14 | Corrección path resolution en train_lstm.py | Claude Code |
| 2026-04-14 | Implementación mejoras regularización en train_lstm.py | Claude Code |
| 2026-04-14 | Ajuste hiperparámetros para mejorar recall | Claude Code |
| 2026-04-14 | Creación este documento de análisis | Claude Code |
| 2026-04-13 | Script merge_csvs.py para unificación CSVs | Claude Code |
| 2026-04-08 | Grabación sesiones squat para dataset | Usuario |

---

## 11. Apéndices

### 11.1. Comandos Útiles
```bash
# Entrenamiento rápido (test)
.venv/bin/python training/scripts/train_lstm.py \
  --input_csv training/data/merged_squats.csv \
  --output_model /tmp/test.h5 \
  --epochs 5 \
  --test_size 0.99

# Unificación CSVs
python merge_csvs.py --output_file training/data/merged.csv --add_source_file

# Ejecución aplicación
python main.py

# Tests
pytest tests/ -v
```

### 11.2. Estructura Dataset
```
training/data/
├── merged_squats.csv          # Dataset unificado (2,149 frames × 75 features)
├── session1_cleaned.csv       # Sesión individual procesada
├── session2_cleaned.csv       # Otra sesión
└── merged_results/            # Carpeta para resultados (opcional)
```

### 11.3. Configuración Ejercicio
```yaml
# config/exercises.yml
exercises:
  squat:
    enabled: true
    evaluator_type: lstm        # o "angle" para evaluación por ángulos
    lstm_model_path: "models/lstm/squat_small_dataset.h5"
    # ... otros parámetros
```

---

*Documento vivo - actualizar con cada hallazgo significativo*