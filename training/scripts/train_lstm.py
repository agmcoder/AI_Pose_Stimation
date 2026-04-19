# ── Configurar LD_LIBRARY_PATH para CUDA/cuDNN de pip ────────────────
# Las librerías NVIDIA instaladas via pip (tensorflow[and-cuda]) quedan
# dentro del venv pero TensorFlow no las encuentra automáticamente.
#
# NOTA CRÍTICA: os.environ["LD_LIBRARY_PATH"] NO afecta al dlopen del
# proceso actual — el linker dinámico cachea las rutas de búsqueda al
# arrancar el proceso.  La solución es re-ejecutar el script con el
# LD_LIBRARY_PATH ya presente en el entorno del proceso (os.execvp).
# Se usa un centinela (_CUDA_PATHS_SET) para evitar bucle infinito.
import os
import sys
import pathlib as _pathlib

def _ensure_cuda_paths():
    """Re-lanza el proceso con LD_LIBRARY_PATH correcto si es necesario."""
    # Si ya re-ejecutamos, no repetir
    if os.environ.get("_CUDA_PATHS_SET") == "1":
        return

    nvidia_base = None
    for sp in (
        _pathlib.Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia",
        _pathlib.Path(sys.prefix) / "lib" / "site-packages" / "nvidia",
    ):
        if sp.is_dir():
            nvidia_base = sp
            break

    if nvidia_base is None:
        return

    lib_dirs = sorted(str(p) for p in nvidia_base.glob("*/lib") if p.is_dir())
    if not lib_dirs:
        return

    new_paths = ":".join(lib_dirs)
    existing = os.environ.get("LD_LIBRARY_PATH", "")

    # Comprobar si ya están todas las rutas
    if existing and all(d in existing for d in lib_dirs):
        os.environ["_CUDA_PATHS_SET"] = "1"
        print(f"🔧 LD_LIBRARY_PATH ya contiene las {len(lib_dirs)} rutas NVIDIA")
        return

    # Actualizar y re-ejecutar para que dlopen las vea
    os.environ["LD_LIBRARY_PATH"] = f"{new_paths}:{existing}" if existing else new_paths
    os.environ["_CUDA_PATHS_SET"] = "1"
    print(f"🔧 Re-lanzando con LD_LIBRARY_PATH ({len(lib_dirs)} rutas NVIDIA del venv)…")
    os.execvp(sys.executable, [sys.executable] + sys.argv)

_ensure_cuda_paths()
# ─────────────────────────────────────────────────────────────────────

import argparse
import json
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import tf2onnx

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenar modelo LSTM para squat con CV por grupos y fit final"
    )

    parser.add_argument("--input_csv", default="training/data/train_squats_cleaned.csv")
    parser.add_argument("--output_dir", default="models/lstm",
                        help="Directorio de salida para los artefactos del modelo")
    parser.add_argument("--model_name", default="lstm_exercise_detector",
                        help="Nombre base del modelo (sin extensión)")

    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--decision_threshold", type=float, default=0.5)

    parser.add_argument("--lstm_units1", type=int, default=16)
    parser.add_argument("--lstm_units2", type=int, default=8)
    parser.add_argument("--dropout_rate", type=float, default=0.35)
    parser.add_argument("--dense_units", type=int, default=8)
    parser.add_argument("--l2_reg", type=float, default=0.001)

    parser.add_argument("--early_stopping", action="store_true", default=True)
    parser.add_argument("--early_stopping_patience", type=int, default=8)
    parser.add_argument("--reduce_lr", action="store_true", default=True)
    parser.add_argument("--reduce_lr_patience", type=int, default=4)
    parser.add_argument("--reduce_lr_factor", type=float, default=0.5)

    parser.add_argument("--class_weight", action="store_true", default=True)
    parser.add_argument("--max_class_weight", type=float, default=2.0)

    parser.add_argument(
        "--cv_folds",
        type=int,
        default=0,
        help="0 = Leave-One-Group-Out; >1 = GroupKFold(n_splits=cv_folds)",
    )
    parser.add_argument(
        "--final_fit_full_data",
        action="store_true",
        default=True,
        help="Tras la CV, reentrena con todo el dataset",
    )
    parser.add_argument(
        "--final_val_fraction",
        type=float,
        default=0.1,
        help="Fracción final de validación interna al entrenar con todo el dataset",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--multiclass", dest="multiclass", action="store_true")
    mode_group.add_argument("--binary", dest="multiclass", action="store_false")
    parser.set_defaults(multiclass=False)

    return parser.parse_args()


def choose_group_columns(df: pd.DataFrame):
    # If person_id exists, use it as the SOLE group key for CV.
    # This enables true Leave-One-Person-Out evaluation: the model is validated
    # on a person it has never seen during training, which is the correct
    # generalisation test for a pose-based exercise detector.
    if "person_id" in df.columns and df["person_id"].notna().any() and df["person_id"].nunique() > 1:
        print(f"👤 Agrupando CV por person_id ({df['person_id'].nunique()} personas: {sorted(df['person_id'].unique())})")
        return ["person_id"]

    # Fallback: group by session + video + track (old behaviour)
    group_cols = []
    if "session_id" in df.columns:
        group_cols.append("session_id")
    if "video_id" in df.columns:
        group_cols.append("video_id")

    if not group_cols:
        raise ValueError(
            "El CSV debe contener al menos 'person_id', 'session_id' o 'video_id' para agrupar correctamente."
        )

    if "track_id" in df.columns:
        group_cols.append("track_id")

    return group_cols


def build_group_key(df: pd.DataFrame, group_cols):
    return df[group_cols].astype(str).fillna("NA").agg("||".join, axis=1)


def get_sort_columns(df: pd.DataFrame, group_cols):
    cols = list(group_cols)
    if "frame_number" in df.columns:
        cols.append("frame_number")
    elif "timestamp" in df.columns:
        cols.append("timestamp")
    return cols


def select_feature_columns(df: pd.DataFrame):
    exact_exclude = {
        "session_id",
        "video_id",
        "timestamp",
        "frame_number",
        "track_id",
        "fps",
        "exercise_label",
        "phase_label",
        "rep_id",
        "_group_key",
        "_target",
    }
    prefix_exclude = ("bbox_", "debug_")

    feature_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in exact_exclude:
            continue
        if any(col_lower.startswith(prefix) for prefix in prefix_exclude):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    if not feature_cols:
        raise ValueError("No se encontraron columnas numéricas válidas para features.")

    print(f"✅ Seleccionadas {len(feature_cols)} columnas de características")
    print(f"   Primeras 10: {feature_cols[:10]}")
    if len(feature_cols) > 10:
        print(f"   ... y {len(feature_cols) - 10} más")

    return feature_cols


def make_target(df: pd.DataFrame, multiclass: bool):
    if "phase_label" not in df.columns:
        raise ValueError("Columna 'phase_label' no encontrada en el CSV")

    phase = df["phase_label"].fillna("none").astype(str)

    if multiclass:
        preferred = ["none", "phase_1", "phase_2"]
        present = phase.unique().tolist()
        ordered = [x for x in preferred if x in present] + sorted([x for x in present if x not in preferred])
        label2id = {label: idx for idx, label in enumerate(ordered)}
        y = phase.map(label2id).astype(int).values
        print(f"✅ Target multiclase: {label2id}")
    else:
        label2id = {"none": 0, "squat": 1}
        y = (phase != "none").astype(int).values
        print(f"✅ Target binario: positivos={int(y.sum())}, negativos={int(len(y) - y.sum())}")

    return y, label2id


def preprocess_groupwise(df: pd.DataFrame, feature_cols, group_cols):
    sort_cols = get_sort_columns(df, group_cols)
    cleaned_groups = []
    total_nan_before = int(df[feature_cols].isna().sum().sum())
    total_rows_before = len(df)
    total_dropped_rows = 0

    for _, group_df in df.groupby("_group_key", sort=False):
        group_df = group_df.sort_values(sort_cols).copy()
        group_df[feature_cols] = group_df[feature_cols].ffill()
        before_rows = len(group_df)
        group_df = group_df.dropna(subset=feature_cols).copy()
        total_dropped_rows += before_rows - len(group_df)

        if not group_df.empty:
            cleaned_groups.append(group_df)

    if not cleaned_groups:
        raise ValueError("Tras el preprocesado no quedó ninguna fila válida.")

    cleaned_df = pd.concat(cleaned_groups, ignore_index=True)
    total_nan_after = int(cleaned_df[feature_cols].isna().sum().sum())

    print(f"⚠️  NaN antes del preprocesado: {total_nan_before}")
    print(f"✅ NaN después del preprocesado: {total_nan_after}")
    print(f"🧹 Filas descartadas por NaN residuales al inicio de grupo: {total_dropped_rows}")
    print(f"📉 Filas finales tras limpieza: {len(cleaned_df)} / {total_rows_before}")

    return cleaned_df


def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols):
    scaler = StandardScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[feature_cols] = train_df[feature_cols].astype(np.float32)
    test_df[feature_cols] = test_df[feature_cols].astype(np.float32)

    train_scaled = scaler.fit_transform(train_df[feature_cols].to_numpy(dtype=np.float32))
    test_scaled = scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float32))

    train_df[feature_cols] = train_scaled.astype(np.float32)
    test_df[feature_cols] = test_scaled.astype(np.float32)

    return train_df, test_df, scaler


def scale_features_full(df: pd.DataFrame, feature_cols):
    scaler = StandardScaler()
    df = df.copy()
    df[feature_cols] = df[feature_cols].astype(np.float32)
    scaled = scaler.fit_transform(df[feature_cols].to_numpy(dtype=np.float32))
    df[feature_cols] = scaled.astype(np.float32)
    return df, scaler


def create_sequences_from_df(df: pd.DataFrame, feature_cols, window_size: int):
    X_seq, y_seq = [], []
    group_used = []
    skipped_groups = 0

    for group_key, group_df in df.groupby("_group_key", sort=False):
        sort_cols = get_sort_columns(group_df, [])
        group_df = group_df.sort_values(sort_cols).reset_index(drop=True)

        X = group_df[feature_cols].values.astype(np.float32)
        y = group_df["_target"].values.astype(np.int64)

        if len(group_df) < window_size:
            skipped_groups += 1
            continue

        for i in range(len(group_df) - window_size + 1):
            X_seq.append(X[i:i + window_size])
            y_seq.append(y[i + window_size - 1])
            group_used.append(group_key)

    if not X_seq:
        raise ValueError("No se pudieron crear secuencias. Revisa window_size o el tamaño de los grupos.")

    X_seq = np.asarray(X_seq, dtype=np.float32)
    y_seq = np.asarray(y_seq)
    group_used = np.asarray(group_used)

    print(f"🕒 Ventanas creadas: {len(X_seq)}")
    print(f"ℹ️  Grupos omitidos por ser más cortos que window_size: {skipped_groups}")

    return X_seq, y_seq, group_used


def build_model(args, n_features: int, n_classes: int):
    l2_reg = regularizers.l2(args.l2_reg) if args.l2_reg > 0 else None

    layers = [Input(shape=(args.window_size, n_features))]

    if args.lstm_units2 > 0:
        layers.append(
            LSTM(
                args.lstm_units1,
                return_sequences=True,
                kernel_regularizer=l2_reg,
                recurrent_regularizer=l2_reg,
            )
        )
        layers.append(Dropout(args.dropout_rate))
        layers.append(
            LSTM(
                args.lstm_units2,
                kernel_regularizer=l2_reg,
                recurrent_regularizer=l2_reg,
            )
        )
    else:
        layers.append(
            LSTM(
                args.lstm_units1,
                kernel_regularizer=l2_reg,
                recurrent_regularizer=l2_reg,
            )
        )

    layers.append(Dropout(args.dropout_rate))

    if args.dense_units > 0:
        layers.append(Dense(args.dense_units, activation="relu", kernel_regularizer=l2_reg))

    if args.multiclass:
        layers.append(Dense(n_classes, activation="softmax"))
        loss = "sparse_categorical_crossentropy"
    else:
        layers.append(Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"

    model = Sequential(layers)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def compute_soft_class_weights(y_train, max_weight=2.0):
    classes = np.unique(y_train)
    counts = np.array([(y_train == c).sum() for c in classes], dtype=np.float32)
    weights = np.sqrt(counts.max() / counts)
    weights = np.clip(weights, 1.0, max_weight)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def get_group_splits(group_keys, cv_folds=0):
    unique_groups = np.unique(group_keys)
    dummy_x = np.zeros(len(group_keys))

    if len(unique_groups) < 2:
        raise ValueError("Se necesitan al menos 2 grupos para evaluar por grupos.")

    if cv_folds and cv_folds > 1 and len(unique_groups) >= cv_folds:
        splitter = GroupKFold(n_splits=cv_folds)
        splits = list(splitter.split(dummy_x, groups=group_keys))
        mode = f"GroupKFold({cv_folds})"
    else:
        splitter = LeaveOneGroupOut()
        splits = list(splitter.split(dummy_x, groups=group_keys))
        mode = "LeaveOneGroupOut"

    return splits, mode


def evaluate_predictions(y_true, y_pred, multiclass):
    if multiclass:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

    acc = accuracy_score(y_true, y_pred)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def build_callbacks(args, verbose=0):
    callbacks = []

    if args.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=args.early_stopping_patience,
                restore_best_weights=True,
                verbose=verbose,
            )
        )

    if args.reduce_lr:
        callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss",
                patience=args.reduce_lr_patience,
                factor=args.reduce_lr_factor,
                min_lr=1e-6,
                verbose=verbose,
            )
        )

    return callbacks


def train_one_split(df_train, df_test, feature_cols, args, label2id):
    df_train, df_test, scaler = scale_features(df_train, df_test, feature_cols)

    X_train, y_train, _ = create_sequences_from_df(df_train, feature_cols, args.window_size)
    X_test, y_test, _ = create_sequences_from_df(df_test, feature_cols, args.window_size)

    n_features = X_train.shape[2]
    n_classes = len(label2id) if args.multiclass else 2
    model = build_model(args, n_features=n_features, n_classes=n_classes)

    class_weight = None
    if args.class_weight and len(np.unique(y_train)) > 1:
        class_weight = compute_soft_class_weights(y_train, max_weight=args.max_class_weight)

    model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        class_weight=class_weight,
        callbacks=build_callbacks(args, verbose=0),
        verbose=0,
    )

    if args.multiclass:
        y_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
    else:
        y_prob = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_prob >= args.decision_threshold).astype(int)

    metrics = evaluate_predictions(y_test, y_pred, args.multiclass)
    return metrics


def export_to_onnx(model, output_path: pathlib.Path, window_size: int, n_features: int):
    """Exporta el modelo Keras a formato ONNX para inferencia en producción."""
    spec = (tf.TensorSpec((None, window_size, n_features), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    output_path.write_bytes(model_proto.SerializeToString())
    print(f"✅ Modelo ONNX guardado como: {output_path}")


def save_side_files(output_dir: pathlib.Path, model_name: str, scaler: StandardScaler, feature_cols, label2id, group_cols, args, cv_summary=None):
    scaler_path = output_dir / f"{model_name}_scaler.npz"
    meta_path = output_dir / f"{model_name}_meta.json"

    np.savez(
        scaler_path,
        mean=scaler.mean_,
        scale=scaler.scale_,
        features=np.array(feature_cols, dtype=object),
    )

    meta = {
        "feature_cols": feature_cols,
        "label2id": label2id,
        "id2label": {str(v): k for k, v in label2id.items()},
        "group_cols": group_cols,
        "window_size": args.window_size,
        "multiclass": args.multiclass,
        "decision_threshold": args.decision_threshold,
        "cv_summary": cv_summary,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return scaler_path, meta_path


def print_eval_summary(metrics):
    print(f"   • Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"   • Precision: {metrics['precision']*100:.2f}%")
    print(f"   • Recall:    {metrics['recall']*100:.2f}%")
    print(f"   • F1:        {metrics['f1']*100:.2f}%")


def final_train_on_full_data(df, feature_cols, label2id, group_cols, args, output_dir, model_name, cv_summary):
    print("\n🚀 Entrenando modelo final con todo el dataset...")

    df_all, scaler = scale_features_full(df, feature_cols)
    X_all, y_all, all_groups = create_sequences_from_df(df_all, feature_cols, args.window_size)

    n_features = X_all.shape[2]
    n_classes = len(label2id) if args.multiclass else 2
    model = build_model(args, n_features=n_features, n_classes=n_classes)

    class_weight = None
    if args.class_weight and len(np.unique(y_all)) > 1:
        class_weight = compute_soft_class_weights(y_all, max_weight=args.max_class_weight)
        print(f"⚖️  Pesos finales: {class_weight}")

    fit_kwargs = dict(
        x=X_all,
        y=y_all,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        verbose=1,
    )

    if args.final_val_fraction > 0:
        fit_kwargs["validation_split"] = args.final_val_fraction
        fit_kwargs["shuffle"] = False
        fit_kwargs["callbacks"] = build_callbacks(args, verbose=1)

    history = model.fit(**fit_kwargs)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Guardar .keras ────────────────────────────────────────────────
    keras_path = output_dir / f"{model_name}.keras"
    model.save(keras_path)
    print(f"\n✅ Modelo Keras guardado como: {keras_path}")

    # ── Guardar archivos auxiliares (scaler + meta) ───────────────────
    # Se guardan ANTES de ONNX para que un fallo en la exportación no
    # bloquee los ficheros imprescindibles para inferencia.
    scaler_path, meta_path = save_side_files(
        output_dir=output_dir,
        model_name=model_name,
        scaler=scaler,
        feature_cols=feature_cols,
        label2id=label2id,
        group_cols=group_cols,
        args=args,
        cv_summary=cv_summary,
    )

    print(f"✅ Scaler guardado como: {scaler_path}")
    print(f"✅ Metadata guardada como: {meta_path}")

    # ── Exportar .onnx (opcional, no crítico) ─────────────────────────
    onnx_path = output_dir / f"{model_name}.onnx"
    try:
        export_to_onnx(model, onnx_path, args.window_size, n_features)
    except Exception as e:
        print(f"⚠️  Exportación ONNX falló (no crítico): {e}")

    if args.multiclass:
        y_prob = model.predict(X_all, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        target_names = [k for k, _ in sorted(label2id.items(), key=lambda x: x[1])]
        print("\n🧾 Classification report sobre todas las ventanas:")
        print(classification_report(y_all, y_pred, target_names=target_names, zero_division=0))
        print("🧩 Matriz de confusión:")
        print(confusion_matrix(y_all, y_pred))
    else:
        y_prob = model.predict(X_all, verbose=0).ravel()
        y_pred = (y_prob >= args.decision_threshold).astype(int)
        print("\n🧾 Classification report sobre todas las ventanas:")
        print(classification_report(y_all, y_pred, target_names=["none", "squat"], zero_division=0))
        print("🧩 Matriz de confusión:")
        print(confusion_matrix(y_all, y_pred))


def main():
    args = parse_args()
    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)

    print("=" * 70)
    print("🎯 ENTRENAMIENTO LSTM PARA DETECCIÓN DE SQUATS")
    print("=" * 70)

    csv_path = pathlib.Path(args.input_csv).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    model_name = args.model_name

    if not csv_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {csv_path}")

    print(f"📂 Cargando datos de: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"📈 Total de filas originales: {len(df)}")

    group_cols = choose_group_columns(df)
    print(f"🔗 Columnas de grupo: {group_cols}")

    df["_group_key"] = build_group_key(df, group_cols)

    sort_cols = get_sort_columns(df, group_cols)
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
        print(f"📊 Datos ordenados por: {sort_cols}")

    feature_cols = select_feature_columns(df)
    y, label2id = make_target(df, multiclass=args.multiclass)
    df["_target"] = y

    df = preprocess_groupwise(df, feature_cols, group_cols)

    print("\n🔁 Iniciando validación cruzada por grupos...")
    row_group_keys = df["_group_key"].values
    splits, split_mode = get_group_splits(row_group_keys, cv_folds=args.cv_folds)
    print(f"✅ Estrategia CV: {split_mode} con {len(splits)} folds")

    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        df_train = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()

        print(f"\n--- Fold {fold_idx}/{len(splits)} ---")
        print(f"Grupos train: {df_train['_group_key'].nunique()} | grupos test: {df_test['_group_key'].nunique()}")
        print(f"Filas train: {len(df_train)} | filas test: {len(df_test)}")

        metrics = train_one_split(df_train, df_test, feature_cols, args, label2id)
        fold_metrics.append(metrics)

        print_eval_summary(metrics)

    cv_summary = {
        key: float(np.mean([m[key] for m in fold_metrics]))
        for key in fold_metrics[0].keys()
    }

    print("\n📊 Media CV:")
    print_eval_summary(cv_summary)

    if args.final_fit_full_data:
        final_train_on_full_data(
            df=df,
            feature_cols=feature_cols,
            label2id=label2id,
            group_cols=group_cols,
            args=args,
            output_dir=output_dir,
            model_name=model_name,
            cv_summary=cv_summary,
        )

    print("\n🎉 Entrenamiento completado!")


if __name__ == "__main__":
    main()