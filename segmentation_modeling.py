import os
import gc
import time
import datetime
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# -----------------------
# SETTINGS
# -----------------------
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# HYPERPARAMETERS (copiados del notebook)
HYPERPARAMETERS = {
    'window_size': [3, 5, 7],  # ejemplo desde notebook
    'arquitectura': [
        [64, 32, 16],
        [128, 64],
        [32, 32, 32],
        [64, 32],
    ],
    'learning_rate': [0.001, 0.0001],
}
N_SPLITS = 5
EPOCHS = 100
BATCH_SIZE = 32
VERBOSE = 0

# -----------------------
# TimeSeriesKFold (igual que en notebook)
# -----------------------
class TimeSeriesKFold:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X):
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        for i in range(self.n_splits):
            train_end = fold_size * (i + 2)
            train_indices = np.arange(0, train_end - fold_size)
            val_start = train_end - fold_size
            val_end = train_end
            val_indices = np.arange(val_start, val_end)
            yield train_indices, val_indices

# -----------------------
# Función estación (igual al notebook)
# -----------------------
def obtener_estacion(fecha):
    mes = fecha.month
    dia = fecha.day
    # misma lógica del notebook (dejada tal cual)
    if (mes == 12 and dia >= 21) or (mes <= 3 and (mes < 3 or dia <= 20)):
        return "verano"
    elif (mes == 3 and dia >= 21) or (mes <= 6 and (mes < 6 or dia <= 20)):
        return "otoño"
    elif (mes == 6 and dia >= 21) or (mes <= 9 and (mes < 9 or dia <= 20)):
        return "invierno"
    else:
        return "primavera"

# -----------------------
# crear_ventanas (según notebook original)
# - historial: últimos window_size días (i-window_size .. i-1)
# - futuro: columnas future_cols en i
# - target: altura_value en i
# -----------------------
def crear_ventanas(df, features, window_size=3):
    hist_cols = features
    future_cols = features[1:]  # quitamos altura
    X = []
    y = []
    # iteramos i desde window_size hasta len(df)-1 (igual al notebook)
    for i in range(window_size, len(df) - 0):
        # historia: i-window_size .. i-1
        ventana = df[hist_cols].iloc[i - window_size:i].values.flatten()
        # futuro = características del día i (precip + estaciones)
        futuro = df[future_cols].iloc[i].values.flatten()
        features_vector = np.concatenate([ventana, futuro])
        X.append(features_vector)
        y.append(df['altura_value'].iloc[i])
    if len(X) == 0:
        return np.empty((0,)), np.empty((0,))
    return np.array(X), np.array(y)

# -----------------------
# crear_modelo (igual que notebook)
# -----------------------
def crear_modelo(input_shape, arquitectura, learning_rate=0.001):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    for neurons in arquitectura:
        model.add(layers.Dense(neurons, activation='relu'))
    model.add(layers.Dense(1))
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# -----------------------
# Pipeline que replica el notebook (grid + k-fold + final train + evaluación)
# -----------------------
def run_notebook_pipeline(csv_path="registros_rio_6746.csv"):
    # Cargar datos (igual que notebook)
    df = pd.read_csv(csv_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)

    # Crear estación y one-hot encoding (igual al notebook)
    df['estacion'] = df['date'].apply(obtener_estacion)
    df = pd.get_dummies(df, columns=['estacion'], prefix='est')

    # Eliminar columnas innecesarias si existen (igual notebook)
    columnas_a_eliminar = ['Unnamed: 0', 'rio_id', 'lat', 'lon', 'altura_7_dias', 'precipitaciones_7_dias']
    df = df.drop(columns=columnas_a_eliminar, errors='ignore')

    print(f"Datos cargados: {len(df)} registros. Rango: {df['date'].min()} a {df['date'].max()}")
    # Split temporal (80/20) - igual que notebook
    test_size = 0.20
    split_point = int(len(df) * (1 - test_size))
    df_train_val = df.iloc[:split_point].copy()
    df_test = df.iloc[split_point:].copy()
    print(f"Train+Val: {len(df_train_val)}, Test: {len(df_test)}")

    # Features como en notebook
    features = ['altura_value', 'precipitaciones_value',
                'est_verano', 'est_otoño', 'est_invierno', 'est_primavera']

    # Grid search + K-Fold
    resultados = []
    total_configs = (len(HYPERPARAMETERS['window_size']) *
                     len(HYPERPARAMETERS['arquitectura']) *
                     len(HYPERPARAMETERS['learning_rate']))
    config_num = 0
    kfold = TimeSeriesKFold(n_splits=N_SPLITS)

    for window_size, arquitectura, lr in product(HYPERPARAMETERS['window_size'],
                                                 HYPERPARAMETERS['arquitectura'],
                                                 HYPERPARAMETERS['learning_rate']):
        config_num += 1
        print("\n" + "=" * 60)
        print(f"CONFIG {config_num}/{total_configs} -> window={window_size}, arch={arquitectura}, lr={lr}")
        start_cfg = time.time()

        try:
            X, y = crear_ventanas(df_train_val, features, window_size=window_size)
            if X.size == 0:
                print("No hay suficientes ventanas para esta configuración. Saltando.")
                continue

            fold_scores = []
            fold_idx = 0
            for train_idx, val_idx in kfold.split(X):
                fold_idx += 1
                if len(train_idx) == 0 or len(val_idx) == 0:
                    continue

                X_train_fold = X[train_idx]
                X_val_fold = X[val_idx]
                y_train_fold = y[train_idx]
                y_val_fold = y[val_idx]

                scaler_X_fold = MinMaxScaler()
                scaler_y_fold = MinMaxScaler()

                X_train_scaled = scaler_X_fold.fit_transform(X_train_fold)
                X_val_scaled = scaler_X_fold.transform(X_val_fold)

                y_train_scaled = scaler_y_fold.fit_transform(y_train_fold.reshape(-1,1)).flatten()
                y_val_scaled = scaler_y_fold.transform(y_val_fold.reshape(-1,1)).flatten()

                # crear y entrenar modelo
                K.clear_session()
                tf.random.set_seed(SEED + fold_idx)
                model = crear_modelo(X_train_scaled.shape[1], arquitectura, learning_rate=lr)

                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)
                ]

                model.fit(X_train_scaled, y_train_scaled,
                          epochs=EPOCHS, batch_size=BATCH_SIZE,
                          validation_data=(X_val_scaled, y_val_scaled),
                          verbose=VERBOSE, callbacks=callbacks)

                y_pred_scaled = model.predict(X_val_scaled, verbose=0)
                y_pred = scaler_y_fold.inverse_transform(y_pred_scaled)
                y_val_real = scaler_y_fold.inverse_transform(y_val_fold.reshape(-1,1))

                mae = mean_absolute_error(y_val_real, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val_real, y_pred))
                fold_scores.append({'mae': float(mae), 'rmse': float(rmse)})

                # limpiar fold
                K.clear_session()
                del model
                gc.collect()

            if not fold_scores:
                print("No se obtuvieron folds válidos para esta config.")
                continue

            mae_mean = float(np.mean([f['mae'] for f in fold_scores]))
            mae_std = float(np.std([f['mae'] for f in fold_scores]))
            rmse_mean = float(np.mean([f['rmse'] for f in fold_scores]))
            rmse_std = float(np.std([f['rmse'] for f in fold_scores]))

            resultados.append({
                'window_size': window_size,
                'arquitectura': str(arquitectura),
                'learning_rate': lr,
                'mae_mean': mae_mean,
                'mae_std': mae_std,
                'rmse_mean': rmse_mean,
                'rmse_std': rmse_std,
                'mae_folds': [f['mae'] for f in fold_scores],
                'rmse_folds': [f['rmse'] for f in fold_scores],
            })

            elapsed = time.time() - start_cfg
            print(f"Config done in {elapsed:.1f}s | MAE: {mae_mean:.6f} ± {mae_std:.6f}")

        except Exception as e:
            print("ERROR en configuración:", e)
            traceback.print_exc()
            resultados.append({
                'window_size': window_size,
                'arquitectura': str(arquitectura),
                'learning_rate': lr,
                'error': str(e)
            })
            continue

    # Guardar resultados y elegir mejor (por mae_mean)
    df_resultados = pd.DataFrame(resultados)
    df_resultados_sorted = df_resultados.dropna(subset=['mae_mean']).sort_values('mae_mean')
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df_resultados.to_csv(os.path.join(MODELS_DIR, f"resultados_kfold_{ts}.csv"), index=False)

    if df_resultados_sorted.empty:
        print("No se encontraron configuraciones válidas.")
        return

    mejor = df_resultados_sorted.iloc[0]
    best_window = int(mejor['window_size'])
    best_arch = eval(mejor['arquitectura'])
    best_lr = float(mejor['learning_rate'])
    print("\n" + "="*60)
    print("MEJOR CONFIGURACIÓN ENCONTRADA:")
    print(mejor.to_dict())

    # Entrenar modelo final con train+val
    X_train_final, y_train_final = crear_ventanas(df_train_val, features, window_size=best_window)
    scaler_X_final = MinMaxScaler().fit(X_train_final)
    scaler_y_final = MinMaxScaler().fit(y_train_final.reshape(-1,1))

    X_train_final_scaled = scaler_X_final.transform(X_train_final)
    y_train_final_scaled = scaler_y_final.transform(y_train_final.reshape(-1,1)).flatten()

    K.clear_session()
    tf.random.set_seed(SEED)
    modelo_final = crear_modelo(X_train_final_scaled.shape[1], best_arch, learning_rate=best_lr)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]

    history_final = modelo_final.fit(X_train_final_scaled, y_train_final_scaled,
                                     epochs=EPOCHS, batch_size=BATCH_SIZE,
                                     validation_split=0.2, verbose=1, callbacks=callbacks)

    # Guardar modelos y scalers
    model_fname = os.path.join(MODELS_DIR, f"modelo_final_{ts}.h5")
    scalerX_fname = os.path.join(MODELS_DIR, f"scaler_X_{ts}.pkl")
    scalerY_fname = os.path.join(MODELS_DIR, f"scaler_y_{ts}.pkl")
    modelo_final.save(model_fname)
    joblib.dump(scaler_X_final, scalerX_fname)
    joblib.dump(scaler_y_final, scalerY_fname)
    print(f"Modelo guardado en {model_fname}")

    # Evaluación en test (mínima)
    X_test, y_test = crear_ventanas(df_test, features, window_size=best_window)
    if X_test.size > 0:
        X_test_scaled = scaler_X_final.transform(X_test)
        y_pred_scaled = modelo_final.predict(X_test_scaled, verbose=0)
        y_pred = scaler_y_final.inverse_transform(y_pred_scaled)
        mae_test = mean_absolute_error(y_test.reshape(-1,1), y_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test.reshape(-1,1), y_pred))
        print(f"Test MAE: {mae_test:.6f}, RMSE: {rmse_test:.6f}")
    else:
        print("No hay suficientes registros en test para evaluación.")

    # Opcional: puedes añadir la sección de predicciones iterativas y gráficas tal como en el notebook
    # (omito por brevedad, pero se puede copiar directamente desde el notebook si la querés)

    K.clear_session()
    gc.collect()
    return df_resultados_sorted

if __name__ == "__main__":
    # Ejecutar pipeline (ajusta csv si es necesario)
    run_notebook_pipeline("registros_rio_6746.csv")
