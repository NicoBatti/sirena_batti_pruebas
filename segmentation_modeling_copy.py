import os
import gc
import time
import datetime
import traceback
import pandas as pd
import numpy as np
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

# HYPERPARAMETERS
HYPERPARAMETERS = {
    'window_size': [3, 5, 7],
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
# TimeSeriesKFold
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
# Función estación
# -----------------------
def obtener_estacion(fecha):
    mes = fecha.month
    dia = fecha.day
    if (mes == 12 and dia >= 21) or (mes <= 3 and (mes < 3 or dia <= 20)):
        return "verano"
    elif (mes == 3 and dia >= 21) or (mes <= 6 and (mes < 6 or dia <= 20)):
        return "otoño"
    elif (mes == 6 and dia >= 21) or (mes <= 9 and (mes < 9 or dia <= 20)):
        return "invierno"
    else:
        return "primavera"


# -----------------------
# Crear ventanas
# -----------------------
def crear_ventanas(df, features, window_size=3):
    hist_cols = features
    future_cols = features[1:]
    X, y = [], []
    for i in range(window_size, len(df)):
        ventana = df[hist_cols].iloc[i - window_size:i].values.flatten()
        futuro = df[future_cols].iloc[i].values.flatten()
        X.append(np.concatenate([ventana, futuro]))
        y.append(df['altura_value'].iloc[i])
    if len(X) == 0:
        return np.empty((0,)), np.empty((0,))
    return np.array(X), np.array(y)


# -----------------------
# Crear modelo
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
# Pipeline para UN río
# -----------------------
def run_pipeline_for_rio(df_rio, rio_id):
    df = df_rio.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)
    df['estacion'] = df['date'].apply(obtener_estacion)
    df = pd.get_dummies(df, columns=['estacion'], prefix='est')

    columnas_a_eliminar = ['Unnamed: 0', 'lat', 'lon', 'altura_7_dias', 'precipitaciones_7_dias']
    df = df.drop(columns=columnas_a_eliminar, errors='ignore')

    print(f"\n=== Entrenando modelo para río {rio_id} ===")
    print(f"Registros: {len(df)} | Fechas: {df['date'].min()} a {df['date'].max()}")

    # Split 80/20 temporal
    split_point = int(len(df) * 0.8)
    df_train_val = df.iloc[:split_point].copy()
    df_test = df.iloc[split_point:].copy()

    features = ['altura_value', 'precipitaciones_value', 'est_verano', 'est_otoño', 'est_invierno', 'est_primavera']

    # GRID SEARCH + KFOLD
    resultados = []
    kfold = TimeSeriesKFold(n_splits=N_SPLITS)

    for window_size, arquitectura, lr in product(
        HYPERPARAMETERS['window_size'],
        HYPERPARAMETERS['arquitectura'],
        HYPERPARAMETERS['learning_rate']
    ):
        X, y = crear_ventanas(df_train_val, features, window_size)
        if X.size == 0:
            continue
        fold_scores = []

        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler_X = MinMaxScaler().fit(X_train)
            scaler_y = MinMaxScaler().fit(y_train.reshape(-1, 1))

            X_train_scaled = scaler_X.transform(X_train)
            X_val_scaled = scaler_X.transform(X_val)
            y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
            y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

            K.clear_session()
            model = crear_modelo(X_train_scaled.shape[1], arquitectura, lr)

            model.fit(
                X_train_scaled, y_train_scaled,
                validation_data=(X_val_scaled, y_val_scaled),
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                verbose=0,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)
                ]
            )

            y_pred_scaled = model.predict(X_val_scaled, verbose=0)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_val_real = scaler_y.inverse_transform(y_val.reshape(-1, 1))

            mae = mean_absolute_error(y_val_real, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val_real, y_pred))
            fold_scores.append({'mae': mae, 'rmse': rmse})

        if not fold_scores:
            continue

        mae_mean = np.mean([f['mae'] for f in fold_scores])
        rmse_mean = np.mean([f['rmse'] for f in fold_scores])
        resultados.append((window_size, arquitectura, lr, mae_mean, rmse_mean))

    if not resultados:
        print(f"⚠️ No se pudo entrenar ningún modelo para río {rio_id}")
        return

    # Mejor configuración
    best_config = sorted(resultados, key=lambda x: x[3])[0]
    best_window, best_arch, best_lr = best_config[:3]

    print(f"✅ Mejor configuración para río {rio_id}: window={best_window}, arch={best_arch}, lr={best_lr}")

    # Entrenamiento final
    X_train_final, y_train_final = crear_ventanas(df_train_val, features, best_window)
    scaler_X_final = MinMaxScaler().fit(X_train_final)
    scaler_y_final = MinMaxScaler().fit(y_train_final.reshape(-1, 1))

    X_train_scaled = scaler_X_final.transform(X_train_final)
    y_train_scaled = scaler_y_final.transform(y_train_final.reshape(-1, 1)).flatten()

    K.clear_session()
    model_final = crear_modelo(X_train_scaled.shape[1], best_arch, best_lr)
    model_final.fit(
        X_train_scaled, y_train_scaled,
        validation_split=0.2,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)
        ]
    )

    # Guardado
    model_path = os.path.join(MODELS_DIR, f"modelo_{rio_id}.h5")
    scalerX_path = os.path.join(MODELS_DIR, f"scaler_X_{rio_id}.pkl")
    scalerY_path = os.path.join(MODELS_DIR, f"scaler_y_{rio_id}.pkl")
    model_final.save(model_path)
    joblib.dump(scaler_X_final, scalerX_path)
    joblib.dump(scaler_y_final, scalerY_path)

    # Evaluación en test
    X_test, y_test = crear_ventanas(df_test, features, best_window)
    if X_test.size > 0:
        X_test_scaled = scaler_X_final.transform(X_test)
        y_pred_scaled = model_final.predict(X_test_scaled, verbose=0)
        y_pred = scaler_y_final.inverse_transform(y_pred_scaled)
        mae_test = mean_absolute_error(y_test.reshape(-1, 1), y_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test.reshape(-1, 1), y_pred))
        print(f"📊 Test MAE: {mae_test:.4f} | RMSE: {rmse_test:.4f}")

        metrics_path = os.path.join(MODELS_DIR, f"metricas_{rio_id}.csv")
        pd.DataFrame([{
            'rio_id': rio_id,
            'mae_test': mae_test,
            'rmse_test': rmse_test,
            'best_window': best_window,
            'best_arch': str(best_arch),
            'best_lr': best_lr
        }]).to_csv(metrics_path, index=False)
    else:
        print("⚠️ No hay datos suficientes en test.")


# -----------------------
# MAIN LOOP: Entrenar por cada río
# -----------------------
if __name__ == "__main__":
    df_full = pd.read_csv("rios_potables_filtrados.csv")
    rios_unicos = df_full['rio_id'].unique()
    print(f"\nSe encontraron {len(rios_unicos)} ríos: {rios_unicos}")

    for rio_id in rios_unicos:
        df_rio = df_full[df_full['rio_id'] == rio_id]
        run_pipeline_for_rio(df_rio, rio_id)
        gc.collect()
