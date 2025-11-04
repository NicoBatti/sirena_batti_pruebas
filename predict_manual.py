"""
Script para predecir la altura del río de mañana usando el modelo
entrenado `modelo_final_optimizado.h5` y los scalers guardados.

Cómo usar:
  - Asegurate de tener en el mismo directorio:
      - modelo_final_optimizado.h5
      - scaler_X_final.pkl
      - scaler_y_final.pkl
  - Ejecutar: python predict_manual.py
  - El script preguntará los valores manuales (últimos N días + precip y estación de mañana).

Notas:
  - El script intenta inferir automáticamente el `window_size` a partir
    de la forma de entrada del modelo. Si no puede, te pedirá que lo indiques.
  - Para la estación podés ingresar: verano, otoño/otono, invierno, primavera
"""

import os
import sys
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import date, timedelta
import json
from datetime import datetime


#Por el momento los HARDCODEAMOS. Esto debería obtenerse automaticamente.
series_id = "6746"
site_code = "2123"

def limpiar_estacion(text):
    if text is None:
        return ''
    t = text.strip().lower()
    t = t.replace('ó', 'o')
    return t


def estacion_one_hot(est):
    t = limpiar_estacion(est)
    mapping = {
        'verano': [1, 0, 0, 0],
        'otoño': [0, 1, 0, 0],
        'otono': [0, 1, 0, 0],
        'invierno': [0, 0, 1, 0],
        'primavera': [0, 0, 0, 1]
    }
    vec = mapping.get(t)
    if vec is None:
        raise ValueError(f"Estación desconocida: '{est}'. Usar: verano, otoño, invierno, primavera")
    return vec


def pedir_float(prompt):
    while True:
        val = input(prompt)
        try:
            return float(val)
        except Exception:
            print("Entrada inválida. Ingresá un número (ej: 1.23).")

def llamar_api_alturas(window_size):
    import requests
    alturas = []
    print(f"\nLlamando a la API para obtener las alturas de los últimos {window_size} días:")
    hoy = date.today()
    # calcular fecha de inicio (oldest) considerando window_size días históricos
    # usamos window_size días incluyendo hoy- (window_size-1) .. hoy
    dia_mas_antiguo = hoy - timedelta(days=(window_size - 1))
    dia_7 = hoy + timedelta(days=7)
    # Formatear fechas en ISO para la query
    time_start = dia_mas_antiguo.isoformat()
    time_end = dia_7.isoformat()
    print(f"  Fechas para la API: desde {time_start} hasta {time_end}")
    url = f"https://alerta.ina.gob.ar/pub/datos/datos&timeStart={time_start}&timeEnd={time_end}&seriesId={series_id}&siteCode={site_code}&varId=2&format=json"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as ex:
        print(f"Error al llamar a la API: {ex}")
        return []
        url = f"https://alerta.ina.gob.ar/pub/datos/datos&timeStart={time_start}&timeEnd={time_end}&seriesId={series_id}&siteCode={site_code}&varId=2&format=json"
    # parsear JSON
    try:
        data = response.json()
    except Exception:
        print("Respuesta de la API no es JSON válido")
        return []

    # Normalizar 'data' a una lista de registros llamada 'records'
    records = None
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        # preferimos la clave 'data' si existe
        if 'data' in data and isinstance(data['data'], list):
            records = data['data']
        else:
            # buscar la primera lista en los valores del dict
            for v in data.values():
                if isinstance(v, list):
                    records = v
                    break

    if records is None:
        print("No se encontró una lista de registros en la respuesta JSON. Tipo recibido:", type(data))
        print("Respuesta (sample):", str(data)[:500])
        return []
    data = data["data"]

    promedios, vals, dia = [], [], None
    for r in data:
        # Esperamos que cada r tenga 'timeend' y 'valor'
        try:
            f = datetime.fromisoformat(r.get("timeend")).date()
            valor = r.get("valor")
        except Exception:
            # si el formato no coincide, saltamos el registro
            continue

        if valor is None:
            continue

        if dia is None:
            dia = f

        if f == dia:
            vals.append(float(valor))
        else:
            if len(vals) > 0:
                promedios.append(sum(vals) / len(vals))
            dia, vals = f, [float(valor)]

    if len(vals) > 0:
        promedios.append(sum(vals) / len(vals))

    print(f"  Alturas obtenidas (promedio por día): {promedios}")
    return promedios



def construir_vector_input(window_size):
    """Pide por consola los valores de los últimos `window_size` días y los de mañana.
    Forma esperada por el notebook original: por cada día histórico ->
      [altura_value, precipitaciones_value, est_verano, est_otoño, est_invierno, est_primavera]
    y al final se agregan los 5 valores futuros: [precipitaciones_tomorrow, est_* (4 valores)].
    Devuelve un array shape (1, input_dim).
    """
    features = []
    print(f"\nIngresá los datos de los últimos {window_size} días (del más antiguo al más reciente):")
    for i in range(window_size):
        print(f"\nDía {i+1}:")
        altura = pedir_float("  Altura (m): ")
        precip = pedir_float("  Precipitación (mm): ")
        est = input("  Estación (verano/otono/invierno/primavera): ")
        onehot = estacion_one_hot(est)
        features.extend([altura, precip] + onehot)

    print('\nAhora ingresá la precipitación y la estación previstas para mañana:')
    precip_manana = pedir_float("  Precipitación mañana (mm): ")
    est_manana = input("  Estación mañana (verano/otono/invierno/primavera): ")
    onehot_manana = estacion_one_hot(est_manana)
    futuro = [precip_manana] + onehot_manana

    features.extend(futuro)
    X = np.array(features, dtype=float).reshape(1, -1)
    return X


def llamar_api_alturas_range(start_date, end_date):
    """Llama a la API y devuelve una lista de alturas (promedio diario) para cada fecha
    en el rango [start_date, end_date] (inclusive). Si la API no tiene datos para
    una fecha en particular, el valor será None.

    start_date, end_date: objetos datetime.date
    Devuelve: lista de floats/None en orden cronológico (oldest -> newest)
    """
    import requests

    # Formatear fechas en ISO para la query
    time_start = start_date.isoformat()
    time_end = end_date.isoformat()
    url = f"https://alerta.ina.gob.ar/pub/datos/datos&timeStart={time_start}&timeEnd={time_end}&seriesId={series_id}&siteCode={site_code}&varId=2&format=json"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as ex:
        print(f"Error al llamar a la API: {ex}")
        return []

    # parsear JSON
    try:
        data = response.json()
    except Exception:
        print("Respuesta de la API no es JSON válido")
        return []

    # Normalizar 'data' a una lista de registros llamada 'records'
    records = None
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        if 'data' in data and isinstance(data['data'], list):
            records = data['data']
        else:
            for v in data.values():
                if isinstance(v, list):
                    records = v
                    break

    if records is None:
        print("No se encontró una lista de registros en la respuesta JSON. Tipo recibido:", type(data))
        print("Respuesta (sample):", str(data)[:500])
        return []

    # Agrupar por día y calcular promedio por día
    daily = {}
    for r in records:
        try:
            f = datetime.fromisoformat(r.get("timeend")).date()
            valor = r.get("valor")
        except Exception:
            continue
        if valor is None:
            continue
        daily.setdefault(f, []).append(float(valor))

    # Construir lista ordenada de fechas en el rango
    n_days = (end_date - start_date).days + 1
    out = []
    fechas = [start_date + timedelta(days=i) for i in range(n_days)]
    for d in fechas:
        if d in daily and len(daily[d]) > 0:
            out.append(sum(daily[d]) / len(daily[d]))
        else:
            out.append(None)

    print(f"  Alturas obtenidas por fecha ({time_start} .. {time_end}): {out}")
    return out


def llamar_api_alturas(window_size, include_future_days=0):
    """Compatibilidad: devuelve alturas desde (today - (window_size-1)) hasta today+include_future_days
    Por defecto include_future_days=0 (solo histórico). Si pedís include_future_days>0,
    la función devolverá None en las fechas futuras si la API no tiene datos.
    """
    hoy = date.today()
    start = hoy - timedelta(days=(window_size - 1))
    end = hoy + timedelta(days=include_future_days)
    return llamar_api_alturas_range(start, end)


def main():
    cwd = os.path.dirname(__file__)
    model_path = os.path.join(cwd, 'modelo_final_optimizado.h5')
    scaler_X_path = os.path.join(cwd, 'scaler_X_final.pkl')
    scaler_y_path = os.path.join(cwd, 'scaler_y_final.pkl')

    # Verificaciones básicas
    for p in (model_path, scaler_X_path, scaler_y_path):
        if not os.path.exists(p):
            print(f"ERROR: No se encontró '{os.path.basename(p)}' en {cwd}.")
            print("Colocá el modelo y los scalers en el mismo directorio que este script.")
            sys.exit(1)

    print("Cargando modelo y scalers...")
    # Algunos modelos guardados incluyen métricas que pueden fallar al deserializar.
    # Para predecir no necesitamos compilar el modelo, así que cargamos con compile=False
    # Esto evita errores como: "Could not deserialize 'keras.metrics.mse'..."
    try:
        model = load_model(model_path)
    except ValueError as e:
        msg = str(e)
        if 'Could not deserialize' in msg or 'keras.metrics' in msg:
            print("Advertencia: error al deserializar métricas del modelo. Cargando con compile=False para continuar (solo predicción).")
            model = load_model(model_path, compile=False)
        else:
            raise
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    # Determinar dimensión de entrada
    try:
        input_dim = model.input_shape[1]
    except Exception:
        print("No se pudo determinar la forma de entrada del modelo.")
        sys.exit(1)

    # Intentar inferir window_size: (input_dim - 5) debe ser divisible por 6
    inferred = None
    if (input_dim - 5) > 0 and (input_dim - 5) % 6 == 0:
        inferred = (input_dim - 5) // 6
        print(f"Se detectó input_dim={input_dim}, infiriendo window_size={inferred} (automático).")
    else:
        # Ambigüedad: pedir al usuario
        print(f"No fue posible inferir automáticamente window_size desde input_dim={input_dim}.")
        while True:
            try:
                inferred = int(input("Ingresá manualmente window_size (ej: 3): "))
                if inferred > 0:
                    break
            except Exception:
                pass
            print("Valor inválido. Intentá nuevamente.")

    # Construir vector de entrada pidiendo datos manuales
    X_manual = construir_vector_input(inferred)

    # Validar tamaño
    if X_manual.shape[1] != input_dim:
        print(f"ERROR: El vector ingresado tiene dimensión {X_manual.shape[1]} pero el modelo espera {input_dim}.")
        print("Verificá window_size e ingresá nuevamente.")
        sys.exit(1)

    # Escalar, predecir e invertir escala
    X_scaled = scaler_X.transform(X_manual)
    y_pred_scaled = model.predict(X_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()[0]

    print("\n========================================")
    print(f"Predicción -> Altura prevista para mañana: {y_pred:.4f} m")
    print("========================================\n")

    # -----------------------------
    # Predicciones iterativas hasta 7 días
    # -----------------------------
    preds = [y_pred]

    # Flattened raw feature vector
    current_flat = X_manual.flatten()
    total_days = (input_dim - 5) // 6  # inferred window_size

    # Extraer el bloque 'future' que usó la primera predicción (precip y estación de día+1)
    prev_future_precip = float(current_flat[6 * total_days])
    prev_future_est = current_flat[6 * total_days + 1: 6 * total_days + 5].tolist()

    print("Iniciando predicciones iterativas usando la predicción anterior como input...")

    # Repetir hasta tener 7 predicciones en total
    while len(preds) < 7:
        dia_objetivo = len(preds) + 1  # si preds tiene 1 -> dia_objetivo=2 (día+2)
        print(f"\n--- Preparando predicción para Día +{dia_objetivo} ---")

        # Pedir al usuario la precipitación y la estación para el día a predecir
        precip_next = pedir_float(f"Ingresá precipitación para Día +{dia_objetivo} (mm): ")
        est_next = input(f"Ingresá estación para Día +{dia_objetivo} (verano/otono/invierno/primavera): ")
        onehot_next = estacion_one_hot(est_next)

        # Construir nueva ventana: desplazar los días históricos y agregar el día creado a partir
        # de la última predicción + los valores (precip y estación) que antes fueron el 'future'
        # days part: tomar desde el segundo día histórico hasta el último
        days_shifted = current_flat[6: 6 * total_days]  # elimina el primer día (6 valores)

        # Nuevo día histórico que entra en la ventana: [pred_último, prev_future_precip, prev_future_est(4)]
        nuevo_dia = np.array([preds[-1], prev_future_precip] + prev_future_est, dtype=float)

        new_window_days = np.concatenate([days_shifted, nuevo_dia])

        # Nuevo bloque future para la predicción actual: precip_next + onehot_next
        new_future_block = np.array([precip_next] + onehot_next, dtype=float)

        # Nuevo vector de entrada completo
        new_input = np.concatenate([new_window_days, new_future_block]).reshape(1, -1)

        if new_input.shape[1] != input_dim:
            raise ValueError(f"Dimensión incorrecta del input iterativo: {new_input.shape[1]} vs esperado {input_dim}")

        # Escalar y predecir
        new_input_scaled = scaler_X.transform(new_input)
        y_pred_scaled_next = model.predict(new_input_scaled, verbose=0)
        y_pred_next = scaler_y.inverse_transform(y_pred_scaled_next).flatten()[0]

        print(f"Predicción Día +{dia_objetivo}: {y_pred_next:.4f} m")

        # Guardar predicción
        preds.append(float(y_pred_next))

        # Actualizar variables para la siguiente iteración
        prev_future_precip = float(new_future_block[0])
        prev_future_est = new_future_block[1:].tolist()
        current_flat = new_input.flatten()

    # Mostrar array final de 7 predicciones
    print("\n========================================")
    print("Array de predicciones (Día +1 .. Día +7):")
    print(np.array(preds))
    print("========================================\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Ocurrió un error: {e}")
        raise
