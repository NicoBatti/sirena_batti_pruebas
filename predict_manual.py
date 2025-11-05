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
import math
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import date, timedelta
import json
from datetime import datetime
import argparse
import csv
import time
import hashlib
from pathlib import Path


#Por el momento los HARDCODEAMOS. Esto debería obtenerse automaticamente.
series_id = "6746"
site_code = "2123"
lat = "-36.3977777777778"
lon = "-67.1402777777778"

# Runtime configurable globals (will be set from CLI args in main)
RETRY_COUNT = 3
TIMEOUT = 15
CACHE_DIR = Path('./cache')


class FetchError(Exception):
    pass


def _cache_path_for_url(url: str, cache_dir: Path) -> Path:
    key = hashlib.md5(url.encode('utf-8')).hexdigest()
    return cache_dir / f"api_{key}.json"


def fetch_json_with_cache(url: str, retries: int = 3, timeout: int = 15, cache_dir: Path = Path('./cache')):
    """Fetch JSON from URL with simple retries and a local file cache.

    Returns the parsed JSON on success. If network fails and cache exists, returns cached JSON.
    Raises FetchError if neither network nor cache available.
    """
    import requests

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path_for_url(url, cache_dir)

    attempt = 0
    last_exc = None
    backoff = 1.0
    while attempt < retries:
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            # write cache
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump({'fetched_at': str(datetime.utcnow()), 'url': url, 'data': data}, f, ensure_ascii=False)
            except Exception:
                pass
            return data
        except Exception as ex:
            last_exc = ex
            attempt += 1
            time.sleep(backoff)
            backoff *= 2

    # retries exhausted
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
                print(f"Usando cache local {cache_path} después de fallo al conectar: {last_exc}")
                return cached.get('data', cached)
        except Exception:
            pass

    raise FetchError(f"No se pudo obtener JSON de {url}: {last_exc}")

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


def get_season_onehot_for_date(dt):
    """Devuelve el one-hot [verano, otono, invierno, primavera] para una fecha (datetime.date)."""
    mes = dt.month
    dia = dt.day
    # Usamos la misma lógica que en el notebook
    if (mes == 12 and dia >= 21) or (mes <= 3 and (mes < 3 or dia <= 20)):
        nombre = 'verano'
    elif (mes == 3 and dia >= 21) or (mes <= 6 and (mes < 6 or dia <= 20)):
        nombre = 'otono'
    elif (mes == 6 and dia >= 21) or (mes <= 9 and (mes < 9 or dia <= 20)):
        nombre = 'invierno'
    else:
        nombre = 'primavera'
    return estacion_one_hot(nombre)


def llamar_api_alturas(window_size, include_future_days=0):
    """Obtener alturas promedio por día desde (today-(window_size-1)) hasta today+include_future_days.
    Por defecto include_future_days=0 (solo pasado y hoy). La API de alturas normalmente no
    entrega datos futuros; si se solicitan días futuros, la función devolverá None para esas fechas.
    """
    import requests
    alturas = []
    hoy = date.today()
    # calcular fecha de inicio (oldest) considerando window_size días históricos
    dia_mas_antiguo = hoy - timedelta(days=(window_size - 1))
    # calcular fecha final (por defecto hoy)
    dia_final = hoy + timedelta(days=include_future_days)
    # Formatear fechas en ISO para la query
    time_start = dia_mas_antiguo.isoformat()
    time_end = dia_final.isoformat()
    print(f"\nLlamando a la API para obtener las alturas desde {time_start} hasta {time_end} (window_size={window_size})")
    url = f"https://alerta.ina.gob.ar/pub/datos/datos&timeStart={time_start}&timeEnd={time_end}&seriesId={series_id}&siteCode={site_code}&varId=2&format=json"
    try:
        data = fetch_json_with_cache(url, retries=RETRY_COUNT, timeout=TIMEOUT, cache_dir=CACHE_DIR)
    except FetchError as ex:
        print(f"Error al llamar a la API de alturas: {ex}")
        raise

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

    # Construir un mapa fecha -> lista de valores para promediar
    daily_vals = {}
    for r in records:
        try:
            t_str = r.get('timeend') or r.get('time') or r.get('date')
            if t_str is None:
                continue
            f = datetime.fromisoformat(t_str).date()
            valor = r.get('valor') if 'valor' in r else r.get('value') if 'value' in r else None
            if valor is None:
                continue
            daily_vals.setdefault(f, []).append(float(valor))
        except Exception:
            continue

    # Construir la lista de fechas solicitadas (desde dia_mas_antiguo hasta dia_final)
    fechas = []
    n_days = (dia_final - dia_mas_antiguo).days + 1
    for i in range(n_days):
        fechas.append(dia_mas_antiguo + timedelta(days=i))

    promedios = []
    for d in fechas:
        vals = daily_vals.get(d)
        if vals and len(vals) > 0:
            promedios.append(sum(vals) / len(vals))
        else:
            promedios.append(None)

    print(f"  Alturas obtenidas (promedio por día, con None para faltantes): {promedios}")
    # Truncar a 3 decimales (no redondear)
    def _truncate_list(lst, ndigits=3):
        factor = 10 ** ndigits
        outt = []
        for v in lst:
            if v is None:
                outt.append(None)
            else:
                outt.append(math.trunc(float(v) * factor) / factor)
        return outt

    promedios_trunc = _truncate_list(promedios, 3)
    print(f"  Alturas truncadas (3 decimales): {promedios_trunc}")
    return promedios_trunc

def llamar_api_precipitaciones(window_size, include_future_days=7):
    """Obtiene precipitaciones diarias (suma por día) desde (today-(window_size-1))
    hasta today+include_future_days. Devuelve lista de longitud window_size+include_future_days
    con valores (float) o None si faltan datos.
    """
    import requests

    hoy = date.today()
    start = hoy - timedelta(days=(window_size - 1))
    end = hoy + timedelta(days=include_future_days)

    time_start = start.isoformat()
    time_end = end.isoformat()
    print(f"\nLlamando a Open-Meteo para precipitación desde {time_start} hasta {time_end}")

    # Usamos hourly precipitation y luego sumamos por fecha. Incluimos start/end para limitar rango.
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&hourly=precipitation&timezone=UTC&start_date={time_start}&end_date={time_end}"
    )

    try:
        data = fetch_json_with_cache(url, retries=RETRY_COUNT, timeout=TIMEOUT, cache_dir=CACHE_DIR)
    except FetchError as ex:
        print(f"Error al llamar a la API de precipitaciones: {ex}")
        raise

    # Preferimos daily precipitation sums si están disponibles
    if isinstance(data, dict) and 'daily' in data and 'precipitation_sum' in data['daily']:
        dates = data['daily'].get('time', [])
        sums = data['daily'].get('precipitation_sum', [])
        daily_map = {datetime.fromisoformat(d).date(): float(s) for d, s in zip(dates, sums)}
    else:
        # Agregar desde hourly
        if 'hourly' not in data or 'time' not in data['hourly'] or 'precipitation' not in data['hourly']:
            print("Respuesta de Open-Meteo no tiene datos horarios esperados")
            return []
        times = data['hourly']['time']
        precs = data['hourly']['precipitation']
        daily_map = {}
        for t, p in zip(times, precs):
            try:
                d = datetime.fromisoformat(t).date()
            except Exception:
                continue
            daily_map.setdefault(d, 0.0)
            # p puede ser None
            try:
                daily_map[d] += float(p)
            except Exception:
                pass

    # Construir lista de salida ordenada por fecha
    n_days = (end - start).days + 1
    fechas = [start + timedelta(days=i) for i in range(n_days)]
    out = [daily_map.get(d, None) for d in fechas]

    # Construir array de estaciones (one-hot) para cada fecha en el mismo orden
    estaciones = [get_season_onehot_for_date(d) for d in fechas]

    # Truncar a 3 decimales (no redondear) antes de imprimir/devolver
    def _truncate_list(lst, ndigits=3):
        factor = 10 ** ndigits
        outt = []
        for v in lst:
            if v is None:
                outt.append(None)
            else:
                outt.append(math.trunc(float(v) * factor) / factor)
        return outt

    out_trunc = _truncate_list(out, 3)
    print(f"  Precipitaciones diarias ({time_start}..{time_end}): {out_trunc}")
    print(f"  Estaciones (one-hot) por fecha: {estaciones}")
    return out_trunc, estaciones

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


def _forward_fill_list(lst, fill_value=0.0):
    """Forward-fill None values in a list. Leading None replaced with first non-None or fill_value."""
    out = list(lst)
    # find first non-None
    first_val = None
    for v in out:
        if v is not None:
            first_val = v
            break
    if first_val is None:
        # all None
        return [fill_value for _ in out]
    # replace leading None
    for i in range(len(out)):
        if out[i] is None:
            out[i] = first_val
        else:
            break
    # forward fill rest
    for i in range(1, len(out)):
        if out[i] is None:
            out[i] = out[i-1]
    return out


def _impute_heights_option1(lst):
    """Imputación Option 1: forward-fill, backward-fill, median fallback.

    - If there are some non-None values: forward-fill then backward-fill to fill gaps.
    - If all values are None: fallback to median of training/present values; if none, use 0.0.
    """
    out = list(lst)
    # collect non-None values
    non_none = [float(x) for x in out if x is not None]
    if len(non_none) == 0:
        # no data at all, fallback to 0.0
        return [0.0 for _ in out]

    # forward fill leading/trailing/internal
    # first, replace leading None with first non-none
    first_val = non_none[0]
    for i in range(len(out)):
        if out[i] is None:
            out[i] = first_val
        else:
            break
    # forward fill rest
    for i in range(1, len(out)):
        if out[i] is None:
            out[i] = out[i-1]

    # backward fill pass to ensure trailing None handled (safety)
    for i in range(len(out)-2, -1, -1):
        if out[i] is None:
            out[i] = out[i+1]

    # final sanity: any remaining None -> median
    for i in range(len(out)):
        if out[i] is None:
            med = float(np.median(non_none))
            out[i] = med

    return out


def automatic_predict(model, scaler_X, scaler_y, window_size, forecast_horizon=7):
    """Ejecuta la pipeline automática usando las funciones de API y el modelo.

    Retorna lista de predicciones de longitud `forecast_horizon` (Día +1 .. Día +H).
    """
    # 1) Obtener datos desde las APIs
    print(f"\n[Automatic] Solicitando alturas históricas (window_size={window_size})...")
    hist_heights = llamar_api_alturas(window_size)
    if not hist_heights or len(hist_heights) < window_size:
        print(f"Advertencia: alturas históricas retornaron {len(hist_heights)} valores; se intentará imputar (ffill/bfill/median).")
    # imputar alturas usando Option 1 (forward-fill, backward-fill, median fallback)
    hist_heights_filled = _impute_heights_option1(hist_heights)

    print(f"[Automatic] Solicitando precipitaciones y estaciones (hist+{forecast_horizon} días)...")
    precs_all, estaciones_all = llamar_api_precipitaciones(window_size, include_future_days=forecast_horizon)

    expected_len = window_size + forecast_horizon
    if len(precs_all) != expected_len or len(estaciones_all) != expected_len:
        raise ValueError(f"Datos de precipitacion/estacion tienen longitudes inesperadas: precs={len(precs_all)}, est={len(estaciones_all)}, esperado={expected_len}")

    # Prepare mapping for predicted future heights (for indices >= window_size)
    predicted_map = {}
    preds = []
    # inputs_record: list of dicts with the input data used for each prediction
    inputs_record = []

    # For each horizon day k (1..forecast_horizon)
    for k in range(1, forecast_horizon + 1):
        # historical indices for this iteration: start_idx .. end_idx
        start_idx = k - 1
        hist_indices = list(range(start_idx, start_idx + window_size))

        # Build flattened feature vector: for each hist idx, get height, precip, station one-hot
        features = []
        for idx in hist_indices:
            # height: from hist_heights_filled if idx < window_size else from predicted_map
            if idx < window_size:
                h = hist_heights_filled[idx]
            else:
                h = predicted_map.get(idx)
                if h is None:
                    # fallback
                    h = 0.0
            p = precs_all[idx]
            # convert None precip to 0.0 as requested
            if p is None:
                p = 0.0
            s = estaciones_all[idx]
            # ensure station one-hot length ==4
            if s is None:
                s = [0,0,0,0]
            features.extend([h, p] + s)

        # future block index for this prediction
        future_idx = window_size + (k - 1)
        future_prec = precs_all[future_idx]
        if future_prec is None:
            future_prec = 0.0
        future_est = estaciones_all[future_idx]
        if future_est is None:
            future_est = [0,0,0,0]
        features.extend([future_prec] + future_est)

        X = np.array(features, dtype=float).reshape(1, -1)

        # scale and predict
        X_scaled = scaler_X.transform(X)
        y_pred_scaled = model.predict(X_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()[0]

        # truncate to 3 decimals (consistent with other functions)
        y_pred_trunc = math.trunc(float(y_pred) * 1000) / 1000
        preds.append(y_pred_trunc)

        # store predicted height at global index future_idx so next iterations use it
        predicted_map[future_idx] = y_pred_trunc
        # record the input used for this prediction (human-friendly)
        # historic per-day list
        hist_by_day = []
        for idx in hist_indices:
            # compute same values used above (heights already inlined)
            if idx < window_size:
                h_rec = hist_heights_filled[idx]
            else:
                h_rec = predicted_map.get(idx, 0.0)
            p_rec = precs_all[idx]
            if p_rec is None:
                p_rec = 0.0
            s_rec = estaciones_all[idx] or [0,0,0,0]
            hist_by_day.append({'height': h_rec, 'precip': p_rec, 'station_onehot': s_rec})

        future_block = {'precip': future_prec, 'station_onehot': future_est}
        inputs_record.append({'day_offset': k, 'historical_window': hist_by_day, 'future_block': future_block, 'flat_input': features})

        print(f"[Automatic] Predicción Día +{k}: {y_pred_trunc} m")

    print("\n[Automatic] Predicciones completas (Día +1 .. +{0}):".format(forecast_horizon))
    print(preds)
    return preds, inputs_record


def main():
    cwd = os.path.dirname(__file__)
    global RETRY_COUNT, TIMEOUT, CACHE_DIR
    model_path = os.path.join(cwd, 'modelo_final_optimizado.h5')
    scaler_X_path = os.path.join(cwd, 'scaler_X_final.pkl')
    scaler_y_path = os.path.join(cwd, 'scaler_y_final.pkl')

    # Verificaciones básicas
    for p in (model_path, scaler_X_path, scaler_y_path):
        if not os.path.exists(p):
            print(f"ERROR: No se encontró '{os.path.basename(p)}' en {cwd}.")
            print("Colocá el modelo y los scalers en el mismo directorio que este script.")
            sys.exit(1)
    # Parse CLI args
    parser = argparse.ArgumentParser(description='Predict river heights using saved model. Use --auto to run automatic pipeline using APIs.')
    parser.add_argument('--auto', action='store_true', help='Run automatic prediction pipeline (no interactive prompts).')
    parser.add_argument('--horizon', type=int, default=7, help='Forecast horizon in days when using --auto (default: 7)')
    parser.add_argument('--window-size', type=int, default=None, help='Override inferred window_size')
    parser.add_argument('--out', type=str, default=None, help='Optional CSV output path to save automatic predictions')
    parser.add_argument('--cache-dir', type=str, default=str(CACHE_DIR), help='Directory to store API caches (default: ./cache)')
    parser.add_argument('--retry-count', type=int, default=RETRY_COUNT, help='Number of retries for API calls (default: 3)')
    parser.add_argument('--timeout', type=int, default=TIMEOUT, help='Timeout seconds for API calls (default: 15)')
    parser.add_argument('--impute-mode', type=str, default='ffill_bfill_median', help='Imputation mode for heights: ffill_bfill_median | last | zero')
    args = parser.parse_args()

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

    # Allow CLI override of window_size
    if args.window_size is not None:
        print(f"Sobrescribiendo window_size inferido {inferred} por valor provisto en CLI: {args.window_size}")
        inferred = int(args.window_size)

    # Apply runtime options
    RETRY_COUNT = int(args.retry_count)
    TIMEOUT = int(args.timeout)
    CACHE_DIR = Path(args.cache_dir)

    # If automatic mode requested, run automatic_predict and exit
    if args.auto:
        print("Modo automático activado: ejecutando pipeline automatic_predict(...) usando las APIs.")
        try:
            preds, inputs_used = automatic_predict(model, scaler_X, scaler_y, inferred, forecast_horizon=args.horizon)
        except Exception as ex:
            # Save diagnostics and exit gracefully without prompting
            diag = {'error': str(ex), 'stage': 'automatic_predict', 'timestamp': str(datetime.utcnow())}
            diag_path = os.path.join(cwd, f'prediction_failed_{datetime.utcnow().strftime("%Y%m%dT%H%M%S")}.json')
            try:
                with open(diag_path, 'w', encoding='utf-8') as jf:
                    json.dump(diag, jf, ensure_ascii=False, indent=2)
                print(f"No se pudo completar la predicción automática. Diagnóstico guardado en: {diag_path}")
            except Exception:
                print(f"No se pudo completar la predicción automática. Error: {ex}")
            return

        # guardar a CSV si se indicó
        if args.out:
            out_path = os.path.join(cwd, args.out) if not os.path.isabs(args.out) else args.out
            try:
                # CSV con predicciones simples
                with open(out_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['day_offset', 'prediction_m'])
                    for i, p in enumerate(preds, start=1):
                        writer.writerow([i, p])
                print(f"Predicciones guardadas en: {out_path}")
                # Guardar inputs usados en JSON junto al CSV para facilitar repro y pasar a otros scripts
                json_path = out_path if out_path.lower().endswith('.json') else (out_path.rsplit('.', 1)[0] + '_inputs.json')
                try:
                    with open(json_path, 'w', encoding='utf-8') as jf:
                        json.dump({'predictions': preds, 'inputs': inputs_used}, jf, ensure_ascii=False, indent=2, default=str)
                    print(f"Inputs usados guardados en: {json_path}")
                except Exception as ex:
                    print(f"Error guardando JSON de inputs: {ex}")
            except Exception as ex:
                print(f"Error guardando CSV de predicciones: {ex}")
        else:
            # no out path: still print inputs_used summary
            print("Inputs usados (por día):")
            for rec in inputs_used:
                print(f" Día +{rec['day_offset']}: future_precip={rec['future_block']['precip']}, future_station={rec['future_block']['station_onehot']}")
        return

    # Si no se pidió --auto explícitamente, intentamos usar las APIs automáticamente.
    # Si las APIs responden con datos coherentes ejecutamos automatic_predict sin pedir entradas manuales.
    if not args.auto:
        try:
            print("Intentando obtener datos automáticamente desde las APIs (sin prompts)...")
            precs_try, ests_try = llamar_api_precipitaciones(inferred, include_future_days=args.horizon)
            heights_try = llamar_api_alturas(inferred)
        except Exception as ex:
            # Cannot reliably fetch APIs: save diagnostic and exit without prompting
            diag = {
                'error': 'API fetch failed',
                'exception': str(ex),
                'stage': 'fetch_apis',
                'timestamp': str(datetime.utcnow())
            }
            diag_path = os.path.join(cwd, f'prediction_no_data_{datetime.utcnow().strftime("%Y%m%dT%H%M%S")}.json')
            try:
                with open(diag_path, 'w', encoding='utf-8') as jf:
                    json.dump(diag, jf, ensure_ascii=False, indent=2)
                print(f"No se pudo obtener datos de las APIs. Diagnóstico guardado en: {diag_path}")
            except Exception:
                print(f"No se pudo obtener datos de las APIs y no se pudo guardar diagnóstico. Error: {ex}")
            return

        expected_len = inferred + args.horizon
        ok_precs = isinstance(precs_try, list) and len(precs_try) == expected_len
        ok_ests = isinstance(ests_try, list) and len(ests_try) == expected_len
        ok_heights = isinstance(heights_try, list) and len(heights_try) >= inferred

        if ok_precs and ok_ests and ok_heights:
            print("APIs proporcionaron datos completos: ejecutando pipeline automática.")
            try:
                preds, inputs_used = automatic_predict(model, scaler_X, scaler_y, inferred, forecast_horizon=args.horizon)
            except Exception as ex:
                diag = {'error': 'automatic_predict failed', 'exception': str(ex), 'stage': 'automatic_predict', 'timestamp': str(datetime.utcnow())}
                diag_path = os.path.join(cwd, f'prediction_failed_{datetime.utcnow().strftime("%Y%m%dT%H%M%S")}.json')
                try:
                    with open(diag_path, 'w', encoding='utf-8') as jf:
                        json.dump(diag, jf, ensure_ascii=False, indent=2)
                    print(f"La predicción automática falló. Diagnóstico guardado en: {diag_path}")
                except Exception:
                    print(f"La predicción automática falló: {ex}")
                return

            if args.out:
                out_path = os.path.join(cwd, args.out) if not os.path.isabs(args.out) else args.out
                try:
                    with open(out_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['day_offset', 'prediction_m'])
                        for i, p in enumerate(preds, start=1):
                            writer.writerow([i, p])
                    print(f"Predicciones guardadas en: {out_path}")
                    json_path = out_path if out_path.lower().endswith('.json') else (out_path.rsplit('.', 1)[0] + '_inputs.json')
                    with open(json_path, 'w', encoding='utf-8') as jf:
                        json.dump({'predictions': preds, 'inputs': inputs_used}, jf, ensure_ascii=False, indent=2, default=str)
                    print(f"Inputs usados guardados en: {json_path}")
                except Exception as ex:
                    print(f"Error guardando salida automática: {ex}")
            else:
                print("Predicciones automáticas:")
                print(preds)
            return
        else:
            # APIs returned unexpected structure/length: do not prompt, save diagnostic and exit
            diag = {
                'error': 'APIs returned incomplete data',
                'precs_len': len(precs_try) if isinstance(precs_try, list) else 'invalid',
                'ests_len': len(ests_try) if isinstance(ests_try, list) else 'invalid',
                'heights_len': len(heights_try) if isinstance(heights_try, list) else 'invalid',
                'expected_len': expected_len,
                'stage': 'validate_api_shapes',
                'timestamp': str(datetime.utcnow())
            }
            diag_path = os.path.join(cwd, f'prediction_no_data_{datetime.utcnow().strftime("%Y%m%dT%H%M%S")}.json')
            try:
                with open(diag_path, 'w', encoding='utf-8') as jf:
                    json.dump(diag, jf, ensure_ascii=False, indent=2)
                print(f"APIs no devolvieron datos completos. Diagnóstico guardado en: {diag_path}")
            except Exception:
                print("APIs no devolvieron datos completos y no se pudo guardar diagnóstico.")
            return

    # Manual prompts have been disabled earlier; exit.
    print("Modo no interactivo: no se solicitarán datos manuales. Si necesitas predicciones, ejecutá con --auto y verifica conectividad a las APIs.")
    return


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Ocurrió un error: {e}")
        raise
