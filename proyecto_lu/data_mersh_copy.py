import ee
import pandas as pd
import json
import time
import sys
import argparse
from datetime import datetime

try:
    ee.Initialize()
except Exception as e:
    # No fallamos aquí porque el usuario puede no tener credenciales; informamos claramente.
    print("Advertencia: ee.Initialize() falló. Asegurate de haber autenticado Earth Engine. Error:", e)

# Cargar el archivo inundaciones.json
with open('inundaciones_unidas.json', 'r', encoding='utf-8') as f:
    inundaciones_data = json.load(f)

# Variables a extraer
variables = [
    'total_precipitation_sum',
    'surface_runoff_sum',
    'total_evaporation_sum',
    'volumetric_soil_water_layer_1',
    'temperature_2m',
]

def get_climate_values(lat, lon, date):
    """Solicita ERA5 solo si hay datos para ese día."""
    
    # Construir point y collection
    point = ee.Geometry.Point(float(lon), float(lat))  # lon, lat
    collection = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(date, ee.Date(date).advance(1, 'day'))  # rango de 24 h
        .select(variables)
    )

    # Reintentos porque getInfo() puede fallar o retrasarse
    max_retries = 3
    delay = 1.0
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            # si no hay imágenes para esa fecha, devolvemos variables vacías
            size = collection.size().getInfo()
            if size == 0:
                return {var: None for var in variables}

            image = collection.first()
            values = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=1000
            ).getInfo()

            if not isinstance(values, dict):
                return {var: None for var in variables}

            # Asegurar que todas las variables estén presentes
            for v in variables:
                if v not in values:
                    values[v] = None

            return values
        except Exception as ex:
            last_exc = ex
            print(f"Warning: fallo al obtener datos ERA5 para ({lat},{lon}) {date} (intento {attempt}/{max_retries}): {ex}")
            time.sleep(delay)
            delay *= 2

    print(f"Error persistente obteniendo ERA5 para ({lat},{lon}) {date}: {last_exc}")
    return {var: None for var in variables}


def _extract_base_data(obj):
    # Extraer todos los atributos excepto 'fechas'
    return {
        'Date (YMD)': obj.get('Date (YMD)'),
        'Provincia': obj.get('Provincia'),
        'Code Provincia': obj.get('Code Provincia'),
        'Departamento': obj.get('Departamento'),
        'Code Departamento': obj.get('Code Departamento'),
        'lat': obj.get('lat'),
        'lon': obj.get('lon'),
        'hayRioCercano': obj.get('hayRioCercano'),
        'distanciaRio': obj.get('distanciaRio'),
        'codeRio': obj.get('codeRio'),
        'tipoDeSuelo': obj.get('tipoDeSuelo'),
        'estacion': obj.get('estacion'),
        'inundacion': obj.get('inundacion'),
    }


def _date_is_valid_and_after_threshold(date_str, threshold=datetime(1990, 1, 1)):
    """Return True if date_str represents a date >= threshold. Handles common formats."""
    if not date_str:
        return False
    try:
        dt = pd.to_datetime(date_str, dayfirst=False, errors='coerce')
        if pd.isna(dt):
            return False
        return dt >= threshold
    except Exception:
        return False


def process_items(items, output_csv="data_with_climate.csv"):
    """Procesa una lista de objetos (pre-filtrada) y escribe `output_csv`.

    Esta función no imprime durante el procesamiento; al final escribe el CSV
    y devuelve el conteo de filas con precipitación nula.
    """
    rows = []

    total = len(items)
    for idx, obj in enumerate(items, start=1):
        if idx % 50 == 0 or idx == 1:
            print(f"Procesando objeto {idx}/{total}...")
        fechas = obj.get('fechas', [])
        if not fechas:
            continue

        base_data = _extract_base_data(obj)

        # Primera fila: usa la primera fecha y obtiene datos climáticos
        primera_fecha = fechas[0]
        try:
            climate = get_climate_values(obj.get('lat'), obj.get('lon'), primera_fecha)
        except Exception as e:
            print(f"Error obteniendo clima para primera fecha {primera_fecha} del objeto {idx}: {e}")
            climate = {v: None for v in variables}

        primera_fila = {
            **base_data,
            'fecha': primera_fecha,
            **(climate or {})
        }
        rows.append(primera_fila)

        # Filas restantes: una por cada fecha restante
        for fecha in fechas[1:]:
            try:
                climate_for_date = get_climate_values(obj.get('lat'), obj.get('lon'), fecha)
            except Exception as e:
                print(f"Error obteniendo clima para fecha {fecha} del objeto {idx}: {e}")
                climate_for_date = {v: None for v in variables}

            precip = climate_for_date.get('total_precipitation_sum') if climate_for_date else None

            fila = {
                **base_data,
                'fecha': fecha,
                **(climate_for_date or {})
            }

            # Asegurar que todas las variables estén presentes en la fila
            for var in variables:
                if var not in fila:
                    fila[var] = None

            rows.append(fila)

    df_final = pd.DataFrame(rows)

    # Contar cuántas filas no tienen valor de precipitación (la API no devolvió nada)
    null_precip_count = df_final['total_precipitation_sum'].isna().sum() if 'total_precipitation_sum' in df_final.columns else 0

    # Guardar CSV
    df_final.to_csv(output_csv, index=False)

    return null_precip_count


def get_filtered_items(min_date_str='1990-01-01'):
    """Devuelve los objetos de `inundaciones_data` cuyo `Date (YMD)` >= min_date_str."""
    threshold = pd.to_datetime(min_date_str)
    filtered = []
    for obj in inundaciones_data:
        date_value = obj.get('Date (YMD)')
        if _date_is_valid_and_after_threshold(date_value, threshold):
            filtered.append(obj)
    return filtered


def run_preview(n=10, min_date_str='1990-01-01'):
    """Procesa solo las primeras `n` filas válidas (tras el filtrado por fecha).

    Guarda el CSV de prueba en `data_with_climate_preview.csv` y devuelve el conteo
    de filas con precipitación nula.
    """
    items = get_filtered_items(min_date_str=min_date_str)
    items_preview = items[:n]
    out_csv = "data_with_climate_preview.csv"
    nulls = process_items(items_preview, output_csv=out_csv)
    print(f"Preview escrito en {out_csv}. Filas con 'total_precipitation_sum' nulo: {nulls}")
    return nulls


def run_full(min_date_str='1990-01-01'):
    """Procesa todos los objetos válidos y guarda `data_with_climate.csv`.

    Devuelve el conteo de filas con precipitación nula.
    """
    items = get_filtered_items(min_date_str=min_date_str)
    out_csv = "data_with_climate.csv"
    nulls = process_items(items, output_csv=out_csv)
    print(f"Salida completa escrita en {out_csv}. Filas con 'total_precipitation_sum' nulo: {nulls}")
    return nulls


def cli_main():
    parser = argparse.ArgumentParser(description='Generar CSV de inundaciones con datos ERA5 por fecha.')
    parser.add_argument('--mode', choices=['preview', 'full'], default='preview', help='preview (rápido) o full (completo)')
    parser.add_argument('-n', type=int, default=10, help='Número de objetos a procesar en preview')
    parser.add_argument('--min-date', default='1990-01-01', help='Fecha mínima (inclusive) para filtrar objetos')
    args = parser.parse_args()

    if args.mode == 'preview':
        run_preview(n=args.n, min_date_str=args.min_date)
    else:
        run_full(min_date_str=args.min_date)


if __name__ == '__main__':
    cli_main()


# Nota: no ejecutamos `run_full()` automáticamente para evitar llamadas pesadas.
# Uso recomendado (PowerShell):
# python -c "from proyecto_lu import data_mersh_copy as dm; dm.run_preview(10)"
