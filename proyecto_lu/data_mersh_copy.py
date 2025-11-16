import ee
import pandas as pd
import json

ee.Initialize()

# Cargar el archivo inundaciones.json
with open('inundaciones.json', 'r', encoding='utf-8') as f:
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
    
    point = ee.Geometry.Point(float(lon), float(lat))  # lon, lat

    collection = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(date, ee.Date(date).advance(1, 'day'))  # rango de 24 h
        .select(variables)
    )

    # ❗ Si no hay imágenes para esa fecha, devolvemos variables vacías
    if collection.size().getInfo() == 0:
        return {var: None for var in variables}

    image = collection.first()

    values = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=1000
    ).getInfo()

    # Si una variable falta → None
    for v in variables:
        if v not in values:
            values[v] = None

    return values


# 🚀 Crear nuevo DataFrame con datos climáticos anexados
rows = []

for obj in inundaciones_data:
    fechas = obj['fechas']
    
    # Extraer todos los atributos excepto 'fechas'
    base_data = {
        'Date (YMD)': obj['Date (YMD)'],
        'Provincia': obj['Provincia'],
        'Code Provincia': obj['Code Provincia'],
        'Departamento': obj['Departamento'],
        'Code Departamento': obj['Code Departamento'],
        'lat': obj['lat'],
        'lon': obj['lon'],
        'hayRioCercano': obj['hayRioCercano'],
        'distanciaRio': obj['distanciaRio'],
        'codeRio': obj['codeRio'],
        'tipoDeSuelo': obj['tipoDeSuelo'],
        'estacion': obj['estacion']
    }
    
    # Primera fila: usa la primera fecha y obtiene datos climáticos
    primera_fecha = fechas[0]
    
    climate = get_climate_values(obj['lat'], obj['lon'], primera_fecha)
    
    primera_fila = {
        **base_data,
        'fecha': primera_fecha,
        **climate
    }
    rows.append(primera_fila)
    
    # Filas restantes: una por cada fecha restante
    # Para garantizar que la variable 'total_precipitation_sum' esté llena,
    # hacemos la solicitud para esa fecha y guardamos solo ese feature en cada fila.
    for fecha in fechas[1:]:
        # solicitar solo para obtener total_precipitation_sum
        climate_for_date = get_climate_values(obj['lat'], obj['lon'], fecha)
        precip = climate_for_date.get('total_precipitation_sum') if climate_for_date else None
        if precip is None:
            # Si la API no devuelve precipitación, dejamos el valor vacío (None)
            precip = None

        fila = {
            **base_data,
            'fecha': fecha,
            # Llenamos `total_precipitation_sum` y dejamos las demás variables como None
            'total_precipitation_sum': precip,
        }

        # Asegurar que todas las variables estén presentes en la fila (si faltan, agregarlas con None)
        for var in variables:
            if var not in fila:
                fila[var] = None

        rows.append(fila)

df_final = pd.DataFrame(rows)

# Contar cuántas filas no tienen valor de precipitación (la API no devolvió nada)
null_precip_count = df_final['total_precipitation_sum'].isna().sum() if 'total_precipitation_sum' in df_final.columns else 0

# Guardar CSV
df_final.to_csv("data_with_climate.csv", index=False)

# Imprimir solo el resumen final: cuántas filas tienen precipitación nula
print(f"Filas con 'total_precipitation_sum' nulo (API no devolvió): {null_precip_count}")
