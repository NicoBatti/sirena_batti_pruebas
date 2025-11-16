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
        print(f"⚠ No hay datos para {date} en ERA5. Se llena con None.")
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
    print(f"📍 Procesando {obj['Departamento']}, {obj['Provincia']} | {primera_fecha}")
    
    climate = get_climate_values(obj['lat'], obj['lon'], primera_fecha)
    
    primera_fila = {
        **base_data,
        'fecha': primera_fecha,
        **climate
    }
    rows.append(primera_fila)
    
    # Filas restantes: una por cada fecha restante con valores climáticos en None
    for fecha in fechas[1:]:
        fila_null = {
            **base_data,
            'fecha': fecha,
            **{var: None for var in variables}
        }
        rows.append(fila_null)

df_final = pd.DataFrame(rows)

print(df_final)

df_final.to_csv("data_with_climate.csv", index=False)
