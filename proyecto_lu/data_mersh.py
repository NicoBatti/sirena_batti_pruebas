import ee
import pandas as pd

ee.Initialize()

# Simulación de tu DataFrame real
df = pd.DataFrame({
    'date': ["1990-01-01", "1990-01-01", "1990-01-01"],
    'localidad': ["Bahía Blanca", "Comuna 15", "Iruya"],
    'lat_lon': [(-38.7183, -62.2683), (-34.591, -58.475), (-22.7913, -65.2159)],
    'has_rio': [True, False, True],
})

# Variables a extraer
variables = [
    'total_precipitation_sum',
    'surface_runoff_sum',
    'total_evaporation_sum',
    'volumetric_soil_water_layer_1',
    'temperature_2m',
]

def get_climate_values(lat_lon, date):
    """Solicita ERA5 solo si hay datos para ese día."""
    
    point = ee.Geometry.Point(lat_lon[1], lat_lon[0])  # lon, lat

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

for index, row in df.iterrows():
    print(f"📍 Procesando {row['localidad']} | {row['date']}")

    climate = get_climate_values(row['lat_lon'], row['date'])

    combined = {
        **row.to_dict(),  # todas las columnas originales
        **climate         # todas las variables del clima
    }

    rows.append(combined)

df_final = pd.DataFrame(rows)

print(df_final)

df_final.to_csv("data_with_climate.csv", index=False)