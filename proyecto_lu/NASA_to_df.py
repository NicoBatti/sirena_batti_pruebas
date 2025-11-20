import os
import requests
import netrc
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta, UTC

# ============================
# 1. ENCONTRAR ÚLTIMO DOMINGO QUE EXISTA EN NASA
# ============================

def ultimo_domingo_existente():
    hoy = datetime.now(UTC)
    
    # primer intento: último domingo
    d = hoy - timedelta(days=(hoy.weekday() + 1) % 7)

    for _ in range(10):  # probar hasta 10 domingos hacia atrás
        fecha_str = d.strftime("%Y%m%d")
        url = (
            f"https://data.nsidc.earthdatacloud.nasa.gov/"
            f"nsidc-cumulus-prod-protected/SMAP/SPL4SMGP.008/"
            f"{d.year}/{d.month:02d}/{d.day:02d}/"
            f"SMAP_L4_SM_gph_{fecha_str}T223000_Vv8011_001.h5"
        )

        print("Probando archivo:", url)
        r = requests.head(url)

        if r.status_code == 200:
            print("✓ Archivo encontrado:", fecha_str)
            return d, url
        
        print("✗ No disponible, probando domingo anterior...")
        d -= timedelta(days=7)

    raise Exception("No se encontraron archivos SMAP válidos en las últimas 10 semanas.")

fecha, url = ultimo_domingo_existente()
fecha_str = fecha.strftime("%Y%m%d")
output_file = f"smap_{fecha_str}.h5"

print("\nArchivo elegido:", url)

# ============================
# 2. DESCARGA AUTÉNTICA
# ============================

netrc_path = os.path.expanduser("~/.netrc")
username, _, password = netrc.netrc(netrc_path).authenticators("urs.earthdata.nasa.gov")

session = requests.Session()
session.auth = (username, password)

response = session.get(url, stream=True)
if response.status_code != 200:
    raise Exception(f"Error en descarga: HTTP {response.status_code}")

with open(output_file, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

print("✓ Descarga completa →", output_file)

# ============================
# 3. EXTRAER VARIABLES ÚTILES
# ============================

ds = xr.open_dataset(output_file, engine="h5netcdf")

vars_utiles = [
    "cell_lat", "cell_lon",
    "sm_surface", "sm_rootzone",
    "sm_surface_wetness", "sm_rootzone_wetness",
    "overland_runoff_flux", "baseflow_flux",
    "land_evapotranspiration_flux"
]

ds = ds[vars_utiles]

# ============================
# 4. FILTRAR ARGENTINA
# ============================

lat_min, lat_max = -55, -22
lon_min, lon_max = -75, -53

ds = ds.where(
    (ds["cell_lat"] >= lat_min) &
    (ds["cell_lat"] <= lat_max) &
    (ds["cell_lon"] >= lon_min) &
    (ds["cell_lon"] <= lon_max),
    drop=True
)

# ============================
# 5. LIMPIAR Y EXPORTAR
# ============================

df = ds.to_dataframe().reset_index(drop=True)
df = df[df.replace(-9999, pd.NA).notna().all(axis=1)]

df.to_csv(f"smap_argentina_{fecha_str}.csv", index=False)
print("\n✓ CSV listo →", f"smap_argentina_{fecha_str}.csv")
print(df.head())
