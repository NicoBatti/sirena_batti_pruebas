import requests
import os
from requests.auth import HTTPBasicAuth

EARTHDATA_USER = "sirenaai"
EARTHDATA_PASS = "ProyectoTic!2025"

def obtener_granulo_smap(fecha):
    """
    fecha: string YYYY-MM-DD
    """
    cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json"

    params = {
        "short_name": "SPL4SMGP",
        "version": "008",
        "temporal": f"{fecha}T00:00:00Z,{fecha}T23:59:59Z",
        "page_size": 2000
    }

    r = requests.get(cmr_url, params=params)
    js = r.json()

    items = js.get("feed", {}).get("entry", [])

    if not items:
        print(f"No se encontraron granulos para {fecha}")
        return None

    # Harmony download link
    links = items[0]["links"]
    data_links = [l["href"] for l in links if l["href"].endswith(".h5")]

    if not data_links:
        print("Granulo encontrado pero sin link visible.")
        return None

    return data_links[0]


def descargar_archivo(url, output_file):
    with requests.get(
        url,
        auth=HTTPBasicAuth(EARTHDATA_USER, EARTHDATA_PASS),
        stream=True
    ) as r:
        r.raise_for_status()
        with open(output_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"Archivo descargado: {output_file}")


if __name__ == "__main__":
    fecha = "2025-11-16"
    link = obtener_granulo_smap(fecha)

    if link:
        out = f"smap_{fecha}.h5"
        descargar_archivo(link, out)
