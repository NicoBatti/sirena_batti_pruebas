import os
import requests
from netrc import netrc
from urllib.parse import urlparse

# --------------------------------------------------------------------
# Cargar credenciales automáticamente desde el archivo .netrc
# --------------------------------------------------------------------
def get_credentials():
    try:
        info = netrc(os.path.expanduser("~/.netrc")).authenticators("urs.earthdata.nasa.gov")
        if info:
            username, _, password = info
            print(f"Credenciales cargadas desde .netrc → usuario: {username}")
            return username, password
    except Exception as e:
        print("No se pudo leer .netrc:", e)

    raise Exception("No se encontraron credenciales en ~/.netrc")


# --------------------------------------------------------------------
# Descarga un archivo protegido de Earthdata
# --------------------------------------------------------------------
def earthdata_download(url, output_path):
    username, password = get_credentials()

    with requests.Session() as session:
        # 1) Preparar autenticación automática (.netrc)
        session.auth = (username, password)

        # 2) Intento inicial (sigue redirecciones hacia OAuth)
        response = session.get(url, stream=True, allow_redirects=True)

        # 3) Si redirige a login → autenticarse
        if "urs.earthdata.nasa.gov" in response.url:
            print("Redireccionado a Earthdata Login → autenticando...")

            response = session.get(response.url, auth=(username, password), allow_redirects=True)

        # 4) Verificar autorización final
        if response.status_code != 200:
            raise Exception(f"Error en descarga: HTTP {response.status_code} → {response.url}")

        # 5) Guardar archivo
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Descarga completa → {output_path}")


# --------------------------------------------------------------------
# Buscar el link del archivo SMAP L4 usando CMR
# --------------------------------------------------------------------
def buscar_granulo(fecha):
    url = "https://cmr.earthdata.nasa.gov/search/granules.json"

    params = {
        "short_name": "SPL4SMGP",
        "version": "008",
        "temporal": f"{fecha}T00:00:00Z,{fecha}T23:59:59Z",
        "page_size": 2000,
    }

    r = requests.get(url, params=params)
    js = r.json()
    entries = js.get("feed", {}).get("entry", [])

    if not entries:
        print("No se encontraron datos para esa fecha.")
        return None

    links = entries[0]["links"]
    h5_links = [x["href"] for x in links if x["href"].endswith(".h5")]

    if not h5_links:
        print("Granulo encontrado pero no tiene link HDF5.")
        return None

    return h5_links[0]


# --------------------------------------------------------------------
# FLUJO PRINCIPAL
# --------------------------------------------------------------------
if __name__ == "__main__":
    fecha = "2025-11-16"

    link = buscar_granulo(fecha)
    if not link:
        exit()

    output = f"smap_{fecha}.h5"
    earthdata_download(link, output)
