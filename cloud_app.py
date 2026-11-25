"""Entrada web: acceso por enlace y datos cifrados, sin modificar la app local."""
import hashlib
import hmac
import io
from pathlib import Path, PurePosixPath
import re
import runpy
import shutil
import tempfile
import zipfile

import gdown
import streamlit as st
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


ORIGEN = Path(__file__).resolve().parent
CONTEXTO_CIFRADO = b"CancerVerso V1 datos completos"
TABLAS = {
    "stats_organos.csv", "resumen_dataset.json", "casos_higado.csv",
    "seguimiento_higado.csv", "checklist_higado.csv",
}


def extraer_datos(contenido, destino):
    """Solo admite las tablas y volúmenes que utiliza CancerVerso."""
    with zipfile.ZipFile(io.BytesIO(contenido)) as archivo:
        nombres = archivo.namelist()
        if len(nombres) != len(set(nombres)):
            raise ValueError("El paquete contiene rutas duplicadas.")
        if sum(x.file_size for x in archivo.infolist()) > 256 * 1024 * 1024:
            raise ValueError("El paquete supera el tamaño permitido.")
        for nombre in nombres:
            ruta = PurePosixPath(nombre)
            tabla = len(ruta.parts) == 3 and ruta.parts[:2] == ("datos", "stats") and ruta.name in TABLAS
            volumen = re.fullmatch(r"datos/procesados/higado/CV_[0-9]{8}/volumen\.npz", nombre)
            if not tabla and not volumen:
                raise ValueError("El paquete contiene una ruta no permitida.")
        if not {"datos/stats/" + x for x in TABLAS}.issubset(nombres):
            raise ValueError("Faltan tablas del dashboard.")
        if not any(n.endswith("/volumen.npz") for n in nombres):
            raise ValueError("Faltan los volúmenes del dashboard.")
        for nombre in nombres:
            salida = destino.joinpath(*PurePosixPath(nombre).parts)
            salida.parent.mkdir(parents=True, exist_ok=True)
            salida.write_bytes(archivo.read(nombre))


@st.cache_resource(show_spinner=False)
def preparar_app(file_id, huella, clave_hex, version_app):
    # version_app invalida esta copia si cambia app.py, estilo.css o algo de resultados/.
    with tempfile.TemporaryDirectory(prefix="cancerverso-descarga-") as descarga:
        cifrado = Path(descarga) / "datos.enc"
        resultado = gdown.download(id=file_id, output=str(cifrado), quiet=True, use_cookies=False)
        if not resultado:
            raise RuntimeError("No se pudo descargar el paquete.")
        contenido = cifrado.read_bytes()
        if not hmac.compare_digest(hashlib.sha256(contenido).hexdigest(), huella):
            raise ValueError("El paquete descargado no coincide con la versión esperada.")
        datos = AESGCM(bytes.fromhex(clave_hex)).decrypt(contenido[:12], contenido[12:], CONTEXTO_CIFRADO)
    # La copia original de datos/ nunca se sobrescribe, ni siquiera al probar localmente.
    entorno = tempfile.TemporaryDirectory(prefix="cancerverso-web-")
    destino = Path(entorno.name)
    try:
        extraer_datos(datos, destino)
        for nombre in ("app.py", "estilo.css"):
            shutil.copyfile(ORIGEN / nombre, destino / nombre)
        # Resultados del modelo (métricas e imagen de ejemplo): vienen de GitHub, no del paquete cifrado
        shutil.copytree(ORIGEN / "resultados", destino / "resultados")
    except Exception:
        entorno.cleanup()
        raise
    return entorno


def main():
    try:
        configuracion = st.secrets["cancerverso"]
        esperado = str(configuracion["access_token_sha256"])
        if len(esperado) != 64:
            raise ValueError("Configuración de acceso incompleta.")
    except (KeyError, FileNotFoundError, ValueError):
        st.title("CancerVerso")
        st.info("La aplicación está pendiente de configuración por su responsable.")
        st.stop()

    recibido = hashlib.sha256(str(st.query_params.get("acceso", "")).encode()).hexdigest()
    if not hmac.compare_digest(recibido, esperado):
        st.title("CancerVerso")
        st.info("Abre el enlace completo que te compartió el equipo para acceder al proyecto.")
        st.stop()

    contenido = (ORIGEN / "app.py").read_bytes() + (ORIGEN / "estilo.css").read_bytes()
    for archivo in sorted((ORIGEN / "resultados").iterdir()):
        contenido = contenido + archivo.read_bytes()
    version = hashlib.sha256(contenido).hexdigest()
    try:
        with st.spinner("Preparando CancerVerso. La primera apertura puede tardar un poco…"):
            entorno = preparar_app(
                str(configuracion["drive_file_id"]),
                str(configuracion["data_sha256"]),
                str(configuracion["data_key_hex"]),
                version,
            )
    except Exception:
        # No imprimir secretos, enlaces de acceso ni reportes en la página o los logs.
        st.error("No se pudieron preparar los datos. Intenta recargar la página; si persiste, avisa al equipo.")
        st.stop()
    runpy.run_path(str(Path(entorno.name) / "app.py"), run_name="__main__")


if __name__ == "__main__":
    main()
