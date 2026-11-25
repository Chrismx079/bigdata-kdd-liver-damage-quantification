# Acceso web a CancerVerso

La entrada para Streamlit Community Cloud es `cloud_app.py`, en la rama `main`.
`app.py` sigue siendo la entrada local y conserva su comportamiento original.

Repositorio de referencia: `Chrismx079/bigdata-kdd-liver-damage-quantification`.
Para alojamiento se usa su fork `I-say/bigdata-kdd-liver-damage-quantification`,
porque Streamlit exige permisos de administrador del repositorio. Los cambios
se realizan en el repositorio de referencia y se sincronizan al fork antes de
publicarlos en la web. No editar ambas copias por separado.

```bash
gh repo sync I-say/bigdata-kdd-liver-damage-quantification --source Chrismx079/bigdata-kdd-liver-damage-quantification --branch main
```

La versión web muestra el contenido completo de la V1 y los resultados de la U-Net (V3): estadísticas de las 3 partes unidas,
los 20 casos procesados, sus imágenes, métricas, seguimiento y reportes existentes.
El visor de cortes y 3D solo incluye esos 20 casos de demostración.

## Datos y acceso

El paquete completo se almacena cifrado con AES-256-GCM en Google Drive. El servidor
lo descarga, verifica su SHA-256 y lo descifra en una carpeta temporal. No requiere
ejecutar el notebook ni una GPU. No se incluyen datos ni claves en GitHub.

La aplicación exige un enlace con una clave aleatoria de acceso. Quien reciba o
reenvíe el enlace podrá entrar: no identifica personas y no equivale a una lista
de usuarios autorizados. El enlace completo no debe publicarse en el README.
El acceso al dashboard se comprueba antes de descargar o mostrar datos.

Las claves se configuran en **Advanced settings → Secrets** de Streamlit:

```toml
[cancerverso]
drive_file_id = "ID_DEL_ARCHIVO_CIFRADO"
data_sha256 = "SHA256_DEL_ARCHIVO_CIFRADO"
data_key_hex = "CLAVE_AES_DE_32_BYTES_EN_HEXADECIMAL"
access_token_sha256 = "SHA256_DE_LA_CLAVE_DEL_ENLACE"
```

Formato del enlace: `https://NOMBRE.streamlit.app/?acceso=CLAVE_DEL_ENLACE`.
Estos ejemplos son marcadores, no credenciales reales. Para revocar los enlaces
compartidos, cambiar `access_token_sha256` y distribuir un enlace con clave nueva.

Los datos descifrados no se sirven como archivos estáticos. Las imágenes se
entregan al navegador para su visualización; un visitante autorizado puede guardar
o compartir lo que ve. No debe interpretarse como protección contra copias.

## Operación

- Configurar Python 3.12 e instalar `requirements.txt`.
- Los archivos de entrada son `cloud_app.py`, `app.py`, `estilo.css` y la carpeta `resultados/`
  (métricas e imagen de la U-Net; viene de GitHub, no del paquete cifrado).
- La primera apertura o un reinicio vuelve a descargar aproximadamente 54 MB.
- Si Drive limita la descarga o el servidor está suspendido, la primera visita
  puede tardar o requerir reintentar. Drive no ofrece una garantía de disponibilidad.
- Mantener la carpeta original `datos/` fuera de Git; el despliegue no la modifica.

Fuente: [CancerVerse, BodyMaps](https://huggingface.co/datasets/BodyMaps/CancerVerse).
Se mantienen la atribución y las condiciones de la licencia
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
El control de acceso no cambia los derechos de uso o redistribución del material.
