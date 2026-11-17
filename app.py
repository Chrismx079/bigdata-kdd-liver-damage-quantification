"""
CancerVerso · Explorador multiórgano de tumores en tomografía (V1)

Cómo correrla:
    pip install -r requirements.txt
    streamlit run app.py

La app solo LEE lo que generó el notebook de Colab (CancerVerso_V1.ipynb).
Busca los datos en la carpeta "datos", junto a este archivo:
    datos/stats/                 tablas CSV y JSON
    datos/procesados/higado/     un volumen reducido por cada CT procesado
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from scipy import ndimage
from skimage import measure

# ───────────────────────── 1. Configuración ─────────────────────────
NOMBRE_PROYECTO = "CancerVerso"
CREDITO = "Licenciatura en Inteligencia Artificial y Ciencia de Datos"
META_GB = 300                 # meta mínima de datos procesados
META_GB_DESEABLE = 1000       # meta deseable

CARPETA_APP = Path(__file__).parent
DATOS = CARPETA_APP / "datos"
STATS = DATOS / "stats"
PROC = DATOS / "procesados" / "higado"

# Colores (los mismos que usa estilo.css)
PANEL = "#0E1524"
BORDE = "#1C2740"
TEXTO = "#E6EDF7"
TENUE = "#8A9BB8"
CIAN = "#22D3EE"
ROJO = "#FF3B5C"
VERDE = "#34D399"
AMBAR = "#FBBF24"
GRIS_PUNTO = "rgba(138,155,184,0.28)"
GRIS_BARRA = "rgba(138,155,184,0.35)"

# Umbrales clínicos. Son LOS MISMOS valores que la sección 2 del notebook.
# Semáforo de HTB%: CRITERIO PROVISIONAL DEL EQUIPO, sin fuente clínica todavía.
HTB_MODERADA = 25
HTB_ALTA = 50
# Tendencia del volumen: equivalente en volumen de RECIST 1.1 (Eisenhauer et al., Eur J Cancer 2009).
# RECIST mide DIÁMETRO: progresión = +20%, respuesta = -30%. En una esfera, +20% de diámetro es
# 1.2^3 = 1.73 veces el volumen (+73%) y -30% es 0.7^3 = 0.34 veces (-66%). CRITERIO PROVISIONAL.
CAMBIO_CRECE_PCT = 73
CAMBIO_REDUCE_PCT = -66

# Órganos con página completa. Los demás muestran "Próximamente".
ORGANOS_ACTIVOS = ["liver"]

# Posición (x, y) de cada órgano sobre la silueta. Vista de frente: la derecha del paciente queda a la izquierda.
POS_ORGANOS = {
    "esophagus": (0.0, 8.3), "liver": (-1.25, 6.55), "stomach": (0.95, 6.75), "spleen": (2.05, 6.05),
    "gallbladder": (-1.05, 5.55), "pancreatic": (0.3, 5.75), "adrenal_gland": (1.75, 5.05),
    "kidney": (-1.75, 4.7), "duodenum": (-0.35, 4.95), "colon": (1.2, 3.9),
    "uterus": (0.0, 2.75), "bladder": (0.0, 2.05), "prostate": (0.0, 1.45),
}
# De qué lado del punto se escribe el nombre de cada órgano (para que no se encimen)
POS_TEXTO = {
    "esophagus": "middle right", "liver": "middle left", "stomach": "top right", "spleen": "middle right",
    "gallbladder": "middle left", "pancreatic": "bottom center", "adrenal_gland": "middle right",
    "kidney": "middle left", "duodenum": "bottom center", "colon": "middle right",
    "uterus": "middle left", "bladder": "middle right", "prostate": "middle right",
}

st.set_page_config(page_title=NOMBRE_PROYECTO, layout="wide", initial_sidebar_state="collapsed")

# Los estilos están en estilo.css; aquí solo se cargan
css = (CARPETA_APP / "estilo.css").read_text(encoding="utf-8")
st.markdown("<style>" + css + "</style>", unsafe_allow_html=True)


# ───────────────────────── 2. Carga de datos ─────────────────────────
@st.cache_data
def cargar_tablas():
    """Lee las tablas que generó el notebook. Devuelve None si todavía no existen."""
    if not (STATS / "stats_organos.csv").exists():
        return None

    organos = pd.read_csv(STATS / "stats_organos.csv").set_index("organo")
    resumen = json.loads((STATS / "resumen_dataset.json").read_text())

    # Casos de hígado procesados (solo los que tienen su volumen reducido en la carpeta)
    casos = pd.DataFrame()
    if (STATS / "casos_higado.csv").exists():
        casos = pd.read_csv(STATS / "casos_higado.csv", parse_dates=["fecha"])
        casos = casos.drop_duplicates("caso", keep="last")
        tiene_volumen = []
        for caso in casos["caso"]:
            tiene_volumen.append((PROC / caso / "volumen.npz").exists())
        casos = casos[tiene_volumen].sort_values(["paciente", "fecha"])
        if "ts_rapido" not in casos.columns:
            casos["ts_rapido"] = True          # datos de la V0: se hicieron con --fast

    seguimiento = pd.DataFrame()
    seguimiento_viejo = False
    if (STATS / "seguimiento_higado.csv").exists():
        seguimiento = pd.read_csv(STATS / "seguimiento_higado.csv", parse_dates=["fecha"])
        # La V1 necesita estas columnas. Una tabla de la V0 no las trae: en ese caso no se dibuja la curva.
        for columna in ["fase", "revisar", "misma_fase", "salto_10x"]:
            if columna not in seguimiento.columns:
                seguimiento_viejo = True
        if seguimiento_viejo:
            seguimiento = pd.DataFrame()

    # GB de CT procesados: se suma el último estado "ok" de cada caso del checklist
    gb = 0.0
    if (STATS / "checklist_higado.csv").exists():
        checklist = pd.read_csv(STATS / "checklist_higado.csv").drop_duplicates("caso", keep="last")
        gb = float(checklist[checklist["estado"] == "ok"]["gb_ct"].sum())

    return {"organos": organos, "resumen": resumen, "casos": casos, "seguimiento": seguimiento, "gb": gb,
            "seguimiento_viejo": seguimiento_viejo}


@st.cache_data
def cargar_volumen(caso):
    """Volumen reducido de un CT: tomografía, máscara de tumor, máscara de hígado y tamaño de vóxel (mm)."""
    v = np.load(PROC / caso / "volumen.npz")
    return v["ct"], v["tumor"], v["higado"], v["espaciado_mm"]


@st.cache_data
def malla_3d(mascara, espaciado, paso):
    """Convierte una máscara en una superficie 3D (vértices y triángulos) con marching cubes."""
    con_borde = np.pad(mascara.astype(np.float32), 1)      # un borde de ceros cierra la superficie
    tamano_voxel = (float(espaciado[0]), float(espaciado[1]), float(espaciado[2]))
    vertices, caras, _, _ = measure.marching_cubes(con_borde, level=0.5, spacing=tamano_voxel, step_size=paso)
    return vertices, caras


# ───────────────────────── 3. Funciones de apoyo ─────────────────────────
def miles(numero):
    """Número con separador de miles: 24420 -> '24,420'."""
    return f"{numero:,.0f}"


def id_corto(paciente):
    """ID corto del paciente: sus primeros 6 caracteres. Es el mismo que usa el notebook."""
    return paciente[:6]


def nivel_htb(pct):
    """Semáforo de HTB%. Devuelve (nombre del nivel, color, clase CSS). Criterio provisional del equipo."""
    if pct < HTB_MODERADA:
        return "Baja", VERDE, "verde"
    elif pct < HTB_ALTA:
        return "Moderada", AMBAR, "ambar"
    else:
        return "Alta", ROJO, "rojo"


def tendencia_volumen(vol_inicial, vol_final):
    """Clasifica el cambio de volumen entre el primer y el último estudio. Devuelve (texto, clase CSS, cambio %)."""
    cambio = 100 * (vol_final - vol_inicial) / vol_inicial
    if cambio >= CAMBIO_CRECE_PCT:
        return "Crece", "rojo", cambio
    elif cambio <= CAMBIO_REDUCE_PCT:
        return "Se reduce", "verde", cambio
    else:
        return "Estable", "ambar", cambio


def tarjetas(lista):
    """Dibuja una fila de tarjetas. Cada elemento es (valor, texto, unidad)."""
    columnas = st.columns(len(lista))
    for columna, (valor, texto, unidad) in zip(columnas, lista):
        html = '<div class="kpi"><div class="kpi-num">' + str(valor) + "<small>" + unidad + "</small></div>"
        html = html + '<div class="kpi-txt">' + texto + "</div></div>"
        columna.markdown(html, unsafe_allow_html=True)


def estilo_grafica(fig, alto):
    """Estilo común de las gráficas de Plotly: fondo transparente y letra de la app."""
    fig.update_layout(height=alto, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", color=TENUE, size=12),
                      margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDE, font_color=TEXTO))


def ir_a(vista):
    """Cambia de pantalla. 'vista' es 'portada' o la clave de un órgano (por ejemplo 'liver')."""
    st.session_state["vista"] = vista
    # El mapa de órganos "recuerda" el último punto en el que se hizo clic. Si no se borrara ese recuerdo,
    # al volver a la portada la app creería que se acaba de hacer clic otra vez y saltaría sola al órgano.
    # Streamlit guarda ese recuerdo bajo la clave (key) de la gráfica; al sumar 1 a n_mapa la clave cambia
    # ("mapa_0" -> "mapa_1") y la gráfica nace de nuevo, sin ningún punto seleccionado.
    st.session_state["n_mapa"] = st.session_state.get("n_mapa", 0) + 1
    # st.rerun() vuelve a ejecutar el archivo desde arriba, ya con la nueva vista guardada.
    st.rerun()


def barra_superior(mostrar_inicio):
    izquierda, derecha = st.columns([6, 1])
    marca = NOMBRE_PROYECTO.replace("Verso", '<span class="acento">Verso</span>')
    izquierda.markdown('<div class="marca">' + marca + "<small>" + CREDITO + "</small></div>", unsafe_allow_html=True)
    if mostrar_inicio:
        if derecha.button("← Inicio", width="stretch"):
            ir_a("portada")
    st.markdown('<div class="linea"></div>', unsafe_allow_html=True)


# ───────────────────────── 4. Gráficas ─────────────────────────
def mapa_cuerpo(organos):
    """Silueta del torso con un punto por órgano. El tamaño del punto indica cuántos CT tienen tumor ahí."""
    # Puntos de la mitad izquierda de la silueta; la mitad derecha es su espejo
    mitad = [(0, 11.35), (-0.55, 11.2), (-0.78, 10.6), (-0.6, 10.0), (-0.45, 9.55), (-1.5, 9.25), (-2.55, 8.95),
             (-3.05, 8.2), (-3.3, 6.6), (-3.55, 4.9), (-3.75, 3.5), (-3.2, 3.4), (-2.95, 4.9), (-2.7, 6.4),
             (-2.45, 7.6), (-2.3, 6.2), (-2.2, 4.6), (-2.4, 3.2), (-2.55, 1.9), (-2.45, 0.2), (-0.35, 0.2), (0, 1.0)]
    xs, ys = [], []
    for x, y in mitad:
        xs.append(x)
        ys.append(y)
    for x, y in reversed(mitad):
        xs.append(-x)
        ys.append(y)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor="rgba(34,211,238,0.045)",
                             line=dict(color="rgba(34,211,238,0.35)", width=1.5, shape="spline", smoothing=0.7),
                             hoverinfo="skip"))

    # Una lista por cada propiedad de los puntos
    claves, px, py, nombres, lados, tamanos, colores, colores_texto, textos_hover = [], [], [], [], [], [], [], [], []
    maximo = organos["n_con_tumor"].max()
    for clave, (x, y) in POS_ORGANOS.items():
        fila = organos.loc[clave]
        if clave in ORGANOS_ACTIVOS:
            color, color_texto, estado = CIAN, TEXTO, "Disponible"
        else:
            color, color_texto, estado = GRIS_PUNTO, TENUE, "Próximamente"
        claves.append(clave)
        px.append(x)
        py.append(y)
        nombres.append(fila["nombre"])
        lados.append(POS_TEXTO[clave])
        tamanos.append(16 + 34 * np.sqrt(fila["n_con_tumor"] / maximo))
        colores.append(color)
        colores_texto.append(color_texto)
        textos_hover.append("<b>" + fila["nombre"] + "</b><br>" + miles(fila["n_con_tumor"]) + " CT con tumor<br>" + estado)

    fig.add_trace(go.Scatter(
        x=px, y=py, mode="markers+text", text=nombres, textposition=lados,
        textfont=dict(color=colores_texto, size=12),
        customdata=claves,                          # la clave del órgano viaja con cada punto: se lee al hacer clic
        marker=dict(size=tamanos, color=colores, line=dict(color="rgba(138,155,184,0.7)", width=1.5), opacity=0.9),
        hovertext=textos_hover, hoverinfo="text",
        selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=0.9)),
    ))
    estilo_grafica(fig, 640)
    fig.update_xaxes(visible=False, range=[-6.2, 6.2], fixedrange=True)
    fig.update_yaxes(visible=False, range=[-0.2, 11.8], fixedrange=True, scaleanchor="x")
    fig.update_layout(dragmode=False)
    return fig


def grafica_organos(organos, resaltar):
    """Barras horizontales: CT con tumor por órgano. 'resaltar' es la clave del órgano que va en cian."""
    orden = organos.sort_values("n_con_tumor")
    colores, textos = [], []
    for clave, fila in orden.iterrows():
        if clave == resaltar:
            colores.append(CIAN)
        else:
            colores.append(GRIS_BARRA)
        textos.append(miles(fila["n_con_tumor"]))
    fig = go.Figure(go.Bar(x=orden["n_con_tumor"], y=orden["nombre"], orientation="h", marker_color=colores,
                           text=textos, textposition="outside", textfont_color=TENUE,
                           hovertemplate="%{y}: %{x:,} CT con tumor<extra></extra>"))
    estilo_grafica(fig, 430)
    fig.update_xaxes(showgrid=True, gridcolor=BORDE, zeroline=False, range=[0, orden["n_con_tumor"].max() * 1.15])
    fig.update_yaxes(tickfont=dict(color=TEXTO, size=12))
    return fig


def imagen_corte(ct, tumor, higado, ver_tumor, ver_higado, opacidad):
    """Corte axial en grises, con el tumor en rojo y el contorno del hígado en cian."""
    rgb = np.stack([ct, ct, ct], axis=2).astype(np.float32)
    if ver_higado:
        # Contorno = la máscara menos su versión "encogida" un píxel (erosión)
        contorno = higado & ~ndimage.binary_erosion(higado)
        rgb[contorno] = (34, 211, 238)
    if ver_tumor:
        rgb[tumor] = (1 - opacidad) * rgb[tumor] + opacidad * np.array([255, 59, 92])
    imagen = Image.fromarray(rgb.astype(np.uint8))
    return imagen.resize((imagen.width * 3, imagen.height * 3), Image.LANCZOS)      # se amplía para verse nítido


def figura_3d(caso):
    """Hígado translúcido con los tumores en rojo."""
    _, tumor, higado, espaciado = cargar_volumen(caso)
    fig = go.Figure()
    # Cada elemento: (máscara, color, opacidad, nombre, paso de marching cubes)
    capas = [(higado, CIAN, 0.16, "Hígado", 2), (tumor, ROJO, 1.0, "Tumor", 1)]
    for mascara, color, opacidad, nombre, paso in capas:
        if mascara.sum() < 8:
            continue                                   # máscara casi vacía: no hay superficie que dibujar
        vertices, caras = malla_3d(mascara, espaciado, paso)
        # vertices[:, 0] = corte (z), vertices[:, 1] = fila, vertices[:, 2] = columna.
        # La fila va con signo menos para que la parte anterior del cuerpo quede al frente.
        fig.add_trace(go.Mesh3d(x=vertices[:, 2], y=-vertices[:, 1], z=vertices[:, 0],
                                i=caras[:, 0], j=caras[:, 1], k=caras[:, 2],
                                color=color, opacity=opacidad, name=nombre, hoverinfo="name",
                                lighting=dict(ambient=0.55, diffuse=0.8, specular=0.25, roughness=0.6)))
    estilo_grafica(fig, 470)
    eje_oculto = dict(visible=False)
    fig.update_layout(scene=dict(xaxis=eje_oculto, yaxis=eje_oculto, zaxis=eje_oculto, aspectmode="data",
                                 bgcolor="rgba(0,0,0,0)", camera=dict(eye=dict(x=0.2, y=-1.9, z=0.5))))
    return fig


def figura_htb(pct):
    """Medidor de HTB% con las tres zonas del semáforo."""
    _, color, _ = nivel_htb(pct)
    zonas = [dict(range=[0, HTB_MODERADA], color="rgba(52,211,153,0.16)"),
             dict(range=[HTB_MODERADA, HTB_ALTA], color="rgba(251,191,36,0.16)"),
             dict(range=[HTB_ALTA, 100], color="rgba(255,59,92,0.16)")]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        number=dict(suffix="%", valueformat=".1f", font=dict(size=46, color=TEXTO, family="Sora, sans-serif")),
        gauge=dict(axis=dict(range=[0, 100], tickvals=[0, HTB_MODERADA, HTB_ALTA, 75, 100], tickcolor=TENUE, tickfont=dict(color=TENUE)),
                   bar=dict(color=color, thickness=0.32), bgcolor="rgba(0,0,0,0)", borderwidth=0, steps=zonas)))
    estilo_grafica(fig, 215)
    fig.update_layout(margin=dict(l=28, r=28, t=22, b=0))
    return fig


def figura_seguimiento(curva, dia_actual, vol_actual):
    """Volumen tumoral del paciente en el tiempo. Los puntos con revisar=True van en rojo."""
    colores, textos = [], []
    for _, fila in curva.iterrows():
        texto = "Día " + str(int(fila["dias"])) + "<br>" + f"{fila['vol_cm3']:,.1f}" + " cm³<br>Fase: " + str(fila["fase"])
        if fila["revisar"]:
            colores.append(ROJO)
            texto = texto + "<br><b>Revisar</b>: salto de más de 10x o volumen menor a 1 cm³"
        else:
            colores.append(CIAN)
        textos.append(texto)

    fig = go.Figure()
    # Estudio que se está viendo arriba: un aro blanco alrededor
    fig.add_trace(go.Scatter(x=[dia_actual], y=[vol_actual], mode="markers", hoverinfo="skip",
                             marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(color=TEXTO, width=2))))
    fig.add_trace(go.Scatter(x=curva["dias"], y=curva["vol_cm3"], mode="lines+markers",
                             line=dict(color="rgba(34,211,238,0.55)", width=2.5),
                             marker=dict(size=10, color=colores),
                             fill="tozeroy", fillcolor="rgba(34,211,238,0.07)",
                             hovertext=textos, hoverinfo="text"))
    estilo_grafica(fig, 300)
    margen = max(curva["dias"].max() * 0.04, 1)
    fig.update_xaxes(title="Días desde el primer estudio", gridcolor=BORDE, zeroline=False,
                     range=[-margen, curva["dias"].max() + margen])
    fig.update_yaxes(title="Volumen tumoral (cm³)", gridcolor=BORDE, zeroline=False, rangemode="tozero")
    return fig


# ───────────────────────── 5. Pantalla: portada ─────────────────────────
def vista_portada(datos):
    organos = datos["organos"]
    r = datos["resumen"]
    barra_superior(False)

    marca = NOMBRE_PROYECTO.replace("Verso", '<span class="acento">Verso</span>')
    st.markdown('<div class="etiqueta">Inteligencia artificial en imagen médica</div>'
                '<div class="titulo">' + marca + "</div>"
                '<p class="bajada">Explorador multiórgano de tumores en tomografía computarizada. Construido sobre CancerVerse, '
                "un conjunto de datos de TC totalmente abierto, longitudinal y multimodal para el cribado de múltiples cánceres.</p>",
                unsafe_allow_html=True)
    st.write("")
    tarjetas([
        (miles(r["n_cts"]), "tomografías (CT) con máscaras", ""),
        (miles(r["n_pacientes"]), "pacientes", ""),
        (r["n_organos"], "órganos con tumores anotados", ""),
        (miles(r["n_anotaciones_tumor"]), "anotaciones de tumor (CT × órgano) hechas por expertos", ""),
        (r["seguimiento_max_anios"], "de seguimiento máximo", "años"),
    ])
    # Nota sobre la base de las cifras
    n_fuera = r["n_cts_dataset"] - r["n_cts"]
    if "parte" in r:
        # Los datos vienen de UNA parte del reparto del notebook (por ejemplo la parte 0 de 3), no del dataset completo
        st.markdown('<p class="nota"><span class="fuerte">Avance parcial: parte ' + str(r["parte"] + 1) + " de " + str(r["n_partes"])
                    + " del dataset.</span> Las cifras de esta pantalla son de los " + miles(r["n_cts"]) + " CT con máscaras de esa parte ("
                    + str(n_fuera) + " sin máscaras, excluidos). El dataset completo tiene 24,422 CT.</p>", unsafe_allow_html=True)
    elif "n_partes" in r and r["union_completa"] == False:
        # Los datos vienen de la UNIÓN de varias partes (sección 13 del notebook), pero todavía faltan partes por juntar
        st.markdown('<p class="nota"><span class="fuerte">Avance parcial: ' + str(len(r["partes_incluidas"])) + " de " + str(r["n_partes"])
                    + " partes del dataset unidas.</span> Las cifras de esta pantalla son de los " + miles(r["n_cts"]) + " CT con máscaras de esas partes ("
                    + str(n_fuera) + " sin máscaras, excluidos). El dataset completo tiene 24,422 CT.</p>", unsafe_allow_html=True)
    else:
        # Dataset completo: sin reparto, o con todas las partes ya unidas
        st.markdown('<p class="nota">Base de todas las cifras: los ' + miles(r["n_cts"]) + " CT que traen máscaras. El dataset publica "
                    + miles(r["n_cts_dataset"]) + "; " + str(n_fuera) + " no traen máscaras (" + ", ".join(r["casos_sin_mascara"])
                    + ") y se excluyen.</p>", unsafe_allow_html=True)

    # Avance de datos procesados respecto a las metas
    gb = datos["gb"]
    ancho = min(100, 100 * gb / META_GB)
    st.markdown('<div class="panel"><span class="titulo-sec">Datos procesados</span>'
                '<span class="sub-sec" style="float:right"><span class="fuerte">' + f"{gb:,.1f}" + " GB</span> de " + str(META_GB)
                + " GB · meta deseable " + miles(META_GB_DESEABLE) + " GB</span>"
                '<div class="progreso"><div style="width:' + f"{ancho:.2f}" + '%"></div></div>'
                '<span class="nota">' + str(len(datos["casos"])) + " tomografías de hígado procesadas en esta versión</span></div>",
                unsafe_allow_html=True)

    st.write("")
    izquierda, derecha = st.columns([5, 4], gap="large")
    with izquierda:
        st.markdown('<p class="titulo-sec">Mapa de órganos</p><p class="sub-sec">Haz clic en un órgano para abrir su ficha. '
                    "El tamaño del punto indica cuántas tomografías tienen tumor.</p>", unsafe_allow_html=True)
        # CÓMO FUNCIONA EL CLIC EN EL MAPA
        # 1) on_select="rerun": cuando alguien hace clic en un punto, Streamlit vuelve a ejecutar el archivo
        #    y st.plotly_chart devuelve qué puntos quedaron seleccionados.
        # 2) Cada punto lleva en "customdata" la clave de su órgano ("liver", "kidney"...). Con ella se cambia de vista.
        # 3) La key incluye el contador n_mapa. ir_a() le suma 1 en cada cambio de pantalla para que el mapa
        #    arranque sin selección al volver (ver el comentario dentro de ir_a).
        clave_mapa = "mapa_" + str(st.session_state.get("n_mapa", 0))
        evento = st.plotly_chart(mapa_cuerpo(organos), width="stretch", on_select="rerun", selection_mode="points",
                                 key=clave_mapa, config={"displayModeBar": False})
        puntos = evento["selection"]["points"]
        if len(puntos) > 0 and "customdata" in puntos[0]:
            ir_a(puntos[0]["customdata"])
    with derecha:
        st.markdown('<p class="titulo-sec">Órganos</p><p class="sub-sec">Tomografías con tumor anotado en cada órgano.</p>',
                    unsafe_allow_html=True)
        columnas = st.columns(2)
        numero = 0
        for clave, fila in organos.iterrows():
            if clave in ORGANOS_ACTIVOS:
                tipo = "primary"
            else:
                tipo = "secondary"
            with columnas[numero % 2]:
                if st.button(fila["nombre"] + " · " + miles(fila["n_con_tumor"]), key="boton_" + clave, width="stretch", type=tipo):
                    ir_a(clave)
            numero = numero + 1
        st.markdown('<p class="nota"><span class="chip chip-on">Disponible</span> Hígado &nbsp;&nbsp; '
                    '<span class="chip chip-off">Próximamente</span> los otros 12 órganos</p>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<p class="titulo-sec">Tomografías con tumor por órgano</p>', unsafe_allow_html=True)
    st.plotly_chart(grafica_organos(organos, "liver"), width="stretch", config={"displayModeBar": False})


# ───────────────────────── 6. Pantalla: ficha de órgano ─────────────────────────
def encabezado_organo(fila, activo):
    if activo:
        chip = '<span class="chip chip-on">Disponible</span>'
    else:
        chip = '<span class="chip chip-off">Próximamente</span>'
    st.markdown('<div class="etiqueta">Ficha de órgano</div><div class="titulo-organo">' + fila["nombre"] + " " + chip + "</div>",
                unsafe_allow_html=True)
    st.write("")
    tarjetas([
        (miles(fila["n_con_tumor"]), "CT con tumor, de " + miles(fila["n_casos"]) + " con máscaras (" + str(fila["pct_con_tumor"]) + "%)", ""),
        (miles(fila["n_pacientes_con_tumor"]), "pacientes con tumor", ""),
        (f"{fila['vol_tumoral_medio_cm3']:,.1f}", "volumen tumoral medio (mediana " + f"{fila['vol_tumoral_mediana_cm3']:,.1f}" + ")", "cm³"),
        (miles(fila["pacientes_2mas_fechas"]), "pacientes con tumor en 2 o más fechas", ""),
        (f"{fila['edad_media']:.0f}", "edad media por paciente · " + f"{fila['pct_mujeres']:.0f}" + "% mujeres", "años"),
    ])


def vista_organo(datos, clave):
    organos = datos["organos"]
    fila = organos.loc[clave]
    barra_superior(True)
    encabezado_organo(fila, False)
    st.write("")
    izquierda, derecha = st.columns([4, 5], gap="large")
    with izquierda:
        st.markdown('<div class="panel"><p class="titulo-sec">Próximamente</p><p class="sub-sec" style="font-size:.92rem;line-height:1.6;margin:0">'
                    "Las estadísticas de " + fila["nombre"].lower() + " ya salen de las máscaras del dataset. En las siguientes versiones "
                    "se agregan aquí el visor de cortes, la reconstrucción 3D y las curvas de seguimiento, igual que en la página de hígado.</p></div>",
                    unsafe_allow_html=True)
        st.write("")
        if st.button("Ver la página de hígado →", type="primary"):
            ir_a("liver")
    with derecha:
        st.markdown('<p class="titulo-sec">' + fila["nombre"] + " frente a los demás órganos</p>", unsafe_allow_html=True)
        st.plotly_chart(grafica_organos(organos, clave), width="stretch", config={"displayModeBar": False})


# ───────────────────────── 7. Pantalla: hígado ─────────────────────────
def vista_higado(datos):
    casos = datos["casos"]
    barra_superior(True)
    encabezado_organo(datos["organos"].loc["liver"], True)
    st.write("")
    if len(casos) == 0:
        st.info("Todavía no hay tomografías de hígado procesadas. Corre el notebook de Colab y copia la carpeta "
                "`procesados/higado` dentro de `datos`.")
        return

    # Aviso mientras haya casos segmentados en modo rápido
    if casos["ts_rapido"].any():
        st.markdown('<div class="aviso"><b>Valores provisionales.</b> El hígado de estos casos se segmentó en modo rápido (--fast). '
                    "Los volúmenes de hígado y el HTB% pueden cambiar al reprocesarlos en Colab a resolución completa.</div>",
                    unsafe_allow_html=True)

    # ── Selector de paciente y de estudio ──
    # Etiqueta de cada paciente: su ID corto (el mismo que usa el notebook) y cuántos estudios tiene
    etiquetas = {}
    for paciente, grupo in casos.groupby("paciente"):
        if len(grupo) == 1:
            etiquetas["Paciente " + id_corto(paciente) + " · 1 estudio"] = paciente
        else:
            etiquetas["Paciente " + id_corto(paciente) + " · " + str(len(grupo)) + " estudios"] = paciente
    columna_1, columna_2, _ = st.columns([2, 2, 3])
    paciente = etiquetas[columna_1.selectbox("Paciente", list(etiquetas.keys()))]

    estudios = casos[casos["paciente"] == paciente].sort_values("fecha")
    opciones = {}
    for _, fila in estudios.iterrows():
        opciones[fila["fecha"].strftime("%d/%m/%Y") + " · " + fila["caso"]] = fila["caso"]
    caso = opciones[columna_2.selectbox("Estudio", list(opciones.keys()))]
    estudio = estudios[estudios["caso"] == caso].iloc[0]
    ct, tumor, higado, _ = cargar_volumen(caso)

    # ── Fila 1: visor de cortes | 3D y HTB% ──
    st.write("")
    izquierda, derecha = st.columns(2, gap="large")
    with izquierda:
        st.markdown('<p class="titulo-sec">Visor de cortes</p><p class="sub-sec">Tomografía axial con el tumor en rojo '
                    "y el contorno del hígado en cian.</p>", unsafe_allow_html=True)
        k = st.slider("Corte", 0, ct.shape[0] - 1, int(estudio["corte_max_tumor"]), key="corte_" + caso)
        a, b, c = st.columns([1, 1, 2])
        ver_tumor = a.toggle("Tumor", True)
        ver_higado = b.toggle("Hígado", True)
        opacidad = c.slider("Opacidad del tumor", 0.1, 1.0, 0.5, 0.05, label_visibility="collapsed")
        st.image(imagen_corte(ct[k], tumor[k], higado[k], ver_tumor, ver_higado, opacidad), width="stretch")
    with derecha:
        st.markdown('<p class="titulo-sec">Reconstrucción 3D</p><p class="sub-sec">Hígado translúcido con los tumores en rojo. '
                    "Arrastra para girar.</p>", unsafe_allow_html=True)
        st.plotly_chart(figura_3d(caso), width="stretch", config={"displayModeBar": False})

        st.markdown('<p class="titulo-sec">Carga tumoral hepática (HTB%)</p>'
                    '<p class="sub-sec">HTB% = volumen de tumor ÷ volumen de hígado × 100. El hígado es la máscara de '
                    "TotalSegmentator unida con el tumor anotado.</p>", unsafe_allow_html=True)
        nombre_nivel, _, clase = nivel_htb(estudio["htb_pct"])
        medidor, semaforo = st.columns([3, 2])
        medidor.plotly_chart(figura_htb(estudio["htb_pct"]), width="stretch", config={"displayModeBar": False})
        luces = ""
        for color_luz in ["verde", "ambar", "rojo"]:
            if color_luz == clase:
                luces = luces + '<div class="luz luz-' + color_luz + '"></div>'      # la luz encendida
            else:
                luces = luces + '<div class="luz"></div>'
        semaforo.markdown('<div style="margin-top:2.2rem"><div class="semaforo">' + luces + "</div>"
                          '<div class="nivel texto-' + clase + '">Carga ' + nombre_nivel.lower() + "</div>"
                          '<p class="nota">Cortes en ' + str(HTB_MODERADA) + "% y " + str(HTB_ALTA)
                          + "%: criterio provisional del equipo, sin fuente clínica.</p></div>", unsafe_allow_html=True)

    # ── Fila 2: cifras del estudio ──
    st.write("")
    if estudio["sexo"] == "M":
        sexo = "Hombre"
    elif estudio["sexo"] == "F":
        sexo = "Mujer"
    else:
        sexo = "Sin dato"
    if pd.notna(estudio["edad"]):
        edad = str(int(estudio["edad"])) + " años"
    else:
        edad = "edad sin dato"
    tarjetas([
        (f"{estudio['vol_tumor_cm3']:,.1f}", "volumen tumoral", "cm³"),
        (f"{estudio['vol_higado_cm3']:,.0f}", "volumen del hígado", "cm³"),
        (int(estudio["n_lesiones"]), "lesiones separadas", ""),
        (sexo, edad + " · fase " + str(estudio["fase"]), ""),
    ])

    # ── Fila 3: seguimiento del paciente ──
    st.write("")
    st.markdown('<p class="titulo-sec">Seguimiento del paciente</p>', unsafe_allow_html=True)
    if datos["seguimiento_viejo"]:
        st.warning("El archivo seguimiento_higado.csv es de una versión anterior. Corre la sección 5 del notebook V1 "
                   "y vuelve a copiar la carpeta stats.")
    seguimiento = datos["seguimiento"]
    curva = pd.DataFrame()
    if len(seguimiento) > 0:
        curva = seguimiento[seguimiento["paciente"] == paciente].sort_values("fecha")
    if len(curva) < 2:
        st.markdown('<p class="sub-sec">Este paciente no tiene 2 o más fechas con tumor hepático anotado, así que no hay curva.</p>',
                    unsafe_allow_html=True)
    else:
        texto, clase, cambio = tendencia_volumen(curva["vol_cm3"].iloc[0], curva["vol_cm3"].iloc[-1])
        st.markdown('<p class="sub-sec">' + str(len(curva)) + " estudios en " + str(int(curva["dias"].max())) + " días · tendencia: "
                    '<span class="texto-' + clase + ' fuerte">' + texto + "</span> (de " + f"{curva['vol_cm3'].iloc[0]:,.1f}" + " a "
                    + f"{curva['vol_cm3'].iloc[-1]:,.1f}" + " cm³, " + f"{cambio:+.0f}" + "%)</p>", unsafe_allow_html=True)
        # Día del estudio seleccionado, contado desde la primera fecha de la curva
        dia_actual = (estudio["fecha"] - curva["fecha"].min()).days
        st.plotly_chart(figura_seguimiento(curva, dia_actual, estudio["vol_tumor_cm3"]), width="stretch",
                        config={"displayModeBar": False})
        st.markdown('<p class="nota">Solo entran fechas con tumor hepático anotado. Puntos rojos: revisar (salto de más de 10 veces '
                    "respecto al estudio anterior, o volumen menor a 1 cm³). Aro blanco: el estudio seleccionado. "
                    "Tendencia: crece desde +" + str(CAMBIO_CRECE_PCT) + "% y se reduce desde " + str(CAMBIO_REDUCE_PCT)
                    + "% de volumen, equivalente aproximado de RECIST 1.1 (+20% y −30% de diámetro; Eisenhauer et al., 2009). "
                    "Criterio provisional del equipo.</p>", unsafe_allow_html=True)

    if isinstance(estudio["reporte"], str) and estudio["reporte"].strip() != "":
        with st.expander("Reporte radiológico original (en inglés)"):
            st.write(estudio["reporte"])


# ───────────────────────── 8. Arranque ─────────────────────────
datos = cargar_tablas()
if datos is None:
    st.error("No encontré los datos en la carpeta `datos`. Corre el notebook de Colab y copia ahí las carpetas `stats` y `procesados`.")
    st.stop()

vista = st.session_state.get("vista", "portada")
if vista == "portada":
    vista_portada(datos)
elif vista in ORGANOS_ACTIVOS:
    vista_higado(datos)
else:
    vista_organo(datos, vista)

st.markdown('<div class="pie">Prototipo académico. No es un dispositivo médico ni sustituye el diagnóstico de un especialista. '
            "Datos: CancerVerse (BodyMaps, Johns Hopkins University), licencia CC BY-NC-ND 4.0. "
            "Máscara de hígado: TotalSegmentator 2.18.0.</div>", unsafe_allow_html=True)
