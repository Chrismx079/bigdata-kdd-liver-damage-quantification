# Proceso KDD de CancerVerso

Este documento sigue las cinco etapas del proceso KDD (Knowledge Discovery in Databases) y dice, en cada una, qué se hizo, con qué y qué falta. Versión final, 25 de noviembre de 2026.

**Pregunta del proyecto:** ¿cuánto del hígado de un paciente está ocupado por tumor, y cómo cambia eso entre un estudio y el siguiente?

| Etapa | Qué se hizo | Dónde está | Estado |
|---|---|---|---|
| 1. Selección | Elegir dataset, órgano y casos | Notebook, secciones 2, 3 y 6 | Hecho (3 partes) |
| 2. Preprocesamiento | Limpiar metadatos y filtrar series | Notebook, secciones 3, 5 y 6 | Hecho |
| 3. Transformación | Medir máscaras, segmentar hígado, reducir volúmenes; dataset 2D para la U-Net | V1, secciones 4, 8 y 9 · V2 | Hecho: 2,336 CTs; dataset 2D de 2,264 CTs |
| 4. Minería de datos | TotalSegmentator + U-Net 2D propia (MONAI) y métricas | V1, sección 8 · V3 · V3b | Hecho (U-Net entrenada en T4) |
| 5. Interpretación | Dashboard (con pestañas Modelo IA y Cómo lo hicimos), control de calidad, seguimiento | `app.py` · `resultados/` · Notebook, secciones 10 y 11 | Hecho para hígado |

---

## 1. Selección

- **Dataset:** BodyMaps/CancerVerse. 3.04 TB, 24,422 tomografías de 14,599 pacientes, con tumores anotados por expertos en 13 órganos.
- **Por qué este dataset:** es abierto, es grande (cumple el requisito de volumen del curso) y tiene varios estudios del mismo paciente en fechas distintas.
- **Órgano elegido para la primera versión:** hígado. Es el órgano con más anotaciones de tumor (6,540 de 12,529).
- **Reparto del trabajo:** los pacientes se dividen en 3 partes con una función fija (hash MD5 del ID del paciente). Todos los estudios de un paciente caen en la misma parte, para no romper su seguimiento.
- **Casos de hígado:** primero pacientes con seguimiento limpio (2 o 3 fechas) y después estudios sueltos con tumor. La selección usa una semilla fija, así que siempre salen los mismos casos.

## 2. Preprocesamiento

- **Fechas y edad:** las fechas se convierten a tipo fecha; la edad "above 89" se registra como 90.
- **Datos faltantes:** el sexo y la edad que faltan se rellenan con los otros estudios del mismo paciente.
- **Casos sin máscara:** 2 tomografías del dataset no traen máscaras y se excluyen de todas las cifras.
- **Filtro de series:** solo entran series con cortes de 5 mm o menos, 50 cortes o más y píxel de 3 mm o menos.
- **Filtro de mediciones poco confiables:** se marcan para revisar los tumores menores a 1 cm³ y los cambios de volumen de más de 10 veces entre dos estudios seguidos.
- **Misma fase:** para cada paciente se prefiere una misma fase del estudio en todas las fechas (de preferencia portal venosa), porque el tumor se ve distinto según la fase. Si en una fecha no hay CT en esa fase se usa el que exista y la columna `misma_fase` lo marca: con las 3 partes unidas, 570 de 1,258 pacientes con seguimiento mezclan fases. Para las curvas de la demostración solo se eligen pacientes con una sola fase.

## 3. Transformación

- **De máscara a volumen:** se cuentan los vóxeles de tumor de cada órgano y se multiplican por el tamaño del vóxel para obtener cm³. Esto se hace sobre las máscaras, sin bajar los CTs.
- **Máscara de hígado:** el dataset no la trae, así que se genera con TotalSegmentator.
- **Verificación de alineación:** antes de calcular se comprueba que el CT, la máscara de tumor y la de hígado tengan el mismo tamaño y la misma posición en el espacio.
- **Variable principal:** `HTB% = volumen de tumor ÷ volumen de (hígado ∪ tumor) × 100`.
- **Otras variables por estudio:** volumen de tumor, volumen de hígado, número de lesiones separadas y porcentaje de tumor fuera de la máscara de hígado.
- **Volumen reducido para la app:** cada CT se recorta a la zona del hígado, se pasa a escala de grises con ventana de tejido blando y se reduce a 256 px y máximo 96 cortes. Las métricas se calculan antes de reducir, sobre el CT completo.

## 4. Minería de datos

**Lo que ya está**

- **Modelo de deep learning aplicado:** TotalSegmentator 2.18.0, una red neuronal preentrenada de segmentación, corrida en GPU a resolución completa sobre cada CT de hígado.
- **Patrones obtenidos:** estadísticas por órgano (porcentaje de CTs con tumor, volumen medio y mediano, edad y sexo por paciente) y curvas de volumen tumoral por paciente en el tiempo.
- **Regla de tendencia:** un tumor "crece" si su volumen sube 73% o más y "se reduce" si baja 66% o más. Es el equivalente en volumen de RECIST 1.1 (+20% y −30% de diámetro).

**Modelo propio: U-Net 2D (notebook V3)**

- **Datos:** dataset 2D del notebook V2: 180,238 cortes con hígado de 2,264 CTs y 1,291 pacientes, separados **por paciente** en entrenamiento (904), validación (194) y prueba (193). Se excluyeron 72 CTs con píxel deformado (series adquiridas en otro plano). Detalle en [DATOS.md](DATOS.md).
- **Modelo:** U-Net 2D de MONAI (6.5 millones de parámetros) que clasifica cada píxel como fondo, hígado o tumor. Pérdida Dice + entropía cruzada, precisión mixta.
- **Entrenamiento:** 14 épocas en una GPU T4 (2 h 5 min); se guarda la época con mejor Dice de tumor en validación.
- **Evaluación en prueba (346 CTs de pacientes que el modelo nunca vio):**

| Métrica | Valor |
|---|---|
| Dice de hígado (global / mediana por CT) | 0.949 / 0.969 |
| Dice de tumor (global / mediana por CT) | 0.653 / 0.521 |
| Dice de tumor en tumores chicos / grandes (cuartil inferior / superior) | 0.13 / 0.74 |
| Tumor por píxel: precisión / recall / IoU (notebook V3b) | 0.71 / 0.60 / 0.48 |
| Por corte, ¿tiene tumor?: sensibilidad / especificidad (1 píxel basta; con 50 píxeles) | 95% / 48% (76% / 83%) |
| HTB% predicho vs real (correlación) | 0.85 |
| HTB% predicho vs real (error absoluto mediano) | 0.66 puntos |

- **Referencia:** el tumor se compara contra las máscaras de radiólogos de CancerVerse; el hígado, contra TotalSegmentator (el dataset no trae máscara de hígado). Es la mejor respuesta disponible, no la verdad absoluta.
- **Lectura:** el hígado coincide casi perfecto con TotalSegmentator; el tumor depende de su tamaño y el modelo se queda corto más de lo que inventa (precisión 0.71 > recall 0.60). El HTB%, la variable del proyecto, sale cerca del real en la mayoría de los CTs, pero el modelo subestima los casos con más de 40% de carga tumoral.
- **Límites:** solo se evalúan cortes con hígado y la validación se usó para elegir la mejor época, así que las cifras son algo optimistas.

## 5. Interpretación y evaluación

- **Dashboard (Streamlit):** portada con las cifras del dataset, menú de 13 órganos y página de hígado con tres pestañas:
  - **Paciente:** visor de cortes, reconstrucción 3D, medidor de HTB% y curva de seguimiento.
  - **Modelo IA:** métricas de la U-Net en test, 4 cortes referencia vs U-Net, HTB% real vs predicho, Dice por tamaño de tumor, curva de entrenamiento, dónde falla y matrices de confusión con precisión y recall.
  - **Cómo lo hicimos:** las 5 etapas del KDD, tamaño de los datos en cada etapa, reparto entre 4 cuentas, benchmark de lectura, precisión mixta y problemas resueltos.
- **Control de calidad visual:** imágenes de todos los casos donde más del 10% del tumor quedó fuera de la máscara de hígado. Con las 3 partes son 445 de 2,336 CTs (19%); en esos casos el HTB% puede quedar bajo y conviene revisarlos.
- **Límites que el propio dashboard declara:**
  - Los cortes del semáforo de HTB% (25% y 50%) son un criterio provisional del equipo, sin fuente clínica.
  - La regla de tendencia es una aproximación de RECIST, que mide diámetros y no volumen total.
  - La curva de seguimiento solo incluye fechas con tumor anotado; no muestra cuándo un tumor desapareció.
  - Es un prototipo académico, no un dispositivo médico.

---

## Avance en números

| Concepto | Valor |
|---|---|
| CTs con máscaras medidos (3 partes) | 24,420 |
| Anotaciones de tumor | 12,529 |
| CTs de hígado procesados a detalle | 2,336 (1,319 pacientes) |
| Datos de CT procesados | 301 GB |
| Pacientes con seguimiento (2 o más fechas) | 1,258 (405 con una sola fase y sin marcas de revisar) |
| Seguimiento más largo | 3,497 días (unos 9.6 años) |
| HTB% mediano | 1.8% (40 CTs arriba de 50%) |
| Dataset 2D para la U-Net | 180,238 cortes · 2,264 CTs · 23.6 GB en 90 shards |
| U-Net en prueba | Dice tumor 0.65 · Dice hígado 0.95 · HTB% r = 0.85 · tumor: precisión 0.71, recall 0.60 |
| Meta mínima / deseable | 300 GB / 1 TB |

## Siguientes pasos

1. Correr la U-Net en vivo en la app (hoy la pestaña Modelo IA muestra la evaluación ya hecha; Streamlit Cloud no tiene GPU).
2. Mejorar el tumor chico (más épocas, más peso a la clase tumor o entrada 2.5D con cortes vecinos).
3. Activar más órganos.

La ingeniería de datos (cómo se procesaron los 3.04 TB) está en [INGENIERIA.md](INGENIERIA.md).
