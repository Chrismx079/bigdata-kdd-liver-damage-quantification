# Proceso KDD de CancerVerso

Este documento sigue las cinco etapas del proceso KDD (Knowledge Discovery in Databases) y dice, en cada una, qué se hizo, con qué y qué falta. Avance al 23 de noviembre de 2026.

**Pregunta del proyecto:** ¿cuánto del hígado de un paciente está ocupado por tumor, y cómo cambia eso entre un estudio y el siguiente?

| Etapa | Qué se hizo | Dónde está | Estado |
|---|---|---|---|
| 1. Selección | Elegir dataset, órgano y casos | Notebook, secciones 2, 3 y 6 | Hecho (3 partes) |
| 2. Preprocesamiento | Limpiar metadatos y filtrar series | Notebook, secciones 3, 5 y 6 | Hecho |
| 3. Transformación | Medir máscaras, segmentar hígado, reducir volúmenes | Notebook, secciones 4, 8 y 9 | Hecho para 2,336 CTs (3 partes) |
| 4. Minería de datos | Modelo de deep learning y métricas | Notebook, sección 8 · U-Net pendiente | Parcial |
| 5. Interpretación | Dashboard, control de calidad, seguimiento | `app.py` · Notebook, secciones 10 y 11 | Hecho para hígado |

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

**Lo que falta**

- **Modelo propio:** una U-Net 2D entrenada por el equipo (con MONAI) para segmentar el tumor hepático.
- **Evaluación:** comparar la predicción de la U-Net con la anotación de los expertos usando el coeficiente Dice, con pacientes separados para entrenamiento y prueba.
- **Datos para entrenarla:** ya hay 2,336 CTs de hígado procesados de 1,319 pacientes (3 partes), suficientes para separar entrenamiento y prueba por paciente.

## 5. Interpretación y evaluación

- **Dashboard (Streamlit):** portada con las cifras del dataset, menú de 13 órganos y página de hígado con visor de cortes, reconstrucción 3D, medidor de HTB% y curva de seguimiento.
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
| Meta mínima / deseable | 300 GB / 1 TB |

## Siguientes pasos

1. Entrenar y evaluar la U-Net 2D de tumor hepático.
2. Conectar la U-Net a la app.
3. Activar más órganos.
