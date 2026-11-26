# Guía para defender CancerVerso

Preguntas que nos pueden hacer en la presentación, con respuestas cortas y con las cifras reales. Todas las cifras salen de [KDD.md](KDD.md), [INGENIERIA.md](INGENIERIA.md) y [DATOS.md](DATOS.md). Versión final, 26 de noviembre de 2026.

---

## Guion de 5 minutos para la demo

1. **Portada (30 s).** El problema: CancerVerse pesa 3.04 TB, tiene 24,422 tomografías de 14,599 pacientes y tumores anotados en 13 órganos. Mostrar el mapa de órganos y la barra de datos procesados: 300.9 GB de 300.
2. **Hígado › Paciente (1 min).** Elegir un paciente con varios estudios: visor de cortes, 3D, HTB% y la curva de seguimiento en el tiempo.
3. **Hígado › Cómo lo hicimos (2 min).** Aquí está lo que se califica: cómo cambia el tamaño de los datos en cada etapa, el reparto entre 4 cuentas, por qué shards (Drive leía ~80 cortes/s y con shards más de 1,300), precisión mixta y la tabla de problemas y soluciones.
4. **Hígado › Modelo IA (1.5 min).** La U-Net: Dice de tumor 0.65 (contra radiólogos), Dice de hígado 0.95 (contra TotalSegmentator), HTB% con correlación de 0.85. Mostrar los 4 cortes referencia contra U-Net, decir dónde falla (tumores chicos y cargas mayores a 40%) y cerrar con las matrices: se queda corto más de lo que inventa.

---

## 1. El proyecto

**¿Qué hace CancerVerso, en una frase?**
Mide qué porcentaje del hígado está ocupado por tumor (HTB%) en tomografías y sigue cómo cambia en el tiempo; además entrenamos una red neuronal que aprende a encontrar hígado y tumor por su cuenta.

**¿Por qué solo hígado si el dataset tiene 13 órganos?**
Porque es el órgano con más tumores anotados (6,540 de 12,529) y procesar a fondo un CT cuesta ~84 s y ~129 MB. Con el tiempo y las GPU que teníamos alcanzaba para un órgano bien hecho. Los otros 12 sí tienen sus estadísticas en la app, calculadas desde las máscaras.

**¿Qué es el HTB%?**
Carga tumoral hepática: volumen de tumor ÷ volumen de (hígado + tumor) × 100. Si el hígado mide 1,000 cm³ y el tumor 50 cm³, el HTB% es 50 ÷ 1,050 ≈ 4.8%.

## 2. Big data: los datos y el volumen

**¿Cuántos datos procesaron de verdad?**
300.9 GB de tomografías de hígado (2,336 CTs de 1,319 pacientes), bajados, segmentados y medidos uno por uno. Además, se midieron los tumores de los 24,420 CTs con máscara leyendo solo 0.55 GB.

**¿Por qué no procesaron los 3 TB completos?**
No cabe en Colab: el disco se borra, las sesiones se cortan y solo hay una T4. La meta mínima del curso eran 300 GB y se cumplió. Para todo el dataset sí sacamos estadísticas, pero sin bajar los CTs.

**¿Cómo midieron 24,420 CTs sin bajarlos?**
El dataset trae aparte las máscaras de tumor. El volumen de un tumor es el número de vóxeles marcados × el tamaño del vóxel. Las máscaras vacías pesan exactamente lo mismo en bytes, así que se detectan por su tamaño y no se abren (1 de cada 1,500 sí se abre para verificar el truco).

**¿Cómo se repartieron el trabajo?**
Con el hash MD5 del ID de cada paciente se le asigna una parte (0, 1 o 2). Así todos los estudios de un paciente caen en la misma parte y su seguimiento no se rompe, y cualquiera puede recalcular el reparto sin coordinarse.

**¿Cómo saben que el proceso da el mismo resultado en distintas cuentas?**
173 CTs se procesaron dos veces a propósito, en cuentas distintas. El HTB% difirió como máximo 0.01 puntos.

**¿Qué pasaba si Colab se desconectaba?**
Cada CT terminado se anota en un checklist. Al volver a correr, lo hecho se salta y lo que falló se reintenta. Hubo 9 errores y todos se recuperaron. En el entrenamiento se guarda un checkpoint en Drive cada época.

**¿Qué son los shards y por qué los usaron?**
Son archivos grandes con ~2,000 cortes cada uno (90 en total, 23.6 GB), en lugar de miles de archivos chicos. Leyendo de Drive archivo por archivo salían ~80 cortes por segundo; con shards o en disco local, más de 1,300. Es de 15 a 70 veces más rápido.

**¿Por qué el dataset 2D pesa más (23.6 GB) que los volúmenes reducidos (6 GB)?**
A propósito: los shards van sin comprimir para leerse rápido. Se cambió espacio en disco por velocidad, que era el cuello de botella.

**¿Cómo redujeron cada CT ~50 veces sin perder la medida?**
Primero se calculan todas las métricas sobre el CT completo. Después se guarda una copia reducida (solo la zona del hígado, 256 px, máximo 96 cortes, en 0–255). El HTB% recalculado sobre la copia reducida difiere en una mediana de 0.07 puntos.

## 3. El proceso KDD

**¿Cuáles son las 5 etapas y dónde está cada una?**
1. Selección: dataset, órgano y casos (notebook V1).
2. Preprocesamiento: limpiar fechas, edad y sexo; filtrar series; excluir 72 CTs con píxel deformado (V1 y V2).
3. Transformación: máscara de hígado con TotalSegmentator, HTB% y volúmenes reducidos; dataset 2D en shards (V1 y V2).
4. Minería de datos: U-Net 2D propia (V3).
5. Interpretación: el dashboard.

**¿Por qué excluyeron 72 CTs?**
Eran series adquiridas en otro plano: el píxel medía hasta 10.4 × 1.3 mm, así que la imagen se ve aplastada. Se excluyen con una regla medible (píxel más de 1.5 veces más alto que ancho) y queda la lista de excluidos.

**¿De dónde sale la máscara del hígado si el dataset no la trae?**
De TotalSegmentator 2.18.0, una red preentrenada que segmenta órganos. La corrimos en GPU a resolución completa sobre cada CT.

## 4. El modelo

**¿Es machine learning o deep learning?**
Deep learning. Es una U-Net 2D de MONAI con 6.5 millones de parámetros.

**¿Qué hace la U-Net?**
Recibe un corte de la tomografía y le pone a cada píxel una etiqueta: fondo, hígado o tumor. Primero "encoge" la imagen para entender el contexto y luego la "agranda" para pintar píxel por píxel. Con esos píxeles se calcula el HTB%.

**¿Con qué datos se entrenó y cómo se separaron?**
180,238 cortes de 2,264 CTs y 1,291 pacientes. Se separaron **por paciente** (70/15/15, semilla 42), sin importar qué compañero procesó cada CT:

| Grupo | CTs | Pacientes |
|---|---|---|
| Entrenamiento | 1,582 | 904 |
| Validación | 336 | 194 |
| Prueba | 346 | 193 |
| **Entraron al modelo** | **2,264** | **1,291** |
| Excluidos (píxel deformado) | 72 | 28 |
| **Total procesado** | **2,336** | **1,319** |

Los 28 pacientes excluidos salen de restar 1,319 − 1,291: solo tenían CTs deformados. Ningún paciente está en dos grupos.

**¿Un CT es un estudio? ¿Por qué un paciente tiene varios?**
Un CT es una serie, un escaneo. Un paciente puede tener varios por **seguimiento** (otra fecha) o por **fases** el mismo día (sin contraste, arterial, portal venosa). Los 2,336 CTs son 2,251 estudios (paciente + fecha); 902 pacientes (68%) tienen un solo CT y el que más tiene, 23.

**¿Por qué separar por paciente y no por corte?**
Si cortes de un mismo paciente quedaran en entrenamiento y en prueba, el modelo ya "conocería" a ese paciente y el resultado saldría inflado.

**¿Cómo se entrenó?**
Pérdida Dice + entropía cruzada, AdamW con tasa 0.001 que baja en coseno, precisión mixta fp16 en una T4. 14 épocas en 2 h 5 min (tope de 2 horas). Se guardó la época con mejor Dice de tumor en validación (la 5.ª, 0.751).

**¿Qué es la precisión mixta y qué ganaron?**
La GPU hace parte de las cuentas con números de 16 bits en lugar de 32. En la T4 pasó de 148 a 284 cortes por segundo (1.9 veces) y usa menos memoria.

**¿Por qué no usaron un modelo ya entrenado o LoRA?**
Un modelo chico propio (6.5 millones de parámetros) cabía en una T4 y en 2 horas de entrenamiento, y lo podemos explicar completo. El modelo preentrenado del proyecto ya es TotalSegmentator (para la máscara de hígado). Ajustar uno preentrenado para tumor queda como siguiente paso.

## 5. Resultados

**¿Qué tan bueno es el modelo?**
En la prueba (346 CTs de 193 pacientes que nunca vio):
- Dice de tumor: 0.653 (contra radiólogos). Es el que importa.
- Dice de hígado: 0.949 (contra TotalSegmentator).
- HTB% del modelo contra el real: correlación de 0.85 y error mediano de 0.66 puntos.
- Tumor por píxel: precisión 0.71, recall 0.60, IoU 0.48.

**¿Contra qué lo califican? ¿Lo que marcó el experto es 100% correcto?**
No es la verdad absoluta: es el **estándar de referencia**, la mejor respuesta disponible. Así se evalúa cualquier modelo de segmentación médica. Dos radiólogos tampoco marcan contornos idénticos, así que parte de lo que contamos como error puede ser diferencia de criterio. Por eso decimos "coincide con la referencia" y no "acierta". Hay dos referencias: el **tumor** son las máscaras de CancerVerse anotadas por radiólogos; el **hígado** sale de TotalSegmentator, porque el dataset no trae máscara de hígado. El Dice de hígado mide qué tanto coincidimos con esa herramienta, que tampoco es perfecta: en 445 de 2,336 CTs más del 10% del tumor quedó fuera de su máscara de hígado.

**¿Qué es el Dice?**
2 × (píxeles en común) ÷ (píxeles del modelo + píxeles de la referencia). Vale 1 si coinciden perfecto y 0 si no se tocan. Ejemplo: si la referencia marca 100 píxeles, el modelo 80 y coinciden 70, el Dice es 140 ÷ 180 = 0.78.

**¿Por qué el Dice de tumor es más bajo que el de hígado?**
El tumor es menos del 1% de los píxeles y muchos tumores son chicos. Por tamaño: en el 25% de tumores más chicos el Dice es 0.13; en el 25% más grande, 0.74.

**¿Dónde falla?**
En tumores chicos, que casi no encuentra, y en cargas tumorales mayores a 40%, donde subestima mucho (hay solo 14 casos así en prueba). En la mayoría de los CTs (HTB% menor a 5%) el error es de menos de medio punto.

**¿Qué dice la matriz de confusión por píxel?**
De lo que la referencia marca como tumor, el modelo pinta el 60% como tumor, el 35% como hígado y el 4% como fondo. Es decir, **se queda corto más de lo que inventa**: la precisión de tumor (0.71) es mayor que el recall (0.60). Cuadra con que subestima las cargas altas.

**¿Qué dice la matriz por corte (¿este corte tiene tumor?)?**
Un corte cuenta como "con tumor" si tiene al menos 1 píxel de tumor. De 12,294 cortes con tumor, el modelo detecta 11,681 (sensibilidad 95%) y se le escapan 613. De 15,074 sin tumor, en 7,223 no pinta nada (especificidad 48%) y en 7,851 pinta alguna mancha. Son falsas alarmas chicas: la mitad mide 23 píxeles o menos de los 65,536 del corte. Basta un píxel mal para que el corte entero cuente como falsa alarma; por píxeles, del fondo solo pinta como tumor el 0.02%. Si se exige un mínimo de 50 píxeles para avisar, la especificidad sube a 83% y la sensibilidad baja a 76%: el intercambio de cualquier prueba médica.

**¿Por qué no hicieron la matriz por CT?**
Porque 340 de los 346 CTs de prueba tienen tumor. Habría solo 6 casos sin tumor y la matriz no diría nada.

**¿Por qué no usan la exactitud (accuracy)?**
Porque engaña: el tumor es solo el 0.5% de los píxeles. Un modelo que pinte todo como fondo saca 93.2% de exactitud sin encontrar ni un tumor; la U-Net saca 99.3%. Por eso usamos Dice, precisión y recall.

**¿Las cifras son confiables?**
Son algo optimistas por dos razones: la validación se usó para elegir la mejor época, y solo se evalúan cortes que tienen hígado. Las decimos en la app y en los documentos. Además, encontramos CTs duplicados que se colaron entre grupos (ver la siguiente pregunta).

**¿Hay fuga de datos entre entrenamiento y prueba?**
Un poco, y la medimos. En el dataset, 291 de los 2,336 CTs tienen un gemelo idéntico (mismo volumen de tumor y de hígado, mismas lesiones y cortes), casi siempre con **otro ID de paciente** y fechas corridas: parece la misma tomografía publicada dos veces. Como separamos por ID de paciente, 32 de los 346 CTs de prueba tienen su gemelo en entrenamiento o validación. En esos 32 el modelo saca Dice de tumor 0.83, porque ya los había visto. Sin ellos (314 CTs): Dice de tumor **0.625**, precisión 0.69, recall 0.57, correlación del HTB% 0.84 y error mediano 0.69 puntos. La conclusión no cambia: el hígado sale casi igual y el tumor chico sigue siendo el problema. La solución completa es separar por grupo de gemelos y volver a entrenar. Lo detectamos al revisar nuestros propios resultados: es una prueba de que el proceso se puede auditar.

## 6. La app

**¿La app corre el modelo en vivo?**
No. Streamlit Cloud no tiene GPU. La app muestra la evaluación ya hecha en Colab: métricas, cortes de ejemplo y gráficas. Correrlo en vivo queda para la siguiente versión.

**¿Por qué el visor solo tiene 20 casos? ¿Por qué esos y no otros?**
No se escogieron a mano: salen de una regla escrita en el notebook V1 (sección 6), sorteada con semilla fija (42), así que cualquiera que lo corra obtiene los mismos 20.
- **16 CTs de 6 pacientes con seguimiento limpio** (4 con 3 fechas y 2 con 2): ningún tumor menor a 1 cm³, sin saltos de volumen de más de 10 veces, series normales y **la misma fase en todas sus fechas**, para que la curva compare lo mismo.
- **4 CTs sueltos con tumor grande**, del 10% de tumores más grandes, para que el 3D y el control de calidad se vean bien.

El visor es una vitrina, no la muestra del estudio: **los 20 no cambian ninguna cifra**. El HTB%, las curvas, las estadísticas y la U-Net usan los 2,336 CTs. No se metieron todos porque serían ~6 GB en la web (hoy pesa ~54 MB) y la licencia CC BY-NC-ND no permite redistribuir los datos. No son representativos (HTB% mediano ~12% contra 1.8% del total), porque se buscaron tumores visibles y seguimientos limpios; conviene decirlo antes de que lo pregunten.

**¿Los datos están protegidos?**
El repositorio no trae datos de pacientes. La web descarga un paquete cifrado (AES-256-GCM) desde Drive y pide un enlace con clave. Se respeta la licencia CC BY-NC-ND 4.0 del dataset.

**¿El semáforo de HTB% (25% y 50%) es un criterio médico?**
No. Es un criterio provisional del equipo, sin fuente clínica, y la app lo dice. Igual la regla de "crece / se reduce" es una aproximación de RECIST 1.1, que mide diámetros y no volumen.

## 7. Si nos preguntan qué sigue

- Mejorar el tumor chico: más épocas, más peso a la clase tumor o usar cortes vecinos (2.5D).
- Correr la U-Net en vivo en la app con un servidor con GPU.
- Activar más órganos en el menú.
