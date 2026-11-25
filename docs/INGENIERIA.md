# Cómo lo procesamos: la ingeniería de datos de CancerVerso

Este documento cuenta **qué problemas de volumen de datos encontramos y cómo los resolvimos**, con las cifras medidas en cada paso. El diccionario de cada tabla y archivo está en [DATOS.md](DATOS.md); el proceso KDD completo, en [KDD.md](KDD.md).

Versión final, 25 de noviembre de 2026.

---

## 1. El problema

| Dato | Valor |
|---|---|
| Tamaño del dataset (BodyMaps/CancerVerse) | **3.04 TB** |
| Tomografías (CT) | 24,422 de 14,599 pacientes |
| Tamaño promedio de un CT | ~129 MB comprimido |
| Máquina disponible | Google Colab: una GPU T4, un disco local que se borra al desconectar y sesiones que se cortan solas |
| Almacenamiento permanente | Google Drive (5 TB), lento para muchos archivos chicos |
| Equipo | 4 personas, cada una con su cuenta de Colab |

No cabe ni en el disco ni en el tiempo de una sesión. Todo lo que sigue sale de esa restricción.

---

## 2. Cómo cambió el tamaño de los datos en cada etapa

| Etapa | Qué queda | Tamaño |
|---|---|---|
| Dataset original | 24,422 CTs + máscaras | 3.04 TB |
| Medir tumores sin bajar CTs | 24,420 CTs × 13 órganos, solo máscaras | 0.55 GB leídos |
| CTs de hígado procesados a detalle | 2,336 CTs completos (bajados, segmentados y medidos) | **300.9 GB leídos** |
| Volúmenes reducidos (`volumen.npz`) | 2,336 archivos, uno por CT | 6.0 GB (**~50× menos**) |
| Dataset 2D para la U-Net | 180,238 cortes en 90 shards | 23.6 GB (crece a propósito, ver 3.8) |
| Modelo entrenado | `mejor.pt` | 26 MB |

---

## 3. Las técnicas, una por una

### 3.1 No bajar lo que no hace falta
Para las estadísticas de los 13 órganos **no se bajó ningún CT**. El dataset trae un archivo aparte con todas las máscaras de tumor (0.55 GB), y el volumen de un tumor se calcula contando vóxeles de su máscara × tamaño del vóxel. Así se midieron **24,420 CTs y 12,529 anotaciones de tumor** leyendo menos del 0.02% de los 3.04 TB.

### 3.2 Saltarse las máscaras vacías sin abrirlas
De cada CT hay 13 máscaras, y casi todas están vacías (sin tumor en ese órgano). Todas las vacías de un mismo CT pesan exactamente lo mismo en bytes, así que el tamaño que más se repite es el de "vacía" y esas no se descomprimen. Para comprobar que el truco no se equivoca, 1 de cada 1,500 vacías sí se descomprime y se revisa. De la máscara solo se lee la **cabecera** (forma y tamaño del vóxel), no la imagen completa.

### 3.3 Elegir qué procesar a fondo
El paso caro (bajar el CT, segmentar el hígado con TotalSegmentator y medir) se hizo solo para **hígado**, el órgano con más anotaciones de tumor (6,540). Se procesaron **2,336 CTs (300.9 GB)**, lo que cumple la meta mínima de 300 GB.

### 3.4 Repartir el trabajo entre 4 cuentas sin romper pacientes
Los pacientes se dividen en 3 partes con el **hash MD5 de su ID**: todos los estudios de un paciente caen en la misma parte, así que su seguimiento en el tiempo no se rompe, y cualquiera puede recalcular el reparto sin coordinarse.

| Parte | Quién | CTs |
|---|---|---|
| 0 | Isay | 1,909 |
| 0 | Paloma | 200 (27 nuevos + 173 repetidos a propósito) |
| 1 | Paul | 200 |
| 2 | Christopher | 200 |

Los **173 casos repetidos** sirvieron de **prueba de reproducibilidad**: el HTB% calculado en dos cuentas distintas difirió como máximo 0.01 puntos.

### 3.5 Procesar por lotes y poder reanudar
Colab corta las sesiones. Cada CT terminado se anota en un **checklist** (`checklist_higado.csv`) con su estado, GB y segundos; al volver a correr, lo hecho se salta y lo que falló se reintenta.

- **55 horas** de proceso sumando todos los casos, **mediana de 84 s por CT**.
- **9 errores**, todos recuperados en la siguiente corrida.
- Los archivos se escriben con otro nombre y luego se renombran: nunca queda un archivo a medias con el nombre bueno.

### 3.6 Drive es la bodega, el disco local es la mesa de trabajo
Cada CT se baja al disco local de Colab, se procesa ahí y solo el resultado se guarda en Drive. El CT original se borra del disco local en cuanto termina, para que el disco no se llene.

**Memoria:** en una máquina de 8 GB de RAM, TotalSegmentator falló 8 veces (el sistema lo mataba por falta de memoria, código −9). Se resolvió pidiendo RAM alta.

### 3.7 Reducir cada CT ~50 veces sin perder la medida
Primero se calculan **todas las métricas sobre el CT completo** (volumen de tumor, de hígado, HTB%). Después se guarda una versión reducida para la app y para el modelo:

1. Recorte en z a la zona del hígado (+15 mm de margen).
2. Ventana de tejido blando (−160 a 240 HU) y paso a `uint8` (0–255).
3. 256 px de lado y máximo 96 cortes.
4. Las máscaras se cortan en 0.3 (no 0.5) al reducir, para no perder lesiones chicas.

Resultado: de **~129 MB a ~2.6 MB por CT**. El HTB% recalculado sobre la versión reducida difiere del original en una mediana de **0.07 puntos** (máximo 1.53).

### 3.8 Del volumen al dataset para entrenar (notebook V2)

| Paso | Problema | Solución | Cifra |
|---|---|---|---|
| Inventario | ¿Están todos los archivos de las 4 cuentas? | Atajos de Drive a las carpetas de los compañeros + comparar contra cada checklist | 2,336 de 2,336, en 145 s |
| Copia | Leer miles de archivos chicos de Drive es lento | 16 copias simultáneas al disco local | 6 GB en 85–100 s (~60–70 MB/s) |
| Validación | Un archivo dañado arruina el entrenamiento | Abrir cada uno y revisar claves, tipos, tamaños y HTB% | 0 errores |
| Píxel deformado | Series adquiridas en otro plano se ven aplastadas en axial | Excluir si el píxel es >1.5 veces más alto que ancho | 72 CTs fuera |
| Separación | Fuga de datos si un paciente está en train y en test | Separar por **paciente** (70/15/15, semilla 42) | 0 pacientes en dos grupos |
| Formato | `float32` ocupa 4× más | Guardar `uint8` y dividir entre 255 en la GPU | Imágenes: 11.8 GB en vez de 47 GB |
| Shards | Miles de archivos chicos = lentos | ~2,000 cortes por archivo, **sin comprimir** | 90 archivos |
| Linaje | ¿Se entrenó con lo documentado? | `manifest.json` con huella MD5 de cada shard | 90 de 90 coinciden |

**Por qué shards sin comprimir: benchmark de lectura** (cortes con hígado por segundo):

| Forma de leer | Corrida 1 | Corrida 2 |
|---|---|---|
| 100 `volumen.npz` sueltos desde Drive | 77 | 85 |
| Los mismos, desde disco local | 1,290 | 1,314 |
| Shards desde Drive* | 1,230 | 1,667 |
| Shards desde disco local | 1,813 | 4,184 |
| Ya en RAM | 8,557 | 5,874 |

\* Puede salir optimista: los shards se acababan de escribir y Colab pudo tenerlos en caché.

Lo estable entre corridas: **Drive archivo por archivo da ~80 cortes/s; cualquier otra forma pasa de 1,300 (15 a 70 veces más)**. Por eso todo se copia al disco local y el modelo entrena desde la RAM.

### 3.9 Entrenar con los datos en RAM y precisión mixta (notebook V3)
- Antes de entrenar se **vuelven a verificar las 90 huellas MD5** contra el manifiesto (copia + verificación de 23.6 GB: 154 s).
- Los cortes se cargan a la RAM en `uint8` (92 s) y se convierten a 0–1 en la GPU, lote por lote: la GPU no espera al disco.
- **Precisión mixta** en la T4: de 148 a **284 cortes/s (1.9×)** y menos memoria.

| GPU | Precisión | Lote | Cortes/s | Memoria |
|---|---|---|---|---|
| T4 | fp32 | 24 | 148 | 1.5 GB |
| T4 | fp16 (mixta) | 24 | 284 | 0.9 GB |

- **Checkpoint en Drive cada época** (`ultimo.pt`) y del mejor modelo (`mejor.pt`): si Colab se corta, se retoma.
- **Tope de tiempo** (120 min): termina la época en curso y se detiene, para no gastar de más.
- Resultado: 14 épocas en 2 h 5 min (~490 s por época, ~255 cortes/s); predecir 27,368 cortes de test tomó 42 s.

---

## 4. Lo que salió mal y cómo se resolvió

| Problema | Qué pasó | Solución |
|---|---|---|
| Falta de RAM | TotalSegmentator falló 8 veces con 8 GB | Entorno con RAM alta |
| Sesiones cortadas | Colab desconecta en horas | Checklist reanudable y checkpoints por época |
| Archivos en otras cuentas | Colab solo ve "Mi unidad" | Atajos de Drive en `companeros/` |
| Imágenes aplastadas | 72 CTs con píxel de hasta 10.4 × 1.3 mm | Filtro medible en mm, con lista de excluidos |
| bf16 en la T4 | PyTorch dice que la T4 "soporta" bf16, pero lo emula y sería muy lento | Elegir bf16 solo si la capacidad de cómputo es ≥ 8 (G4, A100, L4); en la T4, fp16 |
| GPU no disponible | La G4 y la A100 no se asignaron en 3 intentos (más de 12 min cada uno) la noche del 24 nov | Entrenar en la T4, que sí estaba disponible; el notebook ajusta solo el lote y la precisión a la GPU que toque |
| Permiso de Drive | El montaje vence si nadie aprueba la ventana a tiempo | Volver a ejecutar; el notebook es reanudable |

---

## 5. Tiempos medidos

| Etapa | Tiempo |
|---|---|
| Procesar 2,336 CTs de hígado (V1, suma de casos) | ~55 h |
| Inventario de 2,509 archivos en Drive | 145 s |
| Copiar 6 GB de `volumen.npz` (16 hilos) | 85–100 s |
| Validar 2,336 volúmenes | 200–218 s |
| Armar 180,238 cortes en 90 shards | 226–263 s |
| MD5 + subir 23.6 GB de shards a Drive | 272–373 s |
| Copiar y verificar shards antes de entrenar | 154 s |
| Entrenar 14 épocas en T4 | 7,498 s (2 h 5 min) |
| Predecir test (27,368 cortes) | 42 s |

---

## 6. Si lo volviéramos a hacer

- Pedir RAM alta desde el inicio.
- Guardar el espaciado en mm desde el principio y filtrar las series anisotrópicas antes de procesarlas (se habrían ahorrado 72 CTs de proceso).
- Empaquetar en shards directamente en la V1, en lugar de un archivo por CT.
- Reservar la GPU grande con anticipación: a medianoche no hubo G4 ni A100.
