"""
CancerVerso · Revisión de CTs duplicados entre entrenamiento y prueba

Qué hace:
  1. Busca CTs que son la misma tomografía publicada dos veces en el dataset:
     mismo volumen de tumor, mismo volumen de hígado, mismo número de lesiones y mismo número de cortes.
  2. Revisa si alguno de esos gemelos quedó de un lado en entrenamiento/validación y del otro en prueba.
     Como el dataset les pone distinto ID de paciente, la separación por paciente no los detecta.
  3. Recalcula las métricas de la U-Net en prueba sin esos CTs.

Archivos que lee (de Drive, carpeta "Big data Zucco"):
  stats/casos_higado.csv           tabla de CTs procesados (notebook V1)
  stats/split_pacientes.csv        a qué grupo quedó cada CT (notebook V2)
  modelos/unet_t4/test_por_caso.csv conteos de píxeles por CT de prueba (notebook V3)

Cómo correrlo en Colab:  !python revisar_duplicados.py "/content/drive/MyDrive/Big data Zucco"
"""
import sys
import json

import numpy as np
import pandas as pd

BASE = sys.argv[1] if len(sys.argv) > 1 else "/content/drive/MyDrive/Big data Zucco"

casos = pd.read_csv(BASE + "/stats/casos_higado.csv").drop_duplicates("caso", keep="last")
split = pd.read_csv(BASE + "/stats/split_pacientes.csv")
prueba = pd.read_csv(BASE + "/modelos/unet_t4/test_por_caso.csv")

# 1. Gemelos: CTs con exactamente las mismas medidas
tabla = casos.merge(split[["caso", "split"]], on="caso", how="left")
CLAVE = ["vol_tumor_cm3", "vol_higado_cm3", "n_lesiones", "n_cortes"]
gemelos = tabla[tabla.duplicated(CLAVE, keep=False)].copy()
gemelos["grupo"] = gemelos.groupby(CLAVE).ngroup()
grupos_mismo_paciente = 0
for _, g in gemelos.groupby("grupo"):
    if g["paciente"].nunique() == 1:
        grupos_mismo_paciente += 1
print("CTs con un gemelo:", len(gemelos), "| grupos:", gemelos["grupo"].nunique(),
      "| grupos con el mismo ID de paciente:", grupos_mismo_paciente)

# 2. CTs de prueba cuyo gemelo quedó en entrenamiento o validación
con_fuga = []
for _, fila in gemelos[gemelos["split"] == "test"].iterrows():
    otros = gemelos[(gemelos["grupo"] == fila["grupo"]) & (gemelos["caso"] != fila["caso"])]
    if otros["split"].isin(["train", "val"]).any():
        con_fuga.append(fila["caso"])
print("CTs de prueba con gemelo en entrenamiento o validación:", len(con_fuga), "de", len(prueba))


# 3. Métricas con y sin esos CTs
def metricas(t):
    s = t.sum(numeric_only=True)
    return {
        "cts": int(len(t)),
        "dice_higado": round(2 * s["comun_hig"] / (s["pred_hig"] + s["real_hig"]), 3),
        "dice_tumor": round(2 * s["comun_tum"] / (s["pred_tum"] + s["real_tum"]), 3),
        "precision_tumor": round(s["comun_tum"] / s["pred_tum"], 3),
        "recall_tumor": round(s["comun_tum"] / s["real_tum"], 3),
        "htb_correlacion": round(float(np.corrcoef(t["htb_real"], t["htb_pred"])[0, 1]), 3),
        "htb_error_absoluto_mediano": round(float(t["error_htb"].median()), 2),
    }


resultado = {
    "_nota": "Revisión de CTs duplicados (misma tomografía con distinto ID de paciente). Script notebooks/revisar_duplicados.py.",
    "cts_con_gemelo": int(len(gemelos)),
    "grupos_de_gemelos": int(gemelos["grupo"].nunique()),
    "grupos_mismo_paciente": grupos_mismo_paciente,
    "cts_prueba_con_gemelo_fuera": len(con_fuga),
    "todos": metricas(prueba),
    "sin_gemelos": metricas(prueba[~prueba["caso"].isin(con_fuga)]),
    "solo_gemelos": metricas(prueba[prueba["caso"].isin(con_fuga)]),
}
print(json.dumps(resultado, indent=2, ensure_ascii=False))
with open("revision_duplicados.json", "w") as f:
    json.dump(resultado, f, indent=2, ensure_ascii=False)
print("Guardado en revision_duplicados.json")
