# -*- coding: utf-8 -*-
"""
T3 — Parte 3 — ¿Recupera LDA los temas reales?

Usa la distribución documento-tema del modelo LDA de tres temas ajustado en la
Parte 1 (disponible en entradas/T1.json, clave 'distribucion_documentos_temas').
Si ese archivo no estuviera disponible o tuviera un formato inesperado, se
reajusta el modelo exactamente con los parámetros del enunciado (determinista:
random_state=0), lo que reproduce el modelo de la Parte 1.

Pasos:
  1) Obtener la distribución documento-tema (12 x 3).
  2) Asignar a cada documento (d01–d12) su tema LDA dominante.
  3) Calcular la pureza: para cada tema LDA, contar los documentos de la
     etiqueta real más frecuente entre los asignados a él, sumar esas cuentas
     y dividir por 12.
  4) Tabla de asignación por documento e identificación de los documentos mal
     agrupados.

Salida: impresiones por pantalla y resultados.json (esta subtarea NO requiere
figura PNG).
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# ----------------------------------------------------------------------------
# 1. Corpus y etiquetas reales (datos del enunciado; LDA no ve las etiquetas)
# ----------------------------------------------------------------------------
ids = ["d%02d" % i for i in range(1, 13)]
textos = [
    "El delantero marcó dos goles en el partido de fútbol y el equipo ganó la liga.",
    "El entrenador del equipo preparó la defensa para el partido final de la liga.",
    "La tenista ganó el torneo después de un partido largo contra la campeona.",
    "El equipo de baloncesto perdió el partido por un punto en el último segundo.",
    "La receta lleva harina, huevos y azúcar; la masa se hornea durante treinta minutos.",
    "Para la salsa se sofríe cebolla y ajo en aceite y se añade tomate.",
    "El pan se hornea con harina, agua, sal y levadura después de amasar la masa.",
    "La sopa de verduras lleva cebolla, zanahoria, ajo y sal, y se cocina a fuego lento.",
    "El telescopio observó una galaxia lejana y varias estrellas de la nebulosa.",
    "Los planetas giran alrededor de la estrella; la órbita de la Tierra dura un año.",
    "La nebulosa es una nube de gas donde nacen estrellas nuevas en la galaxia.",
    "El telescopio espacial fotografió planetas y la órbita de una luna de Júpiter.",
]
etiquetas = ["deporte"] * 4 + ["cocina"] * 4 + ["astronomía"] * 4
n_docs = len(ids)
n_temas = 3

stop_words = [
    "el", "la", "los", "las", "de", "del", "en", "y", "a", "un", "una", "por",
    "para", "con", "se", "su", "al", "es", "después", "durante", "contra",
    "donde", "alrededor",
]

# ----------------------------------------------------------------------------
# 2. Distribución documento-tema de la Parte 1
# ----------------------------------------------------------------------------
def extraer_distribucion(obj):
    """Devuelve la distribución documento-tema como array (12, 3) o None."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        if all(k in obj for k in ids):
            try:
                return np.vstack([np.asarray(obj[k], dtype=float) for k in ids])
            except Exception:
                return None
        return None
    try:
        arr = np.asarray(obj, dtype=float)
    except Exception:
        return None
    if arr.shape == (n_docs, n_temas):
        return arr
    if arr.shape == (n_temas, n_docs):
        return arr.T
    return None


fuente = None
dist_doc_tema = None
ruta_t1 = os.path.join("entradas", "T1.json")
if os.path.exists(ruta_t1):
    try:
        with open(ruta_t1, "r", encoding="utf-8") as f:
            t1 = json.load(f)
        dist_doc_tema = extraer_distribucion(t1.get("distribucion_documentos_temas"))
        if dist_doc_tema is not None:
            fuente = "entradas/T1.json -> 'distribucion_documentos_temas' (modelo de la Parte 1)"
    except Exception:
        dist_doc_tema = None

if dist_doc_tema is None:
    # Respaldo: reajuste determinista del modelo de la Parte 1 con los
    # parámetros exactos del enunciado (CountVectorizer por defecto + lista de
    # palabras vacías; LDA k=3, batch, max_iter=50, random_state=0).
    X = CountVectorizer(stop_words=stop_words).fit_transform(textos)
    lda = LatentDirichletAllocation(
        n_components=3, learning_method="batch", max_iter=50, random_state=0
    )
    lda.fit(X)
    dist_doc_tema = lda.transform(X)
    fuente = ("reajuste local con los parámetros de la Parte 1 "
              "(CountVectorizer + LDA k=3, batch, max_iter=50, random_state=0)")

# Normalización defensiva por documento (cada fila debe sumar 1)
sumas = dist_doc_tema.sum(axis=1, keepdims=True)
if np.all(sumas > 0):
    dist_doc_tema = dist_doc_tema / sumas

# ----------------------------------------------------------------------------
# 3. Tema dominante por documento
# ----------------------------------------------------------------------------
tema_dominante = np.argmax(dist_doc_tema, axis=1)
prob_dominante = dist_doc_tema[np.arange(n_docs), tema_dominante]

# ----------------------------------------------------------------------------
# 4. Pureza
# ----------------------------------------------------------------------------
conteos_por_tema = {}
etiqueta_mayoritaria = {}
numerador = 0
for t in range(n_temas):
    conteos = {}
    for i in range(n_docs):
        if int(tema_dominante[i]) == t:
            conteos[etiquetas[i]] = conteos.get(etiquetas[i], 0) + 1
    conteos_por_tema[t] = conteos
    if conteos:
        # etiqueta real más frecuente; empates rotos alfabéticamente (determinista)
        mejor = sorted(conteos.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        etiqueta_mayoritaria[t] = mejor
        numerador += max(conteos.values())
    else:
        etiqueta_mayoritaria[t] = None

pureza = numerador / float(n_docs)

# ----------------------------------------------------------------------------
# 5. Asignación por documento y documentos mal agrupados
# ----------------------------------------------------------------------------
asignacion = []
mal_agrupados = []
detalle_mal_agrupados = []
for i in range(n_docs):
    t = int(tema_dominante[i])
    etiq_tema = etiqueta_mayoritaria[t]
    correcto = (etiq_tema == etiquetas[i])
    fila = {
        "id": ids[i],
        "etiqueta_real": etiquetas[i],
        "tema_lda_dominante": t,
        "probabilidad_tema_dominante": float(prob_dominante[i]),
        "distribucion_documento_tema": [float(v) for v in dist_doc_tema[i]],
        "etiqueta_mayoritaria_del_tema": etiq_tema,
        "bien_agrupado": bool(correcto),
    }
    asignacion.append(fila)
    if not correcto:
        mal_agrupados.append(ids[i])
        detalle_mal_agrupados.append(fila)

# ----------------------------------------------------------------------------
# 6. Impresión de resultados
# ----------------------------------------------------------------------------
print("=" * 78)
print("T3 · Parte 3 — ¿Recupera LDA los temas reales?")
print("=" * 78)
print("Fuente de la distribución documento-tema:", fuente)
print()
print("Distribución documento-tema (filas = documentos, columnas = temas LDA):")
dist_df = pd.DataFrame(
    np.round(dist_doc_tema, 4),
    index=ids,
    columns=["tema_%d" % t for t in range(n_temas)],
)
print(dist_df.to_string())
print()

tabla = pd.DataFrame(
    {
        "id": ids,
        "etiqueta_real": etiquetas,
        "tema_lda_dominante": tema_dominante.astype(int),
        "p_tema_dominante": np.round(prob_dominante, 4),
        "etiqueta_mayoritaria_del_tema": [
            etiqueta_mayoritaria[int(t)] for t in tema_dominante
        ],
        "bien_agrupado": [
            "sí" if etiqueta_mayoritaria[int(t)] == etiquetas[i] else "NO"
            for i, t in enumerate(tema_dominante)
        ],
    }
)
print("Asignación por documento (índices de tema LDA: 0–2):")
print(tabla.to_string(index=False))
print()
print("Composición real de cada tema LDA (conteo de etiquetas asignadas):")
for t in range(n_temas):
    print("  tema %d -> etiqueta mayoritaria: %s | conteos: %s"
          % (t, etiqueta_mayoritaria[t], conteos_por_tema[t]))
print()
print("Pureza = %d / %d = %.6f" % (numerador, n_docs, pureza))
if mal_agrupados:
    print("Documentos mal agrupados:", ", ".join(mal_agrupados))
    for fila in detalle_mal_agrupados:
        print("  - %s (etiqueta real: %s) asignado al tema %d, cuyo grupo "
              "mayoritario es '%s'"
              % (fila["id"], fila["etiqueta_real"], fila["tema_lda_dominante"],
                 fila["etiqueta_mayoritaria_del_tema"]))
else:
    print("Documentos mal agrupados: ninguno (pureza = 1.0)")

# ----------------------------------------------------------------------------
# 7. Guardado de resultados (tipos nativos, sin redondear)
# ----------------------------------------------------------------------------
resultados = {
    "asignacion_por_documento": asignacion,
    "pureza": float(pureza),
    "documentos_mal_agrupados": mal_agrupados,
    "detalle_documentos_mal_agrupados": detalle_mal_agrupados,
    "pureza_numerador": int(numerador),
    "n_documentos": int(n_docs),
    "mapeo_tema_lda_a_etiqueta": {
        str(t): etiqueta_mayoritaria[t] for t in range(n_temas)
    },
    "conteo_etiquetas_por_tema": {
        str(t): conteos_por_tema[t] for t in range(n_temas)
    },
    "distribucion_documentos_temas_usada": [
        [float(v) for v in fila] for fila in dist_doc_tema
    ],
    "fuente_distribucion": fuente,
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)
print()
print("Resultados guardados en resultados.json")
