# -*- coding: utf-8 -*-
"""
T2 - Parte 2: Perplejidad de LDA con 2, 3 y 4 temas.

Corpus mínimo de 12 oraciones (Blei, Ng y Jordan, 2003 - corpus del curso).
Usa el mismo CountVectorizer de la Parte 1 (misma lista de palabras vacías,
resto de parámetros por defecto) y los mismos hiperparámetros de LDA:
learning_method='batch', max_iter=50, random_state=0.
Reporta la perplejidad de cada modelo sobre el corpus (tal como pide el
enunciado, igual que en la Parte 1).
"""

import json
import math

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ----------------------------------------------------------------------
# 1. Corpus del enunciado (12 documentos; la etiqueta 'tema' NO se usa en LDA)
# ----------------------------------------------------------------------
ids_documentos = ["d01", "d02", "d03", "d04", "d05", "d06",
                  "d07", "d08", "d09", "d10", "d11", "d12"]

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

temas_reales = ["deporte"] * 4 + ["cocina"] * 4 + ["astronomía"] * 4

# Palabras vacías del enunciado, tal cual
palabras_vacias = [
    "el", "la", "los", "las", "de", "del", "en", "y", "a", "un", "una",
    "por", "para", "con", "se", "su", "al", "es", "después", "durante",
    "contra", "donde", "alrededor",
]

# ----------------------------------------------------------------------
# 2. Vectorización: mismo CountVectorizer que en la Parte 1
# ----------------------------------------------------------------------
vectorizador = CountVectorizer(stop_words=palabras_vacias)
X = vectorizador.fit_transform(textos)

vocabulario = list(vectorizador.get_feature_names_out())
tamano_vocabulario = int(X.shape[1])
total_tokens = int(X.sum())
num_documentos = int(X.shape[0])

# ----------------------------------------------------------------------
# 3. LDA con 2, 3 y 4 temas + perplejidad sobre el corpus
#    (el enunciado pide la perplejidad de cada modelo sobre el corpus,
#     con los mismos parámetros de la Parte 1)
# ----------------------------------------------------------------------
num_temas_lista = [2, 3, 4]
perplejidad_por_num_temas = {}
log_verosimilitud_total_por_num_temas = {}
tabla_filas = []

for k in num_temas_lista:
    lda = LatentDirichletAllocation(
        n_components=k,
        learning_method="batch",
        max_iter=50,
        random_state=0,
    )
    lda.fit(X)
    perp = float(lda.perplexity(X))
    # Definición de scikit-learn: perplejidad = exp(-log_vero_total / total_palabras)
    log_vero_total = float(-math.log(perp) * total_tokens)
    perplejidad_por_num_temas[str(k)] = perp
    log_verosimilitud_total_por_num_temas[str(k)] = log_vero_total
    tabla_filas.append({
        "num_temas": int(k),
        "perplejidad": perp,
        "log_verosimilitud_total": log_vero_total,
    })

df_tabla = pd.DataFrame(tabla_filas)

# ----------------------------------------------------------------------
# 4. Coherencia con la Parte 1 (entradas/T1.json), si está disponible
# ----------------------------------------------------------------------
coherencia_t1 = {}
try:
    with open("entradas/T1.json", "r", encoding="utf-8") as f:
        t1 = json.load(f)
    if "tamano_vocabulario" in t1:
        coherencia_t1["tamano_vocabulario_T1"] = int(t1["tamano_vocabulario"])
        coherencia_t1["tamano_vocabulario_coincide"] = bool(
            int(t1["tamano_vocabulario"]) == tamano_vocabulario
        )
    if "vocabulario" in t1:
        coherencia_t1["vocabulario_coincide"] = bool(
            [str(w) for w in t1["vocabulario"]] == vocabulario
        )
    if "perplejidad_3_temas" in t1:
        perp3_t1 = float(t1["perplejidad_3_temas"])
        perp3_t2 = perplejidad_por_num_temas["3"]
        coherencia_t1["perplejidad_3_temas_T1"] = perp3_t1
        coherencia_t1["perplejidad_3_temas_T2"] = perp3_t2
        coherencia_t1["diferencia_perplejidad_3_temas"] = float(abs(perp3_t2 - perp3_t1))
        coherencia_t1["perplejidad_3_temas_coincide"] = bool(
            abs(perp3_t2 - perp3_t1) < 1e-8
        )
except FileNotFoundError:
    coherencia_t1["nota"] = "entradas/T1.json no encontrado; comparación omitida."

# ----------------------------------------------------------------------
# 5. Guardar todos los resultados en resultados.json
# ----------------------------------------------------------------------
resultados = {
    "perplejidad_por_num_temas": perplejidad_por_num_temas,
    "tabla_perplejidad": tabla_filas,
    "log_verosimilitud_total_por_num_temas": log_verosimilitud_total_por_num_temas,
    "tamano_vocabulario": tamano_vocabulario,
    "total_tokens_corpus": total_tokens,
    "num_documentos": num_documentos,
    "ids_documentos": ids_documentos,
    "etiquetas_reales": temas_reales,
    "configuracion": {
        "vectorizador": "CountVectorizer(stop_words=palabras_vacias del enunciado, resto por defecto)",
        "lda": {
            "learning_method": "batch",
            "max_iter": 50,
            "random_state": 0,
        },
        "num_temas_evaluados": num_temas_lista,
        "conjunto_evaluacion": "corpus completo (12 documentos), como pide el enunciado",
    },
    "coherencia_con_T1": coherencia_t1,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 6. Impresión de la tabla y de las cifras principales
# ----------------------------------------------------------------------
print("T2 - Parte 2: perplejidad de LDA con 2, 3 y 4 temas")
print(f"Documentos: {num_documentos} | Vocabulario: {tamano_vocabulario} terminos | Tokens: {total_tokens}")
print()
print("Tabla de perplejidad (menor = mejor):")
print(df_tabla.to_string(index=False))
print()
for k in num_temas_lista:
    print(f"Perplejidad con {k} temas: {perplejidad_por_num_temas[str(k)]}")
print()
if coherencia_t1:
    print("Coherencia con la Parte 1 (T1.json):")
    for clave, valor in coherencia_t1.items():
        print(f"  {clave}: {valor}")
print()
print("Resultados guardados en resultados.json")
