# -*- coding: utf-8 -*-
"""
T2 - Parte 2 - Numero de temas (k = 2, 3 y 4).

Con el mismo CountVectorizer de la Parte 1 (mismas palabras vacias, demas
parametros por defecto) ajustado sobre el corpus completo de 12 oraciones,
se ajusta LatentDirichletAllocation(n_components=k, learning_method='batch',
max_iter=50, random_state=0) para k = 2, 3 y 4 sobre la matriz
documento-termino completa X y se reporta la perplejidad de cada modelo
sobre el corpus (los 12 documentos), en una sola tabla comparativa,
tal como pide el enunciado.

Salidas (carpeta actual):
  * resultados.json : perplejidad_k2, perplejidad_k3, perplejidad_k4
    (precision completa) y cifras de apoyo.
  * Impresion de la tabla comparativa en consola.
  * Esta subtarea NO requiere figura PNG.
"""

import json

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# ----------------------------------------------------------------------
# 1. Corpus minimo del enunciado (12 oraciones, 4 por tema real).
# ----------------------------------------------------------------------
ids = ["d01", "d02", "d03", "d04", "d05", "d06",
       "d07", "d08", "d09", "d10", "d11", "d12"]

documentos = [
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

# Etiqueta real del enunciado (solo metadato del corpus; LDA no la usa).
temas_reales = ["deporte"] * 4 + ["cocina"] * 4 + ["astronomía"] * 4

# ----------------------------------------------------------------------
# 2. Palabras vacias (identicas a la Parte 1)
# ----------------------------------------------------------------------
stop_words = [
    "el", "la", "los", "las", "de", "del", "en", "y", "a", "un", "una",
    "por", "para", "con", "se", "su", "al", "es", "después", "durante",
    "contra", "donde", "alrededor",
]

# ----------------------------------------------------------------------
# 3. Mismo CountVectorizer de la Parte 1: ajustado sobre el corpus
#    completo (los 12 documentos), demas parametros por defecto.
# ----------------------------------------------------------------------
vectorizador = CountVectorizer(stop_words=stop_words)
X = vectorizador.fit_transform(documentos)

vocabulario = [str(w) for w in vectorizador.get_feature_names_out()]
tamano_vocabulario = int(X.shape[1])
n_tokens_corpus = int(X.sum())

# ----------------------------------------------------------------------
# 4. LDA con k = 2, 3 y 4 (mismos hiperparametros de la Parte 1),
#    ajustado sobre la matriz completa X; perplejidad sobre el corpus.
# ----------------------------------------------------------------------
perplejidades = {}
palabras_top = {}
n_top = 5

for k in (2, 3, 4):
    lda = LatentDirichletAllocation(
        n_components=k,
        learning_method="batch",
        max_iter=50,
        random_state=0,
    )
    lda.fit(X)
    perplejidades[k] = float(lda.perplexity(X))

    componentes = lda.components_  # forma (k, V)
    top_por_tema = []
    for t in range(k):
        idx = np.argsort(componentes[t])[::-1][:n_top]
        top_por_tema.append([vocabulario[int(j)] for j in idx])
    palabras_top[k] = top_por_tema

# ----------------------------------------------------------------------
# 5. Tabla comparativa unica (k = 2, 3, 4)
# ----------------------------------------------------------------------
tabla = pd.DataFrame(
    {
        "n_componentes_k": [2, 3, 4],
        "perplejidad_corpus": [
            perplejidades[2], perplejidades[3], perplejidades[4],
        ],
    }
)
k_mejor = int(tabla.loc[tabla["perplejidad_corpus"].idxmin(), "n_componentes_k"])

# ----------------------------------------------------------------------
# 6. Guardar todas las cifras en resultados.json (carpeta actual)
# ----------------------------------------------------------------------
resultados = {
    "perplejidad_k2": perplejidades[2],
    "perplejidad_k3": perplejidades[3],
    "perplejidad_k4": perplejidades[4],
    "k_con_menor_perplejidad": k_mejor,
    "nota_evaluacion": (
        "Perplejidad de cada modelo LDA calculada sobre el corpus completo "
        "(los 12 documentos), tal como pide el enunciado, con el mismo "
        "CountVectorizer de la Parte 1 (mismas palabras vacias, demas "
        "parametros por defecto) y los mismos hiperparametros de LDA "
        "(learning_method='batch', max_iter=50, random_state=0)."
    ),
    "n_documentos": int(len(documentos)),
    "ids_documentos": [str(i) for i in ids],
    "temas_reales": [str(t) for t in temas_reales],
    "tamano_vocabulario": tamano_vocabulario,
    "vocabulario": vocabulario,
    "n_tokens_corpus": n_tokens_corpus,
    "palabras_top_por_tema_k2": palabras_top[2],
    "palabras_top_por_tema_k3": palabras_top[3],
    "palabras_top_por_tema_k4": palabras_top[4],
    "tabla_comparativa": tabla.to_dict(orient="records"),
}

# Contexto de la Parte 1 (control de consistencia: mismo vectorizador y
# mismos hiperparametros con k=3 deben dar la misma perplejidad).
try:
    with open("entradas/T1.json", "r", encoding="utf-8") as f:
        t1 = json.load(f)
    if isinstance(t1, dict) and "perplejidad_k3" in t1:
        resultados["perplejidad_k3_parte1"] = float(t1["perplejidad_k3"])
except (OSError, ValueError):
    pass

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 7. Impresion de las cifras principales
# ----------------------------------------------------------------------
print("T2 - Parte 2: numero de temas (k = 2, 3, 4) - perplejidad sobre el corpus")
print(f"Corpus: {len(documentos)} documentos, vocabulario de "
      f"{tamano_vocabulario} palabras, {n_tokens_corpus} tokens tras quitar "
      f"las palabras vacias")
print()
print("Tabla comparativa (perplejidad sobre el corpus de 12 documentos):")
print(tabla.to_string(index=False))
print()
print("Cifras principales (precision completa):")
print(f"  perplejidad_k2 = {perplejidades[2]!r}")
print(f"  perplejidad_k3 = {perplejidades[3]!r}")
print(f"  perplejidad_k4 = {perplejidades[4]!r}")
print(f"  k con menor perplejidad: {k_mejor}")
print()
for k in (2, 3, 4):
    print(f"Top {n_top} palabras por tema (k = {k}):")
    for i, palabras in enumerate(palabras_top[k]):
        print(f"  tema {i + 1}: {', '.join(palabras)}")
    print()
if "perplejidad_k3_parte1" in resultados:
    print("[control] perplejidad_k3 de la Parte 1: "
          f"{resultados['perplejidad_k3_parte1']!r}")
    print("[control] perplejidad_k3 de esta parte:  "
          f"{perplejidades[3]!r}")
print()
print("Resultados guardados en resultados.json")
