# -*- coding: utf-8 -*-
"""
T2 - Parte 2 - Numero de temas (k = 2, 3 y 4) - version corregida.

CORRECCION de la fuga de datos detectada por la revision automatica:
en el intento anterior la perplejidad se calculaba sobre la MISMA matriz
documento-termino con la que se ajusto LDA (evaluacion sobre entrenamiento).
Ahora se hace una evaluacion honesta sin fuga:

  1. El corpus de 12 oraciones se divide en entrenamiento (75%) y prueba
     (25%), de forma estratificada por el tema real (usado SOLO para
     estratificar la particion; LDA sigue siendo no supervisado y nunca ve
     las etiquetas), con random_state=0 (determinista).
  2. El CountVectorizer (mismas palabras vacias de la Parte 1, demas
     parametros por defecto) se ajusta SOLO con los documentos de
     entrenamiento; los de prueba unicamente se transforman.
  3. Cada LatentDirichletAllocation (learning_method='batch', max_iter=50,
     random_state=0) se ajusta SOLO con el entrenamiento y la perplejidad
     se reporta sobre el conjunto de PRUEBA (datos no vistos durante el
     ajuste), para k = 2, 3 y 4, en una sola tabla comparativa.

Salidas (carpeta actual):
  * resultados.json : perplejidad_k2, perplejidad_k3, perplejidad_k4
    (perplejidad de generalizacion sobre prueba) y cifras de apoyo.
  * Impresion de la tabla comparativa en consola.
  * Esta subtarea NO requiere figura PNG.
"""

import json

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

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

# Etiqueta real (solo para estratificar la particion; LDA no la usa).
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
# 3. Particion entrenamiento/prueba sin fuga de datos.
#    75% entrenamiento / 25% prueba, estratificada por tema real,
#    random_state=0 (reproducible). Nada se ajusta con la prueba.
# ----------------------------------------------------------------------
ids_train, ids_test, docs_train, docs_test, temas_train, temas_test = train_test_split(
    ids, documentos, temas_reales,
    test_size=0.25,
    stratify=temas_reales,
    random_state=0,
)

# ----------------------------------------------------------------------
# 4. Mismo CountVectorizer de la Parte 1 (mismas palabras vacias, demas
#    parametros por defecto). Por la regla anti-fuga, se ajusta SOLO con
#    los documentos de entrenamiento; la prueba solo se transforma.
# ----------------------------------------------------------------------
vectorizador = CountVectorizer(stop_words=stop_words)
X_train = vectorizador.fit_transform(docs_train)   # ajuste: solo entrenamiento
X_test = vectorizador.transform(docs_test)         # prueba: solo transformacion

vocabulario = [str(w) for w in vectorizador.get_feature_names_out()]
n_palabras_prueba = int(X_test.sum())

# ----------------------------------------------------------------------
# 5. LDA con k = 2, 3 y 4 (mismos hiperparametros de la Parte 1).
#    Ajuste en entrenamiento; perplejidad y log-verosimilitud sobre PRUEBA.
# ----------------------------------------------------------------------
perplejidades = {}
log_verosimilitud_prueba = {}
palabras_top = {}
n_top = 5

for k in (2, 3, 4):
    lda = LatentDirichletAllocation(
        n_components=k,
        learning_method="batch",
        max_iter=50,
        random_state=0,
    )
    lda.fit(X_train)                                    # ajuste: solo train
    perplejidades[k] = float(lda.perplexity(X_test))    # evaluacion: solo test
    log_verosimilitud_prueba[k] = float(lda.score(X_test))

    componentes = lda.components_                       # forma (k, V)
    top_por_tema = []
    for t in range(k):
        idx = np.argsort(componentes[t])[::-1][:n_top]
        top_por_tema.append([vocabulario[int(j)] for j in idx])
    palabras_top[k] = top_por_tema

# ----------------------------------------------------------------------
# 6. Tabla comparativa unica (k = 2, 3, 4)
# ----------------------------------------------------------------------
tabla = pd.DataFrame(
    {
        "n_componentes_k": [2, 3, 4],
        "perplejidad_prueba": [
            perplejidades[2], perplejidades[3], perplejidades[4],
        ],
        "log_verosimilitud_prueba": [
            log_verosimilitud_prueba[2],
            log_verosimilitud_prueba[3],
            log_verosimilitud_prueba[4],
        ],
        "palabras_prueba_total": [n_palabras_prueba] * 3,
    }
)
k_mejor = int(tabla.loc[tabla["perplejidad_prueba"].idxmin(), "n_componentes_k"])

# ----------------------------------------------------------------------
# 7. Guardar todas las cifras en resultados.json (carpeta actual)
# ----------------------------------------------------------------------
resultados = {
    "perplejidad_k2": perplejidades[2],
    "perplejidad_k3": perplejidades[3],
    "perplejidad_k4": perplejidades[4],
    "k_con_menor_perplejidad": k_mejor,
    "nota_evaluacion": (
        "Perplejidades calculadas sobre el conjunto de PRUEBA (25% del corpus, "
        "particion estratificada por tema real con random_state=0). El "
        "CountVectorizer y cada LDA se ajustaron SOLO con el 75% de "
        "entrenamiento; no hay evaluacion sobre los datos de ajuste."
    ),
    "particion": {
        "test_size": 0.25,
        "estratificada_por_tema_real": True,
        "random_state": 0,
    },
    "n_documentos_total": int(len(documentos)),
    "n_documentos_entrenamiento": int(X_train.shape[0]),
    "n_documentos_prueba": int(X_test.shape[0]),
    "ids_entrenamiento": [str(i) for i in ids_train],
    "ids_prueba": [str(i) for i in ids_test],
    "temas_reales_prueba": [str(t) for t in temas_test],
    "tamano_vocabulario": int(len(vocabulario)),
    "vocabulario": vocabulario,
    "palabras_prueba_total": n_palabras_prueba,
    "log_verosimilitud_prueba_k2": log_verosimilitud_prueba[2],
    "log_verosimilitud_prueba_k3": log_verosimilitud_prueba[3],
    "log_verosimilitud_prueba_k4": log_verosimilitud_prueba[4],
    "palabras_top_por_tema_k2": palabras_top[2],
    "palabras_top_por_tema_k3": palabras_top[3],
    "palabras_top_por_tema_k4": palabras_top[4],
    "tabla_comparativa": tabla.to_dict(orient="records"),
}

# Contexto de la Parte 1 (solo informativo: esa perplejidad se calculo sobre
# el corpus completo, es decir, en muestra, y no es comparable directamente).
try:
    with open("entradas/T1.json", "r", encoding="utf-8") as f:
        t1 = json.load(f)
    if isinstance(t1, dict):
        if "perplejidad_k3" in t1:
            resultados["perplejidad_k3_parte1_en_muestra"] = float(t1["perplejidad_k3"])
        if "tamano_vocabulario" in t1:
            resultados["tamano_vocabulario_parte1"] = int(t1["tamano_vocabulario"])
        if "n_documentos" in t1:
            resultados["n_documentos_parte1"] = int(t1["n_documentos"])
except (OSError, ValueError):
    pass

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 8. Impresion de las cifras principales
# ----------------------------------------------------------------------
print("T2 - Parte 2: numero de temas (k = 2, 3, 4) - perplejidad sobre PRUEBA")
print(f"Particion: {X_train.shape[0]} entrenamiento / {X_test.shape[0]} prueba "
      f"(estratificada por tema real, random_state=0)")
print(f"Documentos de prueba: {', '.join(ids_test)}  (temas: {', '.join(temas_test)})")
print(f"Tamano del vocabulario (ajustado solo con entrenamiento): {len(vocabulario)}")
print(f"Tokens de prueba evaluados: {n_palabras_prueba}")
print()
print("Tabla comparativa (perplejidad sobre el conjunto de prueba):")
print(tabla.to_string(index=False))
print()
print("Cifras principales (precision completa):")
print(f"  perplejidad_k2 = {perplejidades[2]!r}")
print(f"  perplejidad_k3 = {perplejidades[3]!r}")
print(f"  perplejidad_k4 = {perplejidades[4]!r}")
print(f"  k con menor perplejidad en prueba: {k_mejor}")
print()
for k in (2, 3, 4):
    print(f"Top {n_top} palabras por tema (k = {k}, modelo ajustado solo con entrenamiento):")
    for i, palabras in enumerate(palabras_top[k]):
        print(f"  tema {i + 1}: {', '.join(palabras)}")
    print()
if "perplejidad_k3_parte1_en_muestra" in resultados:
    print("[control] perplejidad_k3 de la Parte 1 (en muestra, corpus completo): "
          f"{resultados['perplejidad_k3_parte1_en_muestra']!r}")
    print("[control] perplejidad_k3 de esta parte (held-out, sobre prueba):     "
          f"{perplejidades[3]!r}")
    print("[control] no son comparables directamente: distinto conjunto de evaluacion.")
print()
print("Resultados guardados en resultados.json")
