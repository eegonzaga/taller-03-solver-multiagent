# -*- coding: utf-8 -*-
"""
T2 - Parte 2 - Numero de temas (k = 2, 3 y 4).

Con el mismo CountVectorizer de la Parte 1 (mismas palabras vacias, resto de
parametros por defecto) y los mismos parametros de LDA
(learning_method='batch', max_iter=50, random_state=0), se ajusta
LatentDirichletAllocation con n_components = 2, 3 y 4 y se reporta la
perplejidad de cada modelo sobre el corpus en una sola tabla comparativa.

Salidas (carpeta actual):
  * resultados.json : perplejidad_k2, perplejidad_k3, perplejidad_k4 y cifras
    de contexto (vocabulario, log-verosimilitudes, tabla comparativa, etc.).
  * Impresion de la tabla comparativa en consola.
Esta subtarea NO requiere figura PNG.
"""

import json

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# ----------------------------------------------------------------------
# 1. Corpus minimo del enunciado (12 oraciones; la etiqueta real de tema no
#    se usa: LDA es no supervisado).
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

temas_reales = ["deporte"] * 4 + ["cocina"] * 4 + ["astronomía"] * 4  # solo referencia

# ----------------------------------------------------------------------
# 2. Palabras vacias (identicas a la Parte 1)
# ----------------------------------------------------------------------
stop_words = [
    "el", "la", "los", "las", "de", "del", "en", "y", "a", "un", "una",
    "por", "para", "con", "se", "su", "al", "es", "después", "durante",
    "contra", "donde", "alrededor",
]

# ----------------------------------------------------------------------
# 3. Mismo CountVectorizer de la Parte 1.
#    Se reconstruye con las mismas palabras vacias (determinista). Si
#    entradas/T1.json trae el vocabulario de la Parte 1 y coincide en
#    contenido, se fija explicitamente para reproducir exactamente la misma
#    matriz documento-termino (mismo orden de columnas que en la Parte 1).
# ----------------------------------------------------------------------
vectorizador_ref = CountVectorizer(stop_words=stop_words)
X_ref = vectorizador_ref.fit_transform(documentos)
vocab_ref = [str(w) for w in vectorizador_ref.get_feature_names_out()]

t1 = None
try:
    with open("entradas/T1.json", "r", encoding="utf-8") as f:
        t1 = json.load(f)
except (OSError, ValueError):
    t1 = None

t1_vocab = t1.get("vocabulario") if isinstance(t1, dict) else None
vocab_valido_t1 = (
    isinstance(t1_vocab, list)
    and len(t1_vocab) > 0
    and all(isinstance(w, str) for w in t1_vocab)
    and len(set(t1_vocab)) == len(t1_vocab)
)
vocab_igual_parte1 = bool(set(t1_vocab) == set(vocab_ref)) if vocab_valido_t1 else None

if vocab_valido_t1 and vocab_igual_parte1:
    vectorizador = CountVectorizer(stop_words=stop_words, vocabulary=list(t1_vocab))
    X = vectorizador.fit_transform(documentos)
    vocabulario = [str(w) for w in vectorizador.get_feature_names_out()]
else:
    vectorizador, X, vocabulario = vectorizador_ref, X_ref, vocab_ref

# ----------------------------------------------------------------------
# 4. LDA con k = 2, 3 y 4 (mismos hiperparametros de la Parte 1)
# ----------------------------------------------------------------------
perplejidades = {}
log_verosimilitud = {}
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
    perplejidades[k] = float(lda.perplexity(X))   # perplejidad sobre el corpus
    log_verosimilitud[k] = float(lda.score(X))    # log-verosimilitud aproximada

    componentes = lda.components_                 # forma (k, V)
    top_por_tema = []
    for t in range(k):
        idx = np.argsort(componentes[t])[::-1][:n_top]
        top_por_tema.append([vocabulario[j] for j in idx])
    palabras_top[k] = top_por_tema

# ----------------------------------------------------------------------
# 5. Tabla comparativa (una sola tabla con k = 2, 3, 4)
# ----------------------------------------------------------------------
tabla = pd.DataFrame(
    {
        "n_componentes_k": [2, 3, 4],
        "perplejidad": [perplejidades[2], perplejidades[3], perplejidades[4]],
        "log_verosimilitud_total": [
            log_verosimilitud[2], log_verosimilitud[3], log_verosimilitud[4],
        ],
    }
)
k_mejor = int(tabla.loc[tabla["perplejidad"].idxmin(), "n_componentes_k"])

# ----------------------------------------------------------------------
# 6. Guardar todas las cifras en resultados.json (carpeta actual)
# ----------------------------------------------------------------------
resultados = {
    "n_documentos": int(X.shape[0]),
    "tamano_vocabulario": int(X.shape[1]),
    "vocabulario": vocabulario,
    "ids_documentos": ids,
    "temas_reales": temas_reales,
    "perplejidad_k2": perplejidades[2],
    "perplejidad_k3": perplejidades[3],
    "perplejidad_k4": perplejidades[4],
    "log_verosimilitud_total_k2": log_verosimilitud[2],
    "log_verosimilitud_total_k3": log_verosimilitud[3],
    "log_verosimilitud_total_k4": log_verosimilitud[4],
    "k_con_menor_perplejidad": k_mejor,
    "palabras_top_por_tema_k2": palabras_top[2],
    "palabras_top_por_tema_k3": palabras_top[3],
    "palabras_top_por_tema_k4": palabras_top[4],
    "vocabulario_igual_al_de_parte1": vocab_igual_parte1,
    "tabla_comparativa": tabla.to_dict(orient="records"),
}

if isinstance(t1, dict) and "perplejidad_k3" in t1:
    try:
        perp_k3_t1 = float(t1["perplejidad_k3"])
        resultados["perplejidad_k3_parte1_T1"] = perp_k3_t1
        resultados["coincide_perplejidad_k3_con_parte1"] = bool(
            abs(perp_k3_t1 - perplejidades[3]) < 1e-9
        )
    except (TypeError, ValueError):
        pass

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 7. Impresion de las cifras principales
# ----------------------------------------------------------------------
print("T2 - Parte 2: numero de temas (k = 2, 3, 4)")
print(f"Documentos: {X.shape[0]} | Tamano del vocabulario: {X.shape[1]}")
print()
print("Tabla comparativa de perplejidad sobre el corpus (precision completa):")
print(tabla.to_string(index=False))
print()
print("Tabla resumida:")
print(f"{'k':>3} | {'perplejidad':>18} | {'log_verosimilitud':>20}")
for k in (2, 3, 4):
    print(f"{k:>3} | {perplejidades[k]:>18.6f} | {log_verosimilitud[k]:>20.6f}")
print()
print("Perplejidades (precision completa):")
for k in (2, 3, 4):
    print(f"  perplejidad_k{k} = {perplejidades[k]!r}")
print(f"  k con menor perplejidad: {k_mejor}")
print()
for k in (2, 3, 4):
    print(f"Top {n_top} palabras por tema (k = {k}):")
    for i, palabras in enumerate(palabras_top[k]):
        print(f"  tema {i + 1}: {', '.join(palabras)}")
    print()
if "perplejidad_k3_parte1_T1" in resultados:
    print(f"[control] perplejidad_k3 de la Parte 1 (T1.json): "
          f"{resultados['perplejidad_k3_parte1_T1']!r}")
    print(f"[control] perplejidad_k3 recalculada aqui:        {perplejidades[3]!r}")
    print(f"[control] coinciden: {resultados['coincide_perplejidad_k3_con_parte1']}")
print()
print("Resultados guardados en resultados.json")
