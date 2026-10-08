# -*- coding: utf-8 -*-
"""
T1 — Parte 1: LDA con tres temas sobre el corpus mínimo de 12 oraciones (d01–d12).

Pasos:
  1. Construir el corpus tal como aparece en el enunciado (sin descargas).
  2. Vectorizar con CountVectorizer usando la lista de palabras vacías dada
     (resto de parámetros por defecto).
  3. Ajustar LatentDirichletAllocation(n_components=3, learning_method='batch',
     max_iter=50, random_state=0).
  4. Reportar: tamaño del vocabulario, cinco palabras más probables por tema
     y perplejidad del modelo sobre el corpus.
  5. Guardar todas las cifras en resultados.json e imprimirlas.

No se requiere ninguna figura PNG para esta subtarea.
"""

import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ----------------------------------------------------------------------
# 1. Corpus de 12 oraciones (d01–d12), exactamente como en el enunciado
# ----------------------------------------------------------------------
ids = ["d01", "d02", "d03", "d04", "d05", "d06",
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

# Etiquetas reales (LDA no las usa; solo contexto del enunciado)
temas_reales = ["deporte"] * 4 + ["cocina"] * 4 + ["astronomía"] * 4

# ----------------------------------------------------------------------
# 2. Vectorización: CountVectorizer con la lista de palabras vacías dada
# ----------------------------------------------------------------------
stop_words = ["el", "la", "los", "las", "de", "del", "en", "y", "a", "un", "una",
              "por", "para", "con", "se", "su", "al", "es", "después", "durante",
              "contra", "donde", "alrededor"]

vectorizer = CountVectorizer(stop_words=stop_words)  # resto de parámetros por defecto
X = vectorizer.fit_transform(textos)

vocabulario = vectorizer.get_feature_names_out()
tamano_vocabulario = int(X.shape[1])

# ----------------------------------------------------------------------
# 3. Ajuste de LDA con los hiperparámetros exactos del enunciado
# ----------------------------------------------------------------------
lda = LatentDirichletAllocation(
    n_components=3,
    learning_method="batch",
    max_iter=50,
    random_state=0,
)
lda.fit(X)

# Perplejidad del modelo sobre el corpus (tal como pide el enunciado)
perplejidad_3_temas = float(lda.perplexity(X))

# ----------------------------------------------------------------------
# 4. Cinco palabras más probables de cada tema
# ----------------------------------------------------------------------
n_top = 5
palabras_top_por_tema = {}
pesos_top_por_tema = {}
for k, componente in enumerate(lda.components_):
    idx_top = np.argsort(componente)[::-1][:n_top]
    palabras_top_por_tema["tema_%d" % k] = [str(vocabulario[i]) for i in idx_top]
    pesos_top_por_tema["tema_%d" % k] = [float(componente[i]) for i in idx_top]

# Distribución documento-tema (informativa; LDA no usa las etiquetas reales)
doc_topic = lda.transform(X)
distribucion_documento_tema = {
    ids[i]: [float(v) for v in doc_topic[i]] for i in range(len(ids))
}

# ----------------------------------------------------------------------
# 5. Guardar todas las cifras en resultados.json
# ----------------------------------------------------------------------
resultados = {
    "tamano_vocabulario": tamano_vocabulario,
    "palabras_top_por_tema": palabras_top_por_tema,
    "perplejidad_3_temas": perplejidad_3_temas,
    "vocabulario": [str(v) for v in vocabulario],
    "pesos_top_por_tema": pesos_top_por_tema,
    "distribucion_documento_tema": distribucion_documento_tema,
    "n_iter": int(lda.n_iter_),
    "n_documentos": int(X.shape[0]),
    "temas_reales": temas_reales,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 6. Impresión de las cifras principales
# ----------------------------------------------------------------------
print("=" * 64)
print("T1 - Parte 1: LDA con 3 temas (corpus de 12 oraciones)")
print("=" * 64)
print("Tamaño del vocabulario: %d" % tamano_vocabulario)
print("Perplejidad sobre el corpus (3 temas): %.10f" % perplejidad_3_temas)
print()
print("Cinco palabras más probables por tema:")
for k in sorted(palabras_top_por_tema.keys()):
    pares = ["%s (%.4f)" % (p, w)
             for p, w in zip(palabras_top_por_tema[k], pesos_top_por_tema[k])]
    print("  %s: %s" % (k, ", ".join(pares)))
print()
print("Distribución documento-tema (filas = d01..d12):")
for i, doc_id in enumerate(ids):
    print("  %s: [%s]" % (doc_id, ", ".join("%.4f" % v for v in doc_topic[i])))
print()
print("Resultados guardados en resultados.json")
