# -*- coding: utf-8 -*-
"""
T1 — Parte 1 — LDA con tres temas
Corpus de doce oraciones (d01-d12) del enunciado, etiquetas reales deporte/cocina/astronomía
(solo informativas: LDA no las usa). CountVectorizer con la lista de palabras vacías dada,
LatentDirichletAllocation(n_components=3, learning_method='batch', max_iter=50, random_state=0).
Reporta: tamaño del vocabulario, top-5 palabras por tema y perplejidad sobre el corpus.
Guarda todo en resultados.json. No produce figura PNG.
"""

import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ------------------------------------------------------------------
# 1. Corpus del enunciado (12 oraciones) y etiquetas reales (no usadas por LDA)
# ------------------------------------------------------------------
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

etiquetas_reales = (["deporte"] * 4 + ["cocina"] * 4 + ["astronomía"] * 4)

# ------------------------------------------------------------------
# 2. Vectorización: CountVectorizer con las palabras vacías del enunciado
#    (resto de parámetros por defecto)
# ------------------------------------------------------------------
stop_words = ['el', 'la', 'los', 'las', 'de', 'del', 'en', 'y', 'a', 'un', 'una',
              'por', 'para', 'con', 'se', 'su', 'al', 'es', 'después', 'durante',
              'contra', 'donde', 'alrededor']

vectorizer = CountVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(textos)  # matriz documento-término (12 x V)

vocabulario = vectorizer.get_feature_names_out()
tamano_vocabulario = int(len(vocabulario))

# ------------------------------------------------------------------
# 3. Ajuste de LDA con k = 3 (parámetros exactos del enunciado)
# ------------------------------------------------------------------
lda = LatentDirichletAllocation(n_components=3,
                                learning_method='batch',
                                max_iter=50,
                                random_state=0)
lda.fit(X)

# Perplejidad del modelo sobre el corpus (tal como pide el enunciado)
perplejidad_k3 = float(lda.perplexity(X))

# ------------------------------------------------------------------
# 4. Cinco palabras más probables de cada tema
# ------------------------------------------------------------------
n_top = 5
top5_palabras_por_tema = {}
top5_pesos_por_tema = {}
for k, componente in enumerate(lda.components_):
    idx_top = np.argsort(componente)[::-1][:n_top]
    palabras_k = [str(vocabulario[i]) for i in idx_top]
    pesos_k = [float(componente[i]) for i in idx_top]
    top5_palabras_por_tema[f"tema_{k}"] = palabras_k
    top5_pesos_por_tema[f"tema_{k}"] = pesos_k

# Extra útil: mezcla documento-tema (distribución theta estimada por documento)
doc_topic = lda.transform(X)
distribucion_documentos = {
    ids[i]: [float(v) for v in doc_topic[i]] for i in range(len(ids))
}

# ------------------------------------------------------------------
# 5. Guardar resultados en resultados.json (sin redondear)
# ------------------------------------------------------------------
resultados = {
    "subtarea": "T1 - Parte 1 - LDA con tres temas",
    "n_documentos": int(len(textos)),
    "ids_documentos": ids,
    "etiquetas_reales": etiquetas_reales,
    "n_palabras_vacias": int(len(stop_words)),
    "tamano_vocabulario": tamano_vocabulario,
    "vocabulario": [str(w) for w in vocabulario],
    "n_componentes": 3,
    "learning_method": "batch",
    "max_iter": 50,
    "random_state": 0,
    "top5_palabras_por_tema": top5_palabras_por_tema,
    "top5_pesos_por_tema": top5_pesos_por_tema,
    "perplejidad_k3": perplejidad_k3,
    "distribucion_documentos_temas": distribucion_documentos,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------------
# 6. Impresión de las cifras principales
# ------------------------------------------------------------------
print("=" * 60)
print("T1 - Parte 1: LDA con tres temas (k=3)")
print("=" * 60)
print(f"Documentos: {len(textos)} | Palabras vacías: {len(stop_words)}")
print(f"Tamaño del vocabulario: {tamano_vocabulario}")
print(f"Perplejidad (k=3) sobre el corpus: {perplejidad_k3}")
print("-" * 60)
for k in range(3):
    palabras = top5_palabras_por_tema[f"tema_{k}"]
    pesos = top5_pesos_por_tema[f"tema_{k}"]
    pares = ", ".join(f"{p} ({w:.4f})" for p, w in zip(palabras, pesos))
    print(f"Tema {k}: {pares}")
print("-" * 60)
print("Mezcla documento-tema (theta):")
for i, doc_id in enumerate(ids):
    mezcla = ", ".join(f"{v:.4f}" for v in doc_topic[i])
    print(f"  {doc_id} [{etiquetas_reales[i]:>10}]: [{mezcla}]")
print("=" * 60)
print("Resultados guardados en resultados.json")
