# -*- coding: utf-8 -*-
"""
T1 — Parte 1 — LDA con tres temas
Corpus mínimo de doce oraciones (Blei, Ng y Jordan, 2003) dado en el enunciado.
Vectoriza con CountVectorizer (lista de palabras vacías del enunciado), ajusta
LatentDirichletAllocation(n_components=3, learning_method='batch', max_iter=50,
random_state=0) y reporta: tamaño del vocabulario, cinco palabras más probables
por tema y perplejidad del modelo sobre el corpus (tal como pide el enunciado).
No produce figuras.
"""

import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ----------------------------------------------------------------------
# 1. Corpus del enunciado: doce oraciones con su etiqueta real
#    (la etiqueta NO se usa para ajustar LDA; solo se registra)
# ----------------------------------------------------------------------
corpus = [
    ("d01", "El delantero marcó dos goles en el partido de fútbol y el equipo ganó la liga.", "deporte"),
    ("d02", "El entrenador del equipo preparó la defensa para el partido final de la liga.", "deporte"),
    ("d03", "La tenista ganó el torneo después de un partido largo contra la campeona.", "deporte"),
    ("d04", "El equipo de baloncesto perdió el partido por un punto en el último segundo.", "deporte"),
    ("d05", "La receta lleva harina, huevos y azúcar; la masa se hornea durante treinta minutos.", "cocina"),
    ("d06", "Para la salsa se sofríe cebolla y ajo en aceite y se añade tomate.", "cocina"),
    ("d07", "El pan se hornea con harina, agua, sal y levadura después de amasar la masa.", "cocina"),
    ("d08", "La sopa de verduras lleva cebolla, zanahoria, ajo y sal, y se cocina a fuego lento.", "cocina"),
    ("d09", "El telescopio observó una galaxia lejana y varias estrellas de la nebulosa.", "astronomía"),
    ("d10", "Los planetas giran alrededor de la estrella; la órbita de la Tierra dura un año.", "astronomía"),
    ("d11", "La nebulosa es una nube de gas donde nacen estrellas nuevas en la galaxia.", "astronomía"),
    ("d12", "El telescopio espacial fotografió planetas y la órbita de una luna de Júpiter.", "astronomía"),
]

ids = [c[0] for c in corpus]
textos = [c[1] for c in corpus]
etiquetas = [c[2] for c in corpus]

# ----------------------------------------------------------------------
# 2. Vectorización: CountVectorizer con las palabras vacías del enunciado
#    (resto de parámetros por defecto)
# ----------------------------------------------------------------------
stop_words = [
    "el", "la", "los", "las", "de", "del", "en", "y", "a", "un", "una",
    "por", "para", "con", "se", "su", "al", "es", "después", "durante",
    "contra", "donde", "alrededor",
]

vectorizador = CountVectorizer(stop_words=stop_words)
X = vectorizador.fit_transform(textos)

vocabulario = vectorizador.get_feature_names_out()
tamano_vocabulario = int(X.shape[1])

# ----------------------------------------------------------------------
# 3. LDA con k = 3 (parámetros exactos del enunciado)
# ----------------------------------------------------------------------
lda = LatentDirichletAllocation(
    n_components=3,
    learning_method="batch",
    max_iter=50,
    random_state=0,
)
lda.fit(X)

# Perplejidad del modelo sobre el corpus, tal como pide el enunciado
perplejidad_k3 = float(lda.perplexity(X))

# ----------------------------------------------------------------------
# 4. Cinco palabras más probables de cada tema
# ----------------------------------------------------------------------
componentes = lda.components_  # forma (3, V): conteos no normalizados
palabras_top_por_tema = {}
palabras_top_con_prob = {}
for k in range(componentes.shape[0]):
    dist = componentes[k] / componentes[k].sum()  # distribución p(palabra|tema)
    idx_orden = np.argsort(dist)[::-1][:5]
    palabras_top_por_tema[f"tema_{k+1}"] = [str(vocabulario[i]) for i in idx_orden]
    palabras_top_con_prob[f"tema_{k+1}"] = [
        {"palabra": str(vocabulario[i]), "probabilidad": float(dist[i])}
        for i in idx_orden
    ]

# Distribución documento-tema (información adicional; LDA no ve las etiquetas)
doc_topic = lda.transform(X)
distribucion_documento_tema = {
    ids[i]: {f"tema_{k+1}": float(doc_topic[i, k]) for k in range(3)}
    for i in range(len(ids))
}

# ----------------------------------------------------------------------
# 5. Guardar todas las cifras en resultados.json
# ----------------------------------------------------------------------
resultados = {
    "tamano_vocabulario": tamano_vocabulario,
    "palabras_top_por_tema": palabras_top_por_tema,
    "perplejidad_k3": perplejidad_k3,
    "palabras_top_por_tema_con_probabilidad": palabras_top_con_prob,
    "vocabulario": [str(v) for v in vocabulario],
    "distribucion_documento_tema": distribucion_documento_tema,
    "etiquetas_reales": {ids[i]: etiquetas[i] for i in range(len(ids))},
    "n_documentos": int(len(ids)),
    "n_componentes": 3,
    "learning_method": "batch",
    "max_iter": 50,
    "random_state": 0,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 6. Impresión de las cifras principales
# ----------------------------------------------------------------------
print("=" * 70)
print("T1 - Parte 1: LDA con tres temas (corpus de 12 oraciones)")
print("=" * 70)
print(f"Documentos: {len(ids)} -> {ids}")
print(f"Tamaño del vocabulario: {tamano_vocabulario}")
print(f"Vocabulario: {', '.join(str(v) for v in vocabulario)}")
print("-" * 70)
print("Cinco palabras más probables por tema:")
for tema, palabras in palabras_top_por_tema.items():
    detalle = ", ".join(
        f"{d['palabra']} ({d['probabilidad']:.4f})"
        for d in palabras_top_con_prob[tema]
    )
    print(f"  {tema}: {', '.join(palabras)}")
    print(f"      con p(palabra|tema): {detalle}")
print("-" * 70)
print(f"Perplejidad (k=3) sobre el corpus: {perplejidad_k3}")
print("=" * 70)
print("Resultados guardados en resultados.json")
