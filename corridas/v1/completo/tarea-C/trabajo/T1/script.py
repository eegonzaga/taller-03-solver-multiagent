# -*- coding: utf-8 -*-
"""
T1 — Parte 1 — LDA con tres temas (corpus mínimo de doce oraciones, Blei, Ng y Jordan 2003).

Construye el corpus d01–d12 dado en el enunciado, lo vectoriza con CountVectorizer
(usando la lista de palabras vacías proporcionada y el resto de parámetros por defecto),
ajusta LatentDirichletAllocation(n_components=3, learning_method='batch', max_iter=50,
random_state=0) y reporta:
  - tamaño del vocabulario,
  - las cinco palabras más probables de cada tema,
  - la perplejidad del modelo sobre el corpus.

Guarda todas las cifras en resultados.json y las imprime. No produce figura PNG.
"""

import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ----------------------------------------------------------------------
# 1. Corpus: doce oraciones (d01–d12) tal como aparecen en el enunciado
# ----------------------------------------------------------------------
corpus = {
    "d01": "El delantero marcó dos goles en el partido de fútbol y el equipo ganó la liga.",
    "d02": "El entrenador del equipo preparó la defensa para el partido final de la liga.",
    "d03": "La tenista ganó el torneo después de un partido largo contra la campeona.",
    "d04": "El equipo de baloncesto perdió el partido por un punto en el último segundo.",
    "d05": "La receta lleva harina, huevos y azúcar; la masa se hornea durante treinta minutos.",
    "d06": "Para la salsa se sofríe cebolla y ajo en aceite y se añade tomate.",
    "d07": "El pan se hornea con harina, agua, sal y levadura después de amasar la masa.",
    "d08": "La sopa de verduras lleva cebolla, zanahoria, ajo y sal, y se cocina a fuego lento.",
    "d09": "El telescopio observó una galaxia lejana y varias estrellas de la nebulosa.",
    "d10": "Los planetas giran alrededor de la estrella; la órbita de la Tierra dura un año.",
    "d11": "La nebulosa es una nube de gas donde nacen estrellas nuevas en la galaxia.",
    "d12": "El telescopio espacial fotografió planetas y la órbita de una luna de Júpiter.",
}

ids_docs = list(corpus.keys())
textos = [corpus[i] for i in ids_docs]
n_documentos = len(textos)

# ----------------------------------------------------------------------
# 2. Vectorización: CountVectorizer con la lista de palabras vacías dada
#    (el resto de parámetros por defecto)
# ----------------------------------------------------------------------
stop_words = [
    "el", "la", "los", "las", "de", "del", "en", "y", "a", "un", "una",
    "por", "para", "con", "se", "su", "al", "es", "después", "durante",
    "contra", "donde", "alrededor",
]

vectorizador = CountVectorizer(stop_words=stop_words)
X = vectorizador.fit_transform(textos)

vocabulario = vectorizador.get_feature_names_out()
tamano_vocabulario = int(vocabulario.shape[0])

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

# Perplejidad del modelo sobre el corpus (tal como pide el enunciado)
perplejidad_k3 = float(lda.perplexity(X))

# ----------------------------------------------------------------------
# 4. Cinco palabras más probables de cada tema
#    components_ contiene los conteos (no normalizados) palabra-tema;
#    se normaliza por fila para obtener p(palabra | tema).
# ----------------------------------------------------------------------
componentes = lda.components_  # forma (3, V)
p_palabra_dado_tema = componentes / componentes.sum(axis=1, keepdims=True)

n_top = 5
palabras_top_por_tema = {}
prob_palabras_top_por_tema = {}
for k in range(componentes.shape[0]):
    idx_top = np.argsort(componentes[k])[::-1][:n_top]
    clave = "tema_%d" % (k + 1)
    palabras_top_por_tema[clave] = [str(vocabulario[i]) for i in idx_top]
    prob_palabras_top_por_tema[clave] = [float(p_palabra_dado_tema[k, i]) for i in idx_top]

# Mezcla de temas por documento (LDA no ve las etiquetas reales; solo informativo)
doc_topic = lda.transform(X)
distribucion_documentos_temas = {
    ids_docs[i]: [float(p) for p in doc_topic[i]] for i in range(n_documentos)
}

# ----------------------------------------------------------------------
# 5. Guardar TODAS las cifras en resultados.json (tipos nativos de Python)
# ----------------------------------------------------------------------
resultados = {
    "n_documentos": int(n_documentos),
    "tamano_vocabulario": tamano_vocabulario,
    "vocabulario": [str(w) for w in vocabulario],
    "palabras_top_por_tema": palabras_top_por_tema,
    "prob_palabras_top_por_tema": prob_palabras_top_por_tema,
    "perplejidad_k3": perplejidad_k3,
    "distribucion_documentos_temas": distribucion_documentos_temas,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 6. Impresión de las cifras principales
# ----------------------------------------------------------------------
print("Número de documentos:", n_documentos)
print("Tamaño del vocabulario:", tamano_vocabulario)
print("Vocabulario:", ", ".join(str(w) for w in vocabulario))
print()
for clave in sorted(palabras_top_por_tema.keys()):
    palabras = palabras_top_por_tema[clave]
    probs = prob_palabras_top_por_tema[clave]
    pares = ["%s (%.6f)" % (p, q) for p, q in zip(palabras, probs)]
    print("%s: %s" % (clave, ", ".join(pares)))
print()
print("Perplejidad del modelo (k=3) sobre el corpus:", perplejidad_k3)
print()
print("Mezcla de temas por documento (tema_1, tema_2, tema_3):")
for doc_id in ids_docs:
    mezcla = distribucion_documentos_temas[doc_id]
    print("  %s: [%s]" % (doc_id, ", ".join("%.4f" % p for p in mezcla)))
print()
print("Resultados guardados en resultados.json")
