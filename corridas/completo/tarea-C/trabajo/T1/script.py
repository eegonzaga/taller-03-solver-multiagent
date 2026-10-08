# -*- coding: utf-8 -*-
"""
T1 — Parte 1 — LDA con tres temas sobre el corpus mínimo (Blei, Ng y Jordan, 2003).

Corpus de doce oraciones (d01–d12) con etiquetas reales deporte/cocina/astronomía,
tal como aparece en el enunciado (S0). No se descarga ningún corpus externo.

Pasos:
  1) Construir el corpus (12 documentos, 3 temas).
  2) Vectorizar con CountVectorizer(stop_words=[...]) y resto de parámetros por defecto.
  3) Ajustar LatentDirichletAllocation(n_components=3, learning_method='batch',
     max_iter=50, random_state=0).
  4) Reportar: tamaño del vocabulario, cinco palabras más probables por tema y
     perplejidad del modelo sobre el corpus (como pide el enunciado, S1).
  5) Guardar todas las cifras en resultados.json e imprimirlas.
"""

import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ----------------------------------------------------------------------
# 1. Corpus mínimo del enunciado (S0): doce oraciones con etiqueta real
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
# 2. Vectorización: CountVectorizer con la lista de palabras vacías dada
#    (el resto de parámetros por defecto)
# ----------------------------------------------------------------------
stop_words = ['el', 'la', 'los', 'las', 'de', 'del', 'en', 'y', 'a', 'un', 'una',
              'por', 'para', 'con', 'se', 'su', 'al', 'es', 'después', 'durante',
              'contra', 'donde', 'alrededor']

vectorizador = CountVectorizer(stop_words=stop_words)
X = vectorizador.fit_transform(textos)  # matriz documentos x términos (12 x V)

vocabulario = vectorizador.get_feature_names_out()
tamano_vocabulario = int(len(vocabulario))

# ----------------------------------------------------------------------
# 3. LDA con tres temas (hiperparámetros exactos del enunciado)
# ----------------------------------------------------------------------
lda = LatentDirichletAllocation(n_components=3,
                                learning_method='batch',
                                max_iter=50,
                                random_state=0)
lda.fit(X)

# Perplejidad del modelo sobre el corpus, tal como lo pide el enunciado (S1).
# (LDA es no supervisado; el enunciado pide explícitamente la perplejidad sobre
#  este corpus de doce oraciones.)
perplejidad_3_temas = float(lda.perplexity(X))

# ----------------------------------------------------------------------
# 4. Cinco palabras más probables de cada tema
#    (lda.components_ guarda los parámetros de Dirichlet de cada tema, forma 3 x V)
# ----------------------------------------------------------------------
n_top = 5
componentes = np.asarray(lda.components_)
orden_desc = np.argsort(-componentes, axis=1)  # índices de palabras, mayor a menor

palabras_top_por_tema = {}
for k in range(componentes.shape[0]):
    top_idx = orden_desc[k, :n_top]
    palabras_top_por_tema["tema_%d" % (k + 1)] = [str(vocabulario[i]) for i in top_idx]

# Distribución documento-tema (mezcla de temas por documento; útil para partes posteriores)
doc_topic = lda.transform(X)  # forma (12, 3), cada fila suma 1

# ----------------------------------------------------------------------
# 5. Guardar TODAS las cifras en resultados.json (sin redondear)
# ----------------------------------------------------------------------
resultados = {
    "tamano_vocabulario": tamano_vocabulario,
    "palabras_top_por_tema": palabras_top_por_tema,
    "perplejidad_3_temas": perplejidad_3_temas,
    # Información adicional de trazabilidad / para subtareas posteriores
    "vocabulario": [str(v) for v in vocabulario],
    "ids_documentos": ids,
    "etiquetas_reales": etiquetas,
    "distribucion_documento_tema": {
        ids[i]: [float(x) for x in doc_topic[i]] for i in range(len(ids))
    },
    "configuracion": {
        "n_components": int(lda.n_components),
        "learning_method": "batch",
        "max_iter": int(lda.max_iter),
        "random_state": int(lda.random_state),
        "n_iter_ejecutadas": int(getattr(lda, "n_iter_", -1)),
        "n_documentos": int(X.shape[0]),
        "n_tokens_corpus": int(X.sum()),
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 6. Impresión de las cifras principales
# ----------------------------------------------------------------------
print("=== T1 Parte 1: LDA con 3 temas sobre el corpus mínimo ===")
print("Tamaño del vocabulario:", tamano_vocabulario)
print("Vocabulario:", ", ".join(vocabulario))
print()
for tema in sorted(palabras_top_por_tema.keys()):
    print("Cinco palabras más probables de %s: %s"
          % (tema, ", ".join(palabras_top_por_tema[tema])))
print()
print("Perplejidad del modelo (3 temas) sobre el corpus: %.10f" % perplejidad_3_temas)
print()
print("Mezcla de temas por documento (filas suman 1):")
for i, did in enumerate(ids):
    print("  %s (%s): %s" % (did, etiquetas[i],
                             np.round(doc_topic[i], 4).tolist()))
print()
print("Resultados guardados en resultados.json")
