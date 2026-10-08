# -*- coding: utf-8 -*-
"""
T1 — Parte 1 — LDA con tres temas
Corpus de 12 oraciones del enunciado (d01–d12; temas reales: deporte, cocina, astronomía).
Vectorización con CountVectorizer (lista de palabras vacías dada, resto por defecto) y
LatentDirichletAllocation(n_components=3, learning_method='batch', max_iter=50, random_state=0).
Reporta: tamaño del vocabulario, cinco palabras más probables por tema y perplejidad
del modelo sobre el corpus. Guarda todo en resultados.json. No produce figura PNG.
"""

import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ----------------------------------------------------------------------
# 1. Corpus del enunciado (la etiqueta real NO se usa para ajustar LDA)
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
temas_reales = ["deporte"] * 4 + ["cocina"] * 4 + ["astronomía"] * 4  # solo contexto

# ----------------------------------------------------------------------
# 2. Vectorización: CountVectorizer con la lista de palabras vacías tal cual
# ----------------------------------------------------------------------
stop_words = ["el", "la", "los", "las", "de", "del", "en", "y", "a", "un", "una",
              "por", "para", "con", "se", "su", "al", "es", "después", "durante",
              "contra", "donde", "alrededor"]

vectorizador = CountVectorizer(stop_words=stop_words)  # resto de parámetros por defecto
X = vectorizador.fit_transform(textos)

vocabulario = vectorizador.get_feature_names_out()
tamano_vocabulario = int(len(vocabulario))

# ----------------------------------------------------------------------
# 3. LDA con 3 temas (parámetros exactos del enunciado)
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
# ----------------------------------------------------------------------
n_top = 5
palabras_top_por_tema = {}        # solo las palabras
palabras_top_con_pesos = {}       # palabras con su probabilidad dentro del tema
for k, componente in enumerate(lda.components_):
    probs = componente / componente.sum()  # normaliza el tema a una distribución
    idx_top = np.argsort(componente)[::-1][:n_top]
    palabras = [str(vocabulario[i]) for i in idx_top]
    pesos = [float(probs[i]) for i in idx_top]
    palabras_top_por_tema[f"tema_{k}"] = palabras
    palabras_top_con_pesos[f"tema_{k}"] = [
        {"palabra": p, "probabilidad": w} for p, w in zip(palabras, pesos)
    ]

# ----------------------------------------------------------------------
# 5. Impresión de las cifras principales
# ----------------------------------------------------------------------
print("=" * 60)
print("T1 - Parte 1: LDA con tres temas (corpus de 12 oraciones)")
print("=" * 60)
print(f"Documentos: {len(textos)}")
print(f"Tamaño del vocabulario: {tamano_vocabulario}")
print(f"Perplejidad (k=3) sobre el corpus: {perplejidad_k3:.6f}")
print("-" * 60)
for k in sorted(palabras_top_con_pesos.keys()):
    linea = ", ".join(
        f"{d['palabra']} ({d['probabilidad']:.4f})"
        for d in palabras_top_con_pesos[k]
    )
    print(f"{k}: {linea}")
print("-" * 60)

# ----------------------------------------------------------------------
# 6. Guardar todas las cifras en resultados.json (tipos nativos de Python)
# ----------------------------------------------------------------------
resultados = {
    "tamano_vocabulario": tamano_vocabulario,
    "palabras_top_por_tema": palabras_top_por_tema,
    "palabras_top_por_tema_con_pesos": palabras_top_con_pesos,
    "perplejidad_k3": perplejidad_k3,
    "vocabulario": [str(v) for v in vocabulario],
    "n_documentos": int(len(textos)),
    "n_componentes": 3,
    "learning_method": "batch",
    "max_iter": 50,
    "random_state": 0,
    "n_iteracion_final": int(lda.n_iter_),
    "perplejidad_k3_detalle": {
        "definicion": "perplejidad de sklearn sobre la matriz de conteos del corpus",
        "valor": perplejidad_k3,
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("Resultados guardados en resultados.json")
