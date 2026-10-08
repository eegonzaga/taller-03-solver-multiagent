# -*- coding: utf-8 -*-
"""
T2 - Parte 2: Perplejidad de LDA con 2, 3 y 4 temas.

Reproduce exactamente el CountVectorizer de la Parte 1 (la lista de palabras
vacías del enunciado pasada tal cual, resto de parámetros por defecto) y los
mismos hiperparámetros de LDA (learning_method='batch', max_iter=50,
random_state=0). Ajusta LDA con n_components = 2, 3 y 4 y reporta la
perplejidad de cada modelo sobre el corpus (mismo protocolo que la Parte 1)
en una sola tabla comparativa.

Salidas:
  - resultados.json : perplejidad_2_temas, perplejidad_3_temas,
                      perplejidad_4_temas, tabla_comparativa, etc.
  - impresión de la tabla comparativa en consola.
  - (sin figura PNG: la subtarea indica que no se requiere)
"""

import json

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# ----------------------------------------------------------------------
# 1. Corpus del enunciado: 12 oraciones, 3 temas reales (4 por tema)
# ----------------------------------------------------------------------
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

temas_reales = ["deporte"] * 4 + ["cocina"] * 4 + ["astronomía"] * 4

# Palabras vacías del enunciado, pasadas tal cual a CountVectorizer
stop_words = [
    "el", "la", "los", "las", "de", "del", "en", "y", "a", "un", "una",
    "por", "para", "con", "se", "su", "al", "es", "después", "durante",
    "contra", "donde", "alrededor",
]

# ----------------------------------------------------------------------
# 2. Vectorizador idéntico al de la Parte 1 (mismas stop words, resto por defecto)
# ----------------------------------------------------------------------
vectorizador = CountVectorizer(stop_words=stop_words)
X = vectorizador.fit_transform(documentos)
vocabulario = vectorizador.get_feature_names_out().tolist()
n_tokens = int(X.sum())

# ----------------------------------------------------------------------
# 3. Resultados de la Parte 1 (entradas/T1.json) para verificación cruzada
# ----------------------------------------------------------------------
t1 = None
try:
    with open("entradas/T1.json", "r", encoding="utf-8") as f:
        t1 = json.load(f)
except (OSError, ValueError):
    t1 = None

# ----------------------------------------------------------------------
# 4. LDA con 2, 3 y 4 temas (mismos hiperparámetros de la Parte 1).
#    La perplejidad se reporta sobre el corpus, igual que en la Parte 1.
# ----------------------------------------------------------------------
perplejidades = {}
n_iter_por_k = {}
for k in (2, 3, 4):
    lda = LatentDirichletAllocation(
        n_components=k,
        learning_method="batch",
        max_iter=50,
        random_state=0,
    )
    lda.fit(X)
    perplejidades[k] = float(lda.perplexity(X))
    n_iter_por_k[k] = int(lda.n_iter_)

# Una sola tabla comparativa con los tres modelos
tabla_comparativa = [
    {"n_temas": k, "perplejidad": perplejidades[k], "n_iter": n_iter_por_k[k]}
    for k in (2, 3, 4)
]

mejor_k = int(min(perplejidades, key=perplejidades.get))

# ----------------------------------------------------------------------
# 5. Diccionario de resultados (tipos nativos de Python, sin redondear)
# ----------------------------------------------------------------------
resultados = {
    "perplejidad_2_temas": perplejidades[2],
    "perplejidad_3_temas": perplejidades[3],
    "perplejidad_4_temas": perplejidades[4],
    "tabla_comparativa": tabla_comparativa,
    "n_temas_con_menor_perplejidad": mejor_k,
    "perplejidad_minima": perplejidades[mejor_k],
    "tamano_vocabulario": len(vocabulario),
    "n_documentos": len(documentos),
    "n_tokens_corpus": n_tokens,
    "temas_reales": temas_reales,
}

if t1 is not None:
    if "perplejidad_3_temas" in t1:
        p1 = float(t1["perplejidad_3_temas"])
        resultados["perplejidad_3_temas_parte1"] = p1
        resultados["diferencia_perplejidad_3_vs_parte1"] = float(perplejidades[3] - p1)
    if "vocabulario" in t1:
        resultados["vocabulario_igual_al_de_parte1"] = bool(
            list(t1["vocabulario"]) == vocabulario
        )

# ----------------------------------------------------------------------
# 6. Guardar resultados.json
# ----------------------------------------------------------------------
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 7. Impresión de la tabla comparativa y de las cifras principales
# ----------------------------------------------------------------------
print("Parte 2 - Perplejidad de LDA según el número de temas")
print(f"Documentos: {len(documentos)} | Vocabulario: {len(vocabulario)} palabras "
      f"| Tokens totales: {n_tokens}")
print()
print("Tabla comparativa de perplejidad")
print("-" * 48)
print(f"{'n_temas':>8} | {'perplejidad':>16} | {'n_iter':>6}")
print("-" * 48)
for fila in tabla_comparativa:
    print(f"{fila['n_temas']:>8} | {fila['perplejidad']:>16.6f} | {fila['n_iter']:>6}")
print("-" * 48)
print(f"Modelo con menor perplejidad: {mejor_k} temas "
      f"(perplejidad = {perplejidades[mejor_k]!r})")
print()
print("Cifras exactas (precisión completa):")
for k in (2, 3, 4):
    print(f"  perplejidad_{k}_temas = {perplejidades[k]!r}")

if t1 is not None and "perplejidad_3_temas" in t1:
    print()
    print("Verificación con la Parte 1:")
    print(f"  perplejidad_3_temas (T1) = {float(t1['perplejidad_3_temas'])!r}")
    print(f"  diferencia (este script - T1) = "
          f"{resultados['diferencia_perplejidad_3_vs_parte1']!r}")
if t1 is not None and "vocabulario" in t1:
    print(f"  vocabulario idéntico al de la Parte 1: "
          f"{resultados['vocabulario_igual_al_de_parte1']}")

print()
print("Resultados guardados en resultados.json")
