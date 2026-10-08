```python
# -*- coding: utf-8 -*-
"""
T3 - Parte 3 - ¿Recupera LDA los temas reales?

Con la distribución documento-tema del modelo LDA de tres temas de la Parte 1
(entradas/T1.json), esta subtarea:
  1) asigna a cada documento su tema dominante (argmax de la distribución
     documento-tema);
  2) presenta la asignación por documento en una tabla junto con la etiqueta
     real;
  3) calcula la pureza: para cada tema de LDA cuenta los documentos de la
     etiqueta real más frecuente entre los asignados a él, suma esas cuentas
     y divide por 12;
  4) identifica los documentos mal agrupados (etiqueta real distinta de la
     etiqueta real mayoritaria del tema LDA al que fueron asignados).

Salidas:
  * resultados.json con las claves 'asignacion_por_documento', 'pureza' y
    'documentos_mal_agrupados' (más cifras de apoyo).
  * Impresión por consola de la tabla y de las cifras principales.
Esta subtarea NO requiere ninguna figura PNG.
"""

import json
import os
from collections import Counter

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1) Corpus, etiquetas reales y palabras vacías (exactamente los del enunciado)
# ---------------------------------------------------------------------------
IDS = ["d%02d" % i for i in range(1, 13)]

TEXTOS = [
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

ETIQUETAS_ENUNCIADO = ["deporte"] * 4 + ["cocina"] * 4 + ["astronomía"] * 4

STOP_WORDS = [
    "el", "la", "los", "las", "de", "del", "en", "y", "a", "un", "una", "por",
    "para", "con", "se", "su", "al", "es", "después", "durante", "contra",
    "donde", "alrededor",
]

N_DOCS = len(IDS)   # 12
N_TEMAS = 3


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------
def a_tipos_python(obj):
    """Convierte tipos de NumPy a tipos nativos de Python (para json.dump)."""
    if isinstance(obj, dict):
        return {str(k): a_tipos_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [a_tipos_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return a_tipos_python(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def cargar_t1():
    """Lee entradas/T1.json (resultados de la Parte 1) si está disponible."""
    ruta = os.path.join("entradas", "T1.json")
    if os.path.exists(ruta):
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                datos = json.load(f)
            print("[info] Resultados de la Parte 1 leídos de %s" % ruta)
            return datos
        except Exception as exc:
            print("[aviso] No se pudo leer %s: %s" % (ruta, exc))
    else:
        print("[aviso] No existe %s; se reajustará el modelo de la Parte 1." % ruta)
    return None


def normalizar_filas(dist):
    """Normaliza cada fila de la distribución documento-tema (defensivo)."""
    dist = np.asarray(dist, dtype=float)
    sumas = dist.sum(axis=1, keepdims=True)
    if np.all(sumas > 0):
        dist = dist / sumas
    return dist


def obtener_distribucion_doc_tema(t1):
    """Devuelve (dist 12x3 de p(tema|doc), descripción de la fuente)."""
    if t1 is not None and "distribucion_documento_tema" in t1:
        bruto = t1["distribucion_documento_tema"]
        dist = None
        if isinstance(bruto, dict):
            try:
                dist = np.asarray([bruto[i] for i in IDS], dtype=float)
            except Exception:
                try:
                    dist = np.asarray([bruto[j] for j in range(N_DOCS)], dtype=float)
                except Exception:
                    dist = None
        else:
            try:
                dist = np.asarray(bruto, dtype=float)
            except Exception:
                dist = None
        if dist is not None and dist.ndim == 2:
            if dist.shape == (N_TEMAS, N_DOCS) and N_TEMAS != N_DOCS:
                dist = dist.T
            if dist.shape == (N_DOCS, N_TEMAS):
                return normalizar_filas(dist), ("entradas/T1.json "
                                                "('distribucion_documento_tema' de la Parte 1)")
            print("[aviso] Forma inesperada de 'distribucion_documento_tema': %s"
                  % (dist.shape,))
    # Plan B: reajuste determinista del modelo exacto de la Parte 1
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    print("[info] Reajuste del modelo de la Parte 1: CountVectorizer(stop_words=lista del "
          "enunciado) + LatentDirichletAllocation(n_components=3, learning_method='batch', "
          "max_iter=50, random_state=0).")
    vectorizer = CountVectorizer(stop_words=STOP_WORDS)
    X = vectorizer.fit_transform(TEXTOS)
    lda = LatentDirichletAllocation(n_components=N_TEMAS, learning_method="batch",
                                    max_iter=50, random_state=0)
    lda.fit(X)
    dist = normalizar_filas(lda.transform(X))
    return dist, "reajuste local del modelo de la Parte 1 (random_state=0)"


def obtener_etiquetas_reales(t1):
    """Etiqueta real por documento (enunciado como referencia; T1.json si coincide)."""
    etiquetas = list(ETIQUETAS_ENUNCIADO)
    origen = "enunciado (columna 'tema' del corpus)"
    if t1 is not None and "temas_reales" in t1:
        tr = t1["temas_reales"]
        candidatas = None
        if isinstance(tr, dict):
            try:
                candidatas = [str(tr[i]) for i in IDS]
            except Exception:
                candidatas = None
        elif isinstance(tr, (list, tuple)) and len(tr) == N_DOCS:
            candidatas = [str(x) for x in tr]
        if candidatas is not None:
            if candidatas == etiquetas:
                origen = "entradas/T1.json ('temas_reales'; coincide con el enunciado)"
            else:
                print("[aviso] 'temas_reales' de T1.json difiere del enunciado; "
                      "se usa la etiqueta real del enunciado.")
    return etiquetas, origen


def obtener_palabras_top_por_tema(t1):
    """Palabras top de cada tema según T1.json (para interpretar los temas)."""
    if t1 is not None and "palabras_top_por_tema" in t1:
        pt = t1["palabras_top_por_tema"]
        try:
            if isinstance(pt, (list, tuple)) and len(pt) == N_TEMAS:
                return
