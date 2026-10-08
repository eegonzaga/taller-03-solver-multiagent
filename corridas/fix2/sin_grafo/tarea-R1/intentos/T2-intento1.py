```python
# -*- coding: utf-8 -*-
"""
T2 — Parte 2 — Mini-IVF: recall@10 vs. speedup según nprobe.

Pipeline:
  1) Regenera deterministamente la base normalizada de N=100_000 de T1
     (mismo código y semillas del enunciado: rng(7) para datos_realistas,
     generator(2026) para los 200 "temas") y 50 consultas del mismo stream.
  2) Entrena KMeans con NLIST=64 (n_init=3, random_state=7), normaliza los
     centroides y construye las celdas (índices GLOBALES por cluster).
  3) Implementa knn_ivf(consulta, nprobe, k): scores contra los 64 centroides
     -> nprobe celdas más cercanas -> kNN exacto solo dentro de esas celdas.
  4) Para nprobe en {1,2,4,8,16} mide sobre 50 consultas: recall@10 contra el
     ground truth de knn_exacto y speedup = latencia exacta / latencia IVF.
  5) Guarda resultados.json y la figura PNG (triángulo recall/latencia).
"""

import json
import time
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", message=".*memory leak.*")

# ---------------------------------------------------------------------------
# 1. Parámetros (fijados por el enunciado)
# ---------------------------------------------------------------------------
N = 100_000            # tamaño de la base (T1)
D = 128                # dimensión de los embeddings sintéticos
NLIST = 64             # número de celdas del índice IVF
N_INIT = 3             # KMeans(n_init=...)
RANDOM_STATE = 7       # KMeans(random_state=...)
K = 10                 # top-k -> recall@10
N_CONSULTAS = 50       # consultas de evaluación
NPROBES = [1, 2, 4, 8, 16]
SEMILLA_DATOS = 7      # rng del enunciado que alimenta datos_realistas
SEMILLA_TEMAS = 2026   # generator de los 200 centros "tema"

# ---------------------------------------------------------------------------
# 2. Datos: base de T1 (regeneración determinista) + 50 consultas
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEMILLA_DATOS)
_CENTROS = np.random.default_rng(SEMILLA_TEMAS).normal(size=(200, D))


def datos_realistas(n, d=D):
    """Embeddings sintéticos CON estructura: mezcla de 'temas' gaussianos fijos.
    Base y consultas comparten los mismos temas, como en un corpus real."""
    asignacion = rng.integers(0, len(_CENTROS), size=n)
    X = _CENTROS[asignacion, :d] + 1.5 * rng.normal(size=(n, d))
    return X.astype(np.float32)


def normalizar(X):
    """Vectores a norma 1: así el producto punto es el coseno."""
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


base = normalizar(datos_realistas(N))                 # idéntica a la base de T1
consultas = normalizar(datos_realistas(N_CONSULTAS))  # 50 consultas (mismos temas)
print(f"Base: {base.shape} ({base.dtype}) | Consultas: {consultas.shape}")

# ---------------------------------------------------------------------------
# 3. Contexto de T1 (entradas/T1.json), lectura defensiva
# ---------------------------------------------------------------------------
contexto_t1 = {}
try:
    with open("entradas/T1.json", "r", encoding="utf-8") as f:
        t1 = json.load(f)
    contexto_t1["subtarea_T1"] = t1.get("subtarea")
    contexto_t1["parametros_T1"] = t1.get("parametros")
    lat_t1 = t1.get("latencia_media_ms_por_N")
    if isinstance(lat_t1, dict):
        for clave in ("100000", "100_000", 100000):
            if clave in lat_t1:
                try:
                    contexto_t1["latencia_exacta_T1_ms_N100000"] = float(lat_t1[clave])
                except (TypeError, ValueError):
                    pass
                break
    print(f"Contexto T1 cargado de entradas/T1.json: {contexto_t1.get('subtarea_T1', '')}")
except Exception:
    contexto_t1["nota"] = ("entradas/T1.json no disponible; la base se regeneró "
                           "con las semillas del enunciado (idéntica a la de T1).")

# ---------------------------------------------------------------------------
# 4. kNN exacto (baseline y ground truth) — misma definición de T1
# ---------------------------------------------------------------------------
def knn_exacto(consulta, k, base=base):
    """Top-k por coseno (vectores normalizados): productos punto + argpartition."""
    scores = base @ consulta
    if k < scores.shape[0]:
        idx = np.argpartition(-scores, k)[:k]
    else:
        idx = np.arange(scores.shape[0])
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]


# ---------------------------------------------------------------------------
# 5. Entrenamiento offline del índice IVF (KMeans + celdas)
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
km = KMeans(n_clusters=NLIST, n_init=N_INIT, random_state=RANDOM_STATE).fit(base)
t_entrenamiento_s = time.perf_counter() - t0

centroides = normalizar(km.cluster_centers_.astype(np.float32))  #
