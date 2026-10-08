```python
# -*- coding: utf-8 -*-
"""
T3 — Parte 2 — Mini-IVF: recall@10 vs. speedup según nprobe
============================================================

Ejercicio 2.1  knn_ivf(consulta, nprobe, k):
    (1) scores contra los centroides -> nprobe celdas más cercanas
    (2) kNN exacto solo dentro de esas celdas, devolviendo índices GLOBALES

Ejercicio 2.2  para nprobe in {1, 2, 4, 8, 16}, sobre 50 consultas:
    - recall@10 contra el kNN exacto (referencia de la Parte 1 / T2)
    - speedup = tiempo_exacto / tiempo_IVF
    y reporte del tamaño medio de celda del IVF entrenado.

Setup respetado del código dado: N=100_000, base normalizada, NLIST=64,
KMeans(n_clusters=64, n_init=3, random_state=7), centroides normalizados,
celdas construidas con km.labels_.

Salidas: resultados.json y T3_parte2_recall_vs_speedup_nprobe.png
"""

import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ------------------------------------------------------------------
# 0. Setup dado por el enunciado (código del profesor, sin cambios)
# ------------------------------------------------------------------
rng = np.random.default_rng(7)
D = 128  # dimensión de los embeddings sintéticos
_CENTROS = np.random.default_rng(2026).normal(size=(200, D))  # "temas" fijos


def datos_realistas(n, d=D):
    """Embeddings sintéticos CON estructura: mezcla de 'temas' gaussianos fijos."""
    asignacion = rng.integers(0, len(_CENTROS), size=n)
    X = _CENTROS[asignacion, :d] + 1.5 * rng.normal(size=(n, d))
    return X.astype(np.float32)


def normalizar(X):
    """Vectores a norma 1: así el producto punto es el coseno (sesión 06)."""
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


N = 100_000
base = normalizar(datos_realistas(N))
NLIST = 64
km = KMeans(n_clusters=NLIST, n_init=3, random_state=7).fit(base)
centroides = normalizar(km.cluster_centers_.astype(np.float32))
celdas = [np.where(km.labels_ == c)[0] for c in range(NLIST)]
tamanos_celdas = np.array([len(c) for c in celdas], dtype=np.int64)
tamano_medio_celda = float(np.mean([len(c) for c in celdas]))
print(f"IVF entrenado: {NLIST} celdas, tamaño medio {tamano_medio_celda:.0f}")

# ------------------------------------------------------------------
# 1. Consultas: 50 vectores nuevos que comparten los temas de la base.
#    Continúan el mismo generador rng(seed=7) tras generar la base
#    (mismo patrón del notebook: base y consultas comparten temas).
# ------------------------------------------------------------------
K = 10
N_CONSULTAS = 50
consultas = normalizar(datos_realistas(N_CONSULTAS))

# ------------------------------------------------------------------
# 2. kNN exacto (Ejercicio 1.1 / T2): referencia de vecinos verdaderos
# ------------------------------------------------------------------
def knn_exacto(consulta, base_mat, k=K):
    """kNN exacto por coseno: vectores normalizados -> usar dot."""
    scores = base_mat @ consulta
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


# ------------------------------------------------------------------
# 3. Ejercicio 2.1 — knn_ivf(consulta, nprobe, k)
# ------------------------------------------------------------------
def knn_ivf(consulta, nprobe, k=K):
    """Búsqueda IVF: (1) nprobe celdas más cercanas por sus centroides,
    (2) kNN exacto solo dentro de esas celdas -> índices GLOBALES."""
    nprobe = int(min(nprobe, NLIST))
    # (1) scores contra los centroides -> nprobe celdas más cercanas
    scores_centroides = centroides @ consulta
    celdas_top = np.argpartition(-scores_centroides, nprobe - 1)[:nprobe]
    # (2) kNN exacto solo dentro de esas celdas (¡índices GLOBALES!)
    idx_globales = np.concatenate([celdas[c] for c in celdas_top])
    scores =
