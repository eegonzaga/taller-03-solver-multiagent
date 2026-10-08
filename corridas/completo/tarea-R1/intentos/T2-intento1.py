```python
# -*- coding: utf-8 -*-
"""
T2 — Parte 2: Mini-IVF con KMeans, recall@10 vs. speedup según nprobe.

Receta del enunciado:
  (1) offline: KMeans agrupa la base normalizada en NLIST celdas;
  (2) en consulta: scores contra los centroides -> nprobe celdas más
      cercanas -> kNN exacto dentro de esas celdas (índices globales).

Se mide, sobre 50 consultas y para nprobe en {1, 2, 4, 8, 16}:
  - recall@10 contra el kNN exacto (Parte 1 / T1),
  - speedup de latencia vs. el kNN exacto,
y se reportan la tabla, la gráfica recall vs. speedup (PNG) y el tamaño
medio de celda. Todas las cifras se guardan en resultados.json.
"""

import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 0. Datos sintéticos con estructura (código dado en el enunciado)
# ----------------------------------------------------------------------
rng = np.random.default_rng(7)
D = 128  # dimensión de los embeddings sintéticos
_CENTROS = np.random.default_rng(2026).normal(size=(200, D))  # "temas" fijos


def datos_realistas(n, d=D):
    """Embeddings sintéticos CON estructura: mezcla de 'temas' gaussianos fijos.
    Base y consultas comparten los mismos temas, como en un corpus real."""
    asignacion = rng.integers(0, len(_CENTROS), size=n)
    X = _CENTROS[asignacion, :d] + 1.5 * rng.normal(size=(n, d))
    return X.astype(np.float32)


def normalizar(X):
    """Vectores a norma 1: así el producto punto es el coseno."""
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


# ----------------------------------------------------------------------
# 1. Parámetros (exactamente los del enunciado)
# ----------------------------------------------------------------------
N = 100_000            # vectores de la base
NLIST = 64             # celdas del IVF
K = 10                 # vecinos por consulta (recall@10)
N_CONSULTAS = 50       # consultas de evaluación
NPROBES = [1, 2, 4, 8, 16]
N_INIT_KMEANS = 3
RANDOM_STATE_KMEANS = 7
REPETICIONES = 5       # llamadas por consulta al cronometrar (estabiliza la media)

# ----------------------------------------------------------------------
# 2. Base normalizada + consultas (misma rng: comparten los mismos temas)
# ----------------------------------------------------------------------
base = normalizar(datos_realistas(N))                 # (100000, 128) float32
consultas = normalizar(datos_realistas(N_CONSULTAS))  # (50, 128) float32
print(f"Base: {base.shape[0]} vectores, D={base.shape[1]}, "
      f"norma media={np.linalg.norm(base, axis=1).mean():.6f}")
print(f"Consultas de evaluación: {consultas.shape[0]}")

# ----------------------------------------------------------------------
# 3. Entrenamiento offline del IVF: KMeans + centroides normalizados + celdas
# ----------------------------------------------------------------------
t0 = time.perf_counter()
km = KMeans(n_clusters=NLIST, n_init=N_INIT_KMEANS,
            random_state=RANDOM_STATE_KMEANS).fit(base)
t_kmeans_s = time.perf_counter() - t0

centroides = normalizar(km.cluster_centers_.astype(np.float32))  # (NLIST, D)
celdas = [np.where(km.labels_ == c)[0] for c in range(NLIST)]    # índices globales
tamanos_celda = np.array([len(c) for c in celdas], dtype=np.int64)
tamano_medio_celda = float(tamanos_celda.mean())

print(f"IVF entrenado: {NLIST} celdas, tamaño medio {tamano_medio_celda:.0f} "
      f"(min={tamanos_celda.min()}, max={tamanos_celda.max()}, "
      f"std={tamanos_celda.std():.1f}) — KMeans tardó {t_kmeans_s:.1f} s "
      f"(inercia={km.inertia_:.2f}, iteraciones={int(km.n_iter_)})")

# ----------------------------------------------------------------------
# 4. kNN exacto (T1) y knn_ivf (ejercicio 2.1)
# ----------------------------------------------------------------------
def knn_exacto(consulta, base_mat, k=K):
    """kNN exacto por producto punto (vectores normalizados -> coseno).
    Devuelve los índices globales de los k mejores, ordenados por score desc."""
    scores = base_mat @ consulta
    if k < scores.shape[0]:
        idx = np.argpartition(scores, -k)[-k:]
    else:
        idx = np.arange(scores.shape[0])
    return idx[np.argsort(scores[idx])[::-1]]


def knn_ivf(consulta, nprobe, k=K):
    """IVF: 1) scores contra los NLIST centroides -> nprobe celdas más cercanas;
    2) kNN exacto solo dentro de esas celdas, con índices GLOBALES."""
    scores_centroides = centroides @ consulta              # (NLIST,) — barato
    if nprobe < NLIST:
        elegidas = np.argpartition(scores_centroides, -nprobe)[-nprobe:]
    else:
        elegidas = np.arange(NLIST)                        # nprobe = nlist => exacto
    candidatos = np.concatenate([celdas[c] for c in elegidas])
    scores = base[candidatos] @ consulta                   # exacto dentro de las celdas
    if k < scores.shape[0]:
        idx = np.argpartition(scores, -k)[-k:]
    else:
        idx = np.arange(scores.shape[0])
    return candidatos[idx[np.argsort(scores[idx])[::-1]]]


# --- verificaciones de correctitud (fuera del cronómetro) ---
q0 = consultas[0]
top_argsort = np.argsort(base @ q0)[::-1][:K]
verif_topk = bool(np.array_equal(np.sort(knn_exacto(q0, base, K)),
                                 np.sort(top_argsort)))
verif_ivf_nlist = all(
    set(knn_ivf(q, NLIST, K).tolist()) == set(knn_exacto(q, base, K).tolist())
    for q in consultas
)
print(f"Verificación top-k (argpartition vs argsort completo): {verif_topk}")
print(f"Verificación nprobe=nlist == exacto en 50 consultas: {verif_ivf_nlist}")

# ----------------------------------------------------------------------
# 5. Ground truth exacto y latencia de referencia
# ----------------------------------------------------------------------
exactos_idx = [knn_exacto(q, base, K) for q in consultas]   # ground truth (T1)
exactos_set = [set(a.tolist()) for a in exactos_idx]

# calentamiento (evita penalizar la primera llamada)
_ = knn_exacto(consultas[0], base, K)
_ = knn_ivf(consultas[0], 4, K)

lat_exacta_q = []
for q in consultas:
    t0 = time.perf_counter()
    for _ in range(REPETICIONES):
        knn_exacto(q, base, K)
    lat_exacta_q.append((time.perf_counter() - t0) / REPETICIONES)
lat_exacta_media_ms = float(np.mean(lat_exacta_q) * 1000.0)
lat_exacta_std_ms = float(np.std(lat_exacta_q) * 1000.0)
print(f"kNN exacto (referencia): {lat_exacta_media_ms:.3f} ms/consulta "
      f"(std {lat_exacta_std_ms:.3f})")

# ----------------------------------------------------------------------
# 6. Barrido de nprobe: recall@10 y latencia (ejercicio 2.2)
# ----------------------------------------------------------------------
recalls, speedups, lat_ivf_ms, lat_ivf_std, pools_medios = [], [], [], [], []
lat_por_consulta = {}
for nprobe in NPROBES:
    recs_q, lats_q, pools_q = [], [], []
    for i, q in enumerate(consultas):
        # pool de candidatos que se abriría (solo centroides, sin cronometrar)
        sc = centroides @ q
        elegidas = (np.argpartition(sc, -nprobe)[-nprobe:]
                    if nprobe < NLIST else np.arange(NLIST))
        pools_q.append(int(sum(celdas[c].size for c in elegidas)))
        # latencia
        t0 = time.perf_counter()
        for _ in range(REPETICIONES):
            idx = knn_ivf(q, nprobe, K)
        lats_q.append((time.perf_counter() - t0) / REPETICIONES)
        # recall@10 contra el exacto
        recs_q.append(len(set(idx.tolist()) & exactos_set[i]) / K)
    rec_medio = float(np.mean(recs_q))
    lat_media_ms = float(np.mean(lats_q) * 1000.0)
    lat_std_ms = float(np.std(lats_q) * 1000.0)
    sp = lat_exacta_media_ms / lat_media_ms      # speedup = razón de medias
    recalls.append(rec_medio)
    speedups.append(sp)
    lat_ivf_ms.append(lat_media_ms)
    lat_ivf_std.append(lat_std_ms)
    pools_medios.append(float(np.mean(pools_q)))
    lat_por_consulta[str(nprobe)] = [float(x * 1000.0) for x in lats_q]
    print(f"nprobe={nprobe:>2}: recall@10={rec_medio:.3f} | "
          f"latencia={lat_media_ms:.3f} ms | speedup={sp:.1f}x | "
          f"candidatos medios={np.mean(pools_q):.0f}")

# sanity check teórico: nprobe = nlist debe reproducir el exacto (recall 1.0)
rec_nlist = float(np.mean([
    len(set(knn_ivf(q, NLIST, K).tolist()) & exactos_set[i]) / K
    for i, q in enumerate(consultas)
]))
print(f"Control: recall@10 con nprobe=nlist={NLIST} → {rec_nlist:.4f} (debe ser 1.0)")

# ----------------------------------------------------------------------
# 7. Tabla y gráfica recall vs. speedup
# ----------------------------------------------------------------------
tabla = pd.DataFrame({
    "nprobe": NPROBES,
    "recall@10": np.round(recalls, 4),
    "latencia_ivf_ms": np.round(lat_ivf_ms, 4),
    "speedup_x": np.round(speedups, 2),
    "candidatos_medios": np.round(pools_medios, 0).astype(int),
})
print("\nTabla recall/latencia (50 consultas, k=10, N=100k, NLIST=64):")
print(tabla.to_string(index=False))
print(f"Referencia exacto: recall@10 = 1.000 | "
      f"latencia = {lat_exacta_media_ms:.3f} ms | speedup = 1.00x")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

# Panel 1 (el pedido): recall@10 vs. speedup, cada punto = un nprobe
ax1.plot(speedups, recalls, "o-", color="#1f77b4", lw=1.8, ms=7, zorder=3)
offsets = {1: (7, 5), 2: (7, 5), 4: (7, 5), 8: (7, -14), 16: (7, 5)}
for nb, s, r in zip(NPROBES, speedups, recalls):
    dx, dy = offsets.get(nb, (7, 5))
    ax1.annotate(f"nprobe={nb}", (s, r), textcoords="offset points",
                 xytext=(dx, dy), fontsize=9)
ax1.axhline(1.0, color="gray", ls="--", lw=1, label="recall = 1.0 (exacto)")
ax1.set_xlabel("Speedup de latencia vs. kNN exacto (×)")
ax1.set_ylabel("Recall@10")
ax1.set_title(f"Mini-IVF: recall@10 vs. speedup\n(N={N}, NLIST={NLIST}, k={K}, "
              f"{N_CONSULTAS} consultas)")
ax1.set_ylim(0, 1.06)
ax1.grid(alpha=0.3)
ax1.legend(loc="lower right")

# Panel 2: el dial nprobe — recall sube y latencia (cae el speedup) a la vez
ax2.plot(NPROBES, recalls, "o-", color="#2ca02c", label="recall@10")
ax2.set_xlabel("nprobe (celdas abiertas por consulta)")
ax2.set_ylabel("Recall@10", color="#2ca02c")
ax2.set_xticks
