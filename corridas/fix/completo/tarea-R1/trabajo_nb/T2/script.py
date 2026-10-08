# -*- coding: utf-8 -*-
"""
T2 — Parte 2: Mini-IVF con KMeans — recall@10 vs. speedup según nprobe.

Receta (código dado por el profesor):
  (1) offline: k-means agrupa la base en nlist celdas;
  (2) en consulta: scores contra los centroides (barato) -> nprobe celdas más
      cercanas -> kNN EXACTO solo dentro de esas celdas (índices globales).

Ejercicio 2.1: knn_ivf(consulta, nprobe, k).
Ejercicio 2.2: para nprobe = 1, 2, 4, 8, 16 -> recall@10 (vs. knn_exacto)
               promediado sobre 50 consultas + speedup vs. búsqueda exacta.

Salidas: resultados.json y T2_parte2_recall_vs_speedup.png (carpeta actual).
"""

import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------
# Setup dado por el profesor (semillas y generador EXACTOS del enunciado)
# ----------------------------------------------------------------------
rng = np.random.default_rng(7)
D = 128  # dimensión de los embeddings sintéticos
_CENTROS = np.random.default_rng(2026).normal(size=(200, D))  # "temas" fijos


def datos_realistas(n, d=D):
    """Embeddings sintéticos CON estructura: mezcla de 'temas' gaussianos fijos."""
    asignacion = rng.integers(0, len(_CENTROS), size=n)
    X = _CENTROS[asignacion, :d] + 1.5 * rng.normal(size=(n, d))
    return X.astype(np.float32)


def normalizar(X):
    """Vectores a norma 1: así el producto punto es el coseno."""
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


# ----------------------------------------------------------------------
# Índice IVF: KMeans sobre la base + celdas por labels (código dado)
# ----------------------------------------------------------------------
N = 100_000
base = normalizar(datos_realistas(N))            # (100000, 128) float32, norma 1
NLIST = 64

t_kmeans0 = time.perf_counter()
km = KMeans(n_clusters=NLIST, n_init=3, random_state=7).fit(base)
t_kmeans_s = time.perf_counter() - t_kmeans0

centroides = normalizar(km.cluster_centers_.astype(np.float32))  # (64, 128), norma 1
celdas = [np.where(km.labels_ == c)[0] for c in range(NLIST)]    # índices GLOBALES

tamanos_celda = np.array([len(c) for c in celdas], dtype=float)
tamano_medio_celda = float(tamanos_celda.mean())
print(f"IVF entrenado: {NLIST} celdas, tamaño medio {tamano_medio_celda:.0f} "
      f"(min {int(tamanos_celda.min())}, max {int(tamanos_celda.max())}; "
      f"KMeans tardó {t_kmeans_s:.1f} s)")

# ----------------------------------------------------------------------
# kNN exacto (vectores normalizados -> producto punto = coseno)
# ----------------------------------------------------------------------
def knn_exacto(consulta, base, k):
    """Top-k por coseno contra TODA la base. Devuelve índices globales."""
    scores = base @ consulta
    if k < scores.shape[0]:
        parte = np.argpartition(-scores, k)[:k]
    else:
        parte = np.arange(scores.shape[0])
    return parte[np.argsort(-scores[parte])]


# ----------------------------------------------------------------------
# Ejercicio 2.1 — knn_ivf(consulta, nprobe, k)
# ----------------------------------------------------------------------
def knn_ivf(consulta, nprobe, k):
    """IVF: 1) scores contra centroides -> nprobe celdas más cercanas
            2) kNN exacto SOLO dentro de esas celdas (índices globales)."""
    scores_celdas = centroides @ consulta               # (NLIST,) coseno vs centroides
    orden_celdas = np.argsort(-scores_celdas)[:nprobe]  # las nprobe celdas más cercanas
    listas = [celdas[c] for c in orden_celdas]
    idx_glob = listas[0] if nprobe == 1 else np.concatenate(listas)
    scores = base[idx_glob] @ consulta                  # exacto SOLO dentro de las celdas
    if k < scores.shape[0]:
        parte = np.argpartition(-scores, k)[:k]
    else:
        parte = np.arange(scores.shape[0])
    top_local = parte[np.argsort(-scores[parte])]
    return idx_glob[top_local]                          # índices GLOBALES


# ----------------------------------------------------------------------
# Ejercicio 2.2 — recall@10 y speedup para nprobe = 1, 2, 4, 8, 16
# ----------------------------------------------------------------------
K = 10
N_CONSULTAS = 50
# Consultas NUEVAS del mismo generador (comparten los "temas" con la base, como
# en un corpus real); NO son filas de la base -> no se evalúa sobre lo "entrenado".
consultas = normalizar(datos_realistas(N_CONSULTAS))

# Calentamiento (no se mide): evita pagar inicialización de BLAS en la 1ª medida
_ = knn_exacto(consultas[0], base, K)
_ = knn_ivf(consultas[0], 4, K)

# --- línea base: kNN exacto, cronometrado consulta por consulta ---
lat_exactas_ms = []
topk_exactos = []
for q in consultas:
    t0 = time.perf_counter()
    idx = knn_exacto(q, base, K)
    lat_exactas_ms.append((time.perf_counter() - t0) * 1e3)
    topk_exactos.append(idx)
lat_media_exacta_ms = float(np.mean(lat_exactas_ms))
print(f"\nkNN exacto sobre 100.000 vectores: {lat_media_exacta_ms:.3f} ms/consulta "
      f"(promedio de {N_CONSULTAS} consultas)")

# --- IVF por cada nprobe ---
nprobe_lista = [1, 2, 4, 8, 16]
recall_at10_por_nprobe = {}
recall_std_por_nprobe = {}
latencia_ivf_ms_por_nprobe = {}
speedup_por_nprobe = {}

for nprobe in nprobe_lista:
    recalls_q, lats_q = [], []
    for q, idx_ex in zip(consultas, topk_exactos):
        t0 = time.perf_counter()
        idx_ivf = knn_ivf(q, nprobe, K)
        lats_q.append((time.perf_counter() - t0) * 1e3)
        inter = np.intersect1d(idx_ivf, idx_ex).shape[0]
        recalls_q.append(inter / K)
    lat_m = float(np.mean(lats_q))
    latencia_ivf_ms_por_nprobe[nprobe] = lat_m
    recall_at10_por_nprobe[nprobe] = float(np.mean(recalls_q))
    recall_std_por_nprobe[nprobe] = float(np.std(recalls_q))
    speedup_por_nprobe[nprobe] = lat_media_exacta_ms / lat_m

# Sanidad: nprobe = nlist debe reproducir el kNN exacto (la aproximación es un dial)
verif_nlist = bool(np.array_equal(np.sort(knn_ivf(consultas[0], NLIST, K)),
                                  np.sort(knn_exacto(consultas[0], base, K))))
print(f"Verificación nprobe=nlist={NLIST} == exacto: {verif_nlist}")

print(f"\n{'nprobe':>6} | {'recall@10':>9} | {'lat IVF (ms)':>12} | {'speedup':>9}")
print("-" * 48)
for np_ in nprobe_lista:
    print(f"{np_:>6} | {recall_at10_por_nprobe[np_]:>9.3f} | "
          f"{latencia_ivf_ms_por_nprobe[np_]:>12.3f} | {speedup_por_nprobe[np_]:>8.1f}x")

# Referencia opcional de la subtarea anterior (T1): latencia exacta en N=100k
ref_T1 = None
try:
    with open("entradas/T1.json", "r", encoding="utf-8") as f:
        t1 = json.load(f)
    lat_N = t1.get("latencia_media_ms_por_N")
    lista_N = t1.get("lista_N", [])
    if isinstance(lat_N, dict):
        for clave in ("100000", 100000):
            if clave in lat_N:
                ref_T1 = float(lat_N[clave])
                break
    elif isinstance(lat_N, (list, tuple)) and 100000 in lista_N:
        ref_T1 = float(lat_N[lista_N.index(100000)])
    if ref_T1 is not None:
        print(f"\nReferencia T1 (exacto, N=100k): {ref_T1:.3f} ms/consulta | "
              f"medida hoy aquí: {lat_media_exacta_ms:.3f} ms")
except Exception:
    ref_T1 = None  # T1 no disponible: la línea base medida en este script es la que vale

# ----------------------------------------------------------------------
# Figura: curva recall@10 vs. speedup + la perilla nprobe
# ----------------------------------------------------------------------
speedups = [speedup_por_nprobe[np_] for np_ in nprobe_lista]
recalls = [recall_at10_por_nprobe[np_] for np_ in nprobe_lista]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

ax1.plot(speedups, recalls, "o-", color="#1f77b4", lw=2)
for np_, x, y in zip(nprobe_lista, speedups, recalls):
    ax1.annotate(f"nprobe={np_}", (x, y), textcoords="offset points",
                 xytext=(7, 5), fontsize=9)
ax1.axhline(1.0, ls="--", color="gray", lw=1)
ax1.set_xlabel("Speedup frente al kNN exacto (×)")
ax1.set_ylabel("Recall@10")
ax1.set_title(f"Mini-IVF: curva recall@10 vs. speedup\n"
              f"(N=100.000, nlist={NLIST}, k={K}, {N_CONSULTAS} consultas)")
ax1.grid(alpha=0.3)
ax1.set_xlim(left=0)

ax2.plot(nprobe_lista, recalls, "o-", color="#d62728", label="Recall@10")
ax2.axhline(1.0, ls="--", color="gray", lw=1)
try:
    ax2.set_xscale("log", base=2)
except TypeError:  # matplotlib antiguo
    ax2.set_xscale("log")
ax2.set_xticks(nprobe_lista)
ax2.minorticks_off()
ax2.set_xlabel("nprobe (escala log2)")
ax2.set_ylabel("Recall@10", color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")
ax2b = ax2.twinx()
ax2b.plot(nprobe_lista, speedups, "s--", color="#2ca02c", label="Speedup (×)")
ax2b.set_ylabel("Speedup vs. exacto (×)", color="#2ca02c")
ax2b.tick_params(axis="y", labelcolor="#2ca02c")
ax2.set_title("La perilla nprobe: recall ↑ ⇒ latencia ↑\n"
              f"(con nprobe = nlist = {NLIST} sería el exacto)")
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2b.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc="center left")

fig.tight_layout()
fig.savefig("T2_parte2_recall_vs_speedup.png", dpi=150)
plt.close(fig)
print("\nFigura guardada: T2_parte2_recall_vs_speedup.png")

# ----------------------------------------------------------------------
# Reflexión (con las cifras medidas): triángulo recall/latencia y ef_search
# ----------------------------------------------------------------------
r1, s1 = recall_at10_por_nprobe[1], speedup_por_nprobe[1]
r16, s16 = recall_at10_por_nprobe[16], speedup_por_nprobe[16]
print("\n--- Reflexión ---")
print(f"1) Triángulo recall/latencia: con nprobe=1 se abre UNA celda "
      f"(~{tamano_medio_celda:.0f} de los {N} vectores) y la consulta cuesta "
      f"{latencia_ivf_ms_por_nprobe[1]:.3f} ms ({s1:.1f}x más rápida que el exacto, "
      f"{lat_media_exacta_ms:.3f} ms), pero el recall@10 cae a {r1:.2f}.")
print("   El recall perdido vive en las FRONTERAS entre celdas: un vecino verdadero "
      "puede estar en una celda vecina cuya frontera pasa cerca de la consulta; esa "
      "celda no se abre aunque el punto sí esté entre los 10 más cercanos.")
print(f"2) Subir la perilla recupera recall pagando latencia: nprobe=16 da recall "
      f"{r16:.2f} con speedup {s16:.1f}x. El costo crece ~lineal con nprobe "
      f"(~nprobe x {tamano_medio_celda:.0f} vectores explorados) y con "
      f"nprobe=nlist={NLIST} IVF ES el kNN exacto (verificado: {verif_nlist}) — "
      "la aproximación es un dial, no otro algoritmo.")
print("3) Paralelismo con ef_search de HNSW: es la MISMA clase de perilla de consulta — "
      "cuántos candidatos explora por consulta (celdas abiertas en IVF, nodos visitados "
      "en el grafo de HNSW) — con el mismo intercambio recall por latencia. Lo que cambia "
      "es la estructura subyacente (celdas vs. grafo) y con ello la forma de la curva; "
      "aquí solo medimos IVF, así que la frase 'HNSW ofrece el mismo dial con mejor curva' "
      "queda como hipótesis del material, no como cifra medida en este notebook.")

# ----------------------------------------------------------------------
# resultados.json (cifras con precisión completa, tipos nativos)
# ----------------------------------------------------------------------
resultados = {
    "tarea": "T2 - Parte 2: Mini-IVF con KMeans, recall@10 vs. speedup",
    "N": int(N),
    "D": int(D),
    "NLIST": int(NLIST),
    "k": int(K),
    "n_consultas": int(N_CONSULTAS),
    "nprobe_lista": [int(x) for x in nprobe_lista],
    "tamano_medio_celda": tamano_medio_celda,
    "tamano_std_celda": float(tamanos_celda.std()),
    "tamano_min_celda": int(tamanos_celda.min()),
    "tamano_max_celda": int(tamanos_celda.max()),
    "tamanos_celdas": [int(x) for x in tamanos_celda],
    "tiempo_entrenamiento_kmeans_s": float(t_kmeans_s),
    "latencia_media_exacta_ms": lat_media_exacta_ms,
    "latencia_std_exacta_ms": float(np.std(lat_exactas_ms)),
    "latencia_media_ivf_ms_por_nprobe": {str(np_): float(v)
                                         for np_, v in latencia_ivf_ms_por_nprobe.items()},
    "recall_at10_por_nprobe": {str(np_): float(v)
                               for np_, v in recall_at10_por_nprobe.items()},
    "recall_std_por_nprobe": {str(np_): float(v)
                              for np_, v in recall_std_por_nprobe.items()},
    "speedup_por_nprobe": {str(np_): float(v)
                           for np_, v in speedup_por_nprobe.items()},
    "verificacion_nprobe_nlist_igual_exacto": verif_nlist,
    "T1_referencia_latencia_exacta_ms_N100000": ref_T1,
    "figura_png": "T2_parte2_recall_vs_speedup.png",
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("\nCifras guardadas en resultados.json: tamano_medio_celda, "
      "recall_at10_por_nprobe, speedup_por_nprobe (y extras de latencia/verificación).")
