# -*- coding: utf-8 -*-
"""
T2 — Parte 2: Mini-IVF con KMeans: recall@10 vs. speedup según nprobe.

Receta IVF (slides S2):
  (1) offline  : k-means agrupa los N vectores en nlist celdas;
  (2) consulta : comparar contra los nlist centroides (barato) y buscar
                 EXACTO solo dentro de las nprobe celdas más cercanas.

Se mide sobre 50 consultas, para nprobe in {1, 2, 4, 8, 16}:
  - recall@10 contra kNN exacto,
  - speedup de latencia vs. la búsqueda exacta,
y se reporta la tabla del "triángulo recall/latencia", el tamaño medio de
celda y las figuras PNG correspondientes.

Salidas (carpeta actual, rutas relativas):
  - resultados.json
  - T2_parte2_triangulo_recall_latencia.png
  - T2_parte2_recall_y_speedup_por_nprobe.png
"""

import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ======================================================================
# 0) Setup — replicación EXACTA del código dado en el enunciado
# ======================================================================
rng = np.random.default_rng(7)                                    # generador de datos
D = 128                                                           # dimensión
_CENTROS = np.random.default_rng(2026).normal(size=(200, D))      # "temas" fijos


def datos_realistas(n, d=D):
    """Embeddings sintéticos CON estructura: mezcla de 'temas' gaussianos fijos.
    Base y consultas comparten los mismos temas, como en un corpus real."""
    asignacion = rng.integers(0, len(_CENTROS), size=n)
    X = _CENTROS[asignacion, :d] + 1.5 * rng.normal(size=(n, d))
    return X.astype(np.float32)


def normalizar(X):
    """Vectores a norma 1: así el producto punto es el coseno."""
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


N = 100_000
NLIST = 64
K = 10                    # k del kNN (recall@10)
N_CONSULTAS = 50
NPROBES = [1, 2, 4, 8, 16]

print(f"Generando base sintética con estructura: N={N:,}, D={D} ...")
base = normalizar(datos_realistas(N))
consultas = normalizar(datos_realistas(N_CONSULTAS))   # mismos 'temas' que la base
print(f"base: {base.shape} ({base.dtype}) | consultas: {consultas.shape}")

# ======================================================================
# 1) Entrenamiento offline del Mini-IVF
# ======================================================================
print(f"Entrenando KMeans(n_clusters={NLIST}, n_init=3, random_state=7) "
      f"sobre {N:,} vectores ...")
t0 = time.perf_counter()
km = KMeans(n_clusters=NLIST, n_init=3, random_state=7).fit(base)
t_kmeans_s = time.perf_counter() - t0

centroides = normalizar(km.cluster_centers_.astype(np.float32))
celdas = [np.where(km.labels_ == c)[0] for c in range(NLIST)]   # índices GLOBALES por celda
tamano_celdas = np.array([len(c) for c in celdas], dtype=np.int64)
tamano_medio_celda = float(tamano_celdas.mean())
n_celdas_vacias = int((tamano_celdas == 0).sum())

print(f"IVF entrenado en {t_kmeans_s:.1f} s: {NLIST} celdas, "
      f"tamaño medio {tamano_medio_celda:.0f} "
      f"(min={int(tamano_celdas.min())}, max={int(tamano_celdas.max())}, "
      f"vacías={n_celdas_vacias})")

# ======================================================================
# 2) kNN exacto (referencia) y knn_ivf
# ======================================================================
def knn_exacto(consulta, k=K):
    """kNN exacto sobre TODA la base (vectores normalizados → dot = coseno).
    Devuelve (índices_globales, scores) ordenados por similitud descendente."""
    scores = base @ consulta                        # O(N·d)
    k_eff = min(k, scores.shape[0])
    idx = np.argpartition(-scores, k_eff - 1)[:k_eff]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]


def knn_ivf(consulta, nprobe, k=K):
    """IVF: 1) scores contra los NLIST centroides → nprobe celdas más cercanas
            2) kNN exacto SOLO dentro de esas celdas (¡índices globales!)."""
    scores_centroides = centroides @ consulta               # barato: NLIST·d
    elegidas = np.argsort(-scores_centroides)[:nprobe]
    idx_globales = np.concatenate([celdas[c] for c in elegidas])
    scores_locales = base[idx_globales] @ consulta          # exacto dentro de las celdas
    k_eff = min(k, idx_globales.shape[0])
    parte = np.argpartition(-scores_locales, k_eff - 1)[:k_eff]
    parte = parte[np.argsort(-scores_locales[parte])]
    return idx_globales[parte], scores_locales[parte]


def celdas_mas_cercanas(consulta, nprobe):
    """Solo para contabilidad: cuántos candidatos se escanean con este nprobe."""
    scores_centroides = centroides @ consulta
    elegidas = np.argsort(-scores_centroides)[:nprobe]
    return int(sum(len(celdas[c]) for c in elegidas))


# calentamiento (evita el ruido de la primera llamada en las mediciones)
_ = knn_exacto(consultas[0])
_ = knn_ivf(consultas[0], nprobe=1)

# --- referencia exacta: índices + latencia por consulta (mismas 50 consultas) ---
idx_exactos, lat_exactas_ms = [], []
for q in consultas:
    t0 = time.perf_counter()
    idx_e, _ = knn_exacto(q)
    lat_exactas_ms.append((time.perf_counter() - t0) * 1e3)
    idx_exactos.append(idx_e)
lat_exacta_media_ms = float(np.mean(lat_exactas_ms))
lat_exacta_std_ms = float(np.std(lat_exactas_ms))
print(f"\nkNN exacto: latencia media {lat_exacta_media_ms:.3f} ms/consulta "
      f"(std {lat_exacta_std_ms:.3f} ms)")

# ======================================================================
# 3) Barrido de nprobe: recall@10 y speedup sobre las 50 consultas
# ======================================================================
recall_at10_por_nprobe = {}
recall_std_por_nprobe = {}
speedup_por_nprobe = {}
latencia_ivf_ms_por_nprobe = {}
candidatos_medios_por_nprobe = {}

for nprobe in NPROBES:
    recalls, lats = [], []
    for qi, q in enumerate(consultas):
        t0 = time.perf_counter()
        idx_ivf, _ = knn_ivf(q, nprobe)
        lats.append((time.perf_counter() - t0) * 1e3)
        recalls.append(len(np.intersect1d(idx_ivf, idx_exactos[qi])) / K)
    lat_media = float(np.mean(lats))
    recall_at10_por_nprobe[nprobe] = float(np.mean(recalls))
    recall_std_por_nprobe[nprobe] = float(np.std(recalls))
    latencia_ivf_ms_por_nprobe[nprobe] = lat_media
    speedup_por_nprobe[nprobe] = lat_exacta_media_ms / lat_media
    candidatos_medios_por_nprobe[nprobe] = float(
        np.mean([celdas_mas_cercanas(q, nprobe) for q in consultas]))

# --- tabla del triángulo recall/latencia ---
print("\n=== Triángulo recall/latencia — Mini-IVF "
      f"(N={N:,}, nlist={NLIST}, k={K}, {N_CONSULTAS} consultas) ===")
print(f"{'nprobe':>6} | {'recall@10':>10} | {'lat IVF (ms)':>13} | "
      f"{'speedup×':>9} | {'candidatos':>11}")
print("-" * 68)
print(f"{'exacto':>6} | {'1.0000':>10} | {lat_exacta_media_ms:>13.3f} | "
      f"{1.0:>9.2f} | {N:>11,}")
for nprobe in NPROBES:
    print(f"{nprobe:>6} | {recall_at10_por_nprobe[nprobe]:>10.4f} | "
          f"{latencia_ivf_ms_por_nprobe[nprobe]:>13.3f} | "
          f"{speedup_por_nprobe[nprobe]:>9.2f} | "
          f"{int(round(candidatos_medios_por_nprobe[nprobe])):>11,}")
print(f"\nTamaño medio de celda: {tamano_medio_celda:.1f} vectores "
      f"(N/nlist = {N / NLIST:.0f})")

# ======================================================================
# 4) Figuras
# ======================================================================
lat_ivf = [latencia_ivf_ms_por_nprobe[np_] for np_ in NPROBES]
recalls = [recall_at10_por_nprobe[np_] for np_ in NPROBES]
speedups = [speedup_por_nprobe[np_] for np_ in NPROBES]

# Figura 1: el triángulo recall/latencia
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))

ax = axes[0]
ax.plot(lat_ivf, recalls, "o-", color="tab:blue", lw=2, ms=7, label="IVF (nprobe ↑)")
for np_, x, y in zip(NPROBES, lat_ivf, recalls):
    ax.annotate(f"nprobe={np_}", (x, y), textcoords="offset points",
                xytext=(6, 6), fontsize=9)
ax.plot([lat_exacta_media_ms], [1.0], "*", color="tab:red", ms=14,
        label="kNN exacto (recall=1)")
ax.axvline(lat_exacta_media_ms, color="tab:red", ls="--", lw=1.2,
           label=f"latencia exacta = {lat_exacta_media_ms:.2f} ms")
ax.axhline(1.0, color="gray", ls=":", lw=1)
ax.set_xlabel("Latencia media por consulta (ms)")
ax.set_ylabel("recall@10")
ax.set_title("Triángulo recall/latencia — Mini-IVF\n"
             f"(N={N:,}, nlist={NLIST}, k={K}, {N_CONSULTAS} consultas)")
ax.set_ylim(0, 1.06)
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(speedups, recalls, "s-", color="tab:green", lw=2, ms=7)
for np_, x, y in zip(NPROBES, speedups, recalls):
    ax.annotate(f"nprobe={np_}", (x, y), textcoords="offset points",
                xytext=(6, 6), fontsize=9)
ax.plot([1.0], [1.0], "*", color="tab:red", ms=14)
ax.axhline(1.0, color="gray", ls=":", lw=1)
ax.set_xlabel("Speedup vs. kNN exacto (×)")
ax.set_ylabel("recall@10")
ax.set_title("recall@10 vs. speedup\n(más a la izquierda = más rápido; "
             "más arriba = mejor recall)")
ax.set_ylim(0, 1.06)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("T2_parte2_triangulo_recall_latencia.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nFigura guardada: T2_parte2_triangulo_recall_latencia.png")

# Figura 2: recall y speedup en función de nprobe (la "perilla")
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(NPROBES, recalls, "o-", color="tab:blue", label="recall@10")
ax1.set_xticks(NPROBES)
ax1.set_xticklabels([str(x) for x in NPROBES])
ax1.set_xlabel("nprobe")
ax1.set_ylabel("recall@10", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_ylim(0, 1.06)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(NPROBES, speedups, "s--", color="tab:red", label="speedup vs. exacto (×)")
ax2.set_ylabel("speedup vs. exacto (×)", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="center right")
ax1.set_title("Mini-IVF: la perilla nprobe\n"
              f"(N={N:,}, nlist={NLIST}, k={K}, {N_CONSULTAS} consultas; "
              "nprobe = nlist sería el exacto)")
fig.tight_layout()
fig.savefig("T2_parte2_recall_y_speedup_por_nprobe.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Figura guardada: T2_parte2_recall_y_speedup_por_nprobe.png")

# ======================================================================
# 5) Referencia cruzada con T1 (si el archivo está disponible)
# ======================================================================
ref_t1 = None
ruta_t1 = os.path.join("entradas", "T1.json")
if os.path.exists(ruta_t1):
    try:
        with open(ruta_t1, "r", encoding="utf-8") as f:
            t1 = json.load(f)
        lm = t1.get("latencia_media_ms_por_N")
        lat_t1_100k = None
        if isinstance(lm, dict):
            for clave in (100000, "100000", "100_000", "100k"):
                if clave in lm:
                    lat_t1_100k = float(lm[clave])
                    break
        elif isinstance(lm, (list, tuple)) and len(lm) >= 3:
            # orden del enunciado Parte 1: N = 10k, 50k, 100k, 200k → índice 2 = 100k
            lat_t1_100k = float(lm[2])
        ref_t1 = {"latencia_media_ms_por_N": lm, "latencia_100k_ms": lat_t1_100k}
        if lat_t1_100k is not None:
            print(f"\nReferencia T1 (kNN exacto, N=100k): {lat_t1_100k:.3f} ms/consulta "
                  f"| medido aquí: {lat_exacta_media_ms:.3f} ms")
    except Exception as exc:
        print(f"Aviso: no se pudo usar entradas/T1.json ({exc})")
        ref_t1 = None

# ======================================================================
# 6) Guardar todas las cifras en resultados.json
# ======================================================================
resultados = {
    "subtarea": "T2 - Parte 2: Mini-IVF con KMeans, recall@10 vs. speedup",
    "parametros": {
        "N": int(N),
        "D": int(D),
        "NLIST": int(NLIST),
        "kmeans": {"n_clusters": int(NLIST), "n_init": 3, "random_state": 7},
        "k": int(K),
        "n_consultas": int(N_CONSULTAS),
        "nprobes": [int(x) for x in NPROBES],
        "semilla_rng_datos": 7,
        "semilla_centros_temas": 2026,
    },
    "tamano_medio_celda": tamano_medio_celda,
    "tamano_celdas_min": int(tamano_celdas.min()),
    "tamano_celdas_max": int(tamano_celdas.max()),
    "n_celdas_vacias": n_celdas_vacias,
    "tamano_celdas": [int(v) for v in tamano_celdas],
    "recall_at10_por_nprobe": {int(np_): float(recall_at10_por_nprobe[np_])
                               for np_ in NPROBES},
    "recall_at10_std_por_nprobe": {int(np_): float(recall_std_por_nprobe[np_])
                                   for np_ in NPROBES},
    "speedup_por_nprobe": {int(np_): float(speedup_por_nprobe[np_])
                           for np_ in NPROBES},
    "latencia_ivf_ms_por_nprobe": {int(np_): float(latencia_ivf_ms_por_nprobe[np_])
                                   for np_ in NPROBES},
    "candidatos_medios_por_nprobe": {int(np_): float(candidatos_medios_por_nprobe[np_])
                                     for np_ in NPROBES},
    "latencia_exacta_media_ms": lat_exacta_media_ms,
    "latencia_exacta_std_ms": lat_exacta_std_ms,
    "tiempo_entrenamiento_kmeans_s": float(t_kmeans_s),
    "figura_triangulo": "T2_parte2_triangulo_recall_latencia.png",
    "figura_por_nprobe": "T2_parte2_recall_y_speedup_por_nprobe.png",
    "referencia_T1": ref_t1,
    "notas": ("nprobe pequeño → speedup grande pero se pierden vecinos de las "
              "fronteras entre celdas; subir nprobe recupera recall pagando "
              "latencia. nprobe = nlist (=64) equivale a la búsqueda exacta."),
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)
print("\nResultados guardados en resultados.json")

# ======================================================================
# 7) Cifras principales (resumen final)
# ======================================================================
print("\n=== CIFRAS PRINCIPALES ===")
print(f"tamano_medio_celda = {tamano_medio_celda:.4f}")
print(f"latencia_exacta_media_ms = {lat_exacta_media_ms:.4f}")
for np_ in NPROBES:
    print(f"nprobe={np_:>2}: recall@10 = {recall_at10_por_nprobe[np_]:.4f} | "
          f"speedup = {speedup_por_nprobe[np_]:.2f}x | "
          f"latencia IVF = {latencia_ivf_ms_por_nprobe[np_]:.3f} ms | "
          f"candidatos medios = {candidatos_medios_por_nprobe[np_]:.0f}")
