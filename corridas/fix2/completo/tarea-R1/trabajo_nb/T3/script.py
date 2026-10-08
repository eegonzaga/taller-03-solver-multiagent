# -*- coding: utf-8 -*-
"""
T3 - Parte 2 - Mini-IVF: recall@10 vs. speedup según nprobe
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

Salidas (carpeta actual, rutas relativas):
    - resultados.json
    - T3_parte2_recall_vs_speedup_nprobe.png
"""

import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from sklearn.cluster import KMeans

# ==================================================================
# 0. Setup dado por el enunciado (código del profesor, sin cambios)
# ==================================================================
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
tamano_medio_celda = float(tamanos_celdas.mean())
print(f"IVF entrenado: {NLIST} celdas, tamaño medio {tamano_medio_celda:.0f}")

# ------------------------------------------------------------------
# Consultas: 50 vectores nuevos del MISMO generador (misma semilla 7,
# el flujo continúa tras generar la base) -> comparten los temas con la
# base, como en un corpus real, y la corrida es reproducible.
# ------------------------------------------------------------------
K = 10
N_CONSULTAS = 50
consultas = normalizar(datos_realistas(N_CONSULTAS))

# ==================================================================
# 1. kNN exacto (Ejercicio 1.1 / T2): referencia de vecinos verdaderos
# ==================================================================
def knn_exacto(consulta, base_mat, k=K):
    """kNN exacto por coseno: vectores normalizados -> usar dot."""
    scores = base_mat @ consulta
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


# Vecinos verdaderos de las 50 consultas (mismo knn_exacto de la referencia)
gt_vecinos = np.vstack([knn_exacto(q, base, K) for q in consultas]).astype(np.int64)

# ==================================================================
# 2. Ejercicio 2.1 — knn_ivf(consulta, nprobe, k)
# ==================================================================
def knn_ivf(consulta, nprobe, k=K):
    """Búsqueda IVF:
    (1) scores contra los centroides -> nprobe celdas más cercanas;
    (2) kNN exacto solo dentro de esas celdas -> índices GLOBALES."""
    nprobe = int(min(nprobe, NLIST))
    # (1) scores contra los centroides -> nprobe celdas más cercanas
    scores_centroides = centroides @ consulta
    if nprobe < NLIST:
        celdas_top = np.argpartition(-scores_centroides, nprobe - 1)[:nprobe]
    else:
        celdas_top = np.arange(NLIST)
    # (2) kNN exacto solo dentro de esas celdas (¡índices GLOBALES!)
    idx_globales = np.concatenate([celdas[c] for c in celdas_top])
    scores_candidatos = base[idx_globales] @ consulta
    if k < idx_globales.size:
        parte = np.argpartition(-scores_candidatos, k - 1)[:k]
    else:
        parte = np.arange(idx_globales.size)
    orden = np.argsort(-scores_candidatos[parte])
    return idx_globales[parte[orden]]


# Verificación: con nprobe = NLIST el IVF debe reproducir el exacto
coincide_full = bool(
    np.array_equal(np.sort(knn_ivf(consultas[0], NLIST, K)), np.sort(gt_vecinos[0]))
)
print(f"Verificación nprobe=NLIST == exacto (consulta 0): {coincide_full}")

# ==================================================================
# 3. Ejercicio 2.2 — recall@10 y speedup para nprobe = 1, 2, 4, 8, 16
# ==================================================================
NPROBES = [1, 2, 4, 8, 16]

# -- tiempo exacto por consulta (misma base, mismas 50 consultas) --
tiempos_exacto_ms = []
for q in consultas:
    t0 = time.perf_counter()
    knn_exacto(q, base, K)
    t1 = time.perf_counter()
    tiempos_exacto_ms.append((t1 - t0) * 1000.0)
tiempo_exacto_medio_ms = float(np.mean(tiempos_exacto_ms))
tiempo_exacto_std_ms = float(np.std(tiempos_exacto_ms))
print(f"\nkNN exacto (N={N}): tiempo medio {tiempo_exacto_medio_ms:.4f} ms/consulta")

# -- IVF: recall@10 y tiempo por consulta, para cada nprobe --
recall_at10_por_nprobe = {}
recall_std_por_nprobe = {}
speedup_por_nprobe = {}
tiempo_ivf_medio_ms_por_nprobe = {}
tiempo_ivf_std_ms_por_nprobe = {}
tiempos_ivf_ms_por_consulta_por_nprobe = {}

_ = knn_ivf(consultas[0], NPROBES[0], K)  # warm-up fuera del cronómetro

for nprobe in NPROBES:
    recalls_q = []
    tiempos_ivf_ms = []
    for i, q in enumerate(consultas):
        t0 = time.perf_counter()
        idx_ivf = knn_ivf(q, nprobe, K)
        t1 = time.perf_counter()
        tiempos_ivf_ms.append((t1 - t0) * 1000.0)
        comunes = np.intersect1d(idx_ivf, gt_vecinos[i]).size
        recalls_q.append(comunes / float(K))
    recall_medio = float(np.mean(recalls_q))
    t_ivf_medio = float(np.mean(tiempos_ivf_ms))
    clave = str(nprobe)
    recall_at10_por_nprobe[clave] = recall_medio
    recall_std_por_nprobe[clave] = float(np.std(recalls_q))
    tiempo_ivf_medio_ms_por_nprobe[clave] = t_ivf_medio
    tiempo_ivf_std_ms_por_nprobe[clave] = float(np.std(tiempos_ivf_ms))
    speedup_por_nprobe[clave] = tiempo_exacto_medio_ms / t_ivf_medio
    tiempos_ivf_ms_por_consulta_por_nprobe[clave] = [float(x) for x in tiempos_ivf_ms]

recall_lista = [float(recall_at10_por_nprobe[str(npb)]) for npb in NPROBES]
speedup_lista = [float(speedup_por_nprobe[str(npb)]) for npb in NPROBES]

print("\nResultados sobre 50 consultas (k=10):")
print(f"{'nprobe':>6} | {'recall@10':>10} | {'t IVF (ms)':>11} | {'speedup':>8}")
for npb in NPROBES:
    clave = str(npb)
    print(
        f"{npb:>6} | {recall_at10_por_nprobe[clave]:>10.4f} "
        f"| {tiempo_ivf_medio_ms_por_nprobe[clave]:>11.4f} "
        f"| {speedup_por_nprobe[clave]:>8.2f}"
    )
print(
    f"\nTamaño medio de celda del IVF: {tamano_medio_celda:.4f} "
    f"(min {int(tamanos_celdas.min())}, max {int(tamanos_celdas.max())})"
)

# ==================================================================
# 4. Figura: recall@10 vs. speedup según nprobe
# ==================================================================
FIG_PNG = "T3_parte2_recall_vs_speedup_nprobe.png"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))

# Panel 1: la curva del dial nprobe — recall@10 frente a speedup
ax1.plot(speedup_lista, recall_lista, "o-", color="tab:blue", lw=1.8, ms=7)
for npb, sp, rc in zip(NPROBES, speedup_lista, recall_lista):
    ax1.annotate(f"nprobe={npb}", xy=(sp, rc), xytext=(8, 5),
                 textcoords="offset points", fontsize=9)
ax1.axhline(1.0, color="gray", ls=":", lw=1)
ax1.set_xlabel("Speedup  (tiempo exacto / tiempo IVF)")
ax1.set_ylabel("Recall@10")
ax1.set_title("Trade-off recall@10 vs. speedup\n(Mini-IVF, 50 consultas, k=10)")
ax1.grid(alpha=0.3)
ax1.set_xlim(left=0.0)
ax1.set_ylim(0.0, 1.05)

# Panel 2: ambas métricas contra nprobe (escala log2)
linea_r, = ax2.plot(NPROBES, recall_lista, "s-", color="tab:blue", label="recall@10")
try:
    ax2.set_xscale("log", base=2)
except TypeError:
    ax2.set_xscale("log", basex=2)
ax2.set_xticks(NPROBES)
ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax2.set_xlabel("nprobe")
ax2.set_ylabel("Recall@10", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")
ax2.set_ylim(0.0, 1.05)
ax2b = ax2.twinx()
linea_s, = ax2b.plot(NPROBES, speedup_lista, "^--", color="tab:orange", label="speedup")
ax2b.set_ylabel("Speedup", color="tab:orange")
ax2b.tick_params(axis="y", labelcolor="tab:orange")
ax2.set_title("Recall@10 y speedup vs. nprobe")
ax2.grid(alpha=0.3)
ax2.legend([linea_r, linea_s], ["recall@10", "speedup"], loc="center left")

fig.suptitle(
    "T3 · Parte 2 — Mini-IVF: recall@10 vs. speedup según nprobe "
    f"(N={N}, NLIST={NLIST}, k={K})",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIG_PNG, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada: {FIG_PNG}")

# ==================================================================
# 5. Resultados -> resultados.json (precisión completa, sin redondear)
# ==================================================================
t2_referencia = None
ruta_t2 = Path("entradas") / "T2.json"
if ruta_t2.exists():
    try:
        with open(ruta_t2, "r", encoding="utf-8") as f:
            t2_datos = json.load(f)
        t2_referencia = {}
        for clave in ("tiempo_medio_ms_por_N", "tiempo_std_ms_por_N",
                      "factor_escalamiento_lineal", "pendiente_loglog",
                      "intercepto_loglog"):
            if clave in t2_datos:
                t2_referencia[clave] = t2_datos[clave]
    except Exception:
        t2_referencia = None

resultados = {
    "subtarea": "T3 - Parte 2 - Mini-IVF: recall@10 vs. speedup según nprobe",
    "figura_png": FIG_PNG,
    "setup": {
        "N": int(N),
        "D": int(D),
        "NLIST": int(NLIST),
        "base_normalizada": True,
        "centroides_normalizados": True,
        "celdas_construidas_con": "km.labels_",
        "kmeans_n_init": 3,
        "kmeans_random_state": 7,
        "semilla_generador_datos": 7,
        "semilla_centros_temas": 2026,
    },
    "parametros_experimento": {
        "k": int(K),
        "n_consultas": int(N_CONSULTAS),
        "nprobe_valores": [int(x) for x in NPROBES],
        "metrica_distancia": "coseno (dot sobre vectores normalizados)",
        "referencia_recall": "knn_exacto (Parte 1 / T2)",
        "speedup_definicion": "tiempo_exacto_medio / tiempo_ivf_medio",
    },
    "tamano_medio_celda": tamano_medio_celda,
    "tamano_std_celda": float(tamanos_celdas.std()),
    "tamano_min_celda": int(tamanos_celdas.min()),
    "tamano_max_celda": int(tamanos_celdas.max()),
    "tamanos_celdas": [int(x) for x in tamanos_celdas],
    "nprobe_valores": [int(x) for x in NPROBES],
    "recall_at10_por_nprobe": {str(npb): float(recall_at10_por_nprobe[str(npb)]) for npb in NPROBES},
    "recall_at10_valores": [float(v) for v in recall_lista],
    "recall_std_por_nprobe": {str(npb): float(recall_std_por_nprobe[str(npb)]) for npb in NPROBES},
    "speedup_por_nprobe": {str(npb): float(speedup_por_nprobe[str(npb)]) for npb in NPROBES},
    "speedup_valores": [float(v) for v in speedup_lista],
    "tiempo_exacto_medio_ms": tiempo_exacto_medio_ms,
    "tiempo_exacto_std_ms": tiempo_exacto_std_ms,
    "tiempos_exacto_ms_por_consulta": [float(x) for x in tiempos_exacto_ms],
    "tiempo_ivf_medio_ms_por_nprobe": {str(npb): float(tiempo_ivf_medio_ms_por_nprobe[str(npb)]) for npb in NPROBES},
    "tiempo_ivf_std_ms_por_nprobe": {str(npb): float(tiempo_ivf_std_ms_por_nprobe[str(npb)]) for npb in NPROBES},
    "tiempos_ivf_ms_por_consulta_por_nprobe": tiempos_ivf_ms_por_consulta_por_nprobe,
    "verificacion_nprobe_nlist_igual_exacto": coincide_full,
    "T2_referencia": t2_referencia,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)
print("Resultados guardados en resultados.json")
