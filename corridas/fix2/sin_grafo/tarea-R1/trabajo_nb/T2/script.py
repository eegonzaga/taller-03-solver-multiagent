# -*- coding: utf-8 -*-
"""
T2 - Parte 2 - Mini-IVF: recall@10 vs speedup segun nprobe.

Pipeline:
  1) Regenera deterministicamente la base normalizada de N=100_000 de T1
     (misma receta y semillas del enunciado: rng(7) para datos_realistas,
     generator(2026) para los 200 "temas") y 50 consultas del mismo stream.
  2) Entrena KMeans con NLIST=64 (n_init=3, random_state=7), normaliza los
     centroides y construye las celdas (indices GLOBALES por cluster).
  3) Implementa knn_ivf(consulta, nprobe, k): scores contra los 64 centroides
     -> nprobe celdas mas cercanas -> kNN exacto solo dentro de esas celdas.
  4) Para nprobe en {1,2,4,8,16} mide sobre 50 consultas: recall@10 contra el
     ground truth de knn_exacto y speedup = latencia exacta / latencia IVF.
  5) Guarda resultados.json y la figura PNG (triangulo recall/latencia).
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
# 1) Parametros fijados por el enunciado
# ---------------------------------------------------------------------------
N = 100_000            # tamano de la base (T1)
D = 128                # dimension de los embeddings sinteticos
NLIST = 64             # numero de celdas del indice IVF
N_INIT = 3             # KMeans(n_init=...)
RANDOM_STATE = 7       # KMeans(random_state=...)
K = 10                 # top-k -> recall@10
N_CONSULTAS = 50       # consultas de evaluacion
NPROBES = [1, 2, 4, 8, 16]
SEMILLA_DATOS = 7      # rng del enunciado que alimenta datos_realistas
SEMILLA_TEMAS = 2026   # generator de los 200 centros "tema"
FIG_PNG = "t2_parte2_recall_vs_speedup.png"

# ---------------------------------------------------------------------------
# 2) Datos sinteticos (misma receta y semillas que T1)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEMILLA_DATOS)
_CENTROS = np.random.default_rng(SEMILLA_TEMAS).normal(size=(200, D))


def datos_realistas(n, d=D):
    """Embeddings sinteticos CON estructura: mezcla de 'temas' gaussianos fijos.
    Base y consultas comparten los mismos temas, como en un corpus real."""
    asignacion = rng.integers(0, len(_CENTROS), size=n)
    X = _CENTROS[asignacion, :d] + 1.5 * rng.normal(size=(n, d))
    return X.astype(np.float32)


def normalizar(X):
    """Vectores a norma 1: asi el producto punto es el coseno."""
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


base = normalizar(datos_realistas(N))                 # identica a la base de T1
consultas = normalizar(datos_realistas(N_CONSULTAS))  # 50 consultas (mismos temas)
print(f"Base: {base.shape} {base.dtype} | Consultas: {consultas.shape}")

# ---------------------------------------------------------------------------
# 3) Contexto de T1 (entradas/T1.json) - lectura defensiva
# ---------------------------------------------------------------------------
contexto_t1 = {"cargado": False}
try:
    with open("entradas/T1.json", "r", encoding="utf-8") as f:
        t1 = json.load(f)
    contexto_t1["cargado"] = True
    contexto_t1["subtarea"] = t1.get("subtarea")
    if isinstance(t1.get("parametros"), dict):
        contexto_t1["parametros"] = t1["parametros"]
    lat = t1.get("latencia_media_ms_por_N")
    ref = None
    if isinstance(lat, dict):
        for kk, vv in lat.items():
            if str(kk).replace("_", "") == "100000":
                try:
                    ref = float(vv)
                except (TypeError, ValueError):
                    ref = None
                break
    elif isinstance(lat, list) and len(lat) > 0:
        try:
            ref = float(lat[-1])
        except (TypeError, ValueError):
            ref = None
    if ref is not None:
        contexto_t1["latencia_exacta_T1_ms_N100000"] = ref
        print(f"T1: latencia exacta reportada en N=100000: {ref:.4f} ms/consulta")
except Exception:
    contexto_t1["nota"] = ("entradas/T1.json no disponible; la base se regenero "
                           "con las semillas del enunciado (identica a la de T1).")

# ---------------------------------------------------------------------------
# 4) kNN exacto (baseline de latencia y ground truth de recall)
# ---------------------------------------------------------------------------
def knn_exacto(consulta, k):
    """Top-k por coseno (vectores normalizados): productos punto + argpartition."""
    scores = base @ consulta
    if k < scores.shape[0]:
        idx = np.argpartition(-scores, k)[:k]
    else:
        idx = np.arange(scores.shape[0])
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]


# ---------------------------------------------------------------------------
# 5) Indice IVF: KMeans + celdas con indices globales
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
km = KMeans(n_clusters=NLIST, n_init=N_INIT, random_state=RANDOM_STATE).fit(base)
t_kmeans_s = time.perf_counter() - t0

centroides = normalizar(km.cluster_centers_.astype(np.float32))
celdas = [np.where(km.labels_ == c)[0] for c in range(NLIST)]
tam_celdas = np.array([len(c) for c in celdas], dtype=np.int64)
tamano_medio_celda = float(tam_celdas.mean())
print(f"IVF entrenado en {t_kmeans_s:.1f} s: {NLIST} celdas, "
      f"tamano medio {tamano_medio_celda:.1f} "
      f"(min {int(tam_celdas.min())}, max {int(tam_celdas.max())})")


def _candidatos(consulta, nprobe):
    """Paso 1 de knn_ivf: scores contra centroides -> nprobe celdas mas
    cercanas -> indices GLOBALES concatenados de esas celdas."""
    scores_celdas = centroides @ consulta
    sel = np.argsort(-scores_celdas)[:nprobe]
    return np.concatenate([celdas[c] for c in sel])


def knn_ivf(consulta, nprobe, k):
    """Paso 2: kNN exacto solo dentro de las celdas seleccionadas."""
    idx_glob = _candidatos(consulta, nprobe)
    scores = base[idx_glob] @ consulta
    if k < idx_glob.shape[0]:
        top = np.argpartition(-scores, k)[:k]
    else:
        top = np.arange(idx_glob.shape[0])
    top = top[np.argsort(-scores[top])]
    return idx_glob[top], scores[top]


# ---------------------------------------------------------------------------
# 6) Ground truth + latencia exacta sobre las mismas 50 consultas
# ---------------------------------------------------------------------------
_ = knn_exacto(consultas[0], K)  # warm-up (evita ruido de primera llamada)
gt = []
lat_exactas = []
for q in consultas:
    t0 = time.perf_counter()
    idx, _s = knn_exacto(q, K)
    lat_exactas.append(time.perf_counter() - t0)
    gt.append(idx)
lat_exacta_media_ms = float(np.mean(lat_exactas) * 1000.0)
lat_exacta_std_ms = float(np.std(lat_exactas) * 1000.0)
print(f"kNN exacto: {lat_exacta_media_ms:.4f} ms/consulta "
      f"(std {lat_exacta_std_ms:.4f})")

# ---------------------------------------------------------------------------
# 7) Barrido de nprobe: recall@10 y speedup
# ---------------------------------------------------------------------------
recall_at10_por_nprobe = {}
recall_std_por_nprobe = {}
latencia_ivf_media_ms_por_nprobe = {}
latencia_ivf_std_ms_por_nprobe = {}
speedup_por_nprobe = {}
candidatos_medios_por_nprobe = {}
comparaciones_teoricas_por_nprobe = {}
speedup_teorico_por_nprobe = {}

for nprobe in NPROBES:
    _ = knn_ivf(consultas[0], nprobe, K)  # warm-up
    lats, recs = [], []
    for i in range(N_CONSULTAS):
        t0 = time.perf_counter()
        idx, _s = knn_ivf(consultas[i], nprobe, K)
        lats.append(time.perf_counter() - t0)
        recs.append(len(np.intersect1d(idx, gt[i])) / float(K))
    lat_ms = float(np.mean(lats) * 1000.0)
    lat_std = float(np.std(lats) * 1000.0)
    rec_m = float(np.mean(recs))
    rec_s = float(np.std(recs))
    cand = float(np.mean([_candidatos(q, nprobe).shape[0] for q in consultas]))
    comps_teo = float(NLIST + (N / float(NLIST)) * nprobe)

    recall_at10_por_nprobe[str(nprobe)] = rec_m
    recall_std_por_nprobe[str(nprobe)] = rec_s
    latencia_ivf_media_ms_por_nprobe[str(nprobe)] = lat_ms
    latencia_ivf_std_ms_por_nprobe[str(nprobe)] = lat_std
    speedup_por_nprobe[str(nprobe)] = lat_exacta_media_ms / lat_ms
    candidatos_medios_por_nprobe[str(nprobe)] = cand
    comparaciones_teoricas_por_nprobe[str(nprobe)] = comps_teo
    speedup_teorico_por_nprobe[str(nprobe)] = N / comps_teo

# Listas en el orden de NPROBES (comodidad para tareas posteriores)
recalls = [recall_at10_por_nprobe[str(nb)] for nb in NPROBES]
speedups = [speedup_por_nprobe[str(nb)] for nb in NPROBES]

# ---------------------------------------------------------------------------
# 8) Tabla en consola
# ---------------------------------------------------------------------------
print("\nTriangulo recall/latencia (50 consultas, k=10, base N=100000):")
print(f"{'nprobe':>6} | {'recall@10':>9} | {'lat IVF ms':>11} | "
      f"{'speedup':>8} | {'candidatos':>10} | {'comps teo':>10}")
for nb in NPROBES:
    print(f"{nb:>6} | {recall_at10_por_nprobe[str(nb)]:>9.4f} | "
          f"{latencia_ivf_media_ms_por_nprobe[str(nb)]:>11.4f} | "
          f"{speedup_por_nprobe[str(nb)]:>8.2f} | "
          f"{candidatos_medios_por_nprobe[str(nb)]:>10.0f} | "
          f"{comparaciones_teoricas_por_nprobe[str(nb)]:>10.0f}")
print(f"(exacto: {N} comparaciones, {lat_exacta_media_ms:.4f} ms/consulta)")

# ---------------------------------------------------------------------------
# 9) Figura: curva parametrizada por nprobe + eje doble vs nprobe
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

ax1.plot(speedups, recalls, "o-", color="tab:purple", lw=2, ms=8)
for sp, rc, nb in zip(speedups, recalls, NPROBES):
    ax1.annotate(f"nprobe={nb}", (sp, rc), textcoords="offset points",
                 xytext=(7, 5), fontsize=9)
ax1.axhline(1.0, color="gray", ls="--", lw=1, label="recall exacto = 1.0")
ax1.set_xlabel("Speedup  (latencia exacta / latencia IVF)")
ax1.set_ylabel("recall@10")
ax1.set_title("Triangulo recall/latencia\n(curva parametrizada por nprobe)")
ax1.grid(alpha=0.3)
ax1.legend(loc="lower left")

ax2.plot(NPROBES, recalls, "o-", color="tab:blue", label="recall@10")
ax2.set_xlabel("nprobe")
ax2.set_ylabel("recall@10", color="tab:blue")
ax2.set_xticks(NPROBES)
ax2.set_ylim(0.0, 1.05)
ax2.tick_params(axis="y", labelcolor="tab:blue")
ax2.grid(alpha=0.3)
ax2b = ax2.twinx()
ax2b.plot(NPROBES, speedups, "s--", color="tab:red", label="speedup")
ax2b.set_ylabel("speedup (x)", color="tab:red")
ax2b.tick_params(axis="y", labelcolor="tab:red")
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2b.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc="center right")
ax2.set_title("recall@10 y speedup vs nprobe (eje doble)")

fig.suptitle("Mini-IVF: N=100000, nlist=64, k=10, 50 consultas",
             y=1.02, fontsize=11)
plt.tight_layout()
plt.savefig(FIG_PNG, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada: {FIG_PNG}")

# ---------------------------------------------------------------------------
# 10) resultados.json
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "T2 - Parte 2 - Mini-IVF: recall@10 vs speedup segun nprobe",
    "parametros": {
        "N": int(N), "D": int(D), "NLIST": int(NLIST),
        "n_init": int(N_INIT), "random_state_kmeans": int(RANDOM_STATE),
        "k": int(K), "n_consultas": int(N_CONSULTAS),
        "nprobes": [int(nb) for nb in NPROBES],
        "semilla_datos": int(SEMILLA_DATOS),
        "semilla_temas": int(SEMILLA_TEMAS),
    },
    "tamano_medio_celda": tamano_medio_celda,
    "tamano_celda_std": float(tam_celdas.std()),
    "tamano_celda_min": int(tam_celdas.min()),
    "tamano_celda_max": int(tam_celdas.max()),
    "nprobes": [int(nb) for nb in NPROBES],
    "recall_at10_por_nprobe": recall_at10_por_nprobe,
    "speedup_por_nprobe": speedup_por_nprobe,
    "recall_at10_lista": recalls,
    "speedup_lista": speedups,
    "recall_std_por_nprobe": recall_std_por_nprobe,
    "latencia_exacta_media_ms": lat_exacta_media_ms,
    "latencia_exacta_std_ms": lat_exacta_std_ms,
    "latencia_ivf_media_ms_por_nprobe": latencia_ivf_media_ms_por_nprobe,
    "latencia_ivf_std_ms_por_nprobe": latencia_ivf_std_ms_por_nprobe,
    "candidatos_medios_por_nprobe": candidatos_medios_por_nprobe,
    "comparaciones_teoricas_por_nprobe": comparaciones_teoricas_por_nprobe,
    "speedup_teorico_comparaciones_por_nprobe": speedup_teorico_por_nprobe,
    "t_entrenamiento_kmeans_s": float(t_kmeans_s),
    "ground_truth": "knn_exacto (coseno sobre base normalizada, N=100000)",
    "metodologia": ("speedup = latencia media exacta / latencia media IVF sobre "
                    "las mismas 50 consultas; recall@10 = |top10 IVF inter "
                    "top10 exacto| / 10; nada se ajusta con las consultas"),
    "figura_png": FIG_PNG,
    "T1": contexto_t1,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)
print("Resultados guardados en resultados.json")
