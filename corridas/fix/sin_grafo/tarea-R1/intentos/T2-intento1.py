# -*- coding: utf-8 -*-
"""
T2 — Parte 2: Mini-IVF con KMeans — recall@10 vs. speedup según nprobe.

Reproduce el setup del enunciado (N=100_000, base normalizada, NLIST=64,
KMeans(n_clusters=64, n_init=3, random_state=7), centroides normalizados,
celdas por labels_) y resuelve:

  Ejercicio 2.1 — knn_ivf(consulta, nprobe, k): scores contra los centroides,
                  selección de las nprobe celdas más cercanas y kNN exacto solo
                  dentro de esas celdas, devolviendo índices GLOBALES.
  Ejercicio 2.2 — para nprobe in {1, 2, 4, 8, 16}: recall@10 contra knn_exacto
                  promediado sobre 50 consultas + speedup de latencia vs.
                  búsqueda exacta; tabla recall/speedup y curva
                  recall@10 vs. speedup (triángulo recall/latencia).

Salidas: resultados.json, figura_recall_vs_speedup.png
"""

import json
import time

import numpy as np
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ----------------------------------------------------------------------------
# 1) Setup: datos sintéticos con estructura (código dado en el enunciado)
# ----------------------------------------------------------------------------
rng = np.random.default_rng(7)                                  # rng del notebook
D = 128                                                         # dimensión
_CENTROS = np.random.default_rng(2026).normal(size=(200, D))    # "temas" fijos


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
K = 10              # top-k (recall@10)
N_CONSULTAS = 50
NPROBES = [1, 2, 4, 8, 16]

# Mismo orden de llamadas que el notebook: primero la base, después las consultas
# (la rng global de semilla 7 continúa su estado; consultas y base comparten temas)
base = normalizar(datos_realistas(N))
consultas = normalizar(datos_realistas(N_CONSULTAS))
print(f"Base: {base.shape[0]:,} vectores de dimensión {base.shape[1]} | "
      f"{N_CONSULTAS} consultas | k={K}")

# ----------------------------------------------------------------------------
# 2) Entrenamiento del Mini-IVF (código dado)
# ----------------------------------------------------------------------------
t0 = time.perf_counter()
km = KMeans(n_clusters=NLIST, n_init=3, random_state=7).fit(base)
t_kmeans_s = time.perf_counter() - t0

centroides = normalizar(km.cluster_centers_.astype(np.float32))
celdas = [np.where(km.labels_ == c)[0] for c in range(NLIST)]
tamanos = np.array([len(c) for c in celdas], dtype=np.int64)
tamano_medio_celda = float(tamanos.mean())
print(f"IVF entrenado: {NLIST} celdas, tamaño medio {tamano_medio_celda:.0f} "
      f"(min {int(tamanos.min())}, max {int(tamanos.max())}, std {tamanos.std():.1f}) "
      f"[KMeans: {t_kmeans_s:.1f} s]")

# ----------------------------------------------------------------------------
# 3) Ejercicio 2.1 — knn_ivf(consulta, nprobe, k)  (+ knn_exacto de referencia)
# ----------------------------------------------------------------------------
def knn_exacto(consulta, k=K):
    """kNN exacto por fuerza bruta (coseno: base normalizada). Índices globales."""
    scores = base @ consulta                # (N,) cosenos contra toda la base
    idx = np.argpartition(-scores, k)[:k]   # top-k sin ordenar
    return idx[np.argsort(-scores[idx])]    # top-k ordenado por score


def knn_ivf(consulta, nprobe, k=K):
    """IVF en dos pasos (Ejercicio 2.1).
    1) scores contra los NLIST centroides -> nprobe celdas más cercanas
    2) kNN exacto SOLO dentro de esas celdas, con índices GLOBALES
    """
    scores_celdas = centroides @ consulta                    # paso 1: barato (NLIST comparaciones)
    mejores = np.argpartition(-scores_celdas, nprobe - 1)[:nprobe]
    mejores = mejores[np.argsort(-scores_celdas[mejores])]   # celdas ordenadas por cercanía
    idx = np.concatenate([celdas[c] for c in mejores])       # índices GLOBALES candidatos
    scores = base[idx] @ consulta                            # paso 2: exacto solo dentro de las celdas
    if len(idx) > k:
        top = np.argpartition(-scores, k)[:k]
    else:
        top = np.argsort(-scores)[:k]
    return idx[top[np.argsort(-scores[top])]]


# ----------------------------------------------------------------------------
# 4) Referencia a T1 (latencia del kNN exacto medida en la subtarea anterior)
# ----------------------------------------------------------------------------
t1_ref = {"disponible": False, "latencia_exacta_ms_N100000_T1": None}
try:
    with open("entradas/T1.json", "r", encoding="utf-8") as f:
        t1 = json.load(f)
    t1_ref["disponible"] = True
    tm = t1.get("tiempo_medio_ms_por_N") if isinstance(t1, dict) else None
    if isinstance(tm, dict):
        for clave in ("100000", 100000, "100_000"):
            if clave in tm:
                try:
                    v = tm[clave]
                    if isinstance(v, dict):
                        for sub in ("media", "medio", "ms", "valor"):
                            if sub in v:
                                v = v[sub]
                                break
                    t1_ref["latencia_exacta_ms_N100000_T1"] = float(v)
                except (TypeError, ValueError):
                    pass
                break
except Exception:
    pass

# ----------------------------------------------------------------------------
# 5) Ejercicio 2.2 — latencias y recall@10 sobre las 50 consultas
# ----------------------------------------------------------------------------
# warm-up: evita contabilizar el arranque de BLAS/hilos en la primera llamada
_ = knn_exacto(consultas[0])
_ = knn_ivf(consultas[0], NPROBES[-1])

# --- búsqueda exacta: latencia de referencia + top-10 verdadero por consulta ---
lat_exactas_ms, top10_exactos = [], []
for q in consultas:
    t0 = time.perf_counter()
    idx = knn_exacto(q)
    lat_exactas_ms.append((time.perf_counter() - t0) * 1e3)
    top10_exactos.append(idx)
lat_exacta_ms = float(np.mean(lat_exactas_ms))
lat_exacta_std_ms = float(np.std(lat_exactas_ms))
print(f"\nkNN exacto (N={N:,}, k={K}): {lat_exacta_ms:.3f} ± {lat_exacta_std_ms:.3f} ms/consulta")
if t1_ref["latencia_exacta_ms_N100000_T1"] is not None:
    print(f"  [referencia T1] exacto a N=100000 medido en T1: "
          f"{t1_ref['latencia_exacta_ms_N100000_T1']:.3f} ms/consulta")

recall_at10_por_nprobe = {}
speedup_por_nprobe = {}
latencia_ivf_ms_por_nprobe = {}
latencia_ivf_std_ms_por_nprobe = {}
recall_individual_por_nprobe = {}
candidatos_promedio_por_nprobe = {}

for nprobe in NPROBES:
    lats_ms, recalls = [], []
    for q, idx_exacto in zip(consultas, top10_exactos):
        t0 = time.perf_counter()
        idx_ivf = knn_ivf(q, nprobe)
        lats_ms.append((time.perf_counter() - t0) * 1e3)
        ganados = len(set(idx_ivf.tolist()) & set(idx_exacto.tolist()))
        recalls.append(ganados / K)
    key = str(nprobe)
    recall_at10_por_nprobe[key] = float(np.mean(recalls))
    latencia_ivf_ms_por_nprobe[key] = float(np.mean(lats_ms))
    latencia_ivf_std_ms_por_nprobe[key] = float(np.std(lats_ms))
    speedup_por_nprobe[key] = lat_exacta_ms / float(np.mean(lats_ms))
    recall_individual_por_nprobe[key] = [float(r) for r in recalls]

    # candidatos reales visitados (diagnóstico, fuera del cronómetro)
    cand = []
    for q in consultas:
        sc = centroides @ q
        mej = np.argpartition(-sc, nprobe - 1)[:nprobe]
        cand.append(sum(int(celdas[c].size) for c in mej))
    candidatos_promedio_por_nprobe[key] = float(np.mean(cand))

# aritmética teórica de IVF (slides): comparaciones ≈ nlist + (N/nlist)·nprobe
comparaciones_teoricas_por_nprobe = {str(n): float(NLIST + (N / NLIST) * n) for n in NPROBES}
speedup_teorico_por_nprobe = {str(n): float(N / (NLIST + (N / NLIST) * n)) for n in NPROBES}

# Sanity check: nprobe = nlist debe reproducir el kNN exacto (nprobe=nlist ⇒ exacto)
sanity_recalls = []
for q, idx_exacto in list(zip(consultas, top10_exactos))[:5]:
    idx_ivf = knn_ivf(q, NLIST)
    sanity_recalls.append(len(set(idx_ivf.tolist()) & set(idx_exacto.tolist())) / K)
sanity_ok = bool(np.mean(sanity_recalls) == 1.0)

# --- tabla recall/speedup ---
print("\n=== Ejercicio 2.2 — tabla recall/speedup "
      f"(N={N:,}, nlist={NLIST}, k={K}, {N_CONSULTAS} consultas) ===")
cab = (f"{'nprobe':>7} | {'recall@10':>10} | {'lat IVF (ms)':>13} | {'speedup':>8} | "
       f"{'candidatos':>11} | {'comp.teóricas':>13} | {'speedup teór.':>13}")
print(cab)
print("-" * len(cab))
print(f"{'exacto':>7} | {1.0:>10.4f} | {lat_exacta_ms:>13.3f} | {1.0:>8.2f} | "
      f"{N:>11,} | {float(N):>13.1f} | {1.0:>13.2f}")
for n in NPROBES:
    key = str(n)
    print(f"{n:>7} | {recall_at10_por_nprobe[key]:>10.4f} | "
          f"{latencia_ivf_ms_por_nprobe[key]:>13.3f} | {speedup_por_nprobe[key]:>8.2f} | "
          f"{candidatos_promedio_por_nprobe[key]:>11,.0f} | "
          f"{comparaciones_teoricas_por_nprobe[key]:>13.1f} | "
          f"{speedup_teorico_por_nprobe[key]:>13.2f}")

# ----------------------------------------------------------------------------
# 6) Figura: curva recall@10 vs. speedup (triángulo recall/latencia)
# ----------------------------------------------------------------------------
sp_med = [speedup_por_nprobe[str(n)] for n in NPROBES]
rc_med = [recall_at10_por_nprobe[str(n)] for n in NPROBES]
sp_teo = [speedup_teorico_por_nprobe[str(n)] for n in NPROBES]
lat_ivf = [latencia_ivf_ms_por_nprobe[str(n)] for n in NPROBES]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

ax1.plot(sp_teo, rc_med, "o--", color="gray", alpha=0.75, ms=5,
         label=r"speedup teórico $N/(nlist+(N/nlist)\cdot nprobe)$ (cota superior)")
ax1.plot(sp_med, rc_med, "o-", color="tab:blue", lw=2, ms=8, label="speedup medido")
for n, s, r in zip(NPROBES, sp_med, rc_med):
    ax1.annotate(f"nprobe={n}", (s, r), textcoords="offset points", xytext=(8, 5), fontsize=9)
ax1.axhline(1.0, color="crimson", ls=":", lw=1.2, label="recall perfecto (= exacto)")
ax1.set_xlabel("Speedup de latencia vs. kNN exacto (×)")
ax1.set_ylabel("Recall@10")
ax1.set_title(f"Triángulo recall/latencia — Mini-IVF\n"
              f"(N={N:,}, nlist={NLIST}, k={K}, {N_CONSULTAS} consultas)")
ax1.set_xlim(0, max(max(sp_teo), max(sp_med)) * 1.12)
ax1.set_ylim(0, 1.06)
ax1.grid(alpha=0.3)
ax1.legend(loc="lower left", fontsize=8.5)

ax2.plot(NPROBES, rc_med, "o-", color="tab:green", label="recall@10")
try:
    ax2.set_xscale("log", base=2)
except TypeError:
    ax2.set_xscale("log", basex=2)
ax2.set_xticks(NPROBES)
ax2.get_xaxis().set_major_formatter(ScalarFormatter())
ax2.set_xlabel("nprobe (la perilla de IVF, escala log2)")
ax2.set_ylabel("Recall@10", color="tab:green")
ax2.tick_params(axis="y", labelcolor="tab:green")
ax2.set_ylim(0, 1.06)
ax2.grid(alpha=0.3)
ax2b = ax2.twinx()
ax2b.plot(NPROBES, lat_ivf, "s--", color="tab:orange", label="latencia IVF (ms)")
ax2b.axhline(lat_exacta_ms, color="black", ls=":", label="latencia exacta (ms)")
ax2b.set_ylabel("Latencia por consulta (ms)", color="tab:orange")
ax2b.tick_params(axis="y", labelcolor="tab:orange")
l1, lab1 = ax2.get_legend_handles_labels()
l2, lab2 = ax2b.get_legend_handles_labels()
ax2.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=8.5, framealpha=0.9)
ax2.set_title("El dial nprobe: recall ↑ y latencia ↑\n"
              "(los vecinos perdidos viven en fronteras entre celdas)")

fig.tight_layout()
fig.savefig("figura_recall_vs_speedup.png", dpi=150)
plt.close(fig)
print("\nFigura guardada: figura_recall_vs_speedup.png")

# ----------------------------------------------------------------------------
# 7) resultados.json
# ----------------------------------------------------------------------------
r1, s1 = recall_at10_por_nprobe["1"], speedup_por_nprobe["1"]
r16, s16 = recall_at10_por_nprobe["16"], speedup_por_nprobe["16"]
conclusion = (
    f"Con nprobe=1 el Mini-IVF es ~{s1:.1f}x más rápido que el kNN exacto "
    f"({latencia_ivf_ms_por_nprobe['1']:.3f} ms vs {lat_exacta_ms:.3f} ms) pero el recall@10 cae a "
    f"{r1:.3f}: se pierden los vecinos que caen en fronteras entre celdas. Subiendo la perilla, con "
    f"nprobe=16 el recall sube a {r16:.3f} y el speedup baja a ~{s16:.1f}x. El speedup medido queda por "
    f"debajo del teórico (p. ej. {speedup_teorico_por_nprobe['1']:.1f}x con nprobe=1) porque además de "
    f"las comparaciones el código paga concatenar índices y leer filas dispersas. Con "
    f"nprobe=nlist={NLIST} el IVF reproduce la búsqueda exacta (sanity check en 5 consultas: "
    f"{'OK' if sanity_ok else 'FALLÓ'}). HNSW ofrece el mismo dial (ef_search) con mejor curva."
)

resultados = {
    "tarea": "T2 - Parte 2: Mini-IVF con KMeans — recall@10 vs. speedup según nprobe",
    "setup": {
        "N": int(N), "D": int(D), "NLIST": int(NLIST), "k": int(K),
        "n_consultas": int(N_CONSULTAS), "nprobes": [int(n) for n in NPROBES],
        "kmeans": {"n_clusters": int(NLIST), "n_init": 3, "random_state": 7},
        "semilla_rng_datos": 7, "semilla_centros_temas": 2026,
        "tiempo_entrenamiento_kmeans_s": float(t_kmeans_s),
        "nota_consultas": ("50 consultas generadas con datos_realistas(50) inmediatamente después de la "
                           "base (misma rng global de semilla 7, como el flujo del notebook) y "
                           "normalizadas; comparten los 200 temas con la base."),
        "definicion_recall": "|top10_IVF ∩ top10_exacto| / 10, promediado sobre 50 consultas",
        "definicion_speedup": "latencia_media_exacta / latencia_media_IVF (misma máquina, mismas consultas)",
    },
    "tamano_medio_celda": tamano_medio_celda,
    "tamano_celdas": {
        "min": int(tamanos.min()), "max": int(tamanos.max()),
        "std": float(tamanos.std()), "lista": [int(x) for x in tamanos.tolist()],
    },
    "latencia_exacta_ms_media": lat_exacta_ms,
    "latencia_exacta_ms_std": lat_exacta_std_ms,
    "latencias_exactas_individuales_ms": [float(x) for x in lat_exactas_ms],
    "recall_at10_por_nprobe": recall_at10_por_nprobe,
    "speedup_por_nprobe": speedup_por_nprobe,
    "latencia_ivf_ms_por_nprobe": latencia_ivf_ms_por_nprobe,
    "latencia_ivf_ms_std_por_nprobe": latencia_ivf_std_ms_por_nprobe,
    "recall_at10_individual_por_nprobe": recall_individual_por_nprobe,
    "candidatos_promedio_por_nprobe": candidatos_promedio_por_nprobe,
    "comparaciones_teoricas_por_nprobe": comparaciones_teoricas_por_nprobe,
    "speedup_teorico_por_nprobe": speedup_teorico_por_nprobe,
    "referencia_T1": t1_ref,
    "sanity_check_nprobe_nlist": {
        "recall_promedio_5_consultas": float(np.mean(sanity_recalls)),
        "igual_a_exacto": sanity_ok,
    },
    "figura_recall_vs_speedup": "figura_recall_vs_speedup.png",
    "conclusion": conclusion,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("\nCifras principales:")
print(f"  tamano_medio_celda = {tamano_medio_celda:.1f}")
for n in NPROBES:
    key = str(n)
    print(f"  nprobe={n:>2}: recall@10 = {recall_at10_por_nprobe[key]:.4f} | "
          f"speedup = {speedup_por_nprobe[key]:.2f}x | "
          f"latencia IVF = {latencia_ivf_ms_por_nprobe[key]:.3f} ms")
print("Resultados guardados en resultados.json")
