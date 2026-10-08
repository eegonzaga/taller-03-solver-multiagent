# -*- coding: utf-8 -*-
"""
T2 — Parte 2: Mini-IVF con KMeans, recall@10 vs. speedup según nprobe.

Receta (enunciado):
  (1) offline: KMeans (n_init=3, random_state=7) agrupa la base normalizada
      en NLIST=64 celdas; se normalizan los centroides y se construyen las
      celdas con los índices globales de sus miembros;
  (2) en consulta: scores contra los centroides -> nprobe celdas más
      cercanas -> kNN exacto dentro de esas celdas (índices globales).

Se mide, sobre 50 consultas y para nprobe en {1, 2, 4, 8, 16}:
  - recall@10 contra el kNN exacto (referencia de T1),
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
    """Vectores a norma 1: así el producto punto es el coseno (sesión 06)."""
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


# ----------------------------------------------------------------------
# 1. Parámetros (exactamente los del enunciado)
# ----------------------------------------------------------------------
N = 100_000            # vectores de la base
NLIST = 64             # celdas del IVF
K = 10                 # vecinos por consulta (recall@10)
N_CONSULTAS = 50       # consultas de evaluación
NPROBES = [1, 2, 4, 8, 16]
KMEANS_N_INIT = 3
KMEANS_RANDOM_STATE = 7
REP = 5                # repeticiones por consulta al cronometrar

# ----------------------------------------------------------------------
# 2. Base normalizada + 50 consultas (mismo generador: comparten temas)
# ----------------------------------------------------------------------
base = normalizar(datos_realistas(N))
consultas = normalizar(datos_realistas(N_CONSULTAS))
print(f"Base: {base.shape[0]} vectores x {base.shape[1]} dims (float32), "
      f"norma media = {float(np.linalg.norm(base, axis=1).mean()):.6f}")
print(f"Consultas de evaluación: {consultas.shape[0]} "
      f"(generadas aparte; ninguna fila de la base)")

# ----------------------------------------------------------------------
# 3. IVF offline: KMeans + centroides normalizados + celdas
# ----------------------------------------------------------------------
t0 = time.perf_counter()
km = KMeans(n_clusters=NLIST, n_init=KMEANS_N_INIT,
            random_state=KMEANS_RANDOM_STATE).fit(base)
t_kmeans_s = time.perf_counter() - t0

centroides = normalizar(km.cluster_centers_.astype(np.float32))  # (NLIST, D)
celdas = [np.where(km.labels_ == c)[0] for c in range(NLIST)]    # índices globales
tamanos = np.array([len(c) for c in celdas], dtype=np.int64)
tamano_medio_celda = float(tamanos.mean())

print(f"IVF entrenado: {NLIST} celdas, tamaño medio {tamano_medio_celda:.0f} "
      f"(min={int(tamanos.min())}, max={int(tamanos.max())}, "
      f"std={float(tamanos.std()):.1f}); KMeans: {t_kmeans_s:.1f} s, "
      f"inercia={float(km.inertia_):.2f}, iter={int(km.n_iter_)}")

# ----------------------------------------------------------------------
# 4. kNN exacto (referencia T1) y knn_ivf (ejercicio 2.1)
# ----------------------------------------------------------------------
def knn_exacto(consulta, base_mat, k=K):
    """kNN exacto por producto punto (vectores normalizados -> coseno).
    Devuelve índices globales de los k mejores, ordenados por score desc."""
    scores = base_mat @ consulta
    if k < scores.shape[0]:
        idx = np.argpartition(scores, -k)[-k:]
    else:
        idx = np.arange(scores.shape[0])
    return idx[np.argsort(scores[idx])[::-1]]


def knn_ivf(consulta, nprobe, k=K):
    """IVF: 1) scores contra los NLIST centroides -> nprobe celdas más
    cercanas; 2) kNN exacto solo dentro de esas celdas (índices GLOBALES)."""
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
verif_nlist = all(
    set(knn_ivf(q, NLIST, K).tolist()) == set(knn_exacto(q, base, K).tolist())
    for q in consultas
)
print(f"Verificación top-k (argpartition vs argsort completo): {verif_topk}")
print(f"Control nprobe=nlist={NLIST} == exacto en las {N_CONSULTAS} consultas: "
      f"{verif_nlist}")

# ----------------------------------------------------------------------
# 5. Ground truth exacto y latencia de referencia del exacto
# ----------------------------------------------------------------------
exactos_set = [set(knn_exacto(q, base, K).tolist()) for q in consultas]

_ = knn_exacto(consultas[0], base, K)   # calentamiento (fuera del cronómetro)
_ = knn_ivf(consultas[0], 4, K)

lat_exacta_q = []
for q in consultas:
    t0 = time.perf_counter()
    for _ in range(REP):
        knn_exacto(q, base, K)
    lat_exacta_q.append((time.perf_counter() - t0) / REP)
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
        # pool de candidatos que se abre (misma lógica que knn_ivf, sin cronometrar)
        sc = centroides @ q
        elegidas = (np.argpartition(sc, -nprobe)[-nprobe:]
                    if nprobe < NLIST else np.arange(NLIST))
        pools_q.append(int(sum(int(celdas[c].size) for c in elegidas)))
        # latencia
        t0 = time.perf_counter()
        for _ in range(REP):
            idx = knn_ivf(q, nprobe, K)
        lats_q.append((time.perf_counter() - t0) / REP)
        # recall@10 contra el exacto
        recs_q.append(len(set(idx.tolist()) & exactos_set[i]) / K)
    rec_medio = float(np.mean(recs_q))
    lat_ms = float(np.mean(lats_q) * 1000.0)
    lat_std = float(np.std(lats_q) * 1000.0)
    sp = lat_exacta_media_ms / lat_ms           # speedup = razón de medias
    recalls.append(rec_medio)
    lat_ivf_ms.append(lat_ms)
    lat_ivf_std.append(lat_std)
    speedups.append(sp)
    pools_medios.append(float(np.mean(pools_q)))
    lat_por_consulta[str(nprobe)] = [float(x * 1000.0) for x in lats_q]
    print(f"nprobe={nprobe:>2}: recall@10={rec_medio:.3f} | "
          f"latencia={lat_ms:.3f} ms | speedup={sp:.1f}x | "
          f"candidatos medios={float(np.mean(pools_q)):.0f}")

# ----------------------------------------------------------------------
# 7. Tabla resumen
# ----------------------------------------------------------------------
tabla_df = pd.DataFrame({
    "nprobe": NPROBES,
    "recall@10": [round(r, 4) for r in recalls],
    "latencia_ivf_ms": [round(l, 4) for l in lat_ivf_ms],
    "speedup_x": [round(s, 2) for s in speedups],
    "candidatos_medios": [int(round(p)) for p in pools_medios],
})
print("\nTabla recall/latencia (N=100k, NLIST=64, k=10, 50 consultas):")
print(tabla_df.to_string(index=False))
print(f"Referencia exacto: recall@10 = 1.000 | latencia = "
      f"{lat_exacta_media_ms:.3f} ms | speedup = 1.00x")

# ----------------------------------------------------------------------
# 8. Gráfica recall vs. speedup (PNG)
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

# Panel 1 (el pedido): recall@10 vs. speedup, cada punto = un nprobe
ax1.plot(speedups, recalls, "o-", color="#1f77b4", lw=1.8, ms=7, zorder=3)
offsets = {1: (8, 4), 2: (8, 4), 4: (8, 4), 8: (8, -13), 16: (8, 4)}
for nb, s, r in zip(NPROBES, speedups, recalls):
    dx, dy = offsets.get(nb, (8, 4))
    ax1.annotate(f"nprobe={nb}", (s, r), textcoords="offset points",
                 xytext=(dx, dy), fontsize=9)
ax1.scatter([1.0], [1.0], marker="*", s=230, color="#d62728", zorder=4,
            label="kNN exacto (speedup=1, recall=1)")
ax1.axhline(1.0, color="gray", ls="--", lw=1)
ax1.set_xlabel("Speedup de latencia vs. kNN exacto (x)")
ax1.set_ylabel("Recall@10")
ax1.set_title(f"Mini-IVF: recall@10 vs. speedup\n"
              f"(N={N}, NLIST={NLIST}, k={K}, {N_CONSULTAS} consultas)")
ax1.set_ylim(0, 1.08)
ax1.grid(alpha=0.3)
ax1.legend(loc="lower right", fontsize=8)

# Panel 2: el dial nprobe — recall sube y latencia también
ax2.plot(NPROBES, recalls, "o-", color="#2ca02c", label="recall@10")
ax2.set_xticks(NPROBES)
ax2.set_xticklabels([str(x) for x in NPROBES])
ax2.set_xlabel("nprobe (celdas abiertas por consulta)")
ax2.set_ylabel("Recall@10", color="#2ca02c")
ax2.tick_params(axis="y", labelcolor="#2ca02c")
ax2.set_ylim(0, 1.08)
ax2.grid(alpha=0.3)
ax2b = ax2.twinx()
ax2b.plot(NPROBES, lat_ivf_ms, "s--", color="#ff7f0e", label="latencia IVF (ms)")
ax2b.set_ylabel("Latencia IVF (ms/consulta)", color="#ff7f0e")
ax2b.tick_params(axis="y", labelcolor="#ff7f0e")
lineas1, etiq1 = ax2.get_legend_handles_labels()
lineas2, etiq2 = ax2b.get_legend_handles_labels()
ax2.legend(lineas1 + lineas2, etiq1 + etiq2, loc="center right", fontsize=8)
ax2.set_title("El dial nprobe: recall sube, latencia también")

fig.tight_layout()
plt.savefig("T2_parte2_recall_vs_speedup.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------
# 9. Referencia cruzada con T1 (si está disponible)
# ----------------------------------------------------------------------
t1 = None
try:
    with open("entradas/T1.json", "r", encoding="utf-8") as f:
        t1 = json.load(f)
except Exception:
    t1 = None

referencia_t1 = {}
if isinstance(t1, dict):
    v = t1.get("latencia_ms_por_10k_vectores")
    if isinstance(v, (int, float)):
        referencia_t1["latencia_ms_por_10k_vectores_T1"] = float(v)
        referencia_t1["estimacion_lineal_exacto_N100k_ms_T1"] = float(v) * (N / 10_000.0)
        print(f"\nReferencia T1: {float(v):.4f} ms por 10k vectores -> "
              f"estimación lineal para N=100k: {float(v) * 10.0:.3f} ms "
              f"(medido aquí: {lat_exacta_media_ms:.3f} ms)")

# ----------------------------------------------------------------------
# 10. Guardar todas las cifras en resultados.json
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T2 - Parte 2: Mini-IVF con KMeans, recall@10 vs. speedup",
    "parametros": {
        "N": int(N),
        "D": int(D),
        "NLIST": int(NLIST),
        "k": int(K),
        "nprobe_valores": [int(x) for x in NPROBES],
        "n_consultas": int(N_CONSULTAS),
        "kmeans_n_init": int(KMEANS_N_INIT),
        "kmeans_random_state": int(KMEANS_RANDOM_STATE),
        "repeticiones_cronometro_por_consulta": int(REP),
    },
    "tamano_medio_celda": tamano_medio_celda,
    "tamanos_celda": [int(x) for x in tamanos.tolist()],
    "tamanos_celda_min": int(tamanos.min()),
    "tamanos_celda_max": int(tamanos.max()),
    "tamanos_celda_std": float(tamanos.std()),
    "kmeans_inercia": float(km.inertia_),
    "kmeans_iteraciones": int(km.n_iter_),
    "kmeans_tiempo_entrenamiento_s": float(t_kmeans_s),
    "recall10_por_nprobe": {str(nb): float(r) for nb, r in zip(NPROBES, recalls)},
    "speedup_por_nprobe": {str(nb): float(s) for nb, s in zip(NPROBES, speedups)},
    "latencia_ivf_ms_por_nprobe": {str(nb): float(l) for nb, l in zip(NPROBES, lat_ivf_ms)},
    "latencia_ivf_std_ms_por_nprobe": {str(nb): float(s) for nb, s in zip(NPROBES, lat_ivf_std)},
    "candidatos_medios_por_nprobe": {str(nb): float(p) for nb, p in zip(NPROBES, pools_medios)},
    "latencia_exacta_media_ms": lat_exacta_media_ms,
    "latencia_exacta_std_ms": lat_exacta_std_ms,
    "latencia_exacta_por_consulta_ms": [float(x * 1000.0) for x in lat_exacta_q],
    "latencia_ivf_por_consulta_ms": lat_por_consulta,
    "verificacion_topk_vs_argsort": verif_topk,
    "verificacion_nprobe_nlist_igual_exacto": bool(verif_nlist),
    "referencia_T1": referencia_t1,
    "tabla": [
        {
            "nprobe": int(nb),
            "recall10": float(r),
            "latencia_ivf_ms": float(l),
            "speedup": float(s),
            "candidatos_medios": float(p),
        }
        for nb, r, l, s, p in zip(NPROBES, recalls, lat_ivf_ms, speedups, pools_medios)
    ],
    "figura": "T2_parte2_recall_vs_speedup.png",
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("\nCifras principales:")
print(f"  tamano_medio_celda = {tamano_medio_celda:.2f}")
for nb, r, s in zip(NPROBES, recalls, speedups):
    print(f"  nprobe={nb:>2}: recall@10={r:.4f}  speedup={s:.2f}x")
print("Archivos generados: resultados.json, T2_parte2_recall_vs_speedup.png")
