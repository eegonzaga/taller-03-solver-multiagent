# -*- coding: utf-8 -*-
"""
T1 — Setup + Parte 1: kNN exacto, latencia vs. N
MMIA 6013 · Semana 2 · S2·MAR — ANN: mide el trade-off recall / latencia

Ejercicio 1.1:
  * Setup dado por el profesor: datos_realistas (200 centros gaussianos fijos,
    D=128, rng semilla 7) y normalizar a norma 1.
  * knn_exacto(consulta, base, k) con producto punto sobre vectores normalizados
    (dot = coseno).
  * Tiempo medio de consulta para N = 10k, 50k, 100k, 200k
    (varias consultas por N para promediar).
  * Gráfica latencia vs. N (PNG) verificando el crecimiento lineal O(N·d).
"""

import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================================================================
# 0. Setup (código dado — sin cambios)
# ======================================================================
rng = np.random.default_rng(7)
D = 128  # dimensión de los embeddings sintéticos
_CENTROS = np.random.default_rng(2026).normal(size=(200, D))  # "temas" fijos


def datos_realistas(n, d=D):
    """Embeddings sintéticos CON estructura: mezcla de 'temas' gaussianos fijos.
    Los embeddings reales viven agrupados por tema — crucial para que IVF
    tenga sentido (con ruido uniforme, ningún índice por clusters funciona).
    Base y consultas comparten los mismos temas, como en un corpus real."""
    asignacion = rng.integers(0, len(_CENTROS), size=n)
    X = _CENTROS[asignacion, :d] + 1.5 * rng.normal(size=(n, d))
    return X.astype(np.float32)


def normalizar(X):
    """Vectores a norma 1: así el producto punto es el coseno (sesión 06)."""
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


# ======================================================================
# 1. Ejercicio 1.1 — kNN exacto (dot sobre vectores normalizados)
# ======================================================================
def knn_exacto(consulta, base, k):
    """k vecinos más cercanos por similitud coseno.

    consulta : (d,) float32 con norma 1
    base     : (N, d) float32 con filas de norma 1
    k        : int, 1 <= k < N

    Devuelve (indices, similitudes): los k mayores cosenos, ordenados desc.
    Costo O(N·d): un producto matriz-vector + una selección top-k O(N).
    """
    similitudes = base @ consulta                         # (N,) cosenos
    indices = np.argpartition(-similitudes, k)[:k]        # top-k sin orden
    indices = indices[np.argsort(-similitudes[indices])]  # orden descendente
    return indices, similitudes[indices]


# --- parámetros del experimento (fijos según el enunciado) ---
K = 10                        # top-k (la Parte 2 mide recall@10)
NS = [10_000, 50_000, 100_000, 200_000]
N_MAX = max(NS)
N_CONSULTAS = 30              # varias consultas por N para promediar
N_WARMUP = 3                  # llamadas de calentamiento (no se cronometran)

# --- datos: base de N_MAX y consultas (mismos 'temas'), todos a norma 1 ---
base_total = normalizar(datos_realistas(N_MAX, D))
consultas = normalizar(datos_realistas(N_CONSULTAS, D))

# --- verificación de corrección de knn_exacto contra argsort completo ---
base_check = base_total[:1_000]
idx_fast, sim_fast = knn_exacto(consultas[0], base_check, K)
sim_full = base_check @ consultas[0]
idx_ref = np.argsort(-sim_full)[:K]
topk_correcto = bool(np.array_equal(idx_fast, idx_ref)) and bool(
    np.allclose(sim_fast, sim_full[idx_ref])
)

# ======================================================================
# 2. Medición: tiempo medio de consulta por N
# ======================================================================
lat_media_ms, lat_std_ms, lat_por_consulta_ms = {}, {}, {}
for N in NS:
    base_N = np.ascontiguousarray(base_total[:N])   # prefijo de la base
    for _ in range(N_WARMUP):                       # calentamiento (caché, etc.)
        knn_exacto(consultas[0], base_N, K)
    tiempos = np.empty(N_CONSULTAS, dtype=np.float64)
    for i in range(N_CONSULTAS):
        q = consultas[i]
        t0 = time.perf_counter()
        knn_exacto(q, base_N, K)
        t1 = time.perf_counter()
        tiempos[i] = (t1 - t0) * 1e3                # ms
    lat_por_consulta_ms[str(N)] = tiempos.tolist()
    lat_media_ms[str(N)] = float(tiempos.mean())
    lat_std_ms[str(N)] = float(tiempos.std(ddof=1))

# ======================================================================
# 3. Cifras: factor de escalamiento y ajuste lineal (verificación O(N·d))
# ======================================================================
lat_arr = np.array([lat_media_ms[str(N)] for N in NS], dtype=np.float64)
std_arr = np.array([lat_std_ms[str(N)] for N in NS], dtype=np.float64)
Ns_arr = np.array(NS, dtype=np.float64)

lat_ref = float(lat_arr[0])
factor_escalamiento_N = {str(N): float(l / lat_ref) for N, l in zip(NS, lat_arr)}
factor_teorico_lineal_N = {str(N): float(N / NS[0]) for N in NS}
razon_medida_teorica = {
    str(N): float(factor_escalamiento_N[str(N)] / factor_teorico_lineal_N[str(N)])
    for N in NS
}
factor_consecutivo = {}
for a, b in zip(NS[:-1], NS[1:]):
    factor_consecutivo[f"{a}_a_{b}"] = {
        "medido": float(lat_media_ms[str(b)] / lat_media_ms[str(a)]),
        "teorico_lineal": float(b / a),
    }

pendiente_ms_por_vector, intercepto_ms = np.polyfit(Ns_arr, lat_arr, 1)
pred = pendiente_ms_por_vector * Ns_arr + intercepto_ms
ss_res = float(np.sum((lat_arr - pred) ** 2))
ss_tot = float(np.sum((lat_arr - lat_arr.mean()) ** 2))
r2_lineal = 1.0 - ss_res / ss_tot

ms_por_10k = {str(N): float(lat_media_ms[str(N)] / (N / 10_000)) for N in NS}
throughput_qps = {str(N): float(1000.0 / lat_media_ms[str(N)]) for N in NS}

# ======================================================================
# 4. Figura: latencia vs. N + verificación de linealidad
# ======================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

xs = np.linspace(0.0, float(N_MAX), 200)
ax1.errorbar(Ns_arr / 1000.0, lat_arr, yerr=std_arr, fmt="o-",
             color="#1f77b4", capsize=4, lw=1.8,
             label="latencia media medida (±1 DE)")
ax1.plot(xs / 1000.0, pendiente_ms_por_vector * xs + intercepto_ms, "--",
         color="#d62728", lw=1.5,
         label=f"ajuste lineal (R² = {r2_lineal:.4f})")
ax1.set_xlabel("N — vectores en la base (miles)")
ax1.set_ylabel("latencia media por consulta (ms)")
ax1.set_title(f"kNN exacto (k={K}, d={D}): latencia vs. N")
ax1.set_xticks([n / 1000 for n in NS])
ax1.legend()
ax1.grid(alpha=0.3)

lat_norm = lat_arr / (Ns_arr / 10_000.0)
ax2.plot(Ns_arr / 1000.0, lat_norm, "s-", color="#2ca02c", lw=1.8)
ax2.set_xlabel("N — vectores en la base (miles)")
ax2.set_ylabel("latencia / (N/10k)  (ms por cada 10k vectores)")
ax2.set_title("Verificación de linealidad: ms por 10k vectores ≈ constante")
ax2.set_xticks([n / 1000 for n in NS])
ax2.grid(alpha=0.3)

fig.suptitle("Parte 1 — kNN exacto: el costo O(N·d), medido", y=1.02)
fig.tight_layout()
fig.savefig("parte1_latencia_vs_N.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ======================================================================
# 5. Guardar resultados.json e imprimir cifras principales
# ======================================================================
resultados = {
    "subtarea": "T1 — Setup + Parte 1: kNN exacto, latencia vs. N",
    "parametros": {
        "D": int(D),
        "k": int(K),
        "N_valores": [int(n) for n in NS],
        "n_consultas_por_N": int(N_CONSULTAS),
        "warmups_por_N": int(N_WARMUP),
        "semilla_rng": 7,
        "semilla_centros": 2026,
        "n_centros": int(len(_CENTROS)),
        "metrica": "coseno (producto punto sobre vectores con norma 1)",
    },
    "verificacion_topk_correcta_vs_argsort": topk_correcto,
    "latencia_media_ms_por_N": lat_media_ms,
    "latencia_std_ms_por_N": lat_std_ms,
    "latencia_por_consulta_ms": lat_por_consulta_ms,
    "factor_escalamiento_N": factor_escalamiento_N,
    "factor_teorico_lineal_N": factor_teorico_lineal_N,
    "razon_medida_vs_teorica": razon_medida_teorica,
    "factor_escalamiento_consecutivo": factor_consecutivo,
    "ajuste_lineal": {
        "pendiente_ms_por_vector": float(pendiente_ms_por_vector),
        "intercepto_ms": float(intercepto_ms),
        "r2": float(r2_lineal),
    },
    "latencia_ms_por_10k_vectores": ms_por_10k,
    "throughput_consultas_por_seg": throughput_qps,
    "figura": "parte1_latencia_vs_N.png",
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("=" * 66)
print("T1 · Parte 1 — kNN exacto: latencia vs. N")
print("=" * 66)
print(f"Verificación top-{K} contra argsort completo: "
      f"{'OK' if topk_correcto else 'FALLÓ'}")
print(f"\nLatencia media por consulta (k={K}, d={D}, "
      f"{N_CONSULTAS} consultas por N):")
for N in NS:
    print(f"  N = {N:>7,} : {lat_media_ms[str(N)]:9.4f} ms  "
          f"(±{lat_std_ms[str(N)]:.4f} ms)  "
          f"→ {throughput_qps[str(N)]:8.1f} consultas/s")
print("\nFactor de escalamiento medido (vs. N=10k) frente al teórico lineal N/10k:")
for N in NS:
    print(f"  N = {N:>7,} : medido ×{factor_escalamiento_N[str(N)]:6.2f}   "
          f"teórico ×{factor_teorico_lineal_N[str(N)]:6.2f}   "
          f"razón {razon_medida_teorica[str(N)]:.3f}")
print("\nAjuste lineal  latencia(ms) = a·N + b:")
print(f"  a = {pendiente_ms_por_vector * 1e3:.4f} µs por vector   "
      f"b = {intercepto_ms:.4f} ms   R² = {r2_lineal:.6f}")
print("\nConclusión: la latencia crece linealmente con N (costo O(N·d) medido).")
print("Con 10M de vectores y tráfico real, esto muere. Entra IVF.")
print("\nResultados guardados en resultados.json · figura en parte1_latencia_vs_N.png")
