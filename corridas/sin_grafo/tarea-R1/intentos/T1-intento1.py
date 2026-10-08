# -*- coding: utf-8 -*-
"""
T1 — Setup + Parte 1: kNN exacto, latencia vs. N
=================================================
- Setup (código dado): datos_realistas(N) genera N vectores d=128 alrededor de
  200 centros gaussianos fijos (semilla fija); normalizar() lleva cada vector
  a norma 1.
- knn_exacto(consulta, base, k): kNN exacto sobre vectores normalizados usando
  producto punto (≡ similitud coseno). Costo O(N·d).
- Se mide el tiempo medio de consulta para N = 10_000, 50_000, 100_000, 200_000
  (20 consultas por tamaño, media en ms), se grafica latencia vs. N y se
  verifica que el crecimiento es lineal (O(N·d)).

Salidas (carpeta actual): resultados.json, t1_latencia_vs_N.png
"""

import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Setup (código dado del curso)
# ----------------------------------------------------------------------
D = 128            # dimensión de los vectores
N_CENTROS = 200    # centros gaussianos fijos
SEMILLA_DATOS = 42
SEMILLA_CONSULTAS = 123


def normalizar(X):
    """Normaliza cada fila de X a norma 1 (float32)."""
    X = np.asarray(X, dtype=np.float32)
    normas = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(normas, 1e-12)


def datos_realistas(N, semilla=SEMILLA_DATOS, d=D, n_centros=N_CENTROS):
    """N vectores d-dimensionales 'realistas': mezcla de n_centros gaussianos fijos.

    Cada vector = centro elegido al azar + ruido gaussiano. Con la misma semilla
    los centros (y los datos) son deterministas.
    """
    rng = np.random.default_rng(semilla)
    centros = rng.normal(0.0, 1.0, size=(n_centros, d))        # 200 centros fijos
    asign = rng.integers(0, n_centros, size=N)                 # centro de cada vector
    X = centros[asign] + rng.normal(0.0, 0.3, size=(N, d))     # ruido alrededor del centro
    return X.astype(np.float32)


# ----------------------------------------------------------------------
# Ejercicio 1.1 — kNN exacto (vectores normalizados -> producto punto)
# ----------------------------------------------------------------------
def knn_exacto(consulta, base, k=10):
    """kNN exacto de `consulta` contra `base` (ambas normalizadas a norma 1).

    Similitud = producto punto (≡ coseno). Costo O(N·d): N productos punto
    de dimensión d. Devuelve (índices, scores) de los k mejores, ordenados desc.
    """
    scores = base @ consulta                              # O(N·d)
    k_eff = min(int(k), base.shape[0])
    idx = np.argpartition(-scores, k_eff - 1)[:k_eff]     # top-k sin ordenar
    idx = idx[np.argsort(-scores[idx])]                   # ordenar solo los k
    return idx, scores[idx]


# --- verificación rápida contra una ordenación naive (sanity check) ---
_base_chk = normalizar(datos_realistas(500, semilla=7))
_idx_chk, _sc_chk = knn_exacto(_base_chk[0], _base_chk, 10)
_idx_naive = np.argsort(-(_base_chk @ _base_chk[0]))[:10]
verificacion_knn = bool(np.array_equal(_idx_chk, _idx_naive))
print(f"Verificacion knn_exacto vs. argsort naive: {'OK' if verificacion_knn else 'FALLO'}")
del _base_chk, _idx_chk, _sc_chk, _idx_naive

# ----------------------------------------------------------------------
# Medición de latencia: N = 10k, 50k, 100k, 200k — 20 consultas por tamaño
# ----------------------------------------------------------------------
K = 10
N_VALORES = [10_000, 50_000, 100_000, 200_000]
N_CONSULTAS = 20   # repeticiones por tamaño (se reporta la media en ms)

consultas = normalizar(datos_realistas(N_CONSULTAS, semilla=SEMILLA_CONSULTAS))

latencias_ms = {}  # N -> array con los 20 tiempos individuales (ms)
for N in N_VALORES:
    base = normalizar(datos_realistas(N))                 # base de N vectores normalizados
    for _ in range(3):                                    # warmup (BLAS/caché, no se mide)
        knn_exacto(consultas[0], base, K)
    tiempos = np.empty(N_CONSULTAS, dtype=np.float64)
    for i in range(N_CONSULTAS):
        q = np.ascontiguousarray(consultas[i], dtype=np.float32)
        t0 = time.perf_counter()
        knn_exacto(q, base, K)
        t1 = time.perf_counter()
        tiempos[i] = (t1 - t0) * 1000.0
    latencias_ms[N] = tiempos
    print(f"N = {N:>7,d} | latencia media = {tiempos.mean():8.4f} ms "
          f"| std = {tiempos.std(ddof=1):7.4f} ms | N·d = {N * D:,d} mult./consulta")
    del base

# ----------------------------------------------------------------------
# Análisis: ¿el crecimiento es lineal (O(N·d))?
# ----------------------------------------------------------------------
Ns = np.array(N_VALORES, dtype=np.float64)
medias = np.array([latencias_ms[N].mean() for N in N_VALORES], dtype=np.float64)
stds = np.array([latencias_ms[N].std(ddof=1) for N in N_VALORES], dtype=np.float64)

# Ajuste lineal: latencia(ms) ≈ pendiente·N + intercepto
pendiente, intercepto = np.polyfit(Ns, medias, 1)
pred = pendiente * Ns + intercepto
ss_res = float(np.sum((medias - pred) ** 2))
ss_tot = float(np.sum((medias - medias.mean()) ** 2))
r2 = 1.0 - ss_res / ss_tot

# Factor de crecimiento lineal: crecimiento observado (200k vs 10k) dividido
# por el crecimiento ideal si fuera exactamente O(N) (= 20). ≈ 1 => lineal.
razon_observada = float(medias[-1] / medias[0])
razon_ideal = float(N_VALORES[-1] / N_VALORES[0])
factor_crecimiento_lineal = razon_observada / razon_ideal

# Factores entre tamaños consecutivos (observado vs. ideal lineal)
factores_consecutivos = {}
for i in range(1, len(N_VALORES)):
    n0, n1 = N_VALORES[i - 1], N_VALORES[i]
    factores_consecutivos[f"{n0}_a_{n1}"] = {
        "observado": float(medias[i] / medias[i - 1]),
        "ideal_lineal": float(n1 / n0),
    }

ms_por_vector = medias / Ns  # costo por vector (debe ser ≈ constante si O(N·d))
es_lineal = bool(r2 > 0.99 and 0.5 <= factor_crecimiento_lineal <= 2.0)

print("\n--- Resumen Parte 1: kNN exacto ---")
for N, m, s in zip(N_VALORES, medias, stds):
    print(f"N = {N:>7,d}: {m:.4f} ms de media ({s:.4f} ms std) "
          f"-> {m / N * 1e3:.3f} us por vector")
print(f"Ajuste lineal: latencia(ms) = {pendiente * 1e3:.4f} us/vector * N + {intercepto:.4f} ms")
print(f"R^2 del ajuste lineal: {r2:.6f}")
print(f"Crecimiento observado 200k/10k = {razon_observada:.3f} "
      f"(ideal lineal = {razon_ideal:.0f}) -> factor_crecimiento_lineal = {factor_crecimiento_lineal:.4f}")
print(f"¿Crecimiento lineal (O(N·d))? {'SI' if es_lineal else 'REVISAR'}")

# ----------------------------------------------------------------------
# Figura: latencia vs. N
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

ax = axes[0]
ax.errorbar(Ns, medias, yerr=stds, fmt="o-", color="#1f77b4", capsize=3,
            label="latencia media medida (20 consultas)")
xs = np.linspace(0.0, Ns.max() * 1.06, 200)
ax.plot(xs, pendiente * xs + intercepto, "--", color="crimson",
        label=f"ajuste lineal: {pendiente * 1e3:.3f} us/vector, R$^2$ = {r2:.4f}")
ax.set_xlabel("N (vectores en la base)")
ax.set_ylabel("latencia media por consulta (ms)")
ax.set_title(f"kNN exacto: latencia vs. N  (d = {D}, k = {K})")
ax.set_xlim(left=0)
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(Ns, ms_por_vector * 1e3, "s-", color="#2ca02c")
ax.set_xlabel("N (vectores en la base)")
ax.set_ylabel("latencia / N  (us por vector)")
ax.set_title("Costo por vector ≈ constante  =>  O(N·d)")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("t1_latencia_vs_N.png", dpi=150)
plt.close(fig)
print("\nFigura guardada: t1_latencia_vs_N.png")

# ----------------------------------------------------------------------
# resultados.json
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T1_setup_knn_exacto_latencia_vs_N",
    "parametros": {
        "d": int(D),
        "n_centros_gaussianos": int(N_CENTROS),
        "semilla_datos": int(SEMILLA_DATOS),
        "semilla_consultas": int(SEMILLA_CONSULTAS),
        "k": int(K),
        "valores_N": [int(n) for n in N_VALORES],
        "n_consultas_por_N": int(N_CONSULTAS),
    },
    "latencia_media_ms_por_N": {str(int(N)): float(latencias_ms[N].mean()) for N in N_VALORES},
    "latencia_std_ms_por_N": {str(int(N)): float(latencias_ms[N].std(ddof=1)) for N in N_VALORES},
    "latencias_individuales_ms_por_N": {
        str(int(N)): [float(t) for t in latencias_ms[N]] for N in N_VALORES
    },
    "latencia_por_vector_us_por_N": {
        str(int(N)): float(latencias_ms[N].mean() / N * 1e3) for N in N_VALORES
    },
    "multiplicaciones_por_consulta_por_N": {str(int(N)): int(N * D) for N in N_VALORES},
    "factor_crecimiento_lineal": float(factor_crecimiento_lineal),
    "definicion_factor_crecimiento_lineal": (
        "(latencia_200k / latencia_10k) / (N_200k / N_10k); ≈ 1 confirma crecimiento lineal O(N·d)"
    ),
    "razon_latencia_200k_vs_10k": razon_observada,
    "razon_ideal_lineal_200k_vs_10k": razon_ideal,
    "factores_crecimiento_consecutivos": factores_consecutivos,
    "ajuste_lineal": {
        "pendiente_ms_por_vector": float(pendiente),
        "pendiente_us_por_vector": float(pendiente * 1e3),
        "intercepto_ms": float(intercepto),
        "r_cuadrado": float(r2),
    },
    "crecimiento_lineal_verificado": es_lineal,
    "verificacion_knn_vs_argsort": verificacion_knn,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)
print("Resultados guardados: resultados.json")
