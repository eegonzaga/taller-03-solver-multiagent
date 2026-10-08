# -*- coding: utf-8 -*-
"""
T1 — Parte 1 — kNN exacto: latencia vs. N
=========================================
- Setup dado: datos_realistas con 200 centros gaussianos fijos, D=128,
  vectores normalizados a norma 1.
- knn_exacto(consulta, base, k) por producto punto (vectores normalizados → dot).
- Latencia media de consulta para N = 10_000, 50_000, 100_000, 200_000
  (las MISMAS 20 consultas normalizadas, k=10).
- Verificación del escalamiento lineal O(N·d): factor de crecimiento de la
  latencia entre valores de N comparado con el factor de N.

Salidas: resultados.json, t1_parte1_latencia_vs_N.png
"""

import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Parámetros fijados por el enunciado
# ----------------------------------------------------------------------
D = 128                       # dimensión de los vectores
K = 10                        # vecinos a recuperar
N_VALORES = [10_000, 50_000, 100_000, 200_000]
N_CONSULTAS = 20              # consultas promediadas por tamaño de N
N_CENTROS = 200               # centros gaussianos fijos del setup
SEMILLA = 42
RUIDO_STD = 0.3               # dispersión alrededor del centro
ARCHIVO_RESULTADOS = "resultados.json"
ARCHIVO_FIGURA = "t1_parte1_latencia_vs_N.png"

rng = np.random.default_rng(SEMILLA)

# ----------------------------------------------------------------------
# Setup dado: datos_realistas (200 centros gaussianos fijos, D=128)
# ----------------------------------------------------------------------
centros = rng.normal(0.0, 1.0, size=(N_CENTROS, D)).astype(np.float32)

def normalizar(X):
    """Normaliza las filas de X a norma 1 (evita división por 0)."""
    normas = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(normas, 1e-12)).astype(np.float32)

def datos_realistas(n):
    """n vectores 'realistas': centro gaussiano fijo elegido al azar + ruido gaussiano."""
    idx = rng.integers(0, N_CENTROS, size=n)
    return (centros[idx] + rng.normal(0.0, RUIDO_STD, size=(n, D))).astype(np.float32)

# ----------------------------------------------------------------------
# Ejercicio 1.1 — kNN exacto (vectores normalizados → usar dot)
# ----------------------------------------------------------------------
def knn_exacto(consulta, base, k):
    """
    k vecinos más cercanos por similitud coseno (dot sobre vectores norma 1).
    Costo: O(N·d) — un producto punto de dimensión d contra cada uno de los N vectores.
    Devuelve (índices, scores) ordenados de mayor a menor similitud.
    """
    scores = base @ consulta                      # O(N·d)
    idx = np.argpartition(-scores, k - 1)[:k]     # top-k sin ordenar
    idx = idx[np.argsort(-scores[idx])]           # ordenar los k mejores
    return idx, scores[idx]

# ----------------------------------------------------------------------
# Base máxima (generada UNA vez) y las MISMAS consultas para todos los N
# ----------------------------------------------------------------------
N_MAX = max(N_VALORES)
base_max = normalizar(datos_realistas(N_MAX))          # (N_MAX, D), norma 1
consultas = normalizar(datos_realistas(N_CONSULTAS))   # (20, D), norma 1

# Sanidad: con normas 1, los scores (dots) deben quedar en [-1, 1]
_, scores_check = knn_exacto(consultas[0], base_max, K)
score_max = float(np.max(scores_check))

# ----------------------------------------------------------------------
# Medición: latencia media por consulta para cada N
# ----------------------------------------------------------------------
lat_media_ms = {}
lat_std_ms = {}
for n in N_VALORES:
    base_n = base_max[:n]  # prefijo de la misma base (mismos datos, subconjunto)
    knn_exacto(consultas[0], base_n, K)  # calentamiento (no se mide)
    tiempos_ms = np.empty(N_CONSULTAS, dtype=np.float64)
    for i in range(N_CONSULTAS):
        q = consultas[i]
        t0 = time.perf_counter()
        knn_exacto(q, base_n, K)
        tiempos_ms[i] = (time.perf_counter() - t0) * 1000.0
    lat_media_ms[n] = float(np.mean(tiempos_ms))
    lat_std_ms[n] = float(np.std(tiempos_ms))
    print(f"N = {n:>7,d} | latencia media = {lat_media_ms[n]:8.3f} ms "
          f"(± {lat_std_ms[n]:.3f} ms) | {lat_media_ms[n] / n * 1000.0:.4f} µs/vector")

# ----------------------------------------------------------------------
# Factor de crecimiento entre valores de N (verificación de linealidad O(N·d))
# ----------------------------------------------------------------------
factor_escalamiento = {}
pares = []
for i in range(len(N_VALORES) - 1):
    n1, n2 = N_VALORES[i], N_VALORES[i + 1]
    f_n = float(n2) / float(n1)
    f_lat = lat_media_ms[n2] / lat_media_ms[n1]
    factor_escalamiento[f"{n1}_a_{n2}"] = {
        "factor_N_teorico": f_n,
        "factor_latencia_medido": float(f_lat),
        "razon_medido_teorico": float(f_lat / f_n),
    }
    pares.append((n1, n2, f_n, f_lat))

n1, n2 = N_VALORES[0], N_VALORES[-1]
f_n_global = float(n2) / float(n1)
f_lat_global = lat_media_ms[n2] / lat_media_ms[n1]
factor_escalamiento[f"{n1}_a_{n2}_global"] = {
    "factor_N_teorico": f_n_global,
    "factor_latencia_medido": float(f_lat_global),
    "razon_medido_teorico": float(f_lat_global / f_n_global),
}

# Ajuste lineal latencia ≈ pendiente·N + intercepto (evidencia de la FORMA lineal)
Ns_arr = np.array(N_VALORES, dtype=np.float64)
lats_arr = np.array([lat_media_ms[n] for n in N_VALORES], dtype=np.float64)
pendiente, intercepto = np.polyfit(Ns_arr, lats_arr, 1)
pred = pendiente * Ns_arr + intercepto
ss_res = float(np.sum((lats_arr - pred) ** 2))
ss_tot = float(np.sum((lats_arr - np.mean(lats_arr)) ** 2))
r2 = 1.0 - ss_res / ss_tot

razones = [v["razon_medido_teorico"] for v in factor_escalamiento.values()]
es_lineal = bool(all(0.8 <= r <= 1.25 for r in razones) and r2 > 0.99)
conclusion = ("Escalamiento LINEAL O(N·d): el factor de latencia sigue al factor de N"
              if es_lineal else
              "Desviación de la linealidad (posible ruido de medición en esta máquina)")

# ----------------------------------------------------------------------
# Figura: latencia vs. N + factores de crecimiento
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

ax1.plot(Ns_arr, lats_arr, "o-", color="tab:blue", lw=2, ms=7,
         label="kNN exacto (medido)")
ax1.plot(Ns_arr, lats_arr[0] * Ns_arr / Ns_arr[0], "--", color="tab:orange",
         lw=1.8, label="Referencia lineal O(N·d)")
for x, y in zip(Ns_arr, lats_arr):
    ax1.annotate(f"{y:.2f} ms", (x, y), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=9)
ax1.set_xlabel("N (vectores en la base)")
ax1.set_ylabel("Latencia media por consulta (ms)")
ax1.set_title(f"kNN exacto: latencia vs. N (d={D}, k={K}, "
              f"media de {N_CONSULTAS} consultas)")
ax1.set_xticks(Ns_arr)
ax1.set_xticklabels([f"{int(n) // 1000}k" for n in N_VALORES])
ax1.grid(alpha=0.3)
ax1.legend()

xpos = np.arange(len(pares))
ancho = 0.38
ax2.bar(xpos - ancho / 2, [p[2] for p in pares], ancho,
        label="Factor de N (teórico)", color="tab:orange")
ax2.bar(xpos + ancho / 2, [p[3] for p in pares], ancho,
        label="Factor de latencia (medido)", color="tab:blue")
ax2.set_xticks(xpos)
ax2.set_xticklabels([f"{p[0] // 1000}k→{p[1] // 1000}k" for p in pares])
ax2.set_ylabel("Factor de crecimiento")
ax2.set_title("Crecimiento entre valores de N\n(lineal ⇒ barras iguales)")
ax2.grid(alpha=0.3, axis="y")
ax2.legend()

fig.tight_layout()
fig.savefig(ARCHIVO_FIGURA, dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# Guardar todas las cifras en resultados.json
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T1_parte1_knn_exacto_latencia_vs_N",
    "parametros": {
        "D": int(D),
        "k": int(K),
        "n_centros_gaussianos": int(N_CENTROS),
        "valores_N": [int(n) for n in N_VALORES],
        "n_consultas_promediadas": int(N_CONSULTAS),
        "semilla": int(SEMILLA),
    },
    "latencia_media_ms_por_N": {str(int(n)): lat_media_ms[n] for n in N_VALORES},
    "latencia_std_ms_por_N": {str(int(n)): lat_std_ms[n] for n in N_VALORES},
    "factor_escalamiento_lineal": factor_escalamiento,
    "ajuste_lineal": {
        "pendiente_ms_por_vector": float(pendiente),
        "intercepto_ms": float(intercepto),
        "r2": float(r2),
    },
    "escalamiento_es_lineal": es_lineal,
    "conclusion": conclusion,
    "score_max_primera_consulta": score_max,
    "figura_png": ARCHIVO_FIGURA,
}
with open(ARCHIVO_RESULTADOS, "w") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ----------------------------------------------------------------------
# Resumen impreso
# ----------------------------------------------------------------------
print("\n=== T1 · Parte 1 — kNN exacto: latencia vs. N ===")
print(f"Setup: {N_CENTROS} centros gaussianos fijos, D={D}, vectores normalizados a norma 1")
print(f"k={K}, {N_CONSULTAS} consultas normalizadas (las mismas para cada N)")
for n in N_VALORES:
    print(f"  N = {n:>7,d}: latencia media = {lat_media_ms[n]:.3f} ms "
          f"(± {lat_std_ms[n]:.3f} ms)")
print("\nFactor de crecimiento entre valores de N (lineal ⇒ factor_latencia ≈ factor_N):")
for (n1_, n2_, f_n_, f_lat_) in pares:
    print(f"  {n1_:>7,d} → {n2_:>7,d}: factor_N = {f_n_:5.1f} | "
          f"factor_latencia = {f_lat_:5.2f} | razón = {f_lat_ / f_n_:.3f}")
print(f"  Global {n1:,d} → {n2:,d}: factor_N = {f_n_global:.1f} | "
      f"factor_latencia = {f_lat_global:.2f} | razón = {f_lat_global / f_n_global:.3f}")
print(f"\nAjuste lineal latencia(N): pendiente = {pendiente * 1000.0:.3f} µs/vector, "
      f"intercepto = {intercepto:.3f} ms, R² = {r2:.5f}")
print(f"Conclusión: {conclusion}")
print(f"\nResultados guardados en '{ARCHIVO_RESULTADOS}'")
print(f"Figura guardada como '{ARCHIVO_FIGURA}'")
