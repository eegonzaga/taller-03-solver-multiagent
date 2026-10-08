# -*- coding: utf-8 -*-
"""
T1 — Setup + Parte 1: kNN exacto, latencia vs. N (Ejercicio 1.1)
================================================================
- Setup del curso: rng semilla 7, D=128, datos_realistas con 200 centros
  gaussianos fijos, normalizar.
- Ejercicio 1.1: knn_exacto(consulta, base, k) sobre vectores normalizados
  (dot = coseno) y tiempo medio de consulta para N = 10k, 50k, 100k, 200k
  (media sobre 20 consultas de la misma distribución).
- Tabla de latencias + figura tiempo vs. N (la FORMA: una recta, O(N·d)).

Conclusión: el costo por consulta es LINEAL en N. Con 10M de vectores y
tráfico real, esto muere. Entra IVF (Parte 2).
Nota: los milisegundos los pone cada máquina; lo que se reporta es la forma.
"""

import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================================================================
# Setup dado por el curso (semilla 7, D=128, 200 centros gaussianos fijos)
# ======================================================================
rng = np.random.default_rng(7)   # semilla 7
D = 128                          # dimensión de los vectores
N_CENTROS = 200

# 200 centros gaussianos FIJOS: se generan UNA sola vez con el rng semillado
centros = rng.normal(0.0, 1.0, size=(N_CENTROS, D))

def datos_realistas(n):
    """n vectores 'realistas': cada uno cae cerca de uno de los 200 centros + ruido."""
    idx = rng.integers(0, N_CENTROS, size=n)
    return centros[idx] + rng.normal(0.0, 0.3, size=(n, D))

def normalizar(X):
    """Normalización L2 por filas -> el producto punto equivale al coseno."""
    return X / np.linalg.norm(X, axis=1, keepdims=True)

# ======================================================================
# Ejercicio 1.1 — el costo O(N·d), medido
# ======================================================================
def knn_exacto(consulta, base, k):
    """
    k vecinos más cercanos por similitud coseno.
    Precondición: consulta y base ya normalizadas -> base @ consulta = cosenos.
    Costo: O(N·d): un producto punto de dimensión d contra cada uno de los N.
    Devuelve (indices, scores) ordenados de mayor a menor similitud.
    """
    scores = base @ consulta                     # O(N·d)
    k = min(int(k), base.shape[0])
    idx = np.argpartition(-scores, k - 1)[:k]    # top-k sin orden fino
    orden = np.argsort(-scores[idx])             # ordenar los k mejores
    return idx[orden], scores[idx[orden]]

# --- parámetros del experimento (los del enunciado) ---
TAMANOS_N = [10_000, 50_000, 100_000, 200_000]
K = 10             # vecinos a recuperar (coherente con el recall@10 de la Parte 2)
N_CONSULTAS = 20   # consultas promediadas por tamaño (p. ej. 20, según el enunciado)

# 20 consultas extraídas de la MISMA distribución que la base
consultas = normalizar(datos_realistas(N_CONSULTAS))

# --- verificación rápida de corrección (vs. argsort completo, base chica) ---
base_chk = normalizar(datos_realistas(500))
idx_fast, sc_fast = knn_exacto(consultas[0], base_chk, K)
scores_full = base_chk @ consultas[0]
idx_ref = np.argsort(-scores_full)[:K]
assert set(idx_fast.tolist()) == set(idx_ref.tolist()), "knn_exacto incorrecto"
assert np.allclose(np.sort(sc_fast)[::-1], np.sort(scores_full[idx_ref])[::-1])
print("Check de correccion de knn_exacto vs. fuerza bruta: OK")
del base_chk

# --- calentamiento (evita penalizar la 1ª llamada por arranque de BLAS) ---
base_warm = normalizar(datos_realistas(1_000))
knn_exacto(consultas[0], base_warm, K)
del base_warm

# --- medición de latencias: para cada N, media sobre las 20 consultas ---
mediciones = {}
for N in TAMANOS_N:
    base = normalizar(datos_realistas(N))
    lat_ms = []
    for q in consultas:
        t0 = time.perf_counter()
        knn_exacto(q, base, K)
        t1 = time.perf_counter()
        lat_ms.append((t1 - t0) * 1000.0)
    mediciones[N] = {
        "media_ms": float(np.mean(lat_ms)),
        "std_ms": float(np.std(lat_ms)),
        "us_por_vector": float(np.mean(lat_ms)) * 1000.0 / N,
        "latencias_ms": [float(v) for v in lat_ms],
    }
    del base  # liberar memoria antes de la siguiente N

# --- tabla de latencias ---
print()
print("=" * 80)
print(f" Tabla de latencias — kNN exacto (d={D}, coseno, k={K}, media de {N_CONSULTAS} consultas)")
print("=" * 80)
print(f" {'N':>9} | {'t medio (ms)':>12} | {'std (ms)':>9} | {'us/vector':>10} | {'N·d (mults.)':>16}")
print("-" * 80)
for N in TAMANOS_N:
    m = mediciones[N]
    print(f" {N:>9,} | {m['media_ms']:>12.3f} | {m['std_ms']:>9.3f} | "
          f"{m['us_por_vector']:>10.3f} | {N * D:>16,}")
print("=" * 80)

# --- cifras clave ---
t10k = mediciones[10_000]["media_ms"]
t200k = mediciones[200_000]["media_ms"]
factor_escalado = float(t200k / t10k)     # ideal lineal: 200k/10k = 20
factor_ideal = 200_000 / 10_000

Ns = np.array(TAMANOS_N, dtype=float)
ts = np.array([mediciones[N]["media_ms"] for N in TAMANOS_N])
pendiente, intercepto = [float(v) for v in np.polyfit(Ns, ts, 1)]
pred = pendiente * Ns + intercepto
r2 = float(1.0 - np.sum((ts - pred) ** 2) / np.sum((ts - ts.mean()) ** 2))

# Extrapolación lineal a N = 10M (el escenario del enunciado: "con 10M muere")
t_10M_ms = float(t200k * (10_000_000 / 200_000))

print(f"\nFactor de escalado N=10k -> N=200k : {factor_escalado:.2f}x  (ideal lineal: {factor_ideal:.0f}x)")
print(f"Ajuste lineal  t(N) = {pendiente * 1e3:.4f} us/vector * N + {intercepto:.3f} ms   (R^2 = {r2:.6f})")
print(f"Extrapolacion a N = 10,000,000     : {t_10M_ms:,.1f} ms por consulta (~{t_10M_ms / 1000.0:.1f} s)")
print(f"  ({10_000_000:,} x {D} = {10_000_000 * D:,} multiplicaciones por consulta)")
print("\nCONCLUSION: la latencia crece LINEALMENTE con N (O(N·d)). Con 10M de vectores")
print("y trafico real, esto muere. Entra IVF (Parte 2).")

# --- figura: tiempo vs. N (la forma: una recta) ---
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.plot(Ns, ts, "o-", color="#1f77b4", lw=2.2, ms=9,
        label=f"Tiempo medio medido ({N_CONSULTAS} consultas)")
ax.plot(Ns, ts[0] * Ns / Ns[0], "--", color="gray", lw=1.6,
        label="Recta ideal lineal O(N·d)")
ax.plot(Ns, pendiente * Ns + intercepto, ":", color="crimson", lw=1.8,
        label=f"Ajuste lineal (R$^2$ = {r2:.4f})")
for N, t in zip(Ns, ts):
    ax.annotate(f"{t:.2f} ms", (N, t), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9)
ax.set_xticks(TAMANOS_N)
ax.set_xticklabels([f"{n:,}" for n in TAMANOS_N])
ax.set_xlim(0, max(Ns) * 1.06)
ax.set_ylim(bottom=0)
ax.set_xlabel("N — vectores en la base")
ax.set_ylabel("Tiempo medio por consulta (ms)")
ax.set_title(f"Ejercicio 1.1 — kNN exacto: latencia vs. N\n"
             f"(d={D}, similitud coseno, k={K}) — escalado lineal O(N·d)")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
plt.savefig("figura_latencia_vs_N.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nFigura guardada: figura_latencia_vs_N.png")

# --- guardar todas las cifras en resultados.json ---
resultados = {
    "setup": {
        "semilla": 7,
        "D": int(D),
        "n_centros_gaussianos": int(N_CENTROS),
        "k": int(K),
        "n_consultas_por_tamano": int(N_CONSULTAS),
        "tamanos_N": [int(n) for n in TAMANOS_N],
    },
    "tiempo_medio_ms_por_N": {str(int(N)): mediciones[N]["media_ms"] for N in TAMANOS_N},
    "tiempo_std_ms_por_N": {str(int(N)): mediciones[N]["std_ms"] for N in TAMANOS_N},
    "microsegundos_por_vector_por_N": {str(int(N)): mediciones[N]["us_por_vector"] for N in TAMANOS_N},
    "latencias_individuales_ms_por_N": {str(int(N)): mediciones[N]["latencias_ms"] for N in TAMANOS_N},
    "factor_escalado_N10k_a_N200k": float(factor_escalado),
    "factor_ideal_lineal_10k_a_200k": float(factor_ideal),
    "ajuste_lineal": {
        "pendiente_ms_por_vector": float(pendiente),
        "intercepto_ms": float(intercepto),
        "r2": r2,
    },
    "extrapolacion_N10M_ms_por_consulta": t_10M_ms,
    "figura_latencia_vs_N": "figura_latencia_vs_N.png",
    "conclusion": (
        f"Latencia lineal en N (O(N·d)): el factor medido 10k->200k es "
        f"{factor_escalado:.2f}x frente al 20x ideal (R2={r2:.4f}); us/vector casi constante. "
        f"Extrapolado a 10M de vectores: ~{t_10M_ms / 1000.0:.1f} s por consulta "
        f"({10_000_000 * D:,} multiplicaciones); inviable con trafico real. Entra IVF."
    ),
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)
print("Resultados guardados: resultados.json")

print("\nCifras principales:")
print("  tiempo_medio_ms_por_N:", resultados["tiempo_medio_ms_por_N"])
print("  factor_escalado_N10k_a_N200k:", resultados["factor_escalado_N10k_a_N200k"])
print("  figura_latencia_vs_N:", resultados["figura_latencia_vs_N"])
