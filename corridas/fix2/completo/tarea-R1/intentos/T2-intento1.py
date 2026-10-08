# -*- coding: utf-8 -*-
"""
T2 — Setup + Parte 1: kNN exacto (latencia vs. N)
==================================================
- Setup del profesor (verbatim): datos_realistas (D=128, 200 centros gaussianos
  fijos, semillas 7 y 2026) + normalizar (norma 1 → producto punto = coseno).
- Ejercicio 1.1: knn_exacto(consulta, base, k) con dot y medición del tiempo
  medio de consulta para N = 10_000, 50_000, 100_000, 200_000
  (promedio sobre 50 consultas, k=10). Gráfica latencia vs. N (PNG) y factor
  de crecimiento entre N consecutivos para verificar el escalado lineal O(N·d).

Dependencia de montaje con T1: entradas/T1.json registra que qdrant-client es
una dependencia OPCIONAL que solo usa la Parte 3; aquí NO se importa.
Salidas: resultados.json, T2_latencia_vs_N.png (carpeta actual).
"""
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

T_INICIO = time.perf_counter()

# ----------------------------------------------------------------------------
# SETUP — código dado por el profesor (verbatim)
# ----------------------------------------------------------------------------
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
    """Vectores a norma 1: así el producto punto es el coseno (sesión 06).
    Vive aquí, en el setup, porque las Partes 2 y 3 la usan aunque no hayas
    resuelto el ejercicio 1.1."""
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


# ----------------------------------------------------------------------------
# Montaje: contexto de T1 (anuncia la dependencia opcional qdrant-client que
# usa la Parte 3). Solo se lee para trazabilidad; la Parte 1 no lo necesita.
# ----------------------------------------------------------------------------
contexto_T1 = None
try:
    with open("entradas/T1.json", "r", encoding="utf-8") as f:
        contexto_T1 = json.load(f)
    print("[montaje] entradas/T1.json leído -> modo_qdrant=%s | "
          "qdrant_client_disponible=%s (dependencia opcional, solo Parte 3)"
          % (contexto_T1.get("modo_qdrant"),
             contexto_T1.get("qdrant_client_disponible")))
except Exception:
    print("[montaje] entradas/T1.json no disponible; la Parte 1 no lo requiere.")

# ----------------------------------------------------------------------------
# Ejercicio 1.1 — kNN exacto (vectores normalizados → usar dot)
# ----------------------------------------------------------------------------
def knn_exacto(consulta, base, k):
    """kNN exacto por producto punto.
    Precondición: `consulta` y `base` con norma 1 → dot = similitud coseno.
    Costo O(N·d) del dot + O(N) del top-k con argpartition.
    Devuelve (indices, scores) de los k vecinos más similares (mayor dot)."""
    scores = base @ consulta                 # (N,)  — O(N·d)
    idx = np.argpartition(-scores, k)[:k]    # top-k — O(N)
    orden = np.argsort(-scores[idx])         # ordena solo los k
    return idx[orden], scores[idx][orden]


# Parámetros del experimento (enunciado)
K = 10
N_VALORES = [10_000, 50_000, 100_000, 200_000]
N_CONSULTAS = 50

# Mismas 50 consultas para todos los N (comparación justa de latencia)
consultas = normalizar(datos_realistas(N_CONSULTAS))

# Verificación rápida de corrección de knn_exacto vs. ordenamiento completo
base_mini = normalizar(datos_realistas(1_000))
scores_mini = base_mini @ consultas[0]
idx_ref = np.argsort(-scores_mini)[:K]
idx_fast, sc_fast = knn_exacto(consultas[0], base_mini, K)
sanity_ok = bool(np.array_equal(idx_fast, idx_ref)
                 and np.allclose(sc_fast, scores_mini[idx_ref]))
del base_mini, scores_mini
print(f"[check] knn_exacto coincide con argsort completo en base pequeña: {sanity_ok}")

# ----------------------------------------------------------------------------
# Medición: tiempo medio de consulta vs. N
# ----------------------------------------------------------------------------
tiempos_ms_por_N = {}
checksum_top1 = 0.0
print("\n=== Ejercicio 1.1 — latencia de knn_exacto (dot = coseno) ===")
for N in N_VALORES:
    base = normalizar(datos_realistas(N))
    knn_exacto(consultas[0], base, K)          # warm-up (no se cronometra)
    t = np.empty(N_CONSULTAS, dtype=np.float64)
    for i in range(N_CONSULTAS):
        t0 = time.perf_counter()
        idx, sc = knn_exacto(consultas[i], base, K)
        t[i] = (time.perf_counter() - t0) * 1e3
        checksum_top1 += float(sc[0])
    tiempos_ms_por_N[N] = t
    print(f"N={N:>7,d} | media = {t.mean():.4f} ms/consulta | "
          f"std = {t.std(ddof=1):.4f} ms | min = {t.min():.4f} ms")
    del base  # liberar memoria antes de la siguiente N

medios = np.array([tiempos_ms_por_N[N].mean() for N in N_VALORES])
stds = np.array([tiempos_ms_por_N[N].std(ddof=1) for N in N_VALORES])
mins = np.array([tiempos_ms_por_N[N].min() for N in N_VALORES])
maxs = np.array([tiempos_ms_por_N[N].max() for N in N_VALORES])

# Factor de crecimiento entre N consecutivos (lineal ⇒ ≈ razón de N)
factores = medios[1:] / medios[:-1]
razones_N = np.array(N_VALORES[1:], dtype=float) / np.array(N_VALORES[:-1], dtype=float)
factor_sobre_razon = factores / razones_N            # ≈ 1.0 si el escalado es lineal
logN = np.log(np.array(N_VALORES, dtype=float))
pendiente, intercepto = np.polyfit(logN, np.log(medios), 1)
razon_tiempo_total = float(medios[-1] / medios[0])   # N crece ×20 (10k → 200k)
lineal_ok = bool(np.all(np.abs(factor_sobre_razon - 1.0) < 0.5)
                 and abs(float(pendiente) - 1.0) < 0.2)
latencia_estimada_10M_ms = float(medios[-1] * (10_000_000 / N_VALORES[-1]))

print("\n--- Factores de crecimiento entre N consecutivos ---")
for i in range(len(factores)):
    print(f"N {N_VALORES[i]:>7,d} -> {N_VALORES[i+1]:>7,d}: "
          f"tiempo ×{factores[i]:.3f} (N ×{razones_N[i]:.0f}) | "
          f"factor/razón = {factor_sobre_razon[i]:.3f}")
print(f"Pendiente log-log = {pendiente:.4f} (1.0 ⇒ O(N·d)); "
      f"10k→200k: N ×20, tiempo ×{razon_tiempo_total:.3f}")
print(f"Extrapolación lineal a 10M vectores: ~{latencia_estimada_10M_ms:.1f} ms/consulta "
      f"→ con tráfico real esto muere; entra IVF (Parte 2).")

# ----------------------------------------------------------------------------
# Figura: latencia vs. N (lineal + log-log)
# ----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
N_arr = np.array(N_VALORES, dtype=float)
etiquetas_N = ["10k", "50k", "100k", "200k"]

ax1.plot(N_arr, medios, "o-", color="#1f77b4", lw=2, ms=6,
         label="kNN exacto medido (dot = coseno)")
ax1.plot(N_arr, medios[0] * N_arr / N_arr[0], "k--", lw=1.5,
         label="referencia lineal O(N·d)")
for x, y in zip(N_arr, medios):
    ax1.annotate(f"{y:.2f} ms", (x, y), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=8)
for i in range(len(factores)):
    ax1.annotate(f"×{factores[i]:.2f}",
                 ((N_arr[i] + N_arr[i + 1]) / 2, (medios[i] + medios[i + 1]) / 2),
                 textcoords="offset points", xytext=(0, 16), ha="center",
                 fontsize=8, color="darkred")
ax1.set_xticks(N_arr)
ax1.set_xticklabels(etiquetas_N)
ax1.set_xlabel("N (vectores en la base)")
ax1.set_ylabel("tiempo medio por consulta (ms)")
ax1.set_title(f"Ejercicio 1.1 — kNN exacto: latencia vs. N\n"
              f"(k={K}, D={D}, {N_CONSULTAS} consultas, dot = coseno)")
ax1.grid(alpha=0.3)
ax1.legend(loc="upper left", fontsize=9)

ax2.loglog(N_arr, medios, "s-", color="#d62728", lw=2, ms=6, label="medido")
ax2.loglog(N_arr, np.exp(intercepto) * N_arr ** pendiente, "k--", lw=1.3,
           label=f"ajuste log-log: pendiente = {pendiente:.3f}")
ax2.set_xticks(N_arr)
ax2.set_xticklabels(etiquetas_N)
ax2.xaxis.set_minor_formatter(NullFormatter())
ax2.set_xlabel("N (escala log-log)")
ax2.set_ylabel("ms por consulta")
ax2.set_title("Verificación de escalado lineal\n(pendiente ≈ 1 ⇒ O(N·d))")
ax2.grid(True, which="major", alpha=0.3)
ax2.legend(loc="upper left", fontsize=9)

fig.tight_layout()
plt.savefig("T2_latencia_vs_N.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n[figura] T2_latencia_vs_N.png guardada en la carpeta actual")

# ----------------------------------------------------------------------------
# resultados.json (precisión completa, tipos nativos de Python)
# ----------------------------------------------------------------------------
resultados = {
    "subtarea": "T2 — Setup + Parte 1 — kNN exacto: latencia vs. N",
    "figura_png": "T2_latencia_vs_N.png",
    "setup": {
        "D": int(D),
        "n_centros_gaussianos_fijos": int(len(_CENTROS)),
        "semilla_rng_datos": 7,
        "semilla_centros": 2026,
        "dispersion_intra_tema": 1.5,
        "normalizacion": "norma L2 = 1 → producto punto = similitud coseno",
        "nota_dependencia_T1": ("montaje: T1 registra qdrant-client como dependencia "
                                "opcional usada solo en la Parte 3; no se importa aquí"),
    },
    "contexto_T1": contexto_T1,
    "parametros_experimento": {
        "k": int(K),
        "n_consultas": int(N_CONSULTAS),
        "N_valores": [int(n) for n in N_VALORES],
        "metrica": "producto punto (coseno; vectores normalizados)",
        "implementacion": "numpy: base @ consulta (O(N·d)) + argpartition top-k (O(N))",
        "unidad_tiempo": "milisegundos por consulta",
    },
    "tiempo_medio_ms_por_N": {str(int(n)): float(m) for n, m in zip(N_VALORES, medios)},
    "tiempo_std_ms_por_N": {str(int(n)): float(s) for n, s in zip(N_VALORES, stds)},
    "tiempo_min_ms_por_N": {str(int(n)): float(m) for n, m in zip(N_VALORES, mins)},
    "tiempo_max_ms_por_N": {str(int(n)): float(m) for n, m in zip(N_VALORES, maxs)},
    "tiempos_por_consulta_ms_por_N": {
        str(int(n)): [float(v) for v in tiempos_ms_por_N[n]] for n in N_VALORES
    },
    "factor_escalamiento_lineal": [float(f) for f in factores],
    "razon_N_consecutivos": [float(r) for r in razones_N],
    "factor_escalamiento_sobre_razon_N": [float(f) for f in factor_sobre_razon],
    "pendiente_loglog": float(pendiente),
    "intercepto_loglog": float(intercepto),
    "escalado_total_10k_a_200k": {
        "razon_N": 20.0,
        "razon_tiempo": razon_tiempo_total,
        "factor_normalizado": float(razon_tiempo_total / 20.0),
    },
    "throughput_consultas_por_segundo_por_N": {
        str(int(n)): float(1000.0 / m) for n, m in zip(N_VALORES, medios)
    },
    "latencia_estimada_10M_ms": latencia_estimada_10M_ms,
    "verificacion_escalado_lineal_O_N_d": lineal_ok,
    "sanity_check_knn_exacto_ok": sanity_ok,
    "checksum_scores_top1": float(checksum_top1),
    "tiempo_total_script_s": float(time.perf_counter() - T_INICIO),
    "conclusion": (
        f"La latencia media por consulta crece ×{factores[0]:.3f}, ×{factores[1]:.3f} y "
        f"×{factores[2]:.3f} cuando N crece ×{razones_N[0]:.0f}, ×{razones_N[1]:.0f} y "
        f"×{razones_N[2]:.0f} (factor/razón ≈ "
        f"{', '.join(f'{v:.3f}' for v in factor_sobre_razon)}; pendiente log-log = "
        f"{pendiente:.3f} ≈ 1): el kNN exacto escala linealmente O(N·d). De 10k a 200k "
        f"(N ×20) el tiempo crece ×{razon_tiempo_total:.3f}. Extrapolado a 10M de "
        f"vectores serían ~{latencia_estimada_10M_ms:.1f} ms por consulta: con tráfico "
        f"real esto no es viable — entra IVF (Parte 2)."
    ),
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)
print("[resultados] resultados.json escrito en la carpeta actual")

# ----------------------------------------------------------------------------
# Cifras principales (impresión con precisión completa)
# ----------------------------------------------------------------------------
print("\n=== CIFRAS PRINCIPALES — T2 / Ejercicio 1.1 ===")
print(f"k = {K}, consultas por punto = {N_CONSULTAS}, D = {D}")
for n, m, s in zip(N_VALORES, medios, stds):
    print(f"  N = {n:>7,d}: {m!r} ms/consulta (media)  [std {s!r}]")
print(f"  tiempo_medio_ms_por_N: {resultados['tiempo_medio_ms_por_N']!r}")
print(f"  factor_escalamiento_lineal (N consecutivos): "
      f"{resultados['factor_escalamiento_lineal']!r}")
print(f"  razones de N consecutivas: {resultados['razon_N_consecutivos']!r}")
print(f"  factor/razón (≈1 ⇒ lineal): "
      f"{resultados['factor_escalamiento_sobre_razon_N']!r}")
print(f"  pendiente log-log: {float(pendiente)!r} (≈1 ⇒ O(N·d))")
print(f"  ¿consistente con escalado lineal O(N·d)?: {lineal_ok}")
print(f"  latencia estimada con 10M vectores: {latencia_estimada_10M_ms!r} ms")
