# -*- coding: utf-8 -*-
"""
T1 — Setup + Parte 1: kNN exacto, latencia vs. N
MMIA 6013 · Semana 2 · S2·MAR — ANN: mide el trade-off recall / latencia

Qué hace:
  1. Reproduce el setup dado: datos_realistas (D=128, 200 centros gaussianos
     fijos con semilla 2026, stream rng con semilla 7) y normalizar (norma 1,
     para que el producto punto sea el coseno).
  2. Ejercicio 1.1: knn_exacto(consulta, base, k) con producto punto sobre
     vectores normalizados (top-k vía argpartition, O(N), sin ordenar todo).
  3. Mide el tiempo medio de consulta para N = 10_000, 50_000, 100_000, 200_000
     (mismas 50 consultas para todos los N, 1 calentamiento no cronometrado).
  4. Verifica empíricamente el escalado lineal O(N·d): ajuste lineal + R²,
     exponente log-log ≈ 1, ms por cada 1000 vectores ≈ constante, razón
     latencia(200k)/latencia(10k) ≈ razón N(200k)/N(10k) = 20.
  5. Extrapola a 10M de vectores y cierra con la conclusión de la Parte 1.

Salidas (carpeta actual, rutas relativas):
  - resultados.json            (todas las cifras, precisión completa)
  - parte1_latencia_vs_N.png   (figura requerida)
"""

import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")  # sin pantalla: solo guardar PNG
import matplotlib.pyplot as plt

t_inicio = time.perf_counter()

# ============================================================
# Setup — código dado por el profesor (sin cambios)
# ============================================================
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


# ============================================================
# Ejercicio 1.1 — kNN exacto con producto punto (coseno)
# ============================================================
def knn_exacto(consulta, base, k):
    """Top-k vecinos más cercanos de `consulta` en `base` por similitud coseno.

    Ambos vienen normalizados a norma 1 (setup), así que el producto punto
    ES el coseno y el ranking por dot coincide con el ranking por ángulo.
    Costo dominante: base @ consulta  →  O(N·d) por consulta.

    Parámetros
    ----------
    consulta : (d,) float32 con norma 1
    base     : (N, d) float32 con normas 1
    k        : int, vecinos a devolver

    Retorna
    -------
    indices : (k,) int64   — índices en `base`, de mayor a menor similitud
    scores  : (k,) float32 — similitudes coseno correspondientes
    """
    puntajes = base @ consulta                             # O(N·d): el cuello de botella
    k_eff = min(int(k), base.shape[0])
    idx = np.argpartition(-puntajes, k_eff - 1)[:k_eff]    # top-k en O(N)
    orden = np.argsort(-puntajes[idx])                     # orden final solo los k
    return idx[orden], puntajes[idx][orden]


# ============================================================
# Experimento (enunciado: N = 10k, 50k, 100k, 200k)
# ============================================================
N_LIST = [10_000, 50_000, 100_000, 200_000]
K = 10                 # vecinos por consulta (la Parte 2 evalúa recall@10)
N_CONSULTAS = 50       # consultas por N para el tiempo medio
N_10M = 10_000_000     # extrapolación que pide el enunciado
PRESUPUESTO_MS = 10.0  # 10 ms/consulta ≈ 100 QPS de "tráfico real"

# Mismas consultas para todos los N: las diferencias de latencia se deben
# solo a N, no a consultas distintas. (Base y consultas comparten temas.)
consultas = normalizar(datos_realistas(N_CONSULTAS))

lat_media_ms = {}
lat_std_ms = {}
lat_todas_ms = {}
verificacion_topk = None

for N in N_LIST:
    base = normalizar(datos_realistas(N))  # (N, 128) float32, normas 1

    # --- control de corrección (una sola vez, fuera del cronómetro):
    # knn_exacto debe coincidir con un argsort completo de los puntajes.
    if verificacion_topk is None:
        puntajes0 = base @ consultas[0]
        idx_argsort = np.argsort(-puntajes0)[:K]
        idx_chk, sc_chk = knn_exacto(consultas[0], base, K)
        verificacion_topk = bool(
            set(idx_argsort.tolist()) == set(idx_chk.tolist())
            and np.allclose(puntajes0[idx_argsort], sc_chk, rtol=0, atol=0)
        )

    # calentamiento (hilos de BLAS, asignaciones) — no se cronometra
    knn_exacto(consultas[0], base, K)

    t_ms = np.empty(N_CONSULTAS, dtype=np.float64)
    for i in range(N_CONSULTAS):
        t0 = time.perf_counter()
        knn_exacto(consultas[i], base, K)
        t_ms[i] = (time.perf_counter() - t0) * 1000.0

    lat_media_ms[N] = float(t_ms.mean())
    lat_std_ms[N] = float(t_ms.std(ddof=1))
    lat_todas_ms[N] = [float(v) for v in t_ms]
    del base  # liberar memoria antes de construir la siguiente base

# ============================================================
# Verificación del escalado lineal O(N·d)
# (con d = 128 fijo, O(N·d) se manifiesta como linealidad en N)
# ============================================================
Ns_arr = np.asarray(N_LIST, dtype=np.float64)
medias = np.asarray([lat_media_ms[N] for N in N_LIST])
stds = np.asarray([lat_std_ms[N] for N in N_LIST])

pendiente, intercepto = np.polyfit(Ns_arr, medias, 1)   # ms por vector
pred = pendiente * Ns_arr + intercepto
r2 = float(1.0 - np.sum((medias - pred) ** 2) / np.sum((medias - medias.mean()) ** 2))
a_log, b_log = np.polyfit(np.log10(Ns_arr), np.log10(medias), 1)  # exponente ≈ 1

ms_por_mil = medias / (Ns_arr / 1000.0)   # ≈ constante si el escalado es lineal
qps = 1000.0 / medias

# --- extrapolación a 10M de vectores ---
lat_10M_ajuste = float(pendiente * N_10M + intercepto)
lat_10M_prop = float(medias[-1] * (N_10M / N_LIST[-1]))
qps_10M = 1000.0 / lat_10M_ajuste
veces_presupuesto = lat_10M_ajuste / PRESUPUESTO_MS

# ============================================================
# Figura: latencia vs. N (lineal + log-log con extrapolación a 10M)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

# Panel A — escala lineal: los puntos caen sobre una recta
ax1.errorbar(Ns_arr, medias, yerr=stds, fmt="o-", color="#1f77b4",
             lw=1.8, markersize=6, capsize=4,
             label="medido (media ± desv. est., 50 consultas)")
xs = np.linspace(0.0, Ns_arr.max() * 1.03, 200)
ax1.plot(xs, pendiente * xs + intercepto, "--", color="dimgray", lw=1.6,
         label=f"ajuste lineal: {pendiente * 1000:.3f} ms por 1000 vect. (R² = {r2:.5f})")
ax1.set_xlabel("N — vectores en la base")
ax1.set_ylabel("latencia por consulta (ms)")
ax1.set_title("kNN exacto: latencia vs. N (escala lineal)")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(alpha=0.3)

# Panel B — log-log: pendiente ≈ 1 verifica O(N); extrapolación a 10M
ax2.loglog(Ns_arr, medias, "o", color="#1f77b4", markersize=6, label="medido")
xs_log = np.logspace(np.log10(8_000), 7.0, 200)
ax2.loglog(xs_log, pendiente * xs_log + intercepto, "--", color="dimgray", lw=1.6,
           label=f"misma recta (exponente log-log = {a_log:.3f} ≈ 1)")
ax2.loglog([N_10M], [lat_10M_ajuste], "o", markersize=10, mfc="none",
           mec="crimson", mew=2.2,
           label=f"extrapolado 10M: {lat_10M_ajuste:.0f} ms/consulta")
ax2.axhline(PRESUPUESTO_MS, color="crimson", ls=":", lw=1.5)
ax2.text(9_000, PRESUPUESTO_MS * 1.3, "presupuesto 10 ms/consulta (≈100 QPS)",
         color="crimson", fontsize=9)
ax2.set_xticks([1e4, 1e5, 1e6, 1e7])
ax2.set_xticklabels(["10k", "100k", "1M", "10M"])
ax2.set_xlabel("N — vectores en la base (escala log)")
ax2.set_ylabel("latencia por consulta (ms, escala log)")
ax2.set_title("Verificación O(N·d) y extrapolación a 10M")
ax2.legend(loc="lower right", fontsize=9)
ax2.grid(alpha=0.3, which="both")

fig.suptitle("Parte 1 — kNN exacto (producto punto sobre vectores norma 1, D = 128): "
             "la latencia escala linealmente con N", y=1.02, fontsize=12)
fig.text(0.5, -0.035,
         "Conclusión: lineal en N. Con 10M de vectores y tráfico real, esto muere. Entra IVF.",
         ha="center", fontsize=10.5, color="crimson")
fig.tight_layout()
fig.savefig("parte1_latencia_vs_N.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ============================================================
# resultados.json — todas las cifras, precisión completa
# ============================================================
resultados = {
    "tarea": "T1 — Setup + Parte 1: kNN exacto, latencia vs. N",
    "D": int(D),
    "k": int(K),
    "n_consultas_por_N": int(N_CONSULTAS),
    "semillas": {"rng_datos": 7, "centros": 2026},
    "lista_N": [int(n) for n in N_LIST],
    "latencia_media_ms_por_N": {str(n): lat_media_ms[n] for n in N_LIST},
    "latencia_std_ms_por_N": {str(n): lat_std_ms[n] for n in N_LIST},
    "latencias_individuales_ms_por_N": {str(n): lat_todas_ms[n] for n in N_LIST},
    "latencia_ms_por_mil_vectores_por_N": {
        str(n): float(m) for n, m in zip(N_LIST, ms_por_mil)},
    "throughput_qps_por_N": {str(n): float(q) for n, q in zip(N_LIST, qps)},
    "verificacion_topk_vs_argsort": verificacion_topk,
    "ajuste_lineal": {
        "pendiente_ms_por_vector": float(pendiente),
        "pendiente_ms_por_mil_vectores": float(pendiente * 1000.0),
        "intercepto_ms": float(intercepto),
        "r2": r2,
    },
    "exponente_escala_loglog": float(a_log),
    "razon_latencia_200k_sobre_10k": float(medias[-1] / medias[0]),
    "razon_N_200k_sobre_10k": float(N_LIST[-1] / N_LIST[0]),
    "extrapolacion_10M_vectores": {
        "latencia_ms_por_consulta_ajuste_lineal": lat_10M_ajuste,
        "latencia_ms_por_consulta_proporcional": lat_10M_prop,
        "qps_un_hilo": float(qps_10M),
        "presupuesto_ms_por_consulta_100qps": float(PRESUPUESTO_MS),
        "veces_sobre_presupuesto": float(veces_presupuesto),
    },
    "figura_png": "parte1_latencia_vs_N.png",
}

# conservar resultados de otras subtareas si el archivo ya existe
try:
    with open("resultados.json", "r", encoding="utf-8") as f:
        previos = json.load(f)
    if not isinstance(previos, dict):
        previos = {}
except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
    previos = {}
previos.update(resultados)
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(previos, f, indent=2, ensure_ascii=False)

# ============================================================
# Cifras principales (print)
# ============================================================
print("=" * 74)
print("T1 · Parte 1 — kNN exacto (coseno, D=128): latencia vs. N")
print("=" * 74)
print(f"k = {K} · consultas por N = {N_CONSULTAS} (mismas consultas para todo N)")
print(f"Verificación top-k vs argsort completo: {'OK' if verificacion_topk else 'FALLÓ'}")
print("-" * 74)
print(f"{'N':>9} | {'latencia media (ms)':>20} | {'ms por 1000 vect.':>18} | {'QPS':>8}")
for n in N_LIST:
    print(f"{n:>9,} | {lat_media_ms[n]:>20.4f} | "
          f"{lat_media_ms[n] / (n / 1000.0):>18.4f} | {1000.0 / lat_media_ms[n]:>8.1f}")
print("-" * 74)
print(f"Ajuste lineal:  latencia(ms) = {pendiente * 1000.0:.4f} ms/1000 vect × N "
      f"+ {intercepto:.4f} ms")
print(f"  R² = {r2:.6f}  |  exponente log-log = {a_log:.4f} (≈ 1 ⇒ escalado O(N·d))")
print(f"  razón latencia(200k)/latencia(10k) = {medias[-1] / medias[0]:.2f}  "
      f"vs  razón N = {N_LIST[-1] / N_LIST[0]:.0f}")
print("-" * 74)
print(f"Extrapolación a {N_10M:,} vectores: {lat_10M_ajuste:.1f} ms/consulta "
      f"(escalo proporcional simple: {lat_10M_prop:.1f} ms)")
print(f"  → {qps_10M:.2f} consultas/s por hilo; con presupuesto de {PRESUPUESTO_MS:.0f} ms "
      f"(≈100 QPS) excede ×{veces_presupuesto:.0f}")
print("-" * 74)
print("CONCLUSIÓN: la latencia del kNN exacto es lineal en N (O(N·d), medido).")
print("Lineal en N. Con 10M de vectores y tráfico real, esto muere. Entra IVF.")
print(f"Cifras → resultados.json · Figura → parte1_latencia_vs_N.png")
print(f"Tiempo total del script: {time.perf_counter() - t_inicio:.1f} s")
