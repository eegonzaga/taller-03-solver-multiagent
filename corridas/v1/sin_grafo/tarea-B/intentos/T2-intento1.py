# =====================================================================
# T2 — Parte 2: Efecto de escalar los puntajes de atención por sqrt(d_k)
# =====================================================================
# Se crea UNA sola vez el generador rng = np.random.default_rng(0).
# Para d_k = 4, 64, 512 (en ese orden) se extrae primero q y después K,
# se calculan los pesos softmax de K @ q sin escalar y escalados por
# sqrt(d_k), y se reporta el peso máximo y la entropía en bits.
# No se requiere figura PNG. Resultados -> resultados.json
# =====================================================================

import json
import numpy as np
import pandas as pd


def softmax_estable(x):
    """Softmax numéricamente estable: resta el máximo antes de exponenciar."""
    x = np.asarray(x, dtype=float)
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def entropia_bits(p):
    """Entropía de Shannon en bits; por convención 0*log2(0) = 0."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# Generador único, creado una sola vez (semilla 0, como pide el enunciado)
rng = np.random.default_rng(0)

filas = []            # cifras principales (precisión completa)
pesos_guardados = {}  # distribuciones completas de pesos, para trazabilidad

for d_k in [4, 64, 512]:
    # Orden estricto exigido: primero q, después K, con el mismo rng
    q = rng.standard_normal(d_k)
    K = rng.standard_normal((10, d_k))

    puntajes = K @ q                                    # forma (10,)
    pesos_sin = softmax_estable(puntajes)               # sin escalar
    pesos_esc = softmax_estable(puntajes / np.sqrt(d_k))  # escalados por sqrt(d_k)

    fila = {
        "d_k": int(d_k),
        "peso_maximo_sin_escalar": float(np.max(pesos_sin)),
        "entropia_bits_sin_escalar": entropia_bits(pesos_sin),
        "peso_maximo_escalado_por_raiz_dk": float(np.max(pesos_esc)),
        "entropia_bits_escalado_por_raiz_dk": entropia_bits(pesos_esc),
    }
    filas.append(fila)

    pesos_guardados[f"d_k={d_k}"] = {
        "puntajes_Kq": puntajes.tolist(),
        "pesos_sin_escalar": pesos_sin.tolist(),
        "pesos_escalados_por_raiz_dk": pesos_esc.tolist(),
    }

# --------------------- Tabla con cuatro decimales ---------------------
df = pd.DataFrame(filas)
columnas = [
    "d_k",
    "peso_maximo_sin_escalar",
    "entropia_bits_sin_escalar",
    "peso_maximo_escalado_por_raiz_dk",
    "entropia_bits_escalado_por_raiz_dk",
]
df_tabla = df[columnas]

print("Parte 2 — Peso máximo y entropía (bits) de los pesos de atención")
print("Sin escalar vs. escalados por sqrt(d_k) — cuatro decimales")
print("Entropía máxima posible con 10 claves: log2(10) = {:.4f} bits".format(np.log2(10)))
print("-" * 100)
print(df_tabla.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("-" * 100)

# Impresión explícita de las cifras principales
for f_ in filas:
    print(
        f"d_k={f_['d_k']:>3} | "
        f"sin escalar: peso_max={f_['peso_maximo_sin_escalar']:.4f}, "
        f"H={f_['entropia_bits_sin_escalar']:.4f} bits | "
        f"escalado: peso_max={f_['peso_maximo_escalado_por_raiz_dk']:.4f}, "
        f"H={f_['entropia_bits_escalado_por_raiz_dk']:.4f} bits"
    )

# ------------- Guardado con precisión completa (sin redondear) -------------
resultados = {
    "tabla_pesos_maximos_y_entropias_bits_por_dk": filas,
    "distribuciones_de_pesos": pesos_guardados,
    "entropia_maxima_posible_bits_10_claves": float(np.log2(10)),
}

with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

print("\nResultados guardados en resultados.json")
