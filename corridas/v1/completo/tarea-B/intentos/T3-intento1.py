# -*- coding: utf-8 -*-
"""
T3 — Parte 3: Codificación posicional sinusoidal (Vaswani et al., 2017)

1) Calcula PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
          PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
   con d_model = 16 para las posiciones 0..49.
2) Reporta PE[10, 0], PE[10, 1] y PE[25, 6].
3) Dibuja la matriz 50x16 como mapa de calor (PNG).
4) Reutilizando la función de atención de la Parte 1
   Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V,
   usa PE[10] como consulta y la matriz PE completa como claves y valores,
   y determina qué posición recibe el mayor peso y cuánto vale.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Parámetros del enunciado
# ------------------------------------------------------------------
d_model = 16
n_pos = 50  # posiciones 0 .. 49

# ------------------------------------------------------------------
# Codificación posicional sinusoidal
# ------------------------------------------------------------------
pos = np.arange(n_pos, dtype=float)[:, None]           # (50, 1)
idx = np.arange(d_model // 2, dtype=float)[None, :]    # (1, 8)  -> i = 0..7
angulos = pos / np.power(10000.0, 2.0 * idx / d_model) # (50, 8)

PE = np.zeros((n_pos, d_model), dtype=float)
PE[:, 0::2] = np.sin(angulos)   # dimensiones pares   0, 2, ..., 14
PE[:, 1::2] = np.cos(angulos)   # dimensiones impares 1, 3, ..., 15

PE_10_0 = float(PE[10, 0])
PE_10_1 = float(PE[10, 1])
PE_25_6 = float(PE[25, 6])

# ------------------------------------------------------------------
# Función de atención escalada (reutilizada de la Parte 1)
# softmax numéricamente estable: se resta el máximo de cada fila
# ------------------------------------------------------------------
def softmax_estable(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)

def atencion_escalada(Q, K, V):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V."""
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    pesos = softmax_estable(scores, axis=-1)
    salida = pesos @ V
    return pesos, salida

# ------------------------------------------------------------------
# Verificación de la función reutilizada contra la Parte 1 (T1.json)
# ------------------------------------------------------------------
verificacion_parte1 = None
try:
    with open("entradas/T1.json", "r") as f:
        t1 = json.load(f)
    Q1 = np.array(t1["Q_2x4"], dtype=float)
    K1 = np.array(t1["K_3x4"], dtype=float)
    V1 = np.array(t1["V_3x2"], dtype=float)
    p1, s1 = atencion_escalada(Q1, K1, V1)
    ref_p = np.array(t1["pesos_atencion_2x3"], dtype=float)
    ref_s = np.array(t1["salida_atencion_2x2"], dtype=float)
    verificacion_parte1 = {
        "pesos_coinciden": bool(np.allclose(p1, ref_p, atol=1e-8)),
        "salida_coincide": bool(np.allclose(s1, ref_s, atol=1e-8)),
        "max_error_pesos": float(np.max(np.abs(p1 - ref_p))),
        "max_error_salida": float(np.max(np.abs(s1 - ref_s))),
    }
except (OSError, KeyError, ValueError):
    verificacion_parte1 = None

# ------------------------------------------------------------------
# Atención: consulta = PE[10], claves y valores = matriz PE completa
# ------------------------------------------------------------------
q = PE[10:11, :]      # (1, 16)  -> d_k = 16
K_pe = PE             # (50, 16)
V_pe = PE             # (50, 16)

pesos_pe, salida_pe = atencion_escalada(q, K_pe, V_pe)
pesos_pe = pesos_pe[0]                       # (50,)
salida_pe_vec = salida_pe[0]                 # (16,)

posicion_mayor_peso = int(np.argmax(pesos_pe))
peso_mayor_pe = float(pesos_pe[posicion_mayor_peso])

# ------------------------------------------------------------------
# Mapa de calor de la matriz 50 x 16
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(PE, aspect="auto", cmap="RdBu_r", origin="lower",
               vmin=-1.0, vmax=1.0)
fig.colorbar(im, ax=ax, label="Valor de PE")
ax.set_xlabel("Dimensión $i$ (0–15)")
ax.set_ylabel("Posición $pos$ (0–49)")
ax.set_title("Codificación posicional sinusoidal ($d_{model}=16$, pos 0–49)")
ax.set_xticks(range(0, 16, 2))
ax.set_yticks(range(0, 50, 5))
fig.tight_layout()
fig.savefig("mapa_calor_pe.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# Guardar todas las cifras en resultados.json
# ------------------------------------------------------------------
resultados = {
    "d_model": d_model,
    "n_posiciones": n_pos,
    "PE_10_0": PE_10_0,
    "PE_10_1": PE_10_1,
    "PE_25_6": PE_25_6,
    "posicion_mayor_peso": posicion_mayor_peso,
    "peso_mayor_pe": peso_mayor_pe,
    "mapa_calor_pe_png": "mapa_calor_pe.png",
    "PE_matriz_50x16": PE.tolist(),
    "pesos_atencion_PE10_sobre_PE_50": pesos_pe.tolist(),
    "salida_atencion_PE10_16": salida_pe_vec.tolist(),
    "verificacion_funcion_atencion_parte1": verificacion_parte1,
}
with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

# ------------------------------------------------------------------
# Impresión de las cifras principales
# ------------------------------------------------------------------
print("=== T3 · Parte 3: Codificación posicional sinusoidal ===")
print(f"d_model = {d_model}, posiciones 0..{n_pos - 1}")
print(f"PE[10, 0] = sin(10 / 10000^0)        = {PE_10_0:.10f}")
print(f"PE[10, 1] = cos(10 / 10000^0)        = {PE_10_1:.10f}")
print(f"PE[25, 6] = sin(25 / 10000^(6/16))   = {PE_25_6:.10f}")
print("-" * 60)
print("Atención con PE[10] como consulta y PE completo como claves/valores")
print(f"Posición con mayor peso : {posicion_mayor_peso}")
print(f"Peso mayor              : {peso_mayor_pe:.10f}")
print(f"Suma de los pesos       : {float(np.sum(pesos_pe)):.10f}")
if verificacion_parte1 is not None:
    print("-" * 60)
    print("Verificación de la función de atención contra la Parte 1:")
    print(f"  pesos coinciden : {verificacion_parte1['pesos_coinciden']}")
    print(f"  salida coincide : {verificacion_parte1['salida_coincide']}")
print("-" * 60)
print("Figura guardada   : mapa_calor_pe.png")
print("Resultados en     : resultados.json")
