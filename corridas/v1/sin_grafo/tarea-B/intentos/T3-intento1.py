# -*- coding: utf-8 -*-
"""
SUBTAREA T3 — Parte 3 — Codificación posicional sinusoidal
- Calcula PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) y
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)) con d_model = 16, pos = 0..49.
- Reporta PE[10,0], PE[10,1], PE[25,6].
- Dibuja la matriz PE como mapa de calor (PNG).
- Reutiliza la función de atención de la Parte 1 (softmax estable, escalado por
  sqrt(d_k)) con PE[10] como consulta y la matriz PE completa como claves/valores,
  y determina la posición con mayor peso y su valor.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Función de atención escalada de la Parte 1 (softmax numéricamente
# estable: se resta el máximo de cada fila antes de exponenciar)
# ----------------------------------------------------------------------
def softmax_estable(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)

def atencion_escalada(Q, K, V):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V, con d_k = dim de las claves."""
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    pesos = softmax_estable(scores, axis=-1)
    salida = pesos @ V
    return scores, pesos, salida

# ----------------------------------------------------------------------
# Cargar resultados de la Parte 1 y verificar que la función reutilizada
# reproduce los pesos ya calculados (control de consistencia)
# ----------------------------------------------------------------------
t1 = {}
try:
    with open("entradas/T1.json", "r", encoding="utf-8") as f:
        t1 = json.load(f)
except FileNotFoundError:
    print("Aviso: entradas/T1.json no encontrado; se continúa sin verificación.")

verificacion_diff = None
if t1:
    Q1 = np.array(t1["Q_2x4"], dtype=float)
    K1 = np.array(t1["K_3x4"], dtype=float)
    V1 = np.array(t1["V_3x2"], dtype=float)
    _, pesos1, _ = atencion_escalada(Q1, K1, V1)
    verificacion_diff = float(np.max(np.abs(pesos1 - np.array(t1["pesos_atencion_2x3"], dtype=float))))
    print(f"Verificación función de atención vs Parte 1: max|Δ| = {verificacion_diff:.3e}")

# ----------------------------------------------------------------------
# Parte 3: codificación posicional sinusoidal
# ----------------------------------------------------------------------
d_model = 16
n_pos = 50  # posiciones 0 a 49

pos = np.arange(n_pos, dtype=float)[:, None]            # (50, 1)
idx_i = np.arange(d_model // 2, dtype=float)[None, :]   # (1, 8)
angulos = pos / (10000.0 ** (2.0 * idx_i / d_model))    # (50, 8)

PE = np.zeros((n_pos, d_model), dtype=float)
PE[:, 0::2] = np.sin(angulos)   # columnas pares:   PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE[:, 1::2] = np.cos(angulos)   # columnas impares: PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

PE_10_0 = float(PE[10, 0])
PE_10_1 = float(PE[10, 1])
PE_25_6 = float(PE[25, 6])

# ----------------------------------------------------------------------
# Atención con PE[10] como consulta y PE completa como claves y valores
# (aquí d_k = d_model = 16, luego el escalado es sqrt(16) = 4)
# ----------------------------------------------------------------------
q = PE[10:11, :]      # (1, 16)
K = PE                # (50, 16)
V = PE                # (50, 16)
scores_pe, pesos_pe, salida_pe = atencion_escalada(q, K, V)

pesos_vector = pesos_pe[0]                                  # (50,)
posicion_mayor_peso = int(np.argmax(pesos_vector))
valor_mayor_peso = float(pesos_vector[posicion_mayor_peso])
d_k_pe = int(K.shape[-1])
sqrt_d_k_pe = float(np.sqrt(d_k_pe))

orden = np.argsort(pesos_vector)[::-1]
print("\n--- Parte 3: Codificación posicional sinusoidal (d_model=16, pos 0-49) ---")
print(f"PE[10, 0] = sin(10 / 10000^0)        = {PE_10_0:.10f}")
print(f"PE[10, 1] = cos(10 / 10000^0)        = {PE_10_1:.10f}")
print(f"PE[25, 6] = sin(25 / 10000^(6/16))   = {PE_25_6:.10f}")
print(f"\nAtención con PE[10] como consulta (d_k = {d_k_pe}, sqrt(d_k) = {sqrt_d_k_pe}):")
print(f"Posición con mayor peso: {posicion_mayor_peso}")
print(f"Valor del mayor peso:    {valor_mayor_peso:.10f}")
print("Top-3 posiciones por peso:",
      [(int(p), float(pesos_vector[p])) for p in orden[:3]])

# ----------------------------------------------------------------------
# Guardar todas las cifras en resultados.json
# ----------------------------------------------------------------------
resultados = {
    "d_model": d_model,
    "n_posiciones": n_pos,
    "PE_10_0": PE_10_0,
    "PE_10_1": PE_10_1,
    "PE_25_6": PE_25_6,
    "d_k_atencion_PE": d_k_pe,
    "sqrt_d_k_atencion_PE": sqrt_d_k_pe,
    "posicion_mayor_peso": posicion_mayor_peso,
    "valor_mayor_peso": valor_mayor_peso,
    "pesos_atencion_PE10_50pos": [float(x) for x in pesos_vector],
    "scores_escalados_PE10_50pos": [float(x) for x in scores_pe[0]],
    "salida_atencion_PE10_16dim": [float(x) for x in salida_pe[0]],
    "PE_matriz_50x16": [[float(x) for x in fila] for fila in PE],
    "verificacion_atencion_parte1_max_abs_diff": verificacion_diff,
    "figura_mapa_calor_PE": "figura_mapa_calor_PE.png",
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)
print("\nResultados guardados en resultados.json")

# ----------------------------------------------------------------------
# Mapa de calor de la matriz PE (posiciones x dimensiones)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(PE, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0, origin="upper")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Valor de PE")
ax.set_xlabel("Dimensión i (0–15)")
ax.set_ylabel("Posición pos (0–49)")
ax.set_title("Codificación posicional sinusoidal — d_model = 16, pos = 0…49\n"
             "Pares: sin(pos/10000^(2i/d)) · Impares: cos(pos/10000^(2i/d))")
ax.set_yticks([0, 10, 20, 30, 40, 49])
ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 15])
# Marcar la posición usada como consulta (10) y la de mayor peso
ax.axhline(10 + 0.5, color="black", lw=1.2, ls="--")
ax.axhline(posicion_mayor_peso + 0.5, color="yellow", lw=1.0, ls=":")
plt.tight_layout()
plt.savefig("figura_mapa_calor_PE.png", dpi=150)
plt.close(fig)
print("Figura guardada: figura_mapa_calor_PE.png")

# ----------------------------------------------------------------------
# Figura adicional: distribución de los pesos de atención por posición
# ----------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(9, 4))
ax2.stem(np.arange(n_pos), pesos_vector)
ax2.set_xlabel("Posición (pos)")
ax2.set_ylabel("Peso de atención")
ax2.set_title(f"Atención con PE[10] como consulta — máximo en pos "
              f"{posicion_mayor_peso} (peso = {valor_mayor_peso:.6f})")
plt.tight_layout()
plt.savefig("figura_pesos_atencion_PE10.png", dpi=150)
plt.close(fig2)
print("Figura guardada: figura_pesos_atencion_PE10.png")
