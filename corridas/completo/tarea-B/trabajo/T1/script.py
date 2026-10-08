# -*- coding: utf-8 -*-
"""
T1 — Parte 1 — Atención escalada en NumPy
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
con softmax numéricamente estable (se resta el máximo de cada fila).

Matrices del enunciado:
  Q (2 consultas, dimensión 4)
  K (3 claves,   dimensión 4)
  V (3 valores,  dimensión 2)
"""

import json
import numpy as np

# ----------------------------------------------------------------------
# 1) Datos del enunciado
# ----------------------------------------------------------------------
Q = np.array([
    [1.0, 0.0, 1.0, 0.5],
    [0.0, 2.0, 0.0, 1.0],
], dtype=float)                      # (2, 4)

K = np.array([
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 0.5],
    [1.0, 0.0, 1.0, 1.0],
], dtype=float)                      # (3, 4)

V = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [2.0, 2.0],
], dtype=float)                      # (3, 2)

d_k = Q.shape[1]                     # d_k = 4
sqrt_dk = float(np.sqrt(d_k))        # sqrt(4) = 2.0

# ----------------------------------------------------------------------
# 2) Softmax numéricamente estable (resta el máximo de cada fila)
# ----------------------------------------------------------------------
def softmax_estable(x, axis=-1):
    """softmax(x) con estabilidad numérica: exp(x - max) / sum(exp(x - max))."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)

# ----------------------------------------------------------------------
# 3) Atención escalada
# ----------------------------------------------------------------------
scores = (Q @ K.T) / sqrt_dk         # (2, 3)  = QK^T / sqrt(d_k)
pesos = softmax_estable(scores, axis=1)   # (2, 3)  pesos de atención
salida = pesos @ V                        # (2, 2)  salida de la atención

# ----------------------------------------------------------------------
# 4) Guardar TODAS las cifras en resultados.json (precisión completa)
# ----------------------------------------------------------------------
resultados = {
    "d_k": int(d_k),
    "sqrt_d_k": sqrt_dk,
    "Q_2x4": Q.tolist(),
    "K_3x4": K.tolist(),
    "V_3x2": V.tolist(),
    "scores_escalados_QKt_sqrtdk_2x3": scores.tolist(),
    "pesos_atencion_2x3": pesos.tolist(),
    "salida_atencion_2x2": salida.tolist(),
    # versiones redondeadas a 4 decimales solo para presentación
    "pesos_atencion_2x3_4decimales": np.round(pesos, 4).tolist(),
    "salida_atencion_2x2_4decimales": np.round(salida, 4).tolist(),
}

with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

# ----------------------------------------------------------------------
# 5) Impresión de las cifras principales (4 decimales para el reporte)
# ----------------------------------------------------------------------
np.set_printoptions(precision=4, suppress=True)

print("=" * 60)
print("Atención escalada: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V")
print("=" * 60)
print(f"d_k = {d_k}  ->  sqrt(d_k) = {sqrt_dk:.4f}")
print("\nQ (2x4):")
print(Q)
print("\nK (3x4):")
print(K)
print("\nV (3x2):")
print(V)

print("\nScores escalados QK^T/sqrt(d_k) (2x3):")
print(scores)

print("\nPesos de atención softmax(QK^T/sqrt(d_k)) (2x3):")
print(pesos)
print("Pesos redondeados a 4 decimales:")
print(np.round(pesos, 4))

print("\nSalida Attention(Q,K,V) (2x2):")
print(salida)
print("Salida redondeada a 4 decimales:")
print(np.round(salida, 4))

print("\nVerificación: cada fila de los pesos suma 1 ->",
      pesos.sum(axis=1).tolist())
print("Resultados guardados en 'resultados.json'.")
