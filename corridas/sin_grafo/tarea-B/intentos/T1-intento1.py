# -*- coding: utf-8 -*-
"""
T1 — Parte 1 — Atención escalada en NumPy
Implementa Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V con softmax
numéricamente estable (restando el máximo de cada fila), usando las matrices
del enunciado. Guarda los resultados en resultados.json y los imprime con
cuatro decimales.
"""

import json
import numpy as np
from scipy.special import softmax as scipy_softmax  # solo para verificación


# ----------------------------------------------------------------------
# 1. Matrices del enunciado
# ----------------------------------------------------------------------
Q = np.array([[1.0, 0.0, 1.0, 0.5],
              [0.0, 2.0, 0.0, 1.0]])          # (2, 4)  -> 2 consultas, d_k = 4

K = np.array([[1.0, 1.0, 0.0, 0.0],
              [0.0, 1.0, 1.0, 0.5],
              [1.0, 0.0, 1.0, 1.0]])          # (3, 4)  -> 3 claves

V = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [2.0, 2.0]])                    # (3, 2)  -> 3 valores, d_v = 2

d_k = Q.shape[1]          # dimensión de las claves/consultas = 4
sqrt_dk = float(np.sqrt(d_k))


# ----------------------------------------------------------------------
# 2. Softmax numéricamente estable (resta el máximo de cada fila)
# ----------------------------------------------------------------------
def softmax_estable(x, axis=-1):
    """Softmax estable: exp(x - max_fila) / sum(exp(x - max_fila))."""
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


# ----------------------------------------------------------------------
# 3. Atención escalada
# ----------------------------------------------------------------------
scores = (Q @ K.T) / sqrt_dk          # (2, 3)  puntuaciones QK^T / sqrt(d_k)
pesos = softmax_estable(scores, axis=1)   # (2, 3)  pesos de atención
salida = pesos @ V                        # (2, 2)  salida de la atención

# Verificación cruzada con scipy (misma definición de softmax)
pesos_scipy = scipy_softmax(scores, axis=1)
max_diff_verif = float(np.max(np.abs(pesos - pesos_scipy)))


# ----------------------------------------------------------------------
# 4. Guardar todas las cifras en resultados.json (precisión completa)
# ----------------------------------------------------------------------
resultados = {
    "Q_2x4": Q.tolist(),
    "K_3x4": K.tolist(),
    "V_3x2": V.tolist(),
    "d_k": int(d_k),
    "sqrt_d_k": sqrt_dk,
    "scores_escalados_2x3": scores.tolist(),
    "pesos_atencion_2x3": pesos.tolist(),                 # precisión completa
    "salida_atencion_2x2": salida.tolist(),               # precisión completa
    "pesos_atencion_2x3_4decimales": np.round(pesos, 4).tolist(),
    "salida_atencion_2x2_4decimales": np.round(salida, 4).tolist(),
    "suma_filas_pesos": pesos.sum(axis=1).tolist(),
    "verificacion_max_abs_diff_vs_scipy_softmax": max_diff_verif,
}

with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)


# ----------------------------------------------------------------------
# 5. Impresión de las cifras principales (con cuatro decimales)
# ----------------------------------------------------------------------
np.set_printoptions(precision=4, suppress=True)

print("=" * 62)
print("Atención escalada:  Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V")
print("=" * 62)
print(f"d_k = {d_k}   ->   sqrt(d_k) = {sqrt_dk:.4f}")
print("\nQ (2x4):\n", Q)
print("\nK (3x4):\n", K)
print("\nV (3x2):\n", V)

print("\nPuntuaciones escaladas QK^T/sqrt(d_k) (2x3):\n", scores)

print("\nPesos de atención softmax(QK^T/sqrt(d_k)) (2x3), 4 decimales:")
print(np.round(pesos, 4))

print("\nSalida Attention(Q,K,V) (2x2), 4 decimales:")
print(np.round(salida, 4))

print("\nSuma de cada fila de los pesos (debe ser 1.0):", np.round(pesos.sum(axis=1), 6))
print(f"Verificación vs scipy.special.softmax (max |diff|): {max_diff_verif:.2e}")
print("\nResultados guardados en 'resultados.json'")
