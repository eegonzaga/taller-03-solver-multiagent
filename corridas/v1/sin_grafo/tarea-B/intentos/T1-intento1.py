# -*- coding: utf-8 -*-
"""
T1 — Parte 1 — Atención escalada en NumPy
Implementa Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
con softmax numéricamente estable (restando el máximo de cada fila).
Matrices del enunciado: Q (2x4), K (3x4), V (3x2), d_k = 4.
Guarda resultados en resultados.json e imprime las cifras con 4 decimales.
No se requiere figura PNG para esta subtarea.
"""

import json
import numpy as np


# ----------------------------------------------------------------------
# 1) Datos del enunciado
# ----------------------------------------------------------------------
Q = np.array([
    [1.0, 0.0, 1.0, 0.5],
    [0.0, 2.0, 0.0, 1.0],
], dtype=float)          # (2, 4) -> 2 consultas, d_k = 4

K = np.array([
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 0.5],
    [1.0, 0.0, 1.0, 1.0],
], dtype=float)          # (3, 4) -> 3 claves, d_k = 4

V = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [2.0, 2.0],
], dtype=float)          # (3, 2) -> 3 valores, d_v = 2

d_k = Q.shape[1]         # 4


# ----------------------------------------------------------------------
# 2) Softmax numéricamente estable (por filas) y atención escalada
# ----------------------------------------------------------------------
def softmax_estable(Z):
    """Softmax por filas restando el máximo de cada fila (estabilidad)."""
    Z_max = np.max(Z, axis=1, keepdims=True)
    E = np.exp(Z - Z_max)
    return E / np.sum(E, axis=1, keepdims=True)


def attention(Qm, Km, Vm):
    """Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V."""
    dk = Qm.shape[1]
    scores = (Qm @ Km.T) / np.sqrt(dk)   # (n_q, n_k) puntuaciones escaladas
    pesos = softmax_estable(scores)      # (n_q, n_k) pesos de atención
    salida = pesos @ Vm                  # (n_q, d_v) salida
    return scores, pesos, salida


scores, pesos, salida = attention(Q, K, V)


# ----------------------------------------------------------------------
# 3) Guardar TODAS las cifras en resultados.json (precisión completa)
# ----------------------------------------------------------------------
resultados = {
    "Q_2x4": Q.tolist(),
    "K_3x4": K.tolist(),
    "V_3x2": V.tolist(),
    "d_k": int(d_k),
    "sqrt_d_k": float(np.sqrt(d_k)),
    "scores_escalados_2x3": scores.tolist(),
    "pesos_atencion_2x3": pesos.tolist(),          # precisión completa
    "salida_atencion_2x2": salida.tolist(),        # precisión completa
    # Presentación con cuatro decimales (como pide el enunciado)
    "pesos_atencion_2x3_4dec": np.round(pesos, 4).astype(float).tolist(),
    "salida_atencion_2x2_4dec": np.round(salida, 4).astype(float).tolist(),
    "suma_filas_pesos": pesos.sum(axis=1).tolist(),  # verificación (= 1)
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------
# 4) Impresión de las cifras principales (con cuatro decimales)
# ----------------------------------------------------------------------
np.set_printoptions(precision=4, suppress=True)

print("=" * 60)
print("T1 - Parte 1: Atención escalada (Vaswani et al., 2017)")
print("=" * 60)
print(f"d_k = {d_k}  ->  sqrt(d_k) = {np.sqrt(d_k):.4f}")
print("\nQ (2x4):")
print(Q)
print("\nK (3x4):")
print(K)
print("\nV (3x2):")
print(V)

print("\nPuntuaciones escaladas  Q K^T / sqrt(d_k)  (2x3):")
print(scores)

print("\nPesos de atención  softmax(Q K^T / sqrt(d_k))  (2x3), 4 decimales:")
print(np.round(pesos, 4))
print("Verificación: suma de cada fila de pesos =",
      np.round(pesos.sum(axis=1), 4))

print("\nSalida  Attention(Q, K, V)  (2x2), 4 decimales:")
print(np.round(salida, 4))

print("\nResultados guardados en 'resultados.json' "
      "(claves: pesos_atencion_2x3, salida_atencion_2x2, ...)")
