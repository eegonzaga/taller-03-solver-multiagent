# -*- coding: utf-8 -*-
"""
T1 — Parte 1: Atención escalada (scaled dot-product attention) en NumPy.

Implementa Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
con un softmax numéricamente estable (se resta el máximo de cada fila).

Matrices del enunciado:
  Q (2x4): dos consultas
  K (3x4): tres claves de dimensión 4
  V (3x2): valores asociados

Salidas:
  - Pesos de atención (2x3) y salida (2x2), presentados con cuatro decimales.
  - Todas las cifras se guardan en resultados.json con precisión completa.
"""

import json
import numpy as np


def softmax_estable(x, axis=-1):
    """Softmax numéricamente estable: resta el máximo de cada fila antes de exponenciar."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    """
    Calcula la atención escalada:
      scores = Q K^T / sqrt(d_k)
      pesos  = softmax(scores)  (por filas, estable)
      salida = pesos @ V
    Devuelve (scores, pesos, salida).
    """
    d_k = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d_k)           # (n_consultas, n_claves)
    pesos = softmax_estable(scores, axis=-1)  # (n_consultas, n_claves)
    salida = pesos @ V                        # (n_consultas, d_v)
    return scores, pesos, salida


def main():
    # ------------------------------------------------------------------
    # Matrices del enunciado
    # ------------------------------------------------------------------
    Q = np.array([
        [1.0, 0.0, 1.0, 0.5],
        [0.0, 2.0, 0.0, 1.0],
    ])
    K = np.array([
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.5],
        [1.0, 0.0, 1.0, 1.0],
    ])
    V = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [2.0, 2.0],
    ])

    d_k = Q.shape[1]  # dimensión de las claves = 4

    # ------------------------------------------------------------------
    # Cálculo de la atención escalada
    # ------------------------------------------------------------------
    scores, pesos, salida = scaled_dot_product_attention(Q, K, V)

    # Verificación: cada fila de los pesos debe sumar 1
    sumas_filas = pesos.sum(axis=1)

    # Versiones redondeadas a cuatro decimales (solo para presentación)
    pesos_4dec = np.round(pesos, 4)
    salida_4dec = np.round(salida, 4)

    # ------------------------------------------------------------------
    # Guardar todas las cifras en resultados.json (precisión completa)
    # ------------------------------------------------------------------
    resultados = {
        "d_k": int(d_k),
        "sqrt_d_k": float(np.sqrt(d_k)),
        "Q_2x4": Q.tolist(),
        "K_3x4": K.tolist(),
        "V_3x2": V.tolist(),
        "scores_QKt_sobre_sqrt_dk_2x3": scores.tolist(),
        "pesos_atencion_2x3": pesos.tolist(),
        "salida_atencion_2x2": salida.tolist(),
        "pesos_atencion_2x3_4decimales": pesos_4dec.tolist(),
        "salida_atencion_2x2_4decimales": salida_4dec.tolist(),
        "suma_de_cada_fila_de_pesos": sumas_filas.tolist(),
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Impresión de las cifras principales (con cuatro decimales)
    # ------------------------------------------------------------------
    np.set_printoptions(precision=4, suppress=True)

    print("Q (2x4):")
    print(Q)
    print("\nK (3x4):")
    print(K)
    print("\nV (3x2):")
    print(V)
    print(f"\nd_k = {d_k}   ->   sqrt(d_k) = {np.sqrt(d_k):.4f}")

    print("\nPuntuaciones escaladas Q K^T / sqrt(d_k) (2x3):")
    print(scores)

    print("\nPesos de atención softmax(Q K^T / sqrt(d_k)) (2x3), cuatro decimales:")
    print(pesos_4dec)

    print("\nSalida Attention(Q, K, V) = pesos @ V (2x2), cuatro decimales:")
    print(salida_4dec)

    print("\nComprobación: suma de cada fila de los pesos (debe ser 1):")
    print(sumas_filas)

    print("\nResultados guardados en resultados.json (precisión completa).")


if __name__ == "__main__":
    main()
