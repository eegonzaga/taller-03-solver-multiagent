# -*- coding: utf-8 -*-
"""
SUBTAREA T2 — Parte 2: efecto de escalar por sqrt(d_k) en la atención
=====================================================================
Se crea UNA sola vez el generador rng = np.random.default_rng(0).
Para d_k = 4, 64, 512 (en ese orden) se extraen, consumiendo el mismo
flujo aleatorio:
    q = rng.standard_normal(d_k)          (primero)
    K = rng.standard_normal((10, d_k))    (después)
Se calculan los pesos softmax de los puntajes K @ q:
  - sin escalar
  - escalados por sqrt(d_k)
y se reporta el peso máximo y la entropía en bits de cada distribución.

Salidas:
  - resultados.json (todas las cifras, con precisión completa)
  - tabla impresa en pantalla con cuatro decimales
"""

import json
import numpy as np


# ----------------------------------------------------------------------
# Funciones auxiliares
# ----------------------------------------------------------------------
def softmax(x):
    """Softmax numéricamente estable sobre un vector 1D."""
    x = np.asarray(x, dtype=float)
    z = x - np.max(x)          # estabilidad: restar el máximo
    e = np.exp(z)
    return e / np.sum(e)


def entropia_bits(p):
    """Entropía de Shannon H(p) = -sum p_i log2 p_i, en bits (0·log2(0)=0)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# ----------------------------------------------------------------------
# Experimento: un solo generador, consumido en el orden exacto del enunciado
# ----------------------------------------------------------------------
rng = np.random.default_rng(0)

valores_dk = [4, 64, 512]

# Diccionarios con las cifras principales (claves = str(d_k))
peso_maximo_sin_escalar_por_dk = {}
entropia_bits_sin_escalar_por_dk = {}
peso_maximo_escalado_por_dk = {}
entropia_bits_escalado_por_dk = {}

# Detalle completo (todas las cifras calculadas, para guardarlas)
detalle_por_dk = {}

for d_k in valores_dk:
    # Orden estricto: primero q, después K (mismo flujo del rng)
    q = rng.standard_normal(d_k)
    K = rng.standard_normal((10, d_k))

    # Puntajes de compatibilidad: K @ q  ->  vector de longitud 10
    puntajes = K @ q

    # Distribuciones softmax
    w_sin = softmax(puntajes)                    # sin escalar
    w_esc = softmax(puntajes / np.sqrt(d_k))     # escalado por sqrt(d_k)

    # Cifras a reportar
    pmax_sin = float(np.max(w_sin))
    H_sin = entropia_bits(w_sin)
    pmax_esc = float(np.max(w_esc))
    H_esc = entropia_bits(w_esc)

    clave = str(d_k)
    peso_maximo_sin_escalar_por_dk[clave] = pmax_sin
    entropia_bits_sin_escalar_por_dk[clave] = H_sin
    peso_maximo_escalado_por_dk[clave] = pmax_esc
    entropia_bits_escalado_por_dk[clave] = H_esc

    detalle_por_dk[clave] = {
        "puntajes_Kq": puntajes.tolist(),
        "pesos_softmax_sin_escalar": w_sin.tolist(),
        "pesos_softmax_escalados_por_sqrt_dk": w_esc.tolist(),
        "indice_del_maximo_sin_escalar": int(np.argmax(w_sin)),
        "indice_del_maximo_escalado": int(np.argmax(w_esc)),
    }

# ----------------------------------------------------------------------
# Tabla en pantalla con cuatro decimales
# ----------------------------------------------------------------------
print("Efecto de escalar por sqrt(d_k) en los pesos de atención")
print("q, K ~ N(0, I); 10 claves; softmax de K @ q\n")
encabezado = (
    f"{'d_k':>5} | {'peso_max_sin':>13} | {'H_bits_sin':>11} | "
    f"{'peso_max_esc':>13} | {'H_bits_esc':>11}"
)
print(encabezado)
print("-" * len(encabezado))
for d_k in valores_dk:
    k = str(d_k)
    print(
        f"{d_k:>5} | "
        f"{peso_maximo_sin_escalar_por_dk[k]:>13.4f} | "
        f"{entropia_bits_sin_escalar_por_dk[k]:>11.4f} | "
        f"{peso_maximo_escalado_por_dk[k]:>13.4f} | "
        f"{entropia_bits_escalado_por_dk[k]:>11.4f}"
    )
print()
print("Referencia: entropía máxima posible con 10 claves = log2(10) = "
      f"{np.log2(10):.4f} bits (distribución uniforme).")
print("Sin escalar, al crecer d_k la distribución se concentra (peso máx -> 1, "
      "entropía -> 0); escalando por sqrt(d_k) se mantiene cercana a la uniforme.\n")

# ----------------------------------------------------------------------
# Guardar todas las cifras en resultados.json (precisión completa)
# ----------------------------------------------------------------------
resultados = {
    "peso_maximo_sin_escalar_por_dk": peso_maximo_sin_escalar_por_dk,
    "entropia_bits_sin_escalar_por_dk": entropia_bits_sin_escalar_por_dk,
    "peso_maximo_escalado_por_dk": peso_maximo_escalado_por_dk,
    "entropia_bits_escalado_por_dk": entropia_bits_escalado_por_dk,
    "detalle_por_dk": detalle_por_dk,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("Cifras guardadas en resultados.json")
