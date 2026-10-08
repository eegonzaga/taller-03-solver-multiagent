# -*- coding: utf-8 -*-
"""
SUBTAREA T2 — Parte 2 — Efecto de escalar por sqrt(d_k) en la atención
(Vaswani et al., 2017, "Attention Is All You Need")

Para d_k = 4, 64 y 512 (en ese orden), con un único generador
rng = np.random.default_rng(0):
    q = rng.standard_normal(d_k)
    K = rng.standard_normal((10, d_k))
Se calculan los pesos softmax de K @ q sin escalar y escalados por sqrt(d_k),
y se reporta el peso máximo y la entropía en bits de cada distribución.
"""

import json
import numpy as np


# ----------------------------------------------------------------------
# Funciones auxiliares
# ----------------------------------------------------------------------
def softmax(x):
    """Softmax numéricamente estable (resta el máximo antes de exponenciar)."""
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def entropia_bits(p):
    """Entropía de Shannon en bits: H = -sum(p * log2(p)), con 0*log2(0)=0."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# ----------------------------------------------------------------------
# Experimento: un solo generador, consumido en el orden indicado
# ----------------------------------------------------------------------
rng = np.random.default_rng(0)

dimensiones = [4, 64, 512]  # orden exigido por el enunciado

resultados = {
    "peso_maximo_sin_escalar": {},
    "entropia_bits_sin_escalar": {},
    "peso_maximo_escalado": {},
    "entropia_bits_escalado": {},
}

filas = []  # para la tabla con cuatro decimales

for d_k in dimensiones:
    # Orden de sorteos exigido: primero q, después K
    q = rng.standard_normal(d_k)
    K = rng.standard_normal((10, d_k))

    puntajes = K @ q                      # (10,) productos punto q·k_i

    w_sin = softmax(puntajes)             # sin escalar
    w_esc = softmax(puntajes / np.sqrt(d_k))  # escalado por sqrt(d_k)

    pm_sin = float(np.max(w_sin))
    eb_sin = entropia_bits(w_sin)
    pm_esc = float(np.max(w_esc))
    eb_esc = entropia_bits(w_esc)

    clave = str(d_k)  # las claves JSON deben ser cadenas
    resultados["peso_maximo_sin_escalar"][clave] = pm_sin
    resultados["entropia_bits_sin_escalar"][clave] = eb_sin
    resultados["peso_maximo_escalado"][clave] = pm_esc
    resultados["entropia_bits_escalado"][clave] = eb_esc

    filas.append((d_k, pm_sin, eb_sin, pm_esc, eb_esc))

# Distribuciones completas de pesos (informativo, precisión completa)
resultados["distribucion_pesos_sin_escalar"] = {
    str(d_k): [float(v) for v in softmax((rng is not None) and 0 or 0)] for d_k in []
}  # placeholder vacío (no usado)

# Guardamos también las distribuciones completas de pesos por d_k
dist_sin = {}
dist_esc = {}
rng2 = None  # no se recrean sorteos: reutilizamos los valores ya calculados
# (recalculamos a partir de los mismos sorteos guardando durante el bucle sería
#  equivalente; para no alterar el generador, reconstruimos con los mismos
#  valores ya consumidos no es posible, así que guardamos las distribuciones
#  calculadas arriba reejecutando el mismo orden de sorteos con una semilla
#  idéntica daría lo mismo, pero lo correcto es haberlas guardado en el bucle).
# -> Por simplicidad y fidelidad, se vuelven a calcular con un generador nuevo
#    con la misma semilla y el mismo orden de consumo (mismo resultado exacto).
rng_rep = np.random.default_rng(0)
for d_k in dimensiones:
    q = rng_rep.standard_normal(d_k)
    K = rng_rep.standard_normal((10, d_k))
    puntajes = K @ q
    dist_sin[str(d_k)] = [float(v) for v in softmax(puntajes)]
    dist_esc[str(d_k)] = [float(v) for v in softmax(puntajes / np.sqrt(d_k))]

resultados["distribucion_pesos_sin_escalar"] = dist_sin
resultados["distribucion_pesos_escalado"] = dist_esc

# Entropía máxima posible (10 claves uniformes), como referencia
resultados["entropia_maxima_bits_10_claves"] = float(np.log2(10))

# Tabla formateada con cuatro decimales (tal como pide el enunciado)
resultados["tabla_4_decimales"] = {
    "encabezados": [
        "d_k",
        "peso_maximo_sin_escalar",
        "entropia_bits_sin_escalar",
        "peso_maximo_escalado",
        "entropia_bits_escalado",
    ],
    "filas": [
        [d_k, f"{a:.4f}", f"{b:.4f}", f"{c:.4f}", f"{d:.4f}"]
        for (d_k, a, b, c, d) in filas
    ],
}

# ----------------------------------------------------------------------
# Impresión de resultados
# ----------------------------------------------------------------------
print("Efecto de escalar por sqrt(d_k) — softmax de K @ q (10 claves)")
print(f"Entropía máxima posible (uniforme, 10 claves): {np.log2(10):.4f} bits\n")

encabezado = (
    f"{'d_k':>5} | {'peso_max_sin':>13} | {'H_bits_sin':>11} | "
    f"{'peso_max_esc':>13} | {'H_bits_esc':>11}"
)
print(encabezado)
print("-" * len(encabezado))
for d_k, a, b, c, d in filas:
    print(f"{d_k:>5} | {a:>13.4f} | {b:>11.4f} | {c:>13.4f} | {d:>11.4f}")

print("\nValores con precisión completa:")
for d_k, a, b, c, d in filas:
    print(
        f"d_k={d_k}: sin escalar -> peso_max={a!r}, entropia_bits={b!r} | "
        f"escalado -> peso_max={c!r}, entropia_bits={d!r}"
    )

# ----------------------------------------------------------------------
# Guardado de resultados
# ----------------------------------------------------------------------
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print("\nResultados guardados en 'resultados.json'.")
