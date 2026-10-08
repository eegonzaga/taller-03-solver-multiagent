import json
import numpy as np


def softmax_estable(x):
    """Softmax numéricamente estable: resta el máximo antes de exponenciar."""
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def entropia_bits(p):
    """Entropía de Shannon en bits; términos con p=0 aportan 0."""
    p = np.asarray(p, dtype=float)
    mascara = p > 0
    return float(-np.sum(p[mascara] * np.log2(p[mascara])))


# Generador único, creado una sola vez (semilla 0)
rng = np.random.default_rng(0)

d_k_valores = [4, 64, 512]

peso_maximo_sin_escalar = []
entropia_bits_sin_escalar = []
peso_maximo_escalado = []
entropia_bits_escalado = []
tabla_filas = []

for d_k in d_k_valores:
    # Orden estricto de consumo del generador: primero q, después K
    q = rng.standard_normal(d_k)
    K = rng.standard_normal((10, d_k))

    puntajes = K @ q  # forma (10,)

    # Distribuciones de pesos de atención
    pesos_sin = softmax_estable(puntajes)                 # sin escalar
    pesos_esc = softmax_estable(puntajes / np.sqrt(d_k))  # escalado por sqrt(d_k)

    pm_sin = float(np.max(pesos_sin))
    eb_sin = entropia_bits(pesos_sin)
    pm_esc = float(np.max(pesos_esc))
    eb_esc = entropia_bits(pesos_esc)

    peso_maximo_sin_escalar.append(pm_sin)
    entropia_bits_sin_escalar.append(eb_sin)
    peso_maximo_escalado.append(pm_esc)
    entropia_bits_escalado.append(eb_esc)

    tabla_filas.append({
        "d_k": int(d_k),
        "peso_maximo_sin_escalar": pm_sin,
        "entropia_bits_sin_escalar": eb_sin,
        "peso_maximo_escalado": pm_esc,
        "entropia_bits_escalado": eb_esc,
    })

# Guardar todas las cifras con precisión completa en resultados.json
resultados = {
    "d_k_valores": [int(d) for d in d_k_valores],
    "peso_maximo_sin_escalar": peso_maximo_sin_escalar,
    "entropia_bits_sin_escalar": entropia_bits_sin_escalar,
    "peso_maximo_escalado": peso_maximo_escalado,
    "entropia_bits_escalado": entropia_bits_escalado,
    "tabla": tabla_filas,
}

with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

# Tabla con cuatro decimales
encabezado = (
    f"{'d_k':>6} | {'peso_max_sin':>14} | {'entropia_bits_sin':>18} | "
    f"{'peso_max_esc':>14} | {'entropia_bits_esc':>18}"
)
print(encabezado)
print("-" * len(encabezado))
for fila in tabla_filas:
    print(
        f"{fila['d_k']:>6} | "
        f"{fila['peso_maximo_sin_escalar']:>14.4f} | "
        f"{fila['entropia_bits_sin_escalar']:>18.4f} | "
        f"{fila['peso_maximo_escalado']:>14.4f} | "
        f"{fila['entropia_bits_escalado']:>18.4f}"
    )

print()
print("Cifras guardadas en resultados.json (precisión completa):")
print("  peso_maximo_sin_escalar   =", peso_maximo_sin_escalar)
print("  entropia_bits_sin_escalar =", entropia_bits_sin_escalar)
print("  peso_maximo_escalado      =", peso_maximo_escalado)
print("  entropia_bits_escalado    =", entropia_bits_escalado)
