# -*- coding: utf-8 -*-
"""
SUBTAREA T3 — Parte 3 — Codificación posicional sinusoidal (Vaswani et al., 2017)

1) Calcula PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
          PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
   con d_model = 16 y pos = 0..49.
2) Reporta PE[10,0], PE[10,1], PE[25,6].
3) Dibuja la matriz PE completa (50x16) como mapa de calor -> mapa_calor_pe.png.
4) Reutiliza la función de atención escalada de la Parte 1 (softmax numéricamente
   estable, resta el máximo por fila) con PE[10] como consulta y la matriz PE
   completa (50x16) como claves y valores; reporta la posición con mayor peso.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Función de atención de la Parte 1 (softmax numéricamente estable)
# ----------------------------------------------------------------------
def softmax_estable(x, axis=-1):
    """Softmax estable: resta el máximo a lo largo del eje antes de exponenciar."""
    x = np.asarray(x, dtype=float)
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def atencion_escalada(Q, K, V):
    """Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V  (Parte 1)."""
    d_k = K.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    pesos = softmax_estable(scores, axis=-1)
    salida = pesos @ V
    return scores, pesos, salida


# ----------------------------------------------------------------------
# Cargar resultados de la Parte 1 (T1) y verificar la función reutilizada
# ----------------------------------------------------------------------
with open("entradas/T1.json", "r", encoding="utf-8") as f:
    t1 = json.load(f)

Q1 = np.array(t1["Q_2x4"], dtype=float)
K1 = np.array(t1["K_3x4"], dtype=float)
V1 = np.array(t1["V_3x2"], dtype=float)
_, pesos1, salida1 = atencion_escalada(Q1, K1, V1)
diff_pesos = float(np.max(np.abs(pesos1 - np.array(t1["pesos_atencion_2x3"], dtype=float))))
diff_salida = float(np.max(np.abs(salida1 - np.array(t1["salida_atencion_2x2"], dtype=float))))
print("Verificación de la función de atención vs T1:")
print("  max|Δ pesos|  =", diff_pesos)
print("  max|Δ salida| =", diff_salida)

# ----------------------------------------------------------------------
# Parte 3: codificación posicional sinusoidal
# ----------------------------------------------------------------------
d_model = 16
n_pos = 50                                # posiciones 0..49
pos = np.arange(n_pos, dtype=float)       # (50,)
i_idx = np.arange(d_model // 2)           # i = 0..7

# Ángulos: pos / 10000^(2i/d_model)  -> matriz (50, 8)
ang = pos[:, None] / np.power(10000.0, (2.0 * i_idx)[None, :] / d_model)

PE = np.zeros((n_pos, d_model), dtype=float)
PE[:, 0::2] = np.sin(ang)   # columnas pares  2i   -> sin
PE[:, 1::2] = np.cos(ang)   # columnas impares 2i+1 -> cos

PE_10_0 = float(PE[10, 0])
PE_10_1 = float(PE[10, 1])
PE_25_6 = float(PE[25, 6])
print("\nCodificación posicional (d_model=16, pos=0..49):")
print("  PE[10, 0] =", PE_10_0)
print("  PE[10, 1] =", PE_10_1)
print("  PE[25, 6] =", PE_25_6)

# ----------------------------------------------------------------------
# Atención con PE[10] como consulta; PE completa como claves y valores
# ----------------------------------------------------------------------
q = PE[10:11, :]                                   # (1, 16)
scores_pe, pesos_pe_mat, salida_pe = atencion_escalada(q, PE, PE)
pesos_pe = pesos_pe_mat[0]                         # (50,)

posicion_mayor_peso = int(np.argmax(pesos_pe))
peso_mayor = float(pesos_pe[posicion_mayor_peso])
print("\nAtención con PE[10] como consulta (d_k = 16):")
print("  Posición con mayor peso:", posicion_mayor_peso)
print("  Peso mayor:", peso_mayor)
print("  Suma de los 50 pesos:", float(np.sum(pesos_pe)))

orden = np.argsort(pesos_pe)[::-1]
print("  Top-5 posiciones por peso:")
for p in orden[:5]:
    print(f"    pos {int(p):2d} -> peso {float(pesos_pe[p]):.6f}")

# ----------------------------------------------------------------------
# Figura 1: mapa de calor de la matriz PE (50 x 16)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(PE, aspect="auto", cmap="viridis", interpolation="nearest", origin="upper")
ax.set_xlabel("Dimensión del modelo (0–15)")
ax.set_ylabel("Posición (pos)")
ax.set_title("Codificación posicional sinusoidal — d_model=16, pos=0..49\n"
             "Pares: sin(pos/10000$^{2i/d}$) · Impares: cos(pos/10000$^{2i/d}$)")
ax.set_xticks(range(0, d_model, 2))
ax.set_yticks(range(0, n_pos, 5))
ax.axhline(10 - 0.5, color="red", linestyle="--", linewidth=1.0, label="pos = 10 (consulta)")
ax.legend(loc="lower right", fontsize=9)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Valor de PE")
fig.tight_layout()
fig.savefig("mapa_calor_pe.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# Figura 2 (apoyo): pesos de atención sobre las 50 posiciones
# ----------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 4.5))
ax2.plot(np.arange(n_pos), pesos_pe, marker="o", markersize=3, linewidth=1.2)
ax2.axvline(posicion_mayor_peso, color="red", linestyle="--",
            label=f"máx: pos {posicion_mayor_peso} (peso={peso_mayor:.6f})")
ax2.set_xlabel("Posición (pos)")
ax2.set_ylabel("Peso de atención")
ax2.set_title("Pesos de atención — consulta PE[10], claves/valores = PE completa")
ax2.legend()
fig2.tight_layout()
fig2.savefig("pesos_atencion_pe.png", dpi=150)
plt.close(fig2)

# ----------------------------------------------------------------------
# Guardar todos los resultados en resultados.json
# ----------------------------------------------------------------------
def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Tipo no serializable: {type(o)}")


resultados = {
    "PE_10_0": PE_10_0,
    "PE_10_1": PE_10_1,
    "PE_25_6": PE_25_6,
    "posicion_mayor_peso": posicion_mayor_peso,
    "peso_mayor": peso_mayor,
    "mapa_calor_pe": "mapa_calor_pe.png",
    # información de apoyo
    "d_model": int(d_model),
    "num_posiciones": int(n_pos),
    "d_k_atencion_pe": int(PE.shape[1]),
    "sqrt_d_k_atencion_pe": float(np.sqrt(PE.shape[1])),
    "PE_matriz_50x16": PE.tolist(),
    "scores_atencion_pe_1x50": scores_pe.tolist(),
    "pesos_atencion_pe_50": pesos_pe.tolist(),
    "salida_atencion_pe_1x16": salida_pe.tolist(),
    "suma_pesos_atencion_pe": float(np.sum(pesos_pe)),
    "figura_pesos_atencion": "pesos_atencion_pe.png",
    "verificacion_T1_max_abs_diff_pesos": diff_pesos,
    "verificacion_T1_max_abs_diff_salida": diff_salida,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False, default=_default)

print("\nGuardado: resultados.json, mapa_calor_pe.png, pesos_atencion_pe.png")
