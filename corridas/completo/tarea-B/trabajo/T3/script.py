# -*- coding: utf-8 -*-
"""
SUBTAREA T3 — Parte 3 — Codificación posicional sinusoidal
- Calcula PE(pos,2i)=sin(pos/10000^(2i/d_model)) y PE(pos,2i+1)=cos(pos/10000^(2i/d_model))
  con d_model=16 para posiciones 0..49.
- Reporta PE[10,0], PE[10,1], PE[25,6].
- Dibuja la matriz PE (50x16) como mapa de calor (PNG).
- Reutiliza la función de atención de la Parte 1 (verificada contra entradas/T1.json):
  usa PE[10] como consulta y la matriz PE completa (50x16) como claves y valores,
  e identifica la posición con mayor peso y su valor.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0. Función de atención de la Parte 1 (softmax numéricamente estable)
#    d_k se infiere de la dimensión de los vectores consulta/clave.
# ----------------------------------------------------------------------
def softmax_estable(x, axis=-1):
    """Softmax numéricamente estable: resta el máximo de cada fila."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)

def atencion_escalada(Q, K, V):
    """Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V  (misma función de la Parte 1)."""
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    pesos = softmax_estable(scores, axis=-1)
    salida = pesos @ V
    return scores, pesos, salida

# Verificación: la función debe reproducir los resultados de la Parte 1 (T1.json)
try:
    with open("entradas/T1.json", "r", encoding="utf-8") as f:
        t1 = json.load(f)
    Q1 = np.array(t1["Q_2x4"], dtype=float)
    K1 = np.array(t1["K_3x4"], dtype=float)
    V1 = np.array(t1["V_3x2"], dtype=float)
    _, pesos1, salida1 = atencion_escalada(Q1, K1, V1)
    ok_pesos = bool(np.allclose(pesos1, np.array(t1["pesos_atencion_2x3"], dtype=float)))
    ok_salida = bool(np.allclose(salida1, np.array(t1["salida_atencion_2x2"], dtype=float)))
    print(f"[Verificación Parte 1] pesos coinciden: {ok_pesos} | salida coincide: {ok_salida}")
except FileNotFoundError:
    print("[Aviso] entradas/T1.json no encontrado; se usa la función de atención tal cual.")

# ----------------------------------------------------------------------
# 1. Codificación posicional sinusoidal (d_model=16, posiciones 0..49)
# ----------------------------------------------------------------------
d_model = 16
n_pos = 50
posiciones = np.arange(n_pos)

PE = np.zeros((n_pos, d_model), dtype=float)
for pos in posiciones:
    for i in range(d_model // 2):
        angulo = pos / (10000.0 ** (2.0 * i / d_model))
        PE[pos, 2 * i] = np.sin(angulo)       # PE(pos, 2i)
        PE[pos, 2 * i + 1] = np.cos(angulo)   # PE(pos, 2i+1)

PE_10_0 = float(PE[10, 0])
PE_10_1 = float(PE[10, 1])
PE_25_6 = float(PE[25, 6])

# ----------------------------------------------------------------------
# 2. Atención: consulta = PE[10], claves y valores = PE completa (50x16)
#    Aquí d_k = 16 (dimensión de los vectores), sqrt(d_k) = 4.
# ----------------------------------------------------------------------
Q_pe = PE[10:11, :]   # (1, 16)
K_pe = PE             # (50, 16)
V_pe = PE             # (50, 16)

scores_pe, pesos_pe_mat, salida_pe = atencion_escalada(Q_pe, K_pe, V_pe)
pesos_pe = pesos_pe_mat.flatten()  # (50,)

posicion_mayor_peso = int(np.argmax(pesos_pe))
valor_mayor_peso = float(pesos_pe[posicion_mayor_peso])

# ----------------------------------------------------------------------
# 3. Guardar todas las cifras en resultados.json (precisión completa)
# ----------------------------------------------------------------------
resultados = {
    "d_model": int(d_model),
    "num_posiciones": int(n_pos),
    "PE_10_0": PE_10_0,
    "PE_10_1": PE_10_1,
    "PE_25_6": PE_25_6,
    "posicion_mayor_peso": posicion_mayor_peso,
    "valor_mayor_peso": valor_mayor_peso,
    "pesos_atencion_PE10_sobre_50_posiciones": [float(x) for x in pesos_pe],
    "salida_atencion_PE_16": [float(x) for x in salida_pe.flatten()],
    "PE_matriz_50x16": [[float(x) for x in fila] for fila in PE],
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ----------------------------------------------------------------------
# 4. Impresión de las cifras principales
# ----------------------------------------------------------------------
print("\n--- Parte 3: Codificación posicional sinusoidal (d_model=16, pos 0..49) ---")
print(f"PE[10, 0] = sin(10 / 10000^0)      = {PE_10_0!r}")
print(f"PE[10, 1] = cos(10 / 10000^0)      = {PE_10_1!r}")
print(f"PE[25, 6] = sin(25 / 10000^(6/16)) = {PE_25_6!r}")

print("\nAtención con PE[10] como consulta y PE (50x16) como claves/valores (d_k=16):")
top5 = np.argsort(pesos_pe)[::-1][:5]
for idx in top5:
    print(f"  posición {int(idx):2d} -> peso = {pesos_pe[idx]:.10f}")
print(f"\nPosición con mayor peso: {posicion_mayor_peso}")
print(f"Valor del mayor peso:    {valor_mayor_peso!r}")

# ----------------------------------------------------------------------
# 5. Figuras
# ----------------------------------------------------------------------
# Mapa de calor de la matriz PE (50 x 16)
plt.figure(figsize=(9, 8))
im = plt.imshow(PE, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0, origin="upper")
plt.colorbar(im, label="Valor de PE")
plt.xlabel("Dimensión j (pares sin/cos, i = j // 2)")
plt.ylabel("Posición pos")
plt.title("Codificación posicional sinusoidal\n"
          "PE(pos,2i)=sin(pos/10000^(2i/16)),  PE(pos,2i+1)=cos(pos/10000^(2i/16)),  pos=0..49")
plt.xticks(ticks=np.arange(0, d_model, 2))
plt.yticks(ticks=np.arange(0, n_pos, 5))
plt.tight_layout()
plt.savefig("mapa_calor_codificacion_posicional.png", dpi=150)
plt.close()

# (complemento) Pesos de atención de la consulta PE[10] sobre las 50 posiciones
plt.figure(figsize=(9, 4))
plt.bar(np.arange(n_pos), pesos_pe, color="steelblue")
plt.axvline(posicion_mayor_peso, color="crimson", linestyle="--",
            label=f"máximo: pos {posicion_mayor_peso} (peso = {valor_mayor_peso:.4f})")
plt.xlabel("Posición (clave/valor)")
plt.ylabel("Peso de atención")
plt.title("Atención escalada: consulta PE[10] sobre las 50 posiciones (d_k = 16)")
plt.legend()
plt.tight_layout()
plt.savefig("pesos_atencion_PE10.png", dpi=150)
plt.close()

print("\nFiguras guardadas: mapa_calor_codificacion_posicional.png, pesos_atencion_PE10.png")
print("Resultados guardados en resultados.json")
