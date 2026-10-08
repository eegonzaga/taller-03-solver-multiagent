# -*- coding: utf-8 -*-
"""
T3 — Curva de aprendizaje con submuestras (dígitos de scikit-learn).

Protocolo:
  * Partición única de T1: load_digits, 75%/25%, estratificada, random_state=7
    (parámetros recuperados de entradas/T1.json para reproducirla exactamente).
  * Modelos de T2:
      - GaussianNB (valores por defecto) sobre los atributos originales, sin escalar.
      - LogisticRegression(max_iter=3000) sobre atributos estandarizados con
        StandardScaler ajustado SOLO con los ejemplos de entrenamiento usados
        en cada corrida (nunca con el conjunto de prueba).
  * Submuestras de m = 20, 50, 100, 200, 400 obtenidas con
        train_test_split(X_train, y_train, train_size=m,
                         stratify=y_train, random_state=7)
    y, además, una corrida con el conjunto de entrenamiento completo.
  * Evaluación SIEMPRE sobre el mismo conjunto de prueba de T1.
  * Figura: curva_aprendizaje.png ; cifras: resultados.json
"""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

RANDOM_STATE_SUB = 7  # semilla fijada por el enunciado para todas las submuestras

# ---------------------------------------------------------------------------
# 1) Parámetros de la partición única de T1 (entradas/T1.json)
# ---------------------------------------------------------------------------
random_state_part = 7
test_size_part = 0.25
try:
    with open("entradas/T1.json", "r", encoding="utf-8") as f:
        t1 = json.load(f)
    random_state_part = int(t1.get("random_state", 7))
    test_size_part = float(t1.get("test_size", 0.25))
except Exception:
    pass  # si no está el archivo, se usan los valores del enunciado (75/25, semilla 7)

# Referencia de T2 (para verificación de consistencia, opcional)
t2_ref_gnb, t2_ref_lr = None, None
try:
    with open("entradas/T2.json", "r", encoding="utf-8") as f:
        t2 = json.load(f)
    t2_ref_gnb = float(t2["accuracy_gnb"])
    t2_ref_lr = float(t2["accuracy_lr"])
except Exception:
    pass

# ---------------------------------------------------------------------------
# 2) Datos y partición única (idéntica a la de T1)
# ---------------------------------------------------------------------------
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size_part, stratify=y, random_state=random_state_part
)
n_train = int(len(y_train))
n_test = int(len(y_test))

# ---------------------------------------------------------------------------
# 3) Entrenamiento y evaluación para cada tamaño m y para el train completo
# ---------------------------------------------------------------------------
def entrenar_y_evaluar(X_tr, y_tr):
    """Entrena los dos modelos de T2 con los ejemplos dados y evalúa en el test de T1.

    El StandardScaler de la regresión logística se ajusta únicamente con los
    ejemplos de entrenamiento de esa corrida (la submuestra de m ejemplos, o el
    train completo); el conjunto de prueba solo se transforma, nunca se usa para
    ajustar nada.
    """
    # --- Generativo: GaussianNB sobre atributos originales, sin escalar ---
    gnb = GaussianNB()
    gnb.fit(X_tr, y_tr)
    acc_gnb = float(accuracy_score(y_test, gnb.predict(X_test)))

    # --- Discriminativo: LR sobre atributos estandarizados ---
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)   # ajuste SOLO con el entrenamiento de la corrida
    X_te_s = scaler.transform(X_test)     # el test solo se transforma
    lr = LogisticRegression(max_iter=3000)
    lr.fit(X_tr_s, y_tr)
    acc_lr = float(accuracy_score(y_test, lr.predict(X_te_s)))

    return acc_gnb, acc_lr

m_valores = [20, 50, 100, 200, 400]
assert all(m < n_train for m in m_valores), "m debe ser menor que n_train"

accuracy_gnb_por_m = []
accuracy_lr_por_m = []
for m in m_valores:
    X_sub, _, y_sub, _ = train_test_split(
        X_train, y_train, train_size=m, stratify=y_train, random_state=RANDOM_STATE_SUB
    )
    acc_gnb, acc_lr = entrenar_y_evaluar(X_sub, y_sub)
    accuracy_gnb_por_m.append(acc_gnb)
    accuracy_lr_por_m.append(acc_lr)

# Corrida con el conjunto de entrenamiento completo (mismo protocolo que T2)
acc_gnb_full, acc_lr_full = entrenar_y_evaluar(X_train, y_train)

# ---------------------------------------------------------------------------
# 4) Figura: exactitud contra número de ejemplos (una curva por modelo)
# ---------------------------------------------------------------------------
x_vals = m_valores + [n_train]
gnb_curve = accuracy_gnb_por_m + [acc_gnb_full]
lr_curve = accuracy_lr_por_m + [acc_lr_full]

plt.figure(figsize=(9, 5.5))
plt.plot(x_vals, gnb_curve, "o-", color="tab:blue",
         label="GaussianNB (generativo, atributos originales)")
plt.plot(x_vals, lr_curve, "s-", color="tab:orange",
         label="LogisticRegression (discriminativo, atributos estandarizados)")
plt.plot([n_train], [acc_gnb_full], "*", markersize=16, color="tab:blue")
plt.plot([n_train], [acc_lr_full], "*", markersize=16, color="tab:orange")
plt.axvline(n_train, color="gray", linestyle=":", linewidth=1)
y_lo = max(0.0, min(min(gnb_curve), min(lr_curve)) - 0.05)
plt.ylim(y_lo, 1.02)
plt.text(n_train, y_lo + 0.015, f"train completo\n(n={n_train})",
         fontsize=8, color="gray", ha="right", va="bottom")
plt.xlabel("Número de ejemplos de entrenamiento (m)")
plt.ylabel("Exactitud (accuracy) en el conjunto de prueba")
plt.title("Curva de aprendizaje en dígitos: Naive Bayes gaussiano vs. Regresión logística")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("curva_aprendizaje.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5) Guardar todas las cifras en resultados.json
# ---------------------------------------------------------------------------
resultados = {
    "m_valores": [int(m) for m in m_valores],
    "accuracy_gnb_por_m": [float(a) for a in accuracy_gnb_por_m],
    "accuracy_lr_por_m": [float(a) for a in accuracy_lr_por_m],
    "n_train_completo": n_train,
    "accuracy_gnb_completo": float(acc_gnb_full),
    "accuracy_lr_completo": float(acc_lr_full),
    "x_valores_curva": [int(v) for v in x_vals],
    "accuracy_gnb_curva_completa": [float(a) for a in gnb_curve],
    "accuracy_lr_curva_completa": [float(a) for a in lr_curve],
    "random_state_particion": int(random_state_part),
    "test_size_particion": float(test_size_part),
    "random_state_submuestras": int(RANDOM_STATE_SUB),
    "n_test": n_test,
    "figura": "curva_aprendizaje.png",
}
if t2_ref_gnb is not None:
    resultados["accuracy_gnb_T2_referencia"] = float(t2_ref_gnb)
if t2_ref_lr is not None:
    resultados["accuracy_lr_T2_referencia"] = float(t2_ref_lr)

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# 6) Impresión de las cifras principales
# ---------------------------------------------------------------------------
print("T3 — Curva de aprendizaje (evaluación siempre sobre el mismo test de T1)")
print(f"Partición: test_size={test_size_part}, random_state={random_state_part} "
      f"-> n_train={n_train}, n_test={n_test}")
print(f"Submuestras: train_test_split(X_train, y_train, train_size=m, "
      f"stratify=y_train, random_state={RANDOM_STATE_SUB})")
print("\n   m     acc_GNB     acc_LR")
for m, ag, al in zip(m_valores, accuracy_gnb_por_m, accuracy_lr_por_m):
    print(f"{m:5d}   {ag:.6f}   {al:.6f}")
print(f"{n_train:5d}   {acc_gnb_full:.6f}   {acc_lr_full:.6f}   (entrenamiento completo)")

if t2_ref_gnb is not None:
    print("\nVerificación con T2 (train completo):")
    print(f"  GNB: {acc_gnb_full:.6f} (T3) vs {t2_ref_gnb:.6f} (T2)")
    print(f"  LR : {acc_lr_full:.6f} (T3) vs {t2_ref_lr:.6f} (T2)")

print("\nFigura guardada: curva_aprendizaje.png")
print("Resultados guardados: resultados.json")
