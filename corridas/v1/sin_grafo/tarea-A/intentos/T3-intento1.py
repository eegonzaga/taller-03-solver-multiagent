# -*- coding: utf-8 -*-
# T3 — Parte 3: Curva de aprendizaje
# - Partición: índices de entrenamiento/prueba definidos en T1 (entradas/T1.json)
# - Modelos: los dos de T2 (GaussianNB y Regresión Logística)
# - Submuestras de entrenamiento: m = 20, 50, 100, 200, 400 con
#   train_test_split(X_train, y_train, train_size=m, stratify=y_train, random_state=7)
#   más el entrenamiento completo. Evaluación SIEMPRE sobre el mismo conjunto de prueba.

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------- 1) Cargar la partición única de T1 ----------
with open("entradas/T1.json", "r") as f:
    t1 = json.load(f)

train_indices = np.asarray(t1["train_indices"], dtype=int)
test_indices = np.asarray(t1["test_indices"], dtype=int)

# Referencia de T2 (solo para comparar el punto de entrenamiento completo)
try:
    with open("entradas/T2.json", "r") as f:
        t2 = json.load(f)
except FileNotFoundError:
    t2 = {}

# ---------- 2) Datos y partición (misma de T1, sin rehacer el split) ----------
X, y = load_digits(return_X_y=True)
X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]
n_train = int(X_train.shape[0])
n_test = int(X_test.shape[0])
print(f"Entrenamiento: {n_train} ejemplos | Prueba: {n_test} ejemplos (partición de T1)")

# ---------- 3) Modelos (los mismos de T2) ----------
def nuevo_gnb():
    return GaussianNB()

def nuevo_lr():
    # max_iter alto solo para garantizar convergencia; no se ajusta nada con el test
    return LogisticRegression(max_iter=1000)

# ---------- 4) Curva de aprendizaje ----------
valores_m = [20, 50, 100, 200, 400]
exactitud_gnb_por_m = {}
exactitud_lr_por_m = {}

for m in valores_m:
    X_sub, _, y_sub, _ = train_test_split(
        X_train, y_train,
        train_size=m,
        stratify=y_train,
        random_state=7,
    )
    gnb = nuevo_gnb().fit(X_sub, y_sub)
    lr = nuevo_lr().fit(X_sub, y_sub)
    acc_gnb = float(accuracy_score(y_test, gnb.predict(X_test)))
    acc_lr = float(accuracy_score(y_test, lr.predict(X_test)))
    exactitud_gnb_por_m[str(m)] = acc_gnb
    exactitud_lr_por_m[str(m)] = acc_lr
    print(f"m = {m:>4} | Exactitud GNB = {acc_gnb:.6f} | Exactitud LR = {acc_lr:.6f}")

# Punto final: entrenamiento completo (mismo conjunto de prueba)
gnb_full = nuevo_gnb().fit(X_train, y_train)
lr_full = nuevo_lr().fit(X_train, y_train)
acc_gnb_full = float(accuracy_score(y_test, gnb_full.predict(X_test)))
acc_lr_full = float(accuracy_score(y_test, lr_full.predict(X_test)))
exactitud_gnb_por_m["completo"] = acc_gnb_full
exactitud_lr_por_m["completo"] = acc_lr_full
print(f"m = {n_train} (completo) | Exactitud GNB = {acc_gnb_full:.6f} | Exactitud LR = {acc_lr_full:.6f}")

if t2:
    print(f"[Referencia T2] accuracy_gnb = {t2.get('accuracy_gnb')} | accuracy_lr = {t2.get('accuracy_lr')}")

# ---------- 5) Figura: exactitud vs número de ejemplos ----------
eje_x = valores_m + [n_train]
gnb_y = [exactitud_gnb_por_m[str(m)] for m in valores_m] + [acc_gnb_full]
lr_y = [exactitud_lr_por_m[str(m)] for m in valores_m] + [acc_lr_full]

plt.figure(figsize=(8, 5))
plt.plot(eje_x, gnb_y, "o-", color="tab:blue", label="GaussianNB")
plt.plot(eje_x, lr_y, "s-", color="tab:orange", label="Regresión logística")
plt.xscale("log")
plt.xticks(eje_x, [str(m) for m in valores_m] + [f"{n_train}\n(completo)"])
plt.minorticks_off()
y_min = min(gnb_y + lr_y)
plt.ylim(max(0.0, y_min - 0.05), 1.02)
plt.xlabel("Número de ejemplos de entrenamiento (m)")
plt.ylabel("Exactitud en el conjunto de prueba")
plt.title("Curva de aprendizaje — dígitos (partición T1, random_state=7)")
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("curva_aprendizaje.png", dpi=150)
plt.close()
print("Figura guardada: curva_aprendizaje.png")

# ---------- 6) Guardar todas las cifras en resultados.json ----------
resultados = {
    "exactitud_gnb_por_m": exactitud_gnb_por_m,   # {"20": ..., "50": ..., "100": ..., "200": ..., "400": ..., "completo": ...}
    "exactitud_lr_por_m": exactitud_lr_por_m,     # {"20": ..., "50": ..., "100": ..., "200": ..., "400": ..., "completo": ...}
    "valores_m": [int(m) for m in valores_m],
    "tam_entrenamiento_completo": n_train,
    "tam_test": n_test,
    "random_state_submuestras": 7,
    "figura_png": "curva_aprendizaje.png",
}
if t2:
    resultados["referencia_T2"] = {
        "accuracy_gnb": t2.get("accuracy_gnb"),
        "accuracy_lr": t2.get("accuracy_lr"),
    }

with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)
print("Resultados guardados en resultados.json")
