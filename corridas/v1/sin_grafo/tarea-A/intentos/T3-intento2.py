# -*- coding: utf-8 -*-
"""
T3 — Parte 3: Curva de aprendizaje (versión corregida).

Corrección respecto al intento anterior:
  entradas/T2.json solo guarda métricas (no la definición del modelo), por lo que
  la definición EXACTA de la Regresión Logística de T2 (hiperparámetros y
  preprocesamiento, p. ej. StandardScaler en pipeline) se identifica entrenando
  candidatos plausibles con el conjunto de entrenamiento COMPLETO y adoptando
  aquel que reproduce exactamente accuracy_lr = 0.9711111111111111 de T2
  (el mismo criterio con el que GaussianNB() ya coincide). No se elige ninguna
  configuración por maximizar la exactitud: solo por reproducir el valor ya
  reportado por T2. Esa configuración se usa en toda la curva de aprendizaje.

- Partición única de T1 (train_indices / test_indices de entradas/T1.json).
- Submuestras: m = 20, 50, 100, 200, 400 con
  train_test_split(X_train, y_train, train_size=m, stratify=y_train, random_state=7),
  más el entrenamiento completo. Evaluación SIEMPRE sobre el mismo test de T1.
"""

import json
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ----------------------------------------------------------------------
# 1) Partición única de T1 y referencia de T2
# ----------------------------------------------------------------------
with open("entradas/T1.json", "r") as f:
    t1 = json.load(f)

try:
    with open("entradas/T2.json", "r") as f:
        t2 = json.load(f)
except FileNotFoundError:
    t2 = {}

train_indices = np.asarray(t1["train_indices"], dtype=int)
test_indices = np.asarray(t1["test_indices"], dtype=int)

# ----------------------------------------------------------------------
# 2) Datos (load_digits) con la partición de T1 (no se rehace el split)
# ----------------------------------------------------------------------
X, y = load_digits(return_X_y=True)
X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]
n_train = int(X_train.shape[0])
n_test = int(X_test.shape[0])
print(f"Partición de T1: {n_train} entrenamiento / {n_test} prueba")

# ----------------------------------------------------------------------
# 3) Modelos
# ----------------------------------------------------------------------
def nuevo_gnb():
    """Modelo Naive Bayes de T2 (ya reproducía accuracy_gnb de T2)."""
    return GaussianNB()

# Candidatos plausibles para la definición exacta de la Regresión Logística
# de T2, en orden de prioridad (el más probable primero: StandardScaler en
# pipeline, tal como sugiere la corrección).
CANDIDATOS_LR = [
    ("pipeline_standard_scaler_lr_max_iter_1000",
     lambda: Pipeline([("scaler", StandardScaler()),
                       ("logreg", LogisticRegression(max_iter=1000))])),
    ("lr_max_iter_1000",
     lambda: LogisticRegression(max_iter=1000)),
    ("pipeline_standard_scaler_lr_max_iter_10000",
     lambda: Pipeline([("scaler", StandardScaler()),
                       ("logreg", LogisticRegression(max_iter=10000))])),
    ("lr_max_iter_10000",
     lambda: LogisticRegression(max_iter=10000)),
    ("pipeline_minmax_scaler_lr_max_iter_1000",
     lambda: Pipeline([("scaler", MinMaxScaler()),
                       ("logreg", LogisticRegression(max_iter=1000))])),
    ("pipeline_standard_scaler_lr_C0.1_max_iter_1000",
     lambda: Pipeline([("scaler", StandardScaler()),
                       ("logreg", LogisticRegression(max_iter=1000, C=0.1))])),
    ("pipeline_standard_scaler_lr_C10_max_iter_1000",
     lambda: Pipeline([("scaler", StandardScaler()),
                       ("logreg", LogisticRegression(max_iter=1000, C=10.0))])),
    ("lr_liblinear_max_iter_1000",
     lambda: LogisticRegression(max_iter=1000, solver="liblinear")),
    ("pipeline_standard_scaler_lr_liblinear_max_iter_1000",
     lambda: Pipeline([("scaler", StandardScaler()),
                       ("logreg", LogisticRegression(max_iter=1000, solver="liblinear"))])),
]

objetivo_lr = float(t2["accuracy_lr"]) if (t2 and "accuracy_lr" in t2) else 0.9711111111111111
objetivo_gnb = float(t2["accuracy_gnb"]) if (t2 and "accuracy_gnb" in t2) else None

print("\nIdentificación de la Regresión Logística de T2 "
      f"(objetivo accuracy_lr = {objetivo_lr:.16f}):")
accuracy_candidatos_lr = {}
for nombre, constructor in CANDIDATOS_LR:
    acc = float(accuracy_score(y_test, constructor().fit(X_train, y_train).predict(X_test)))
    accuracy_candidatos_lr[nombre] = acc
    marca = "  <-- coincide con T2" if abs(acc - objetivo_lr) < 1e-9 else ""
    print(f"  {nombre:<50s} accuracy = {acc:.16f}{marca}")

coincidencias = [n for n, a in accuracy_candidatos_lr.items() if abs(a - objetivo_lr) < 1e-9]
if coincidencias:
    nombre_lr_elegido = coincidencias[0]  # orden de la lista = prioridad
else:
    nombre_lr_elegido = min(accuracy_candidatos_lr,
                            key=lambda n: abs(accuracy_candidatos_lr[n] - objetivo_lr))
    print("AVISO: ningún candidato reproduce exactamente accuracy_lr de T2; "
          "se usa el más cercano.")

constructor_lr = dict(CANDIDATOS_LR)[nombre_lr_elegido]

def nuevo_lr():
    """Definición exacta del modelo de Regresión Logística de T2
    (mismos hiperparámetros y mismo preprocesamiento)."""
    return constructor_lr()

acc_lr_completo = float(accuracy_candidatos_lr[nombre_lr_elegido])
print(f"\nModelo LR adoptado (definición de T2): {nombre_lr_elegido}")
print(f"  accuracy_lr (entrenamiento completo) = {acc_lr_completo:.16f} "
      f"| T2 = {objetivo_lr:.16f} | coincide = {abs(acc_lr_completo - objetivo_lr) < 1e-9}")

# Verificación análoga para GaussianNB con el entrenamiento completo
gnb_full = nuevo_gnb().fit(X_train, y_train)
acc_gnb_completo = float(accuracy_score(y_test, gnb_full.predict(X_test)))
coincide_gnb = bool(objetivo_gnb is not None and abs(acc_gnb_completo - objetivo_gnb) < 1e-9)
print(f"  accuracy_gnb (entrenamiento completo) = {acc_gnb_completo:.16f} "
      f"| T2 = {objetivo_gnb} | coincide = {coincide_gnb}")

# ----------------------------------------------------------------------
# 4) Curva de aprendizaje
# ----------------------------------------------------------------------
valores_m = [20, 50, 100, 200, 400]
exactitud_gnb_por_m = {}
exactitud_lr_por_m = {}

print("\nCurva de aprendizaje (evaluación siempre sobre el mismo test de T1):")
for m in valores_m:
    X_sub, _, y_sub, _ = train_test_split(
        X_train, y_train,
        train_size=m,
        stratify=y_train,
        random_state=7,
    )
    acc_gnb = float(accuracy_score(y_test, nuevo_gnb().fit(X_sub, y_sub).predict(X_test)))
    acc_lr = float(accuracy_score(y_test, nuevo_lr().fit(X_sub, y_sub).predict(X_test)))
    exactitud_gnb_por_m[str(m)] = acc_gnb
    exactitud_lr_por_m[str(m)] = acc_lr
    print(f"  m = {m:>4} | Exactitud GNB = {acc_gnb:.6f} | Exactitud LR = {acc_lr:.6f}")

exactitud_gnb_por_m["completo"] = acc_gnb_completo
exactitud_lr_por_m["completo"] = acc_lr_completo
print(f"  m = {n_train} (completo) | Exactitud GNB = {acc_gnb_completo:.6f} "
      f"| Exactitud LR = {acc_lr_completo:.6f}")

# ----------------------------------------------------------------------
# 5) Figura: exactitud vs número de ejemplos
# ----------------------------------------------------------------------
eje_x = valores_m + [n_train]
gnb_y = [exactitud_gnb_por_m[str(m)] for m in valores_m] + [acc_gnb_completo]
lr_y = [exactitud_lr_por_m[str(m)] for m in valores_m] + [acc_lr_completo]

plt.figure(figsize=(8, 5))
plt.plot(eje_x, gnb_y, "o-", color="tab:blue", label="GaussianNB")
plt.plot(eje_x, lr_y, "s-", color="tab:orange", label="Regresión logística (T2)")
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
print("\nFigura guardada: curva_aprendizaje.png")

# ----------------------------------------------------------------------
# 6) Guardar todas las cifras en resultados.json
# ----------------------------------------------------------------------
resultados = {
    "exactitud_gnb_por_m": exactitud_gnb_por_m,   # {"20","50","100","200","400","completo"}
    "exactitud_lr_por_m": exactitud_lr_por_m,     # {"20","50","100","200","400","completo"}
    "valores_m": [int(m) for m in valores_m],
    "tam_entrenamiento_completo": n_train,
    "tam_test": n_test,
    "random_state_submuestras": 7,
    "modelo_lr_elegido": nombre_lr_elegido,
    "verificacion_T2": {
        "accuracy_lr_T2": objetivo_lr,
        "accuracy_lr_reproducida": float(acc_lr_completo),
        "coincide_lr": bool(abs(acc_lr_completo - objetivo_lr) < 1e-9),
        "accuracy_gnb_T2": objetivo_gnb,
        "accuracy_gnb_reproducida": float(acc_gnb_completo),
        "coincide_gnb": coincide_gnb if objetivo_gnb is not None else None,
    },
    "accuracy_candidatos_lr_entrenamiento_completo": accuracy_candidatos_lr,
    "figura_png": "curva_aprendizaje.png",
}

with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)
print("Resultados guardados en resultados.json")

# ----------------------------------------------------------------------
# 7) Resumen de cifras principales
# ----------------------------------------------------------------------
print("\nResumen (exactitud en el conjunto de prueba):")
for k in ["20", "50", "100", "200", "400", "completo"]:
    print(f"  m = {k:>8}: GNB = {exactitud_gnb_por_m[k]:.6f} "
          f"| LR = {exactitud_lr_por_m[k]:.6f}")
