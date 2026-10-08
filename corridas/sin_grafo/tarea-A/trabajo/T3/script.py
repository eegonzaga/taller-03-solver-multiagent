# -*- coding: utf-8 -*-
"""
T3 - Parte 3: Curva de aprendizaje.
- Reproduce la particion de T1 (load_digits, 75/25 estratificado, random_state=7).
- Entrena los dos modelos de T2 (GaussianNB y LogisticRegression) con
  m = 20, 50, 100, 200, 400 ejemplos (submuestras estratificadas con
  train_test_split(..., train_size=m, stratify=y_train, random_state=7))
  y con el entrenamiento completo.
- Evalua SIEMPRE sobre el mismo conjunto de prueba.
- Guarda cifras en resultados.json y la figura en PNG.
"""

import json
import warnings

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_STATE_SUB = 7          # semilla fijada por el enunciado para las submuestras
M_VALORES = [20, 50, 100, 200, 400]


def a_native(x):
    """Convierte tipos de NumPy a tipos nativos de Python."""
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


# ------------------------------------------------------------------
# 1) Entradas de subtareas anteriores (solo lectura, rutas relativas)
# ------------------------------------------------------------------
t1, t2 = {}, {}
try:
    with open("entradas/T1.json", "r", encoding="utf-8") as f:
        t1 = json.load(f)
except Exception:
    t1 = {}
try:
    with open("entradas/T2.json", "r", encoding="utf-8") as f:
        t2 = json.load(f)
except Exception:
    t2 = {}

random_state_split = int(t1.get("random_state", 7))
frac = t1.get("fraccion_test", t1.get("test_size", 0.25))
try:
    frac = float(frac)
except (TypeError, ValueError):
    frac = 0.25
if not (0.0 < frac < 1.0):
    frac = 0.25
test_size_split = frac

# ------------------------------------------------------------------
# 2) Datos y particion unica de T1 (75% train / 25% test, estratificada)
# ------------------------------------------------------------------
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size_split, stratify=y, random_state=random_state_split
)
n_train = int(len(y_train))
n_test = int(len(y_test))

split_coincide = True
if "n_train" in t1 and "n_test" in t1:
    split_coincide = (int(t1["n_train"]) == n_train) and (int(t1["n_test"]) == n_test)

# ------------------------------------------------------------------
# 3) Modelos de T2 (mismos estimadores; max_iter alto solo para converger)
# ------------------------------------------------------------------
def nuevos_modelos():
    return GaussianNB(), LogisticRegression(max_iter=1000)

# ------------------------------------------------------------------
# 4) Curva de aprendizaje: submuestras de tamano m (estratificadas)
# ------------------------------------------------------------------
acc_gnb, acc_lr = [], []
tam_submuestras = []
for m in M_VALORES:
    X_sub, _, y_sub, _ = train_test_split(
        X_train, y_train, train_size=m, stratify=y_train, random_state=RANDOM_STATE_SUB
    )
    tam_submuestras.append(int(len(y_sub)))
    gnb, lr = nuevos_modelos()
    gnb.fit(X_sub, y_sub)          # ajuste SOLO con la submuestra de entrenamiento
    lr.fit(X_sub, y_sub)
    acc_gnb.append(float(gnb.score(X_test, y_test)))   # evaluacion en el mismo test
    acc_lr.append(float(lr.score(X_test, y_test)))

# Entrenamiento con el conjunto de entrenamiento completo
gnb_full, lr_full = nuevos_modelos()
gnb_full.fit(X_train, y_train)
lr_full.fit(X_train, y_train)
acc_gnb_full = float(gnb_full.score(X_test, y_test))
acc_lr_full = float(lr_full.score(X_test, y_test))

# ------------------------------------------------------------------
# 5) Punto de cruce aproximado entre las dos curvas (interpolacion lineal)
# ------------------------------------------------------------------
ms = np.array(M_VALORES + [n_train], dtype=float)
accs_gnb = np.array(acc_gnb + [acc_gnb_full], dtype=float)
accs_lr = np.array(acc_lr + [acc_lr_full], dtype=float)
dif = accs_gnb - accs_lr

m_cruce = None
for i in range(len(ms)):
    if dif[i] == 0.0:
        m_cruce = float(ms[i])
        break
if m_cruce is None:
    for i in range(len(ms) - 1):
        if dif[i] * dif[i + 1] < 0.0:
            m_cruce = float(ms[i] - dif[i] * (ms[i + 1] - ms[i]) / (dif[i + 1] - dif[i]))
            break

# ------------------------------------------------------------------
# 6) Figura PNG: exactitud contra numero de ejemplos (una curva por modelo)
# ------------------------------------------------------------------
plt.figure(figsize=(8.5, 5.2))
plt.plot(ms, accs_gnb, "o-", color="tab:blue", label="GaussianNB")
plt.plot(ms, accs_lr, "s-", color="tab:orange", label="Regresion logistica")
if m_cruce is not None:
    plt.axvline(m_cruce, color="gray", linestyle="--", linewidth=1.2,
                label="cruce aprox. m = %.0f" % m_cruce)
plt.xlabel("Numero de ejemplos de entrenamiento (m)")
plt.ylabel("Exactitud en el conjunto de prueba")
plt.title("Curva de aprendizaje - load_digits (particion T1, random_state=%d)"
          % random_state_split)
plt.xticks(list(ms), [str(int(v)) for v in ms])
plt.ylim(0.0, 1.02)
plt.grid(alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("T3_parte3_curva_aprendizaje.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 7) Guardar todas las cifras en resultados.json
# ------------------------------------------------------------------
exactitud_gnb_por_m = {str(int(m)): float(a) for m, a in zip(ms, accs_gnb)}
exactitud_lr_por_m = {str(int(m)): float(a) for m, a in zip(ms, accs_lr)}

resultados = {
    "subtarea": "T3 - Parte 3 - Curva de aprendizaje",
    "dataset": "sklearn.datasets.load_digits",
    "random_state_particion": random_state_split,
    "test_size_particion": test_size_split,
    "random_state_submuestras": RANDOM_STATE_SUB,
    "split_reproducido_coincide_T1": bool(split_coincide),
    "n_train": n_train,
    "n_test": n_test,
    "m_valores": [int(m) for m in M_VALORES],
    "tam_submuestras": [int(s) for s in tam_submuestras],
    "exactitud_gnb_por_m": exactitud_gnb_por_m,
    "exactitud_lr_por_m": exactitud_lr_por_m,
    "exactitud_gnb_entrenamiento_completo": acc_gnb_full,
    "exactitud_lr_entrenamiento_completo": acc_lr_full,
    "diferencia_gnb_menos_lr_por_m": {str(int(m)): float(d) for m, d in zip(ms, dif)},
    "m_cruce_aproximado": (float(m_cruce) if m_cruce is not None else None),
    "figura": "T3_parte3_curva_aprendizaje.png",
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ------------------------------------------------------------------
# 8) Impresion de las cifras principales
# ------------------------------------------------------------------
print("=== T3 - Parte 3: Curva de aprendizaje ===")
print("Dataset: load_digits | train=%d | test=%d | split coincide con T1: %s"
      % (n_train, n_test, split_coincide))
print("\nExactitud en el conjunto de prueba (mismo test siempre):")
print("%10s | %10s | %10s | %13s" % ("m", "GaussianNB", "RegLog", "dif GNB-LR"))
for m, ag, al, d in zip(ms, accs_gnb, accs_lr, dif):
    etiqueta = "completo" if int(m) == n_train else str(int(m))
    print("%10s | %10.4f | %10.4f | %+13.4f" % (etiqueta, ag, al, d))
print("\nExactitud GNB (entrenamiento completo): %.6f" % acc_gnb_full)
print("Exactitud LR  (entrenamiento completo): %.6f" % acc_lr_full)
if m_cruce is not None:
    print("m de cruce aproximado entre las curvas: %.2f" % m_cruce)
else:
    print("Las curvas no se cruzan en los valores de m evaluados.")
print("\nFigura guardada: T3_parte3_curva_aprendizaje.png")
print("Resultados guardados: resultados.json")
