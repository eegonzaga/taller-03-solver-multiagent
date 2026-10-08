# -*- coding: utf-8 -*-
"""
SUBTAREA T1 — Datos y partición única
Tarea A — ¿Gana el generativo con pocos datos? (Ng y Jordan sobre dígitos)

Carga sklearn.datasets.load_digits, reporta número de ejemplos, atributos y clases,
y crea la ÚNICA partición de la tarea (75% train / 25% test, estratificada por clase,
random_state=7). La partición se guarda (por índices) para reutilizarla en las
subtareas siguientes. El conjunto de prueba no se usa para ajustar ni elegir nada.
"""

import json
import os

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# 1) Cargar los datos
# ----------------------------------------------------------------------
digits = load_digits()
X = digits.data          # matriz (n_ejemplos, n_atributos), valores 0..16
y = digits.target        # etiquetas 0..9

n_ejemplos = int(X.shape[0])
n_atributos = int(X.shape[1])
clases = np.unique(y)
n_clases = int(clases.size)

# ----------------------------------------------------------------------
# 2) Partición única: 75% entrenamiento / 25% prueba, estratificada
#    Se particionan también los índices para poder reconstruir/guardar
#    exactamente la misma división en subtareas posteriores.
# ----------------------------------------------------------------------
indices = np.arange(n_ejemplos)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices,
    test_size=0.25,
    stratify=y,
    random_state=7,
)

n_train = int(X_train.shape[0])
n_test = int(X_test.shape[0])

# Distribución de clases por bloque (verificación de la estratificación)
train_clases, train_cuentas = np.unique(y_train, return_counts=True)
test_clases, test_cuentas = np.unique(y_test, return_counts=True)
train_class_counts = {str(int(c)): int(n) for c, n in zip(train_clases, train_cuentas)}
test_class_counts = {str(int(c)): int(n) for c, n in zip(test_clases, test_cuentas)}

# ----------------------------------------------------------------------
# 3) Guardar todas las cifras en resultados.json (carpeta actual)
# ----------------------------------------------------------------------
resultados = {
    "T1": {
        "n_ejemplos": n_ejemplos,
        "n_atributos": n_atributos,
        "n_clases": n_clases,
        "n_train": n_train,
        "n_test": n_test,
        "particion": {
            "test_size": 0.25,
            "train_size": 0.75,
            "stratify": "y",
            "random_state": 7,
            "unica_particion_de_la_tarea": True,
        },
        "train_indices": [int(i) for i in idx_train],
        "test_indices": [int(i) for i in idx_test],
        "train_class_counts": train_class_counts,
        "test_class_counts": test_class_counts,
    }
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ----------------------------------------------------------------------
# 4) Guardar la partición para reutilizarla en subtareas siguientes
#    (índices de train y de test; los datos se recargan con load_digits)
# ----------------------------------------------------------------------
os.makedirs("entradas", exist_ok=True)
with open(os.path.join("entradas", "T1.json"), "w", encoding="utf-8") as f:
    json.dump(resultados["T1"], f, indent=2, ensure_ascii=False)

# ----------------------------------------------------------------------
# 5) Imprimir las cifras principales
# ----------------------------------------------------------------------
print("=" * 60)
print("T1 — Datos y partición única (load_digits)")
print("=" * 60)
print(f"n_ejemplos  (total de imágenes)      : {n_ejemplos}")
print(f"n_atributos (características 8x8)    : {n_atributos}")
print(f"n_clases    (dígitos 0-9)            : {n_clases}")
print(f"n_train     (75%, estratificado)     : {n_train}")
print(f"n_test      (25%, estratificado)     : {n_test}")
print("-" * 60)
print("Cuentas por clase en entrenamiento:")
for c in sorted(train_class_counts, key=int):
    print(f"  clase {c}: {train_class_counts[c]}")
print("Cuentas por clase en prueba:")
for c in sorted(test_class_counts, key=int):
    print(f"  clase {c}: {test_class_counts[c]}")
print("-" * 60)
print("Partición guardada en 'resultados.json' y 'entradas/T1.json'")
print("(índices de train/test; random_state=7, stratify=y, test_size=0.25)")
print("El conjunto de prueba NO se usa para ajustar ni elegir nada.")
