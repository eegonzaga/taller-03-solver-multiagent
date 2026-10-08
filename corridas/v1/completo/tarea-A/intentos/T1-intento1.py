# -*- coding: utf-8 -*-
"""
SUBTAREA T1 — Datos y partición única
Tarea A — ¿Gana el generativo con pocos datos? (Ng y Jordan sobre dígitos)

Carga sklearn.datasets.load_digits, reporta n_ejemplos, n_atributos y n_clases,
crea la ÚNICA partición de la tarea (train_test_split, test_size=0.25,
stratify=y, random_state=7) y guarda los conjuntos para reuso posterior.
"""

import json
import os

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# 1) Carga de datos
# ----------------------------------------------------------------------
digits = load_digits()
X = digits.data          # matriz (n_ejemplos, n_atributos), valores 0..16
y = digits.target        # etiquetas 0..9

n_ejemplos, n_atributos = X.shape
clases = np.unique(y)
n_clases = int(clases.size)

# ----------------------------------------------------------------------
# 2) Partición única: 75% train / 25% test, estratificada, random_state=7
#    (el conjunto de prueba NO se usa para ajustar ni elegir nada)
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=7,
)

n_train = int(X_train.shape[0])
n_test = int(X_test.shape[0])

# Verificación de la estratificación (proporciones por clase en cada conjunto)
counts_train = np.unique(y_train, return_counts=True)
counts_test = np.unique(y_test, return_counts=True)
dist_train = {str(int(c)): int(v) for c, v in zip(counts_train[0], counts_train[1])}
dist_test = {str(int(c)): int(v) for c, v in zip(counts_test[0], counts_test[1])}

# ----------------------------------------------------------------------
# 3) Guardar cifras en resultados.json (tipos nativos de Python)
# ----------------------------------------------------------------------
resultados = {
    "n_ejemplos": int(n_ejemplos),
    "n_atributos": int(n_atributos),
    "n_clases": int(n_clases),
    "n_train": int(n_train),
    "n_test": int(n_test),
    "test_size": 0.25,
    "random_state": 7,
    "estratificado": True,
    "n_train_mas_n_test": int(n_train + n_test),
    "distribucion_clases_train": dist_train,
    "distribucion_clases_test": dist_test,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ----------------------------------------------------------------------
# 4) Guardar X_train, X_test, y_train, y_test para reuso en subtareas
#    (a) entradas/T1.json en formato JSON, (b) split_digits.npz en binario
# ----------------------------------------------------------------------
os.makedirs("entradas", exist_ok=True)

split_json = {
    "subtarea": "T1",
    "descripcion": "Particion unica de load_digits (test_size=0.25, stratify=y, random_state=7)",
    "random_state": 7,
    "test_size": 0.25,
    "n_train": int(n_train),
    "n_test": int(n_test),
    "n_atributos": int(n_atributos),
    "n_clases": int(n_clases),
    "X_train": X_train.astype(int).tolist(),
    "X_test": X_test.astype(int).tolist(),
    "y_train": y_train.astype(int).tolist(),
    "y_test": y_test.astype(int).tolist(),
}

with open(os.path.join("entradas", "T1.json"), "w", encoding="utf-8") as f:
    json.dump(split_json, f)

np.savez_compressed(
    "split_digits.npz",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)

# ----------------------------------------------------------------------
# 5) Reporte por consola de las cifras principales
# ----------------------------------------------------------------------
print("=" * 60)
print("T1 — Datos y partición única (load_digits)")
print("=" * 60)
print(f"n_ejemplos  : {n_ejemplos}")
print(f"n_atributos : {n_atributos}")
print(f"n_clases    : {n_clases}  (clases: {clases.tolist()})")
print(f"n_train     : {n_train}  ({n_train / n_ejemplos:.4f} del total)")
print(f"n_test      : {n_test}  ({n_test / n_ejemplos:.4f} del total)")
print(f"Verificación n_train + n_test = {n_train + n_test} (debe ser {n_ejemplos})")
print("-" * 60)
print("Distribución por clase (train):", dist_train)
print("Distribución por clase (test) :", dist_test)
print("-" * 60)
print("Archivos generados:")
print("  - resultados.json        (cifras de la subtarea)")
print("  - entradas/T1.json       (split completo para subtareas siguientes)")
print("  - split_digits.npz       (split completo en formato NumPy)")
print("=" * 60)
