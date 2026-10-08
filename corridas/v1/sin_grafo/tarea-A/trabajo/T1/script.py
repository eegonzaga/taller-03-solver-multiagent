# -*- coding: utf-8 -*-
"""
T1 — Parte 1 — Datos y partición
Carga sklearn.datasets.load_digits, reporta número de ejemplos, atributos y clases,
crea la única partición de la tarea (75% train / 25% test, estratificada, random_state=7)
y guarda todo en resultados.json para reutilizarla en las partes siguientes.
"""

import json
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# 1. Carga de datos
# ------------------------------------------------------------------
digits = load_digits()
X = digits.data          # forma (n_ejemplos, n_atributos)
y = digits.target        # etiquetas de clase

n_ejemplos = int(X.shape[0])
n_atributos = int(X.shape[1])
n_clases = int(len(np.unique(y)))

print("Número de ejemplos :", n_ejemplos)
print("Número de atributos:", n_atributos)
print("Número de clases   :", n_clases)

# ------------------------------------------------------------------
# 2. Única partición de la tarea: 75% train / 25% test
#    estratificada por clase, random_state=7
#    Se particiona sobre los índices para poder reconstruir
#    exactamente la misma división en las partes siguientes.
# ------------------------------------------------------------------
indices = np.arange(n_ejemplos)
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.25,
    stratify=y,
    random_state=7,
)

tam_train = int(len(train_idx))
tam_test = int(len(test_idx))

# Reconstrucción de los conjuntos a partir de los índices
X_train = X[train_idx]
y_train = y[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]

# Comprobación de la estratificación (distribución de clases por conjunto)
clases = np.unique(y).tolist()
conteo_train = {int(c): int(np.sum(y_train == c)) for c in clases}
conteo_test = {int(c): int(np.sum(y_test == c)) for c in clases}

print("Tamaño del conjunto de entrenamiento:", tam_train)
print("Tamaño del conjunto de prueba       :", tam_test)
print("Distribución de clases en train     :", conteo_train)
print("Distribución de clases en test      :", conteo_test)

# ------------------------------------------------------------------
# 3. Guardar todas las cifras en resultados.json
#    (los índices permiten reproducir la partición en las partes 2 y 3)
# ------------------------------------------------------------------
resultados = {
    "n_ejemplos": n_ejemplos,
    "n_atributos": n_atributos,
    "n_clases": n_clases,
    "tam_train": tam_train,
    "tam_test": tam_test,
    "proporcion_test": float(tam_test / n_ejemplos),
    "random_state": 7,
    "test_size": 0.25,
    "clases": clases,
    "conteo_clases_train": conteo_train,
    "conteo_clases_test": conteo_test,
    "train_indices": [int(i) for i in train_idx],
    "test_indices": [int(i) for i in test_idx],
}

with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

print("Resultados guardados en resultados.json")
