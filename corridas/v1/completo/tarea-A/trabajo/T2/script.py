# -*- coding: utf-8 -*-
"""
T2 — Clasificadores base: GaussianNB vs regresión logística
------------------------------------------------------------
Con la partición de T1 (load_digits, 75/25 estratificado, random_state=7):
  * GaussianNB (valores por defecto) sobre los atributos ORIGINALES sin escalar.
  * LogisticRegression(max_iter=3000) sobre atributos ESTANDARIZADOS con
    StandardScaler ajustado SOLO con el conjunto de entrenamiento.
Evalúa ambos en el conjunto de prueba y guarda accuracy y F1 macro en
resultados.json. No produce figuras.
"""

import json
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ----------------------------------------------------------------------
# 1) Leer los parámetros de la partición definidos en T1
# ----------------------------------------------------------------------
with open("entradas/T1.json", "r", encoding="utf-8") as f:
    t1 = json.load(f)

random_state = int(t1["random_state"])
test_size = float(t1["test_size"])
estratificado = bool(t1["estratificado"])
n_train_esperado = int(t1["n_train"])
n_test_esperado = int(t1["n_test"])

# ----------------------------------------------------------------------
# 2) Cargar datos y REPRODUCIR exactamente la partición única de T1
#    (el conjunto de prueba no se usa para ajustar nada)
# ----------------------------------------------------------------------
X, y = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state,
    stratify=y if estratificado else None,
)

# Verificaciones de consistencia con T1
assert X.shape[0] == int(t1["n_ejemplos"]), "n_ejemplos no coincide con T1"
assert X.shape[1] == int(t1["n_atributos"]), "n_atributos no coincide con T1"
assert len(np.unique(y)) == int(t1["n_clases"]), "n_clases no coincide con T1"
assert X_train.shape[0] == n_train_esperado, "n_train no coincide con T1"
assert X_test.shape[0] == n_test_esperado, "n_test no coincide con T1"

# ----------------------------------------------------------------------
# 3) Modelo generativo: GaussianNB sobre atributos originales SIN escalar
# ----------------------------------------------------------------------
gnb = GaussianNB()  # valores por defecto
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

accuracy_gnb = float(accuracy_score(y_test, y_pred_gnb))
f1_macro_gnb = float(f1_score(y_test, y_pred_gnb, average="macro"))

# ----------------------------------------------------------------------
# 4) Modelo discriminativo: LogisticRegression sobre atributos estandarizados
#    (StandardScaler ajustado SOLO con el entrenamiento)
# ----------------------------------------------------------------------
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)   # fit solo con train
X_test_std = scaler.transform(X_test)         # transform aplicado a test

lr = LogisticRegression(max_iter=3000)
lr.fit(X_train_std, y_train)
y_pred_lr = lr.predict(X_test_std)

accuracy_lr = float(accuracy_score(y_test, y_pred_lr))
f1_macro_lr = float(f1_score(y_test, y_pred_lr, average="macro"))

# ----------------------------------------------------------------------
# 5) Guardar TODAS las cifras en resultados.json (precisión completa)
# ----------------------------------------------------------------------
resultados = {
    # Cifras principales requeridas
    "accuracy_gnb": accuracy_gnb,
    "f1_macro_gnb": f1_macro_gnb,
    "accuracy_lr": accuracy_lr,
    "f1_macro_lr": f1_macro_lr,
    # Contexto de la partición reproducida
    "n_train": int(X_train.shape[0]),
    "n_test": int(X_test.shape[0]),
    "n_ejemplos": int(X.shape[0]),
    "n_atributos": int(X.shape[1]),
    "n_clases": int(len(np.unique(y))),
    "random_state": random_state,
    "test_size": test_size,
    "estratificado": estratificado,
    # Detalles adicionales de cada modelo
    "f1_por_clase_gnb": [float(v) for v in f1_score(y_test, y_pred_gnb, average=None)],
    "f1_por_clase_lr": [float(v) for v in f1_score(y_test, y_pred_lr, average=None)],
    "confusion_matrix_gnb": confusion_matrix(y_test, y_pred_gnb).tolist(),
    "confusion_matrix_lr": confusion_matrix(y_test, y_pred_lr).tolist(),
    "n_iter_lr": int(np.max(lr.n_iter_)),
    "error_gnb": float(1.0 - accuracy_gnb),
    "error_lr": float(1.0 - accuracy_lr),
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ----------------------------------------------------------------------
# 6) Imprimir las cifras principales
# ----------------------------------------------------------------------
print("=== T2: Clasificadores base sobre dígitos (partición de T1) ===")
print(f"Partición: train={X_train.shape[0]}, test={X_test.shape[0]}, "
      f"test_size={test_size}, random_state={random_state}, estratificado={estratificado}")
print()
print("| Modelo                              | Accuracy   | F1 macro   |")
print("|-------------------------------------|------------|------------|")
print(f"| GaussianNB (sin escalar)            | {accuracy_gnb:.6f} | {f1_macro_gnb:.6f} |")
print(f"| LogisticRegression (estandarizado)  | {accuracy_lr:.6f} | {f1_macro_lr:.6f} |")
print()
print(f"accuracy_gnb = {accuracy_gnb}")
print(f"f1_macro_gnb = {f1_macro_gnb}")
print(f"accuracy_lr  = {accuracy_lr}")
print(f"f1_macro_lr  = {f1_macro_lr}")
print()
print("Resultados guardados en resultados.json")
