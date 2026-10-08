# -*- coding: utf-8 -*-
"""
T2 — Parte 2 — Naive Bayes gaussiano vs regresión logística.

Usa la partición train/test generada en T1 (entradas/T1.json):
- GaussianNB (valores por defecto) sobre los atributos originales, SIN escalar.
- LogisticRegression(max_iter=3000) sobre atributos estandarizados con un
  StandardScaler ajustado SOLO con el conjunto de entrenamiento.
Evalúa ambos modelos sobre el conjunto de prueba y reporta accuracy y F1 macro.
"""

import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# ----------------------------------------------------------------------
# 1) Cargar la partición de T1
# ----------------------------------------------------------------------
with open("entradas/T1.json", "r", encoding="utf-8") as f:
    t1 = json.load(f)

n_ejemplos = int(t1["n_ejemplos"])
n_atributos = int(t1["n_atributos"])
n_clases = int(t1["n_clases"])
train_indices = np.asarray(t1["train_indices"], dtype=int)
test_indices = np.asarray(t1["test_indices"], dtype=int)

# Normalizar etiquetas de clase (por si vinieran como enteros o cadenas)
try:
    clases_norm = sorted(int(c) for c in t1["clases"])
    clases_son_int = True
except (TypeError, ValueError):
    clases_norm = sorted(str(c) for c in t1["clases"])
    clases_son_int = False

print("Partición de T1 cargada:")
print(f"  n_ejemplos={n_ejemplos}, n_atributos={n_atributos}, n_clases={n_clases}")
print(f"  tam_train={len(train_indices)}, tam_test={len(test_indices)}")
print(f"  clases={clases_norm}")

# ----------------------------------------------------------------------
# 2) Identificar el dataset de scikit-learn compatible con la partición
#    (los datos provienen de scikit-learn; se empareja por forma y clases)
# ----------------------------------------------------------------------
candidatos = [
    ("breast_cancer", load_breast_cancer()),
    ("iris", load_iris()),
    ("wine", load_wine()),
    ("digits", load_digits(n_class=n_clases) if 1 <= n_clases <= 10 else None),
]

dataset_name = None
X = None
y = None
for nombre, ds in candidatos:
    if ds is None:
        continue
    if clases_son_int:
        clases_ds = sorted(int(c) for c in np.unique(ds.target))
    else:
        clases_ds = sorted(str(s) for s in ds.target_names)
    if (ds.data.shape[0] == n_ejemplos
            and ds.data.shape[1] == n_atributos
            and clases_ds == clases_norm):
        dataset_name = nombre
        X = ds.data
        y = ds.target
        break

if X is None:
    # Respaldo: el contexto (Ng & Jordan, comparación binaria) sugiere breast_cancer
    print("ADVERTENCIA: no se identificó el dataset por forma; se usa breast_cancer.")
    ds = load_breast_cancer()
    dataset_name = "breast_cancer"
    X = ds.data
    y = ds.target

print(f"Dataset identificado: {dataset_name}, X.shape={X.shape}")

# ----------------------------------------------------------------------
# 3) Partición train/test usando exactamente los índices de T1
# ----------------------------------------------------------------------
X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]

# ----------------------------------------------------------------------
# 4) GaussianNB (valores por defecto) sobre atributos originales sin escalar
# ----------------------------------------------------------------------
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
accuracy_gnb = float(accuracy_score(y_test, y_pred_gnb))
f1_macro_gnb = float(f1_score(y_test, y_pred_gnb, average="macro"))

# ----------------------------------------------------------------------
# 5) LogisticRegression(max_iter=3000) sobre atributos estandarizados
#    (StandardScaler ajustado SOLO con el conjunto de entrenamiento)
# ----------------------------------------------------------------------
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)   # ajuste solo con entrenamiento
X_test_std = scaler.transform(X_test)         # el test solo se transforma

lr = LogisticRegression(max_iter=3000)
lr.fit(X_train_std, y_train)
y_pred_lr = lr.predict(X_test_std)
accuracy_lr = float(accuracy_score(y_test, y_pred_lr))
f1_macro_lr = float(f1_score(y_test, y_pred_lr, average="macro"))

# ----------------------------------------------------------------------
# 6) Tabla resumen de resultados sobre el conjunto de prueba
# ----------------------------------------------------------------------
tabla = pd.DataFrame(
    {
        "Accuracy": [accuracy_gnb, accuracy_lr],
        "F1_macro": [f1_macro_gnb, f1_macro_lr],
    },
    index=["GaussianNB (sin escalar)", "LogisticRegression (estandarizado)"],
)
print("\nTabla de resultados (conjunto de prueba):")
print(tabla.to_string())

print("\nCifras principales:")
print(f"  accuracy_gnb = {accuracy_gnb}")
print(f"  f1_macro_gnb = {f1_macro_gnb}")
print(f"  accuracy_lr  = {accuracy_lr}")
print(f"  f1_macro_lr  = {f1_macro_lr}")

# ----------------------------------------------------------------------
# 7) Guardar todas las cifras en resultados.json
# ----------------------------------------------------------------------
resultados = {
    "dataset": dataset_name,
    "tam_train": int(len(train_indices)),
    "tam_test": int(len(test_indices)),
    "accuracy_gnb": accuracy_gnb,
    "f1_macro_gnb": f1_macro_gnb,
    "accuracy_lr": accuracy_lr,
    "f1_macro_lr": f1_macro_lr,
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print("\nResultados guardados en resultados.json")
