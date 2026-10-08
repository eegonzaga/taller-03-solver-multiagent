# -*- coding: utf-8 -*-
# =====================================================================
# T2 - Parte 2 - Los dos clasificadores
#
# A partir de la configuracion de la particion guardada en entradas/T1.json:
#   * Reproduce el mismo conjunto de entrenamiento y de prueba de T1
#     (mismo random_state, mismo tamano de test, misma estratificacion).
#   * Entrena GaussianNB (valores por defecto) sobre los atributos
#     originales, sin escalar.
#   * Entrena LogisticRegression(max_iter=3000) sobre atributos
#     estandarizados con StandardScaler ajustado SOLO con el train.
#   * Evalua accuracy y F1 macro de ambos modelos SOLO sobre el test.
#   * Guarda todas las cifras en resultados.json y las imprime.
# No se genera ninguna figura (no se pide).
# =====================================================================

import json

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


def norm_texto(v):
    return str(v).strip().lower()


# ---------------------------------------------------------------------
# 1) Leer la configuracion de la particion desde entradas/T1.json
# ---------------------------------------------------------------------
with open("entradas/T1.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

n_ejemplos_meta = int(meta["n_ejemplos"])
n_atributos_meta = int(meta["n_atributos"])
n_clases_meta = int(meta["n_clases"])
n_train_meta = int(meta["n_train"])
n_test_meta = int(meta["n_test"])
random_state = int(meta.get("random_state", 42))
clases_meta = sorted(norm_texto(c) for c in meta.get("clases", []))
dist_train_meta = meta.get("distribucion_clases_train", {}) or {}
dist_test_meta = meta.get("distribucion_clases_test", {}) or {}

# ---------------------------------------------------------------------
# 2) Identificar el dataset de scikit-learn correspondiente a T1
# ---------------------------------------------------------------------
candidatos = {
    "breast_cancer": load_breast_cancer(),
    "iris": load_iris(),
    "wine": load_wine(),
    "digits": load_digits(),
}

mejor_nombre = None
mejor_score = None
mejor_data = None
for nombre, d in candidatos.items():
    Xc = d.data
    yc = d.target
    score = 0.0
    if Xc.shape[0] == n_ejemplos_meta:
        score += 4.0
    if Xc.shape[1] == n_atributos_meta:
        score += 4.0
    if len(np.unique(yc)) == n_clases_meta:
        score += 2.0
    nombres_c = [str(t) for t in getattr(d, "target_names", np.unique(yc))]
    if clases_meta and sorted(norm_texto(t) for t in nombres_c) == clases_meta:
        score += 2.0
    score -= (abs(Xc.shape[0] - n_ejemplos_meta) + abs(Xc.shape[1] - n_atributos_meta)) / 10000.0
    if mejor_score is None or score > mejor_score:
        mejor_score = score
        mejor_nombre = nombre
        mejor_data = d

data = mejor_data
dataset_nombre = mejor_nombre
X = data.data.astype(float)
y = data.target.astype(int)
n_total = X.shape[0]

# ---------------------------------------------------------------------
# 3) Reproducir la particion train/test de T1
# ---------------------------------------------------------------------
def parse_test_size(valor):
    if valor is None:
        return None
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return None
    if 0.0 < v < 1.0:
        return v
    if v >= 1.0 and float(v).is_integer():
        return int(v)
    return None


def error_distribucion(y_vec, dist_meta):
    # Diferencia entre los conteos por clase del split reproducido y los
    # registrados en T1 (admite que T1 guarde conteos o proporciones).
    if not dist_meta:
        return 0
    valores = [float(v) for v in dist_meta.values()]
    total = float(sum(valores))
    if 0.9 < total < 1.1:
        esperados = [v / total * len(y_vec) for v in valores]
    else:
        esperados = valores
    obtenidos = [int(c) for c in np.unique(y_vec, return_counts=True)[1]]
    if len(esperados) != len(obtenidos):
        return 10 ** 6
    esperados = sorted(esperados)
    obtenidos = sorted(obtenidos)
    return int(round(sum(abs(e - o) for e, o in zip(esperados, obtenidos))))


def construir_split(ts_val, estratificar):
    if estratificar:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=ts_val, random_state=random_state, stratify=y
        )
    else:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=ts_val, random_state=random_state
        )
    err = (
        error_distribucion(ytr, dist_train_meta)
        + error_distribucion(yte, dist_test_meta)
        + abs(len(ytr) - n_train_meta)
        + abs(len(yte) - n_test_meta)
    )
    return int(err), (Xtr, Xte, ytr, yte)


ts_fraccion = parse_test_size(meta.get("fraccion_test"))
if ts_fraccion is None:
    ts_fraccion = parse_test_size(meta.get("test_size"))
if ts_fraccion is None:
    ts_fraccion = n_test_meta / float(n_total)

opciones = [(ts_fraccion, True), (ts_fraccion, False)]
if n_test_meta != ts_fraccion:
    opciones.append((n_test_meta, True))
    opciones.append((n_test_meta, False))

mejor_err = None
mejor_split = None
mejor_desc = None
for ts_val, estrat in opciones:
    err, sp = construir_split(ts_val, estrat)
    if mejor_err is None or err < mejor_err:
        mejor_err = err
        mejor_split = sp
        mejor_desc = {"test_size_usado": ts_val, "estratificado": estrat}
    if err == 0:
        break

X_train, X_test, y_train, y_test = mejor_split

# ---------------------------------------------------------------------
# 4) Modelo 1: GaussianNB (por defecto) sobre atributos SIN escalar
# ---------------------------------------------------------------------
gnb = GaussianNB()
gnb.fit(X_train, y_train)
pred_gnb = gnb.predict(X_test)
accuracy_gnb = float(accuracy_score(y_test, pred_gnb))
f1_macro_gnb = float(f1_score(y_test, pred_gnb, average="macro"))

# ---------------------------------------------------------------------
# 5) Modelo 2: LogisticRegression(max_iter=3000) sobre atributos
#    estandarizados (StandardScaler ajustado SOLO con el train)
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_esc = scaler.fit_transform(X_train)
X_test_esc = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=3000)
logreg.fit(X_train_esc, y_train)
pred_lr = logreg.predict(X_test_esc)
accuracy_lr = float(accuracy_score(y_test, pred_lr))
f1_macro_lr = float(f1_score(y_test, pred_lr, average="macro"))

# ---------------------------------------------------------------------
# 6) Guardar todas las cifras en resultados.json
# ---------------------------------------------------------------------
vals_train, cnts_train = np.unique(y_train, return_counts=True)
vals_test, cnts_test = np.unique(y_test, return_counts=True)

resultados = {
    "subtarea": "T2_parte2_clasificadores",
    "dataset": dataset_nombre,
    "random_state": int(random_state),
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "split_test_size_usado": mejor_desc["test_size_usado"],
    "split_estratificado": bool(mejor_desc["estratificado"]),
    "split_reproducido_error": int(mejor_err),
    "distribucion_clases_train_reproducida": {
        str(k): int(v) for k, v in zip(vals_train.tolist(), cnts_train.tolist())
    },
    "distribucion_clases_test_reproducida": {
        str(k): int(v) for k, v in zip(vals_test.tolist(), cnts_test.tolist())
    },
    "accuracy_gnb": accuracy_gnb,
    "f1_macro_gnb": f1_macro_gnb,
    "accuracy_lr": accuracy_lr,
    "f1_macro_lr": f1_macro_lr,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------
# 7) Imprimir la tabla de resultados
# ---------------------------------------------------------------------
tabla = pd.DataFrame(
    {
        "Modelo": [
            "GaussianNB (atributos originales, sin escalar)",
            "LogisticRegression (atributos estandarizados)",
        ],
        "Accuracy": [accuracy_gnb, accuracy_lr],
        "F1_macro": [f1_macro_gnb, f1_macro_lr],
    }
)

print("Dataset identificado:", dataset_nombre)
print(
    "Particion reproducida de T1: n_train =",
    len(y_train),
    ", n_test =",
    len(y_test),
    ", random_state =",
    random_state,
    ", estratificado =",
    mejor_desc["estratificado"],
    ", error_distribucion =",
    mejor_err,
)
print()
print(tabla.to_string(index=False))
print()
print("accuracy_gnb =", accuracy_gnb)
print("f1_macro_gnb =", f1_macro_gnb)
print("accuracy_lr  =", accuracy_lr)
print("f1_macro_lr  =", f1_macro_lr)
print()
print("Cifras guardadas en resultados.json")
