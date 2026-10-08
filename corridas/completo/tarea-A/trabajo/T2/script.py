# -*- coding: utf-8 -*-
"""
T2 — Clasificadores base: GaussianNB vs regresión logística
============================================================
Usando la partición 75 %/25 % estratificada (random_state=7) definida en T1:
  * GaussianNB (valores por defecto) sobre los atributos originales, sin escalar.
  * LogisticRegression(max_iter=3000) sobre atributos estandarizados con un
    StandardScaler ajustado SOLO con el conjunto de entrenamiento.
Evalúa accuracy y F1 macro de ambos modelos sobre el conjunto de prueba,
imprime la tabla comparativa y guarda las cifras en resultados.json.
"""

import json
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def _candidatas_particion(obj):
    """Genera pares (lista_train, lista_test, nombre_claves) hallados en el JSON de T1."""
    if isinstance(obj, dict):
        claves_train = [k for k in obj if "train" in k.lower()]
        claves_test = [k for k in obj if "test" in k.lower()]
        for kt in claves_train:
            for ks in claves_test:
                yield obj[kt], obj[ks], f"{kt}/{ks}"
        for v in obj.values():
            yield from _candidatas_particion(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _candidatas_particion(v)


def _valida_indices(tr, te, n_total):
    """Comprueba que dos listas sean una partición por índices válida y única."""
    try:
        tr = np.asarray(tr, dtype=float)
        te = np.asarray(te, dtype=float)
    except (TypeError, ValueError):
        return None
    if tr.ndim != 1 or te.ndim != 1 or len(tr) == 0 or len(te) == 0:
        return None
    if not (np.all(np.isfinite(tr)) and np.all(np.isfinite(te))):
        return None
    if not (np.all(tr == np.floor(tr)) and np.all(te == np.floor(te))):
        return None
    tr_i = tr.astype(int)
    te_i = te.astype(int)
    if tr_i.min() < 0 or te_i.min() < 0:
        return None
    if tr_i.max() >= n_total or te_i.max() >= n_total:
        return None
    set_tr, set_te = set(tr_i.tolist()), set(te_i.tolist())
    # Índices únicos y disjuntos: descarta etiquetas (y_train/y_test) o máscaras booleanas
    if len(set_tr) != len(tr_i) or len(set_te) != len(te_i) or (set_tr & set_te):
        return None
    return tr_i, te_i


def extraer_particion(t1, n_total):
    """Devuelve (nombre, train_idx, test_idx) de la primera partición válida en T1."""
    validas = []
    for tr_list, te_list, nombre in _candidatas_particion(t1):
        res = _valida_indices(tr_list, te_list, n_total)
        if res is not None:
            validas.append((nombre, res[0], res[1]))
    if not validas:
        return None
    # Preferir claves que sugieren índices (p. ej. train_indices/test_indices)
    validas.sort(key=lambda x: 0 if ("ind" in x[0].lower() or "idx" in x[0].lower()) else 1)
    return validas[0]


def main():
    # ---------- 1. Datos ----------
    digits = load_digits()
    X, y = digits.data, digits.target
    n_total, n_atributos = X.shape
    n_clases = len(np.unique(y))

    # ---------- 2. Recuperar la partición definida en T1 ----------
    train_idx, test_idx = None, None
    fuente = None
    try:
        with open("entradas/T1.json", "r", encoding="utf-8") as f:
            t1_all = json.load(f)
        t1 = t1_all.get("T1", t1_all) if isinstance(t1_all, dict) else t1_all
        extraida = extraer_particion(t1, n_total)
        if extraida is not None:
            nombre, train_idx, test_idx = extraida
            fuente = f"índices de entrenamiento/prueba leídos de entradas/T1.json ({nombre})"
    except Exception:
        pass

    if train_idx is None:
        # La partición de T1 es determinista: 75 %/25 %, estratificada, random_state=7
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=7
        )
        fuente = ("reproducción determinista de la partición de T1: "
                  "train_test_split(test_size=0.25, stratify=y, random_state=7)")
    else:
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

    print("Partición utilizada:", fuente)
    print(f"Ejemplos: {n_total} | Atributos: {n_atributos} | Clases: {n_clases}")
    print(f"Entrenamiento: {X_train.shape[0]} ejemplos | Prueba: {X_test.shape[0]} ejemplos")

    # ---------- 3. GaussianNB sobre atributos originales (sin escalar) ----------
    gnb = GaussianNB()  # valores por defecto
    gnb.fit(X_train, y_train)
    pred_gnb = gnb.predict(X_test)
    accuracy_gnb = accuracy_score(y_test, pred_gnb)
    f1_macro_gnb = f1_score(y_test, pred_gnb, average="macro")

    # ---------- 4. Regresión logística sobre atributos estandarizados ----------
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)  # ajustado SOLO con el entrenamiento
    X_test_std = scaler.transform(X_test)        # misma transformación aplicada a prueba

    lr = LogisticRegression(max_iter=3000)
    lr.fit(X_train_std, y_train)
    pred_lr = lr.predict(X_test_std)
    accuracy_lr = accuracy_score(y_test, pred_lr)
    f1_macro_lr = f1_score(y_test, pred_lr, average="macro")

    # ---------- 5. Tabla de resultados (solo conjunto de prueba) ----------
    print("\nTabla: desempeño sobre el conjunto de prueba")
    print("-" * 58)
    print(f"{'Modelo':<38}{'Accuracy':>10}{'F1 macro':>10}")
    print("-" * 58)
    print(f"{'GaussianNB (atributos sin escalar)':<38}{accuracy_gnb:>10.6f}{f1_macro_gnb:>10.6f}")
    print(f"{'LogisticRegression (atributos estand.)':<38}{accuracy_lr:>10.6f}{f1_macro_lr:>10.6f}")
    print("-" * 58)

    # ---------- 6. Guardar todas las cifras en resultados.json ----------
    resultados = {
        "accuracy_gnb": float(accuracy_gnb),
        "f1_macro_gnb": float(f1_macro_gnb),
        "accuracy_lr": float(accuracy_lr),
        "f1_macro_lr": float(f1_macro_lr),
        "n_ejemplos": int(n_total),
        "n_atributos": int(n_atributos),
        "n_clases": int(n_clases),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "fuente_particion": fuente,
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print("\nCifras guardadas en resultados.json:")
    for k in ["accuracy_gnb", "f1_macro_gnb", "accuracy_lr", "f1_macro_lr"]:
        print(f"  {k} = {resultados[k]!r}")


if __name__ == "__main__":
    main()
