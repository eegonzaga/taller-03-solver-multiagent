# -*- coding: utf-8 -*-
"""
SUBTAREA T1 — Parte 1 — Datos y partición
Carga sklearn.datasets.load_digits, reporta n_ejemplos, n_atributos, n_clases,
crea la partición única (test_size=0.25, stratify=y, random_state=7) y guarda
X_train, X_test, y_train, y_test para reuso en las siguientes subtareas.
No produce figura PNG.
"""

import json
import os

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def main():
    # ------------------------------------------------------------------
    # 1) Cargar el conjunto de dígitos manuscritos de scikit-learn
    # ------------------------------------------------------------------
    digits = load_digits()
    X = digits.data
    y = digits.target

    n_ejemplos = int(X.shape[0])
    n_atributos = int(X.shape[1])
    clases = sorted(int(c) for c in np.unique(y))
    n_clases = int(len(clases))

    # ------------------------------------------------------------------
    # 2) Partición única de toda la tarea: 75% train / 25% test,
    #    estratificada por la clase, random_state=7
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=7
    )

    n_train = int(X_train.shape[0])
    n_test = int(X_test.shape[0])

    # ------------------------------------------------------------------
    # 3) Verificación de la estratificación (distribución de clases)
    # ------------------------------------------------------------------
    dist_total = {str(c): int(np.sum(y == c)) for c in clases}
    dist_train = {str(c): int(np.sum(y_train == c)) for c in clases}
    dist_test = {str(c): int(np.sum(y_test == c)) for c in clases}
    prop_train = {str(c): float(np.sum(y_train == c) / n_train) for c in clases}
    prop_test = {str(c): float(np.sum(y_test == c) / n_test) for c in clases}

    # ------------------------------------------------------------------
    # 4) Guardar cifras en resultados.json (precisión completa, tipos nativos)
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T1_Parte1_datos_y_particion",
        "n_ejemplos": n_ejemplos,
        "n_atributos": n_atributos,
        "n_clases": n_clases,
        "n_train": n_train,
        "n_test": n_test,
        "clases": clases,
        "test_size": 0.25,
        "random_state": 7,
        "fraccion_test": float(n_test / n_ejemplos),
        "distribucion_clases_total": dist_total,
        "distribucion_clases_train": dist_train,
        "distribucion_clases_test": dist_test,
        "proporcion_clases_train": prop_train,
        "proporcion_clases_test": prop_test,
    }
    with open("resultados.json", "w") as f:
        json.dump(resultados, f, indent=2)

    # ------------------------------------------------------------------
    # 5) Guardar la partición para reuso en subtareas siguientes
    #    (archivos .npy en la carpeta actual)
    # ------------------------------------------------------------------
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)

    # Además, copia en entradas/T1.json (listas JSON) por si las siguientes
    # subtareas leen de entradas/<id>.json
    try:
        os.makedirs("entradas", exist_ok=True)
        datos_particion = {
            "X_train": X_train.tolist(),
            "X_test": X_test.tolist(),
            "y_train": y_train.tolist(),
            "y_test": y_test.tolist(),
        }
        with open(os.path.join("entradas", "T1.json"), "w") as f:
            json.dump(datos_particion, f)
    except Exception as e:
        print(f"Aviso: no se pudo escribir entradas/T1.json ({e})")

    # ------------------------------------------------------------------
    # 6) Imprimir las cifras principales
    # ------------------------------------------------------------------
    print("=== T1 — Parte 1: Datos y partición ===")
    print(f"n_ejemplos  : {n_ejemplos}")
    print(f"n_atributos : {n_atributos}")
    print(f"n_clases    : {n_clases}")
    print(f"n_train     : {n_train}  ({n_train / n_ejemplos:.4f} del total)")
    print(f"n_test      : {n_test}  ({n_test / n_ejemplos:.4f} del total)")
    print(f"Clases      : {clases}")
    print("Distribución por clase (total | train | test):")
    for c in clases:
        print(
            f"  clase {c}: {dist_total[str(c)]:4d} | "
            f"{dist_train[str(c)]:4d} ({prop_train[str(c)]:.4f}) | "
            f"{dist_test[str(c)]:3d} ({prop_test[str(c)]:.4f})"
        )
    print("Archivos guardados: resultados.json, X_train.npy, X_test.npy, "
          "y_train.npy, y_test.npy, entradas/T1.json")


if __name__ == "__main__":
    main()
