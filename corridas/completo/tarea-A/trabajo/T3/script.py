# -*- coding: utf-8 -*-
"""
T3 — Curva de aprendizaje con 20, 50, 100, 200, 400 y todos los ejemplos.

Usa la partición de T1 (load_digits, 75%/25% estratificado, random_state=7) y los
dos modelos de T2:
  - GaussianNB (valores por defecto) sobre los atributos originales, sin escalar.
  - LogisticRegression(max_iter=3000) sobre atributos estandarizados, con un
    StandardScaler que se ajusta SOLO con los ejemplos de entrenamiento usados
    en cada corrida (nunca con el conjunto de prueba).
Cada submuestra se obtiene con:
  train_test_split(X_train, y_train, train_size=m, stratify=y_train, random_state=7)
La evaluación se hace siempre sobre el mismo conjunto de prueba.

Salidas:
  - resultados.json
  - curva_aprendizaje.png
"""

import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ------------------- Parámetros fijados por el enunciado -------------------
RANDOM_STATE_PARTICION = 7      # partición 75/25 de T1
RANDOM_STATE_SUBMUESTRA = 7     # train_test_split de cada submuestra
TEST_SIZE = 0.25
TAMANIOS_SUBMUESTRA = [20, 50, 100, 200, 400]
MAX_ITER_LR = 3000


# ------------------- 1) Metadatos de subtareas anteriores ------------------
def leer_json(ruta):
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


t1_crudo = leer_json("entradas/T1.json")
t2 = leer_json("entradas/T2.json")
t1 = t1_crudo.get("T1", {}) if isinstance(t1_crudo, dict) else {}
if isinstance(t1, dict) and t1:
    print("Metadatos T1 leídos:", {k: t1[k] for k in list(t1)[:6]})

# ------------- 2) Datos y partición de T1 (se regenera idéntica) -----------
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE_PARTICION
)
n_train = int(len(y_train))
n_test = int(len(y_test))
print(f"Datos: {X.shape[0]} ejemplos, {X.shape[1]} atributos, "
      f"{len(set(y.tolist()))} clases | n_train={n_train}, n_test={n_test}")

# ------------------- 3) Curva de aprendizaje -------------------------------
acc_gnb_sub = []   # exactitud de GNB con m ejemplos
acc_lr_sub = []    # exactitud de LR con m ejemplos

for m in TAMANIOS_SUBMUESTRA:
    X_m, _, y_m, _ = train_test_split(
        X_train, y_train,
        train_size=m, stratify=y_train, random_state=RANDOM_STATE_SUBMUESTRA
    )

    # Generativo: GaussianNB sobre atributos originales (sin escalar)
    gnb = GaussianNB()
    gnb.fit(X_m, y_m)
    acc_gnb_sub.append(float(accuracy_score(y_test, gnb.predict(X_test))))

    # Discriminativo: LR sobre atributos estandarizados;
    # el escalador se ajusta SOLO con la submuestra de entrenamiento
    scaler = StandardScaler().fit(X_m)
    lr = LogisticRegression(max_iter=MAX_ITER_LR)
    lr.fit(scaler.transform(X_m), y_m)
    acc_lr_sub.append(float(accuracy_score(y_test, lr.predict(scaler.transform(X_test)))))

# Entrenamiento completo (mismo esquema que T2)
gnb_full = GaussianNB().fit(X_train, y_train)
acc_gnb_full = float(accuracy_score(y_test, gnb_full.predict(X_test)))

scaler_full = StandardScaler().fit(X_train)
lr_full = LogisticRegression(max_iter=MAX_ITER_LR)
lr_full.fit(scaler_full.transform(X_train), y_train)
acc_lr_full = float(accuracy_score(y_test, lr_full.predict(scaler_full.transform(X_test))))

# Curva completa (submuestras + entrenamiento completo)
tamanios_curva = [int(t) for t in TAMANIOS_SUBMUESTRA] + [n_train]
acc_gnb_curva = acc_gnb_sub + [acc_gnb_full]
acc_lr_curva = acc_lr_sub + [acc_lr_full]
ventaja_lr = [float(al - ag) for ag, al in zip(acc_gnb_curva, acc_lr_curva)]

# Tamaño de cruce: primer m de la curva donde la LR supera al GNB
tamanio_cruce = None
for m, al, ag in zip(tamanios_curva, acc_lr_curva, acc_gnb_curva):
    if al > ag:
        tamanio_cruce = int(m)
        break

# ------------------- 4) Guardar resultados ---------------------------------
resultados = {
    "tamanios_curva": tamanios_curva,
    "accuracy_curva_gnb": acc_gnb_curva,
    "accuracy_curva_lr": acc_lr_curva,
    "tamanio_cruce": tamanio_cruce,
    "ventaja_lr_menos_gnb": ventaja_lr,
    "tamanios_submuestras": [int(t) for t in TAMANIOS_SUBMUESTRA],
    "accuracy_curva_gnb_submuestras": acc_gnb_sub,
    "accuracy_curva_lr_submuestras": acc_lr_sub,
    "accuracy_gnb_entrenamiento_completo": acc_gnb_full,
    "accuracy_lr_entrenamiento_completo": acc_lr_full,
    "n_train": n_train,
    "n_test": n_test,
    "random_state_particion": int(RANDOM_STATE_PARTICION),
    "random_state_submuestra": int(RANDOM_STATE_SUBMUESTRA),
    "nota_escalador": ("StandardScaler ajustado unicamente con los ejemplos de "
                       "entrenamiento empleados en cada corrida; el conjunto de "
                       "prueba nunca se usa para ajustar nada."),
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ------------------- 5) Impresión de cifras principales --------------------
print("\nCurva de aprendizaje (exactitud sobre el conjunto de prueba):")
print(f"{'m':>6} | {'acc GNB':>9} | {'acc LR':>9} | {'LR-GNB':>9}")
for m, ag, al, d in zip(tamanios_curva, acc_gnb_curva, acc_lr_curva, ventaja_lr):
    print(f"{m:>6} | {ag:>9.4f} | {al:>9.4f} | {d:>+9.4f}")
print(f"\nTamaño de cruce (primer m con acc_LR > acc_GNB): {tamanio_cruce}")
print(f"Entrenamiento completo -> GNB: {acc_gnb_full:.4f} | LR: {acc_lr_full:.4f}")

if isinstance(t2, dict) and "accuracy_gnb" in t2 and "accuracy_lr" in t2:
    print("Verificación contra T2 (mismo entrenamiento completo): "
          f"GNB T2={float(t2['accuracy_gnb']):.4f}, "
          f"LR T2={float(t2['accuracy_lr']):.4f}")

# ------------------- 6) Figura ---------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(tamanios_curva, acc_gnb_curva, marker="o", color="tab:blue",
         label="Naive Bayes gaussiano (generativo)")
plt.plot(tamanios_curva, acc_lr_curva, marker="s", color="tab:orange",
         label="Regresión logística (discriminativo)")
if tamanio_cruce is not None:
    plt.axvline(tamanio_cruce, color="gray", linestyle="--", linewidth=1,
                label=f"Tamaño de cruce = {tamanio_cruce}")
plt.xlabel("Número de ejemplos de entrenamiento (m)")
plt.ylabel("Exactitud en el conjunto de prueba")
plt.title("Curva de aprendizaje: Naive Bayes vs. Regresión logística (dígitos)")
plt.xticks(tamanios_curva)
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("curva_aprendizaje.png", dpi=150)
plt.close()

print("\nFigura guardada: curva_aprendizaje.png")
print("Resultados guardados: resultados.json")
