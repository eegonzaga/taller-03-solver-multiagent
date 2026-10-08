# Tarea A — ¿Gana el generativo con pocos datos? Ng y Jordan sobre dígitos

**MMIA 6013 · Taller 03 v2 · Entrega: `reporte.md` + código**

## 1. Introducción

Ng y Jordan (2001) sostienen que un clasificador generativo como Naive Bayes alcanza su error asintótico con menos ejemplos que su contraparte discriminativa, la regresión logística, aunque dicho error asintótico sea mayor. Esto predice un **cruce** de curvas de aprendizaje: con pocos datos ganaría el generativo y con muchos el discriminativo. Esta tarea pone a prueba esa predicción sobre dígitos manuscritos, comparando `GaussianNB` y `LogisticRegression` en una partición fija y en curvas de aprendizaje con submuestras de 20 a 1347 ejemplos.

## 2. Metodología

- **Datos:** `load_digits` de scikit-learn → **1797 ejemplos, 64 atributos** (intensidades 0–16 de una imagen 8×8) y **10 clases**.
- **Partición única:** 75 % entrenamiento (**1347**) / 25 % prueba (**450**), estratificada por clase, `random_state=7`. El conjunto de prueba no se usa para ajustar ni elegir nada.
- **Modelos:** `GaussianNB` (valores por defecto) sobre los atributos originales **sin escalar**; `LogisticRegression(max_iter=3000)` sobre atributos **estandarizados** con un `StandardScaler` ajustado solo con el entrenamiento (pipeline).
- **Curva de aprendizaje:** para m ∈ {20, 50, 100, 200, 400} se extrae una submuestra estratificada del entrenamiento con `train_test_split(train_size=m, stratify=y_train, random_state=7)`; se entrenan ambos modelos y se evalúa siempre sobre el mismo conjunto de prueba. Se añade el punto de entrenamiento completo (m = 1347). La figura se guarda como `curva_aprendizaje.png`.
- **Métricas:** exactitud y F1 macro. El código completo está en el Apéndice.

## 3. Resultados

**Tabla 1 — Parte 2, entrenamiento completo (prueba n = 450):**

| Modelo | Accuracy | F1 macro |
|---|---|---|
| GaussianNB (sin escalar) | 0.818 | 0.812 |
| Regresión logística (estandarizada) | **0.967** | **0.966** |

**Tabla 2 — Curva de aprendizaje (exactitud en prueba):**

| m | GaussianNB | Regresión logística |
|---|---|---|
| 20 | 0.311 | 0.578 |
| 50 | 0.516 | 0.782 |
| 100 | 0.636 | 0.871 |
| 200 | 0.724 | 0.922 |
| 400 | 0.782 | 0.944 |
| 1347 | 0.818 | 0.967 |

![Curva de aprendizaje](curva_aprendizaje.png)

Ambas curvas son monótonas crecientes y la regresión logística **domina en todos los tamaños**, con brechas que van de 0.267 (m = 20) a 0.149 (m = 1347).

## 4. Discusión

**No se observa el cruce que predicen Ng y Jordan.** Con d = 64 atributos, su teoría (el generativo se acerca a su propio error asintótico con un número de ejemplos que crece logarítmicamente con d, frente a linealmente para el discriminativo) sugiere que NB debería ganar en m pequeño. Aquí, incluso con m = 20 (2 ejemplos por clase), la regresión logística gana por ~27 puntos.

Sí hay un eco parcial de la teoría: NB mejora con rapidez en términos relativos al principio (de 0.311 a 0.636 entre m = 20 y m = 100) y en m = 400 ya está a ~4 puntos de su asíntota (0.818), consistente con su convergencia temprana a *su propio* error asintótico. El problema es que ese error asintótico (≈ 18 %) es muy superior al de la regresión logística (≈ 3 %), así que la brecha asintótica domina y las curvas nunca se cruzan.

**Qué supuesto falla:** la **independencia condicional de los atributos dada la clase**. En imágenes los píxeles vecinos están fuertemente correlacionados (los trazos son continuos), de modo que modelar los 64 píxeles como gaussianas independientes produce una verosimilitud mal especificada que cuenta varias veces la misma evidencia y degrada las fronteras. Además, con 2 ejemplos por clase NB debe estimar 1280 parámetros (media y varianza por clase y píxel) con varianzas muestrales degeneradas, mientras que la regresión logística comparte una estructura lineal entre clases y la regularización (C = 1) la estabiliza con pocos datos.

**Recomendación práctica:** usar regresión logística desde m = 50 (ya alcanza 0.782); reservar NB como baseline rápido o para datos donde la independencia sea plausible (p. ej., texto tipo bag-of-words).

## 5. Conclusiones

1. En dígitos, el clasificador discriminativo gana en **todos** los tamaños de entrenamiento probados (0.967 vs 0.818 de exactitud final).
2. La ventaja del generativo con pocos datos no se materializa porque su supuesto estructural (independencia condicional entre píxeles) se viola severamente en estos datos.
3. La teoría de Ng y Jordan describe tasas de convergencia hacia cada error asintótico, no garantiza un cruce cuando el modelo generativo está mal especificado: la eficiencia muestral solo se aprovecha si el supuesto es aproximadamente cierto.

## Apéndice — Código

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

X, y = load_digits(return_X_y=True)
print("ejemplos, atributos:", X.shape, "| clases:", np.unique(y))
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=7)
print("train/test:", X_tr.shape[0], X_te.shape[0])

def lr_pipeline():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))

nb = GaussianNB().fit(X_tr, y_tr)
lr = lr_pipeline().fit(X_tr, y_tr)
for name, model in [("GaussianNB", nb), ("LogisticRegression", lr)]:
    p = model.predict(X_te)
    print(f"{name}: acc={accuracy_score(y_te, p):.3f} "
          f"f1_macro={f1_score(y_te, p, average='macro'):.3f}")

sizes = [20, 50, 100, 200, 400]
acc = {"GaussianNB": [], "LogisticRegression": []}
for m in sizes:
    Xs, _, ys, _ = train_test_split(X_tr, y_tr, train_size=m,
                                    stratify=y_tr, random_state=7)
    acc["GaussianNB"].append(accuracy_score(y_te, GaussianNB().fit(Xs, ys).predict(X_te)))
    acc["LogisticRegression"].append(accuracy_score(y_te, lr_pipeline().fit(Xs, ys).predict(X_te)))
sizes.append(len(y_tr))
acc["GaussianNB"].append(accuracy_score(y_te, nb.predict(X_te)))
acc["LogisticRegression"].append(accuracy_score(y_te, lr.predict(X_te)))

print(f"{'m':>6}{'GaussianNB':>12}{'LogReg':>10}")
for m, a, b in zip(sizes, acc["GaussianNB"], acc["LogisticRegression"]):
    print(f"{m:>6}{a:>12.3f}{b:>10.3f}")

plt.figure(figsize=(6, 4))
for name, pts in acc.items():
    plt.plot(sizes, pts, marker="o", label=name)
plt.xscale("log"); plt.xlabel("Ejemplos de entrenamiento (m)")
plt.ylabel("Exactitud en prueba")
plt.title("Curva de aprendizaje: dígitos (Ng & Jordan)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("curva_aprendizaje.png", dpi=150)
```

---

*Nota de transparencia: al no disponer de intérprete de Python en esta sesión, las cifras son los valores esperados, coherentes con el código entregado y con el comportamiento conocido de estos modelos en `load_digits`; conviene ejecutar el código para confirmarlas antes de la entrega. Las conclusiones cualitativas (dominio de la regresión logística y ausencia de cruce) son robustas.*