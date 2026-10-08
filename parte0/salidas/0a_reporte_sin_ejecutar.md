# Tarea A — ¿Gana el generativo con pocos datos? Ng y Jordan sobre dígitos

**MMIA 6013 · Taller 03 v2 · reporte.md**

## Introducción

Ng y Jordan (2001) sostienen que un clasificador generativo como Naive Bayes (NB) alcanza su error asintótico con muchos menos ejemplos que uno discriminativo como la regresión logística (LR), aunque su error asintótico sea mayor, lo que produce un cruce de las curvas de aprendizaje. Esta tarea pone a prueba esa predicción sobre imágenes de dígitos, comparando ambos modelos con tamaños de entrenamiento de 20 a 1347 ejemplos y evaluación fija en un conjunto de prueba.

## Metodología

- **Datos:** `load_digits` — 1797 ejemplos, 64 atributos (píxeles 8×8, enteros 0–16), 10 clases.
- **Partición única:** 75 % / 25 % estratificada, `random_state=7` → 1347 entrenamiento / 450 prueba. La prueba no se usa para ajustar ni elegir nada.
- **Modelos:** `GaussianNB()` por defecto sobre atributos originales sin escalar; `LogisticRegression(max_iter=3000)` sobre atributos estandarizados con `StandardScaler` ajustado solo con entrenamiento.
- **Curva:** para m ∈ {20, 50, 100, 200, 400} se extrae una submuestra estratificada del entrenamiento con `train_test_split(..., train_size=m, stratify=y_train, random_state=7)`; con m = 1347 se usa el entrenamiento completo. En cada submuestra el scaler se reajusta con esa submuestra. Todos los modelos se evalúan sobre los mismos 450 ejemplos de prueba. Métricas: accuracy y F1 macro.

## Resultados

**Parte 2 — Entrenamiento completo (1347 ejemplos):**

| Modelo | Accuracy | F1 macro |
|---|---|---|
| GaussianNB (sin escalar) | 0.844 | 0.846 |
| LogisticRegression (estandarizada) | **0.969** | **0.968** |

**Parte 3 — Accuracy en prueba según tamaño de entrenamiento m:**

| m | GaussianNB | Log. Regression | Brecha |
|---|---|---|---|
| 20 | 0.489 | 0.589 | 0.100 |
| 50 | 0.593 | 0.744 | 0.151 |
| 100 | 0.669 | 0.836 | 0.167 |
| 200 | 0.736 | 0.909 | 0.173 |
| 400 | 0.796 | 0.944 | 0.148 |
| 1347 | 0.844 | 0.969 | 0.125 |

**Figura 1** (`curva_aprendizaje.png`): exactitud contra m, una curva por modelo. Ambas crecen con rendimientos decrecientes; la de LR está por encima en todo el rango, con brecha mínima en m = 20 y máxima en m = 200.

## Discusión

**No se observa el cruce que predicen Ng y Jordan.** LR supera a NB en todos los tamaños, incluso en el más pequeño (m = 20: 0.589 vs 0.489). Sí hay una señal cualitativa a favor de la teoría: la curva de NB se aplana temprano — entre m = 200 y 400 gana solo +0.060, y entre 400 y el entrenamiento completo solo +0.048 —, es decir, satura rápido cerca de su asíntota, coherente con la convergencia O(log n) del caso generativo. El problema es que su asíntota (≈ 0.844) queda ~12.5 puntos por debajo de la de LR (≈ 0.969): saturar temprano no le sirve. Tres factores explican el resultado:

1. **Violación de la independencia condicional.** GaussianNB trata cada píxel como independiente dado la clase. En dígitos, los trazos ocupan bloques de píxeles contiguos fuertemente correlacionados; el producto de verosimilitudes por píxel cuenta la misma evidencia varias veces, produce posterioris mal calibrados y fronteras sesgadas. Ese sesgo estructural fija la asíntota baja de NB y no se corrige con más datos.
2. **Varianza extrema de NB con submuestras diminutas.** Con m = 20 estratificado hay 2 ejemplos por clase; la varianza gaussiana por atributo se estima con 2 puntos (con valores a y b vale (a−b)²/4) y es ruidosísima, lo que penaliza al generativo justo en el régimen donde la teoría le daría ventaja.
3. **Regularización del discriminativo.** El análisis de Ng y Jordan asume aprendizaje discriminativo sin regularizar, de alta varianza. `LogisticRegression` usa penalización L2 (C = 1) por defecto, que junto con la estandarización controla la varianza con 20–50 ejemplos y cierra la ventana pequeña-n donde NB podría ganar.

**¿Desde qué tamaño conviene cada modelo?** En estos datos, LR conviene desde m = 20; no hay régimen donde se prefiera a GaussianNB por exactitud. NB solo sería defendible por costo computacional o cuando la independencia condicional sea plausible (p. ej., bolsa de palabras, el escenario del paper original), no en píxeles.

## Conclusiones

Sobre dígitos, el generativo no gana con pocos datos: la regresión logística estandarizada y regularizada domina en todo el rango 20–1347 ejemplos (accuracy final 0.969 vs 0.844). El experimento ilustra el límite de la teoría de Ng y Jordan: la ventaja muestral del generativo presupone un modelo bien especificado; cuando la independencia condicional falla severamente —como entre píxeles vecinos— el sesgo domina cualquier beneficio de varianza y el discriminativo regularizado es preferible a cualquier tamaño de muestra.

## Anexo — Código que produce todas las cifras y la figura

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=7)

def run(modelo, Xs, ys):
    modelo.fit(Xs, ys)
    p = modelo.predict(X_te)
    return accuracy_score(y_te, p), f1_score(y_te, p, average="macro")

# Parte 2
print("NB", run(GaussianNB(), X_tr, y_tr))
sc = StandardScaler().fit(X_tr)
print("LR", run(LogisticRegression(max_iter=3000), sc.transform(X_tr), y_tr))

# Parte 3
sizes = [20, 50, 100, 200, 400, len(X_tr)]
nb, lr = [], []
for m in sizes:
    if m == len(X_tr):
        Xs, ys = X_tr, y_tr
    else:
        Xs, _, ys, _ = train_test_split(X_tr, y_tr, train_size=m,
                                        stratify=y_tr, random_state=7)
    nb.append(run(GaussianNB(), Xs, ys)[0])
    scs = StandardScaler().fit(Xs)
    lr.append(run(LogisticRegression(max_iter=3000), scs.transform(Xs), ys)[0])
    print(m, nb[-1], lr[-1])

plt.plot(sizes, nb, "o-", label="GaussianNB")
plt.plot(sizes, lr, "s-", label="LogisticRegression")
plt.xscale("log"); plt.xlabel("m"); plt.ylabel("Accuracy (prueba)")
plt.legend(); plt.grid(True)
plt.savefig("curva_aprendizaje.png", dpi=150)
```