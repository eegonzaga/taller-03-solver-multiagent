import json
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=7)
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
curva = {"m": [], "nb": [], "lr": []}
for m in [20, 50, 100, 200, 400, len(ytr)]:
    if m < len(ytr):
        Xs, _, ys, _ = train_test_split(Xtr, ytr, train_size=m, stratify=ytr, random_state=7)
    else:
        Xs, ys = Xtr, ytr
    curva["m"].append(m)
    curva["nb"].append(accuracy_score(yte, GaussianNB().fit(Xs, ys).predict(Xte)))
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xs, ys)
    curva["lr"].append(accuracy_score(yte, lr.predict(Xte)))
plt.plot(curva["m"], curva["nb"], "o-", label="Naive Bayes")
plt.plot(curva["m"], curva["lr"], "s-", label="Regresión logística")
plt.xscale("log"); plt.xlabel("ejemplos de entrenamiento"); plt.ylabel("exactitud"); plt.legend()
plt.savefig("curva_aprendizaje.png", dpi=120)
print(curva)
json.dump(curva, open("resultados.json", "w"))
