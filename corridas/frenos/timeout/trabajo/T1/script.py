import json
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=7)
res = {"n": int(X.shape[0]), "atributos": int(X.shape[1]), "clases": int(len(set(y))),
       "n_entrenamiento": len(ytr), "n_prueba": len(yte)}
print(res)
json.dump(res, open("resultados.json", "w"))
