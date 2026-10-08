import json
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=7)
from sklearn.tree import DecisionTreeClassifier
modelo = DecisionTreeClassifier(random_state=7).fit(Xtr, ytr)
acc = float((modelo.predict(Xtr) == ytr).mean())
json.dump({"accuracy": acc}, open("resultados.json", "w"))
