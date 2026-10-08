import json
from sklearn.datasets import fetch_openml
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
json.dump({"n": len(y)}, open("resultados.json", "w"))
