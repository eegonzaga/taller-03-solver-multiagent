import json
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=7)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
res = {}
for nombre, m in [("nb", GaussianNB()), ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)))]:
    p = m.fit(Xtr, ytr).predict(Xte)
    res[nombre + "_accuracy"] = accuracy_score(yte, p)
    res[nombre + "_f1_macro"] = f1_score(yte, p, average="macro")
print(res)
json.dump(res, open("resultados.json", "w"))
