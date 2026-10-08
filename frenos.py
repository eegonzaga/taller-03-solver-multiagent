#!/usr/bin/env python3
"""Parte 3 — Los cuatro frenos, cada uno activado y con su traza. Sin gastar tokens.

    python frenos.py                # los cinco escenarios
    python frenos.py feliz red      # algunos

Usa un **modelo de guion** (`GuionLLM`): respuestas fijas por agente, sobre la Tarea A. El
solver es el mismo de siempre (los mismos nodos, el mismo sandbox, el mismo revisor con
código); solo cambia quién contesta. Los embeddings son TF-IDF, sin red.

Escenarios → corridas/frenos/<escenario>/traza.jsonl
  feliz        el camino sin frenos (comprueba que el flujo completo funciona)
  presupuesto  FRENO 1: el presupuesto se agota a mitad de la corrida; se entrega lo que hay
  intentos     FRENO 2: un script con fuga se rechaza; sus copias idénticas no se ejecutan;
               al tercer intento la subtarea queda fallida y el solver sigue con la siguiente
  timeout      FRENO 3: un script que no termina se mata a los 3 s (y, aparte, una prueba
               directa de que también mueren sus procesos hijos)
  red          FRENO 4: un script con fetch_openml espera la aprobación de una persona; se
               deniega y el programador reescribe con datos locales
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from solver import Config, Solver
from solver.llm import GuionLLM

RAIZ = Path(__file__).resolve().parent
TAREA_A = RAIZ / "enunciados" / "tarea-a-ng-jordan-digitos.pdf"
SALIDA = RAIZ / "corridas" / "frenos"

PLAN = {
    "entrega": {"formato": "md", "archivo": "reporte.md", "max_palabras": 1000, "max_paginas": None,
                "secciones": ["Introducción", "Metodología", "Resultados", "Discusión", "Conclusiones"]},
    "subtareas": [
        {"id": "T1", "titulo": "Datos y partición", "tipo": "calculo", "secciones": ["S1"],
         "depende_de": [], "objetivo": "load_digits, tamaños, partición 75/25 estratificada, random_state=7",
         "figura": False, "resultados": ["n", "atributos", "clases"]},
        {"id": "T2", "titulo": "Dos clasificadores", "tipo": "calculo", "secciones": ["S2"],
         "depende_de": ["T1"], "objetivo": "GaussianNB y regresión logística estandarizada",
         "figura": False, "resultados": ["accuracy", "f1_macro"]},
        {"id": "T3", "titulo": "Curva de aprendizaje", "tipo": "calculo", "secciones": ["S3"],
         "depende_de": ["T1", "T2"], "objetivo": "exactitud con 20, 50, 100, 200, 400 y todos",
         "figura": True, "resultados": ["curva"]},
        {"id": "T4", "titulo": "Discusión y reporte", "tipo": "redaccion", "secciones": ["S0", "S4", "S5"],
         "depende_de": ["T2", "T3"], "objetivo": "discutir el cruce de Ng y Jordan", "figura": False,
         "resultados": []},
    ]}

COMUN = '''import json
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=7)
'''
SCRIPTS = {
    "T1": COMUN + '''res = {"n": int(X.shape[0]), "atributos": int(X.shape[1]), "clases": int(len(set(y))),
       "n_entrenamiento": len(ytr), "n_prueba": len(yte)}
print(res)
json.dump(res, open("resultados.json", "w"))
''',
    "T2": COMUN + '''from sklearn.naive_bayes import GaussianNB
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
''',
    "T3": COMUN + '''import matplotlib.pyplot as plt
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
''',
}
# FRENO 2: evalúa sobre lo mismo con que entrenó (la fuga de la Parte 0.c).
CON_FUGA = COMUN + '''from sklearn.tree import DecisionTreeClassifier
modelo = DecisionTreeClassifier(random_state=7).fit(Xtr, ytr)
acc = float((modelo.predict(Xtr) == ytr).mean())
json.dump({"accuracy": acc}, open("resultados.json", "w"))
'''
# FRENO 3: no termina nunca.
INFINITO = '''import json
x = 0
while True:
    x += 1
'''
# FRENO 4: quiere descargar.
CON_RED = '''import json
from sklearn.datasets import fetch_openml
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
json.dump({"n": len(y)}, open("resultados.json", "w"))
'''

REPORTE = """# Tarea A — Ng y Jordan sobre dígitos

## Introducción
Comparamos un clasificador generativo y uno discriminativo, como proponen Ng y Jordan.

## Metodología
Partición estratificada 75/25 con la semilla del enunciado; escalador ajustado solo con el entrenamiento.

## Resultados
Las cifras están en resultados.json de cada subtarea.

[[FIGURA T3]]

## Discusión
Respuesta de guion: sin cifras, para no inventar ninguna.

## Conclusiones
El flujo completo funciona con el modelo de guion.
"""


def programador(scripts: dict[str, list[str]]):
    """Para cada subtarea, una lista de scripts que se entregan en orden (el último se repite)."""
    usados: dict[str, int] = {}

    def responder(mensajes):
        texto = mensajes[-1]["content"] if "RECHAZADO" not in mensajes[-1]["content"] else mensajes[-1]["content"]
        sid = next(s for s in scripts if f"SUBTAREA {s} " in texto)
        i = usados.get(sid, 0)
        usados[sid] = i + 1
        return "```python\n" + scripts[sid][min(i, len(scripts[sid]) - 1)] + "```"
    return responder


def guion(scripts: dict[str, list[str]]) -> GuionLLM:
    return GuionLLM({
        "indexador": ['{"entidades": []}'],
        "planificador": [json.dumps(PLAN)],
        "programador": programador(scripts),
        "revisor": ['{"aprobado": true, "problemas": [], "correccion": ""}'],
        "redactor": [REPORTE],
    })


def correr(nombre: str, scripts: dict, **config) -> dict:
    salida = SALIDA / nombre
    shutil.rmtree(salida, ignore_errors=True)
    cfg = Config(embeddings="tfidf", **config)
    r = Solver(cfg, servidor=guion(scripts)).solve(str(TAREA_A), str(salida))
    frenos = [e for e in r["trace"] if e["tipo"].startswith("freno_")]
    print(f"\n== {nombre}: status={r['status']} · subtareas="
          + ", ".join(f"{s['id']}:{s['status']}({s['intentos']})" for s in r["subtareas"]))
    for e in frenos:
        print(f"   {e['tipo']}: " + json.dumps({k: v for k, v in e.items()
                                                 if k not in {'n', 't', 't_rel_s', 'tipo'}},
                                                ensure_ascii=False)[:260])
    if r.get("error"):
        print("   ERROR:", r["error"])
    return r


def prueba_directa_killpg() -> None:
    """FRENO 3, a nivel del sandbox: un script que lanza un proceso hijo (algo que la revisión
    estática prohíbe; aquí se llama al ejecutor directamente) muere entero al vencer el tiempo."""
    import os
    from solver.sandbox import ejecutar
    carpeta = SALIDA / "timeout_hijos"
    shutil.rmtree(carpeta, ignore_errors=True)
    carpeta.mkdir(parents=True)
    (carpeta / "script.py").write_text(
        "import subprocess, sys, time\n"
        "hijo = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        "open('pid_hijo.txt', 'w').write(str(hijo.pid))\n"
        "time.sleep(600)\n")
    ej = ejecutar(carpeta / "script.py", timeout_s=3, cache_mpl=SALIDA / ".mpl")
    pid = int((carpeta / "pid_hijo.txt").read_text())
    try:
        os.kill(pid, 0)
        vivo = True
    except ProcessLookupError:
        vivo = False
    informe = {"timeout": ej.timeout, "duracion_s": ej.duracion_s, "pid_hijo": pid, "hijo_sigue_vivo": vivo}
    (carpeta / "prueba_killpg.json").write_text(json.dumps(informe, indent=1))
    print(f"\n== timeout (procesos hijos): {informe}")


ESCENARIOS = {
    "feliz": lambda: correr("feliz", {k: [v] for k, v in SCRIPTS.items()}),
    # Con el corpus en caché, T1 cuesta ~9 000 tokens de guion: el límite de 13 000 (20 000
    # menos la reserva de 7 000 del redactor) alcanza para T1 y se agota antes de programar T2.
    "presupuesto": lambda: correr("presupuesto", {k: [v] for k, v in SCRIPTS.items()},
                                  presupuesto_tokens=20000, reserva_redactor=7000),
    "intentos": lambda: correr("intentos", {"T1": [SCRIPTS["T1"]], "T2": [CON_FUGA],
                                            "T3": [SCRIPTS["T3"]]}),
    "timeout": lambda: (correr("timeout", {"T1": [INFINITO, SCRIPTS["T1"]], "T2": [SCRIPTS["T2"]],
                                           "T3": [SCRIPTS["T3"]]}, timeout_s=3),
                        prueba_directa_killpg()),
    "red": lambda: correr("red", {"T1": [CON_RED, SCRIPTS["T1"]], "T2": [SCRIPTS["T2"]],
                                  "T3": [SCRIPTS["T3"]]},
                          confirmar_red=persona_que_deniega),
}


def persona_que_deniega(subtarea: str, motivos: list[str], codigo: str) -> bool:
    """FRENO 4 en modo demostración: la «persona» ve la petición y responde que no."""
    print(f"   [persona] {subtarea} pide red: {motivos} → NO")
    return False


if __name__ == "__main__":
    for nombre in sys.argv[1:] or list(ESCENARIOS):
        ESCENARIOS[nombre]()
