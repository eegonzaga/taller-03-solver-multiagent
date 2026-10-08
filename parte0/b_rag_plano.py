#!/usr/bin/env python3
"""Parte 0.b — El RAG que pierde la referencia. Sin modelo, sin red.

    python parte0/b_rag_plano.py

Antes de programar la curva de aprendizaje de la Tarea A, el programador necesita saber «¿con
qué datos y con qué partición se calcula?». La Parte 3 solo dice «con la partición de la
Parte 1»: qué datos son y cuál es la proporción de prueba está escrito en la Parte 1, que no se
parece a la pregunta.

1) RAG plano: un fragmento por sección, TF-IDF y coseno, los k más parecidos.
2) Los mismos k, más un salto por las aristas `depende_de` que el solver extrae con una
   regla (`solver.grafo.aristas_depende_de`, la misma función que usa en cada corrida).
"""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from solver.grafo import aristas_depende_de, dividir_en_secciones  # noqa: E402
from solver.lector import leer_entrada  # noqa: E402

PDF = RAIZ / "enunciados" / "tarea-a-ng-jordan-digitos.pdf"
PREGUNTA = "¿Con qué datos y con qué partición se calcula la curva de aprendizaje?"
# Lo que el programador necesita, y el texto que lo prueba (solo está en la Parte 1).
HECHOS = {"conjunto de datos": "load_digits", "proporción de prueba": "25 %",
          "prueba intocable": "no se usa para ajustar"}
K = 2


def cubre(texto: str) -> list[str]:
    texto = " ".join(texto.split())
    return [h for h, marca in HECHOS.items() if marca in texto]


def main() -> int:
    secciones = {s.id: s for s in dividir_en_secciones(leer_entrada(PDF))}
    ids = list(secciones)
    textos = [secciones[i].literal for i in ids]
    tfidf = TfidfVectorizer().fit(textos + [PREGUNTA])
    sim = cosine_similarity(tfidf.transform([PREGUNTA]), tfidf.transform(textos))[0]
    orden = sorted(range(len(ids)), key=lambda i: -sim[i])

    print(f"Pregunta del programador: «{PREGUNTA}»\n")
    print(f"1) RAG plano: los {K} fragmentos más parecidos (TF-IDF, coseno)")
    plano = [ids[i] for i in orden[:K]]
    for i in orden[:K]:
        print(f"   {ids[i]} «{secciones[ids[i]].titulo}»  sim={sim[i]:.3f}")
    hechos_plano = cubre(" ".join(secciones[s].literal for s in plano))
    print(f"   Hechos en el contexto: {len(hechos_plano)} de {len(HECHOS)} {hechos_plano}\n")

    aristas = aristas_depende_de(list(secciones.values()))
    print(f"2) Los mismos {K}, más un salto por depende_de (regla, sin LLM)")
    grafo = list(plano)
    for semilla in plano:
        vecinos = [a["destino"] for a in aristas if a["origen"] == semilla]
        print(f"   {semilla} —depende_de→ {', '.join(vecinos) or '(ninguna)'}")
        grafo += [v for v in vecinos if v not in grafo]
    hechos_grafo = cubre(" ".join(secciones[s].literal for s in grafo))
    print(f"   Hechos en el contexto: {len(hechos_grafo)} de {len(HECHOS)} {hechos_grafo}\n")

    print("Todas las aristas depende_de del enunciado:")
    for a in aristas:
        print(f"   {a['origen']} → {a['destino']}   «…{a['evidencia']}…»")
    print(f"\nRAG plano: {len(hechos_plano)} de {len(HECHOS)} hechos. "
          f"Con el salto: {len(hechos_grafo)} de {len(HECHOS)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
