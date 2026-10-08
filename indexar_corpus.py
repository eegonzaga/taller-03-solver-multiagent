#!/usr/bin/env python3
"""Indexa una vez el corpus de papers (el del Taller 02) para el GraphRAG. Con la H200 (VPN).

    python indexar_corpus.py

Extrae las entidades de cada fragmento con el LLM (en paralelo) y calcula sus embeddings
bge-m3. Todo queda en `.cache/`; el solver lo reutiliza en cada tarea y no lo vuelve a pagar.
Si no se corre antes, la primera tarea lo hace sola. La traza queda en
`.cache/indexado-corpus/traza.jsonl`.
"""
from __future__ import annotations

import time

from solver.config import CACHE
from solver.grafo import IndexadorGraphRAG, fragmentos_del_corpus
from solver.llm import ClienteLLM, Presupuesto, ServidorH200
from solver.rag import CodificadorH200
from solver.traza import Traza

if __name__ == "__main__":
    t0 = time.perf_counter()
    carpeta = CACHE / "indexado-corpus"
    carpeta.mkdir(parents=True, exist_ok=True)
    traza = Traza(carpeta, reiniciar=False)
    llm = ClienteLLM(ServidorH200(), traza, Presupuesto(10**12, 0))
    codificador = CodificadorH200(cache=CACHE)
    indexador = IndexadorGraphRAG(llm, codificador, traza, hilos=8)
    fragmentos = indexador._corpus_con_entidades()
    codificador.encode([f["texto"] for f in fragmentos])
    con_entidades = sum(bool(f["entidades"]) for f in fragmentos)
    print(f"{len(fragmentos)} fragmentos, {con_entidades} con entidades · modelo {llm.modelo} · "
          f"embeddings {codificador.nombre} · {llm.usage['tokens_entrada']} + "
          f"{llm.usage['tokens_salida']} tokens · {time.perf_counter() - t0:.0f} s")
