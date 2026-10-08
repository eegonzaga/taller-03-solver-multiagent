"""Agente INVESTIGADOR — junta el contexto de cada subtarea. Sin LLM.

Con grafo (REGLA 2): la sección literal de la subtarea, las secciones de las que depende
(cierre transitivo de `depende_de`), las secciones de las subtareas de las que depende, el
preámbulo (ahí suelen estar los datos que da el enunciado) y los fragmentos de los papers
del corpus que comparten entidades con la sección. **Cada fragmento lleva su cita.**

Sin grafo (versión recortada de la Parte 2.b): solo los `k` fragmentos más parecidos a la
subtarea, sobre secciones y corpus a la vez, como el RAG plano de la Parte 0.b.
"""
from __future__ import annotations

from dataclasses import dataclass

from .grafo import GrafoTarea


@dataclass
class Fragmento:
    nodo: str
    cita: str
    texto: str
    motivo: str          # por qué está aquí: «propia», «depende_de», «entidades: 3», «similitud 0.61»


def contexto_de(subtarea: dict, plan: dict, grafo: GrafoTarea, usar_grafo: bool,
                k_fragmentos: int, k_corpus: int) -> list[Fragmento]:
    consulta = f"{subtarea.get('titulo', '')}. {subtarea.get('objetivo', '')}"
    if not usar_grafo:
        return [Fragmento(r["nodo"], grafo.cita(r["nodo"]), grafo.texto(r["nodo"]),
                          f"similitud {r['score']}") for r in grafo.buscar(consulta, k_fragmentos)]

    propias = [s for s in subtarea.get("secciones", []) if s in grafo.g]
    por_subtarea = {s["id"]: s for s in plan["subtareas"]}
    de_dependencias = [sec for d in subtarea.get("depende_de", []) if d in por_subtarea
                       for sec in por_subtarea[d].get("secciones", []) if sec in grafo.g]
    enlazadas = grafo.dependencias(propias + de_dependencias)

    fragmentos: list[Fragmento] = []
    vistos: set[str] = set()

    def agregar(nodo: str, motivo: str) -> None:
        if nodo not in vistos:
            vistos.add(nodo)
            fragmentos.append(Fragmento(nodo, grafo.cita(nodo), grafo.texto(nodo), motivo))

    agregar("S0", "preámbulo")
    for s in propias:
        agregar(s, "sección propia")
    for s in de_dependencias:
        agregar(s, "sección de una subtarea de la que depende")
    for s in enlazadas:
        agregar(s, "depende_de")
    # Las entidades del preámbulo (el paper que la tarea pone a prueba) también cuentan: las
    # de una sección como «Datos» (load_digits, partición) no suelen estar en ningún paper.
    for fragmento, compartidas in grafo.corpus_relacionado(propias + ["S0"], consulta, k_corpus):
        agregar(fragmento, f"papers del corpus, {compartidas} entidades compartidas")
    return fragmentos


def como_texto(fragmentos: list[Fragmento], max_corpus: int = 1800) -> str:
    partes = []
    for f in fragmentos:
        texto = f.texto if f.nodo.startswith("S") else f.texto[:max_corpus]
        partes.append(f"{f.cita} ({f.motivo})\n{texto}")
    return "\n\n---\n\n".join(partes)
