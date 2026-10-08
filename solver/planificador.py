"""Agente PLANIFICADOR — divide el enunciado en subtareas con tipo y dependencias.

REGLA 1 — El plan se valida con código (`validar_plan`): ids únicos, dependencias que
existen y sin ciclos, y cada sección del enunciado cubierta por alguna subtarea. Si falla,
el orquestador vuelve al planificador con la lista de problemas.
"""
from __future__ import annotations

import json
import re

import networkx as nx

from .grafo import GrafoTarea
from .llm import ClienteLLM

TIPOS = {"calculo", "redaccion"}
FORMATOS = {"md": ".md", "pdf": ".pdf", "ipynb": ".ipynb"}
MIN_CARACTERES_SECCION = 40      # una sección más corta (p. ej. «Tareas» sin texto) no se exige

PROMPT = """Eres el PLANIFICADOR de un sistema multiagente que resuelve tareas de maestría.
Recibes las secciones de un enunciado (con id) y las dependencias entre ellas, extraídas por
regla. Divide la tarea en subtareas:

- tipo "calculo": requiere ejecutar código (datos, modelos, métricas, figuras). Un
  programador escribirá UN script por subtarea, que se ejecuta en su propia carpeta.
- tipo "redaccion": discusión, preguntas conceptuales o el formato del entregable; las
  escribe el redactor al final usando los resultados de sus dependencias.
- Entre 2 y 8 subtareas. Una por parte del enunciado suele bastar.
- "secciones": los ids de sección que la subtarea cubre. TODA sección del enunciado debe
  quedar cubierta por alguna subtarea (también la del formato del entregable).
- "depende_de": ids de SUBTAREAS (no de secciones) cuyos resultados necesita. Respeta las
  dependencias del enunciado (si la Parte 3 usa la división de la Parte 1, la subtarea de la
  Parte 3 depende de la de la Parte 1). Sin ciclos.
- "figura": true si la subtarea debe producir una figura PNG.
- "resultados": los nombres de las cifras que la subtarea debe medir.

Además describe el ENTREGABLE tal como lo pide el enunciado:
- "formato": "md", "pdf" o "ipynb"; "archivo": el nombre del archivo (con su extensión);
- "secciones": los encabezados obligatorios, con el texto exacto y en orden (si el
  enunciado no los fija, propón uno por parte);
- "max_palabras" y "max_paginas": el límite, o null.

Responde solo JSON:
{"entrega": {"formato": "...", "archivo": "...", "secciones": ["..."], "max_palabras": null, "max_paginas": null},
 "subtareas": [{"id": "T1", "titulo": "...", "tipo": "calculo", "secciones": ["S1"],
   "depende_de": [], "objetivo": "qué hay que hacer, con los parámetros del enunciado",
   "figura": false, "resultados": ["..."]}]}"""


def formato_por_regla(texto: str) -> str | None:
    """Una pista sin LLM: la extensión que el enunciado nombra para la entrega."""
    t = texto.lower()
    if ".ipynb" in t or "notebook" in t:
        return "ipynb"
    if re.search(r"reporte\.pdf|reporte en \**pdf|en pdf", t):
        return "pdf"
    if re.search(r"reporte\.md|en markdown", t):
        return "md"
    return None


def secciones_exigibles(grafo: GrafoTarea) -> list[str]:
    return [s.id for s in grafo.secciones
            if s.id != "S0" and len(s.texto) >= MIN_CARACTERES_SECCION]


def validar_plan(plan: dict, grafo: GrafoTarea) -> list[str]:
    """REGLA 1. Devuelve la lista de problemas; vacía si el plan es válido."""
    problemas: list[str] = []
    subtareas = plan.get("subtareas")
    if not isinstance(subtareas, list) or not subtareas:
        return ["el plan no tiene subtareas"]

    ids = [s.get("id") for s in subtareas]
    repetidos = sorted({i for i in ids if ids.count(i) > 1})
    if repetidos:
        problemas.append(f"ids repetidos: {repetidos}")
    if any(not isinstance(i, str) or not i.strip() for i in ids):
        problemas.append("hay subtareas sin id")

    g = nx.DiGraph()
    g.add_nodes_from(i for i in ids if isinstance(i, str))
    for s in subtareas:
        if s.get("tipo") not in TIPOS:
            problemas.append(f"{s.get('id')}: tipo {s.get('tipo')!r} no es calculo ni redaccion")
        for d in s.get("depende_de") or []:
            if d not in ids:
                problemas.append(f"{s.get('id')}: depende de {d!r}, que no existe")
            elif d == s.get("id"):
                problemas.append(f"{s.get('id')}: depende de sí misma")
            else:
                g.add_edge(s["id"], d)
        for sec in s.get("secciones") or []:
            if sec not in grafo.g or grafo.g.nodes[sec]["tipo"] != "seccion":
                problemas.append(f"{s.get('id')}: la sección {sec!r} no existe")
    try:
        ciclo = nx.find_cycle(g)
        problemas.append(f"hay un ciclo de dependencias: {ciclo}")
    except nx.NetworkXNoCycle:
        pass

    cubiertas = {sec for s in subtareas for sec in (s.get("secciones") or [])}
    sin_cubrir = [sid for sid in secciones_exigibles(grafo) if sid not in cubiertas]
    if sin_cubrir:
        titulos = {s.id: s.titulo for s in grafo.secciones}
        problemas.append("secciones sin cubrir: " +
                         ", ".join(f"{sid} «{titulos[sid]}»" for sid in sin_cubrir))

    entrega = plan.get("entrega") or {}
    formato = entrega.get("formato")
    if formato not in FORMATOS:
        problemas.append(f"entrega.formato {formato!r} no es md, pdf ni ipynb")
    elif not str(entrega.get("archivo", "")).endswith(FORMATOS[formato]):
        problemas.append(f"entrega.archivo {entrega.get('archivo')!r} no termina en {FORMATOS[formato]}")
    if not entrega.get("secciones"):
        problemas.append("entrega.secciones está vacía")
    return problemas


def orden_topologico(plan: dict) -> list[str]:
    """Las subtareas en un orden que respeta las dependencias (y el del plan si empatan)."""
    g = nx.DiGraph()
    posicion = {s["id"]: i for i, s in enumerate(plan["subtareas"])}
    g.add_nodes_from(posicion)
    for s in plan["subtareas"]:
        for d in s.get("depende_de") or []:
            g.add_edge(d, s["id"])
    return list(nx.lexicographical_topological_sort(g, key=lambda n: posicion[n]))


class Planificador:
    def __init__(self, llm: ClienteLLM):
        self.llm = llm

    def planificar(self, grafo: GrafoTarea, texto: str, problemas: list[str] | None = None,
                   anterior: dict | None = None) -> dict:
        secciones = "\n\n".join(f"### {s.id} — {s.titulo}\n{s.texto}" for s in grafo.secciones)
        aristas = [f"{o} depende_de {d}" for o, d, t in grafo.g.edges(data="tipo")
                   if t == "depende_de"]
        usuario = (f"SECCIONES DEL ENUNCIADO\n\n{secciones}\n\n"
                   f"DEPENDENCIAS ENTRE SECCIONES (por regla)\n" + ("\n".join(aristas) or "(ninguna)") +
                   f"\n\nSECCIONES QUE DEBEN QUEDAR CUBIERTAS: {secciones_exigibles(grafo)}"
                   f"\nFORMATO DE ENTREGA DETECTADO POR REGLA: {formato_por_regla(texto) or 'no detectado'}")
        mensajes = [{"role": "system", "content": PROMPT}, {"role": "user", "content": usuario}]
        if problemas:
            mensajes += [{"role": "assistant", "content": json.dumps(anterior, ensure_ascii=False)},
                         {"role": "user", "content": "El plan no pasó la validación:\n- " +
                          "\n- ".join(problemas) + "\nCorrígelo y devuelve el plan completo."}]
        plan = self.llm.pedir_json("planificador", mensajes, max_tokens=24000, salida_estimada=4000)
        for s in plan.get("subtareas", []) if isinstance(plan, dict) else []:
            s["tipo"] = str(s.get("tipo", "")).lower().replace("á", "a").replace("ó", "o")
        return plan
