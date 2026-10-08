"""Agente REDACTOR — escribe el entregable en el formato pedido.

El redactor escribe Markdown; el código lo convierte al formato pedido (formatos.py) y lo
valida (`validar_entregable`) antes de publicarlo:

  - las secciones obligatorias, todas y en orden;
  - el límite de palabras o de páginas;
  - el notebook se ejecuta de principio a fin sin errores;
  - REGLA 5: cada cifra salió de una ejecución o está en el enunciado.

Si algo falla, el orquestador vuelve al redactor con la lista de problemas.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from . import formatos
from .llm import ClienteLLM
from .procedencia import cifras_sin_origen, sin_codigo
from .programador import resumir_json

PROMPT = """Eres el REDACTOR de un sistema multiagente que resuelve tareas de maestría.
Escribe el entregable en Markdown, en español, con estas reglas:

1. Usa como encabezados («## ») EXACTAMENTE las secciones obligatorias, en ese orden. Puedes
   añadir un título «# » al inicio y subsecciones «### » dentro de cada sección.
2. CIFRAS: usa solo las que aparecen en RESULTADOS MEDIDOS o en el enunciado. Cópialas tal
   cual, redondeadas a 4 decimales como máximo, en la misma escala (si el resultado es
   0.947368, escribe 0.9474, no 94.74 %). Nunca inventes ni estimes una cifra. Si falta una
   cifra porque la subtarea falló, dilo explícitamente.
3. Tablas en Markdown cuando el enunciado pida una tabla.
4. Figuras: escribe el marcador [[FIGURA Tn]] donde va la figura de la subtarea Tn.
5. Solo si el formato es notebook: escribe el marcador [[CODIGO Tn]] en su propia línea
   donde debe ir el código de cada subtarea de cálculo (todas deben aparecer), justo después
   del encabezado de la parte que resuelve y antes de comentar sus resultados.
6. Respeta el límite de palabras o de páginas, con margen.
7. Para las partes conceptuales, apóyate en el enunciado y en las notas del curso citadas.
Responde solo con el Markdown del entregable."""


@dataclass
class Entregable:
    ruta: Path | None
    markdown: str
    problemas: list[str] = field(default_factory=list)
    palabras: int = 0
    paginas: int = 0
    cifras_huerfanas: list[str] = field(default_factory=list)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return re.sub(r"[^a-z0-9]+", " ", "".join(c for c in s if not unicodedata.combining(c))).strip()


def posicion_seccion(texto: str, seccion: str) -> int:
    """El mismo criterio que evaluar_solver.py: la primera línea que ES el encabezado."""
    objetivo = _norm(seccion)
    pos = 0
    for linea in texto.splitlines(keepends=True):
        h = re.sub(r"^\d+ ", "", _norm(linea.lstrip("# ")))
        if h and len(h) <= len(objetivo) + 40 and (h == objetivo or h.startswith(objetivo + " ")):
            return pos
        pos += len(linea)
    return -1


class Redactor:
    def __init__(self, llm: ClienteLLM):
        self.llm = llm

    def redactar(self, entrega: dict, subtareas: list[dict], resultados: dict[str, dict],
                 contexto: str, faltantes: list[str], problemas: list[str] | None = None,
                 anterior: str | None = None) -> str:
        bloques = []
        for s in subtareas:
            r = resultados.get(s["id"], {})
            linea = f"### {s['id']} — {s.get('titulo', '')} ({s['tipo']}, estado: {s.get('status', 'pendiente')})"
            if s["tipo"] == "calculo" and r:
                linea += (f"\nresultados.json: {resumir_json(r.get('json', {}), 3500)}"
                          f"\nstdout (final): {r.get('stdout', '')[-1200:]}"
                          f"\nfiguras: {r.get('figuras', [])}")
            bloques.append(linea)
        usuario = (
            f"ENTREGABLE: formato {entrega['formato']}, archivo {entrega['archivo']}\n"
            f"SECCIONES OBLIGATORIAS (en orden): {entrega.get('secciones')}\n"
            f"LÍMITE: palabras={entrega.get('max_palabras')}, páginas={entrega.get('max_paginas')}\n\n"
            f"RESULTADOS MEDIDOS POR SUBTAREA\n\n" + "\n\n".join(bloques) +
            (f"\n\nLO QUE FALTÓ (decláralo en el entregable): {faltantes}" if faltantes else "") +
            f"\n\nENUNCIADO Y NOTAS DEL CURSO (citados)\n\n{contexto}")
        mensajes = [{"role": "system", "content": PROMPT}, {"role": "user", "content": usuario}]
        if problemas and anterior:
            mensajes += [{"role": "assistant", "content": anterior},
                         {"role": "user", "content": "El entregable no pasó la validación:\n- " +
                          "\n- ".join(problemas) + "\nDevuelve el entregable completo corregido."}]
        texto = self.llm.pedir("redactor", mensajes, max_tokens=32000, salida_estimada=6000)
        return re.sub(r"^```(?:markdown|md)?\s*\n(.*)\n```\s*$", r"\1", texto.strip(), flags=re.S)


def validar_entregable(md: str, entrega: dict, salida: Path, figuras: dict[str, list[str]],
                       codigos: dict[str, str], textos_ejecucion: list[str], enunciado: str,
                       timeout_s: int) -> Entregable:
    """Escribe el entregable en su formato y lo revisa con código. REGLA 5 incluida."""
    formato = entrega["formato"]
    destino = salida / Path(entrega["archivo"]).name
    problemas: list[str] = []
    cache_mpl = salida / ".mpl"

    for marca in formatos.MARCADOR_CODIGO.findall(md) + formatos.MARCADOR_FIGURA.findall(md):
        if marca not in codigos:
            problemas.append(f"el marcador {marca} no corresponde a ninguna subtarea con código aprobado")

    if formato == "ipynb":
        faltantes = formatos.construir_notebook(formatos.MARCADOR_FIGURA.sub("", md), codigos, destino)
        if faltantes:
            problemas.append(f"faltan los marcadores [[CODIGO …]] de {faltantes}")
        ok, log = formatos.ejecutar_notebook(destino, timeout_s, cache_mpl)
        if not ok:
            problemas.append(f"el notebook no se ejecutó sin errores: {log[-800:]}")
        texto, salidas_nb = formatos.texto_de_notebook(destino)
        textos_ejecucion = textos_ejecucion + [salidas_nb]
        paginas = 0
    else:
        texto = formatos.insertar_figuras(formatos.quitar_marcadores_codigo(md), figuras)
        if formato == "md":
            destino.write_text(texto, encoding="utf-8")
            paginas = 0
        else:
            paginas = formatos.markdown_a_pdf(texto, destino, salida, entrega.get("max_paginas"))
            if entrega.get("max_paginas") and paginas > entrega["max_paginas"]:
                problemas.append(f"el PDF tiene {paginas} páginas y el máximo es {entrega['max_paginas']}: "
                                 "acorta el texto (un 25 % menos como mínimo)")

    posiciones = [posicion_seccion(texto, s) for s in entrega.get("secciones", [])]
    faltan = [s for s, p in zip(entrega.get("secciones", []), posiciones) if p < 0]
    if faltan:
        problemas.append(f"faltan las secciones (encabezado con el texto exacto): {faltan}")
    elif posiciones != sorted(posiciones):
        problemas.append(f"las secciones no están en el orden pedido: {entrega['secciones']}")

    palabras = len(sin_codigo(texto).split())
    if entrega.get("max_palabras") and palabras > entrega["max_palabras"]:
        problemas.append(f"tiene {palabras} palabras y el máximo es {entrega['max_palabras']}")

    huerfanas = cifras_sin_origen(texto, textos_ejecucion, enunciado)       # REGLA 5
    if huerfanas:
        problemas.append("REGLA 5 — estas cifras no salieron de ninguna ejecución ni están en el "
                         f"enunciado; corrígelas copiando el valor medido o elimínalas: {huerfanas[:25]}")
    return Entregable(destino, texto, problemas, palabras, paginas, huerfanas)


def entregable_minimo(entrega: dict, subtareas: list[dict], resultados: dict, faltantes: list[str]) -> str:
    """Sin presupuesto ni para el redactor: lo que se tiene, escrito por código y sin LLM."""
    partes = ["# Entrega parcial (sin presupuesto para redactar)"]
    for sec in entrega.get("secciones", ["Resultados"]):
        partes.append(f"## {sec}")
        if _norm(sec).startswith("resultado"):
            for s in subtareas:
                if s["id"] in resultados:
                    partes.append(f"### {s['id']} — {s.get('titulo', '')}\n\n```json\n"
                                  f"{json.dumps(resultados[s['id']].get('json', {}), indent=1)[:3000]}\n```")
                    partes.append(f"[[CODIGO {s['id']}]]")
    partes.append("## Lo que faltó\n\n" + "\n".join(f"- {f}" for f in faltantes))
    return "\n\n".join(partes)
