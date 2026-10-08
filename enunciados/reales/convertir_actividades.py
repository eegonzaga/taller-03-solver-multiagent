#!/usr/bin/env python3
"""Las dos tareas reales (Parte 2.a): actividades de clase de la Semana 2 de MMIA 6013,
convertidas de notebook a enunciado en PDF.

    python enunciados/reales/convertir_actividades.py

Fuente: `Semana2/ActividadClase/s2-mar-estudiante.ipynb` y `s2-mie-estudiante.ipynb`
(autor del material: Daniel Riofrío, MMIA 6013). Los notebooks son la copia de la estudiante,
ya resueltos, así que la conversión **retira las soluciones**:

  - celdas de Markdown: se conservan tal cual (son el enunciado del profesor);
  - celdas de código del profesor (datos, setup, andamiaje): se conservan como código dado;
  - celdas de ejercicio (`# Ejercicio …`) y de reflexión: solo sus líneas de comentario
    iniciales, que dicen qué hay que hacer; el código y las respuestas de la estudiante, no.

El resultado (`*.md` y `*.pdf`) queda en esta carpeta, así que el repositorio no depende de
la ruta original.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[1]))
from solver.formatos import markdown_a_pdf  # noqa: E402

ORIGEN = Path("/Users/estefania.gonzaga/Documents/Personal/Master-USFQ-Classes/"
              "AIGenerativa&Agentes/Semana2/ActividadClase")

# Por celda: «dada» (código del profesor), «enunciado» (solo los comentarios iniciales).
# Las celdas de Markdown siempre se conservan.
ACTIVIDADES = {
    "actividad-s2-mar-ann": {
        "notebook": "s2-mar-estudiante.ipynb",
        "celdas": {1: "dada", 3: "enunciado", 5: "dada", 6: "enunciado", 7: "enunciado",
                   9: "dada", 10: "reflexion"},
    },
    "actividad-s2-mie-rag": {
        "notebook": "s2-mie-estudiante.ipynb",
        "celdas": {1: "dada", 2: "dada", 4: "enunciado", 5: "dada", 7: "dada", 9: "dada",
                   11: "enunciado", 13: "dada", 14: "dada", 16: "dada", 18: "reflexion"},
    },
}

ENCABEZADO = """**Tarea real de MMIA 6013** — actividad de clase de la Semana 2, convertida de notebook a
PDF para el solver (apartado 2.a del Taller 03 v2). Se conservan el texto y el código que da el
profesor; las soluciones de los ejercicios se retiraron.

**Entrega:** un notebook de Jupyter (`.ipynb`) **ejecutado** de principio a fin sin errores, con
una celda de Markdown que encabece cada parte, los ejercicios resueltos y las reflexiones
respondidas con las cifras medidas.

"""


def comentarios_iniciales(fuente: str, cortar_en: tuple[str, ...] = ()) -> str:
    """Las líneas de comentario del principio de la celda: el enunciado del ejercicio."""
    salida = []
    for linea in fuente.splitlines():
        if not linea.startswith("#") or linea.lstrip("# ").startswith(cortar_en):
            break
        salida.append(linea)
    return "\n".join(salida)


def convertir(nombre: str, spec: dict) -> Path:
    nb = json.loads((ORIGEN / spec["notebook"]).read_text(encoding="utf-8"))
    partes = []
    for i, celda in enumerate(nb["cells"]):
        fuente = "".join(celda["source"]).strip()
        if celda["cell_type"] == "markdown":
            partes.append(fuente)
            if i == 0:
                partes.append(ENCABEZADO)
            continue
        modo = spec["celdas"].get(i)
        if modo == "dada":
            partes.append(f"Código dado:\n\n```python\n{fuente}\n```")
        elif modo == "enunciado":
            partes.append(f"```python\n{comentarios_iniciales(fuente)}\n# (a completar)\n```")
        elif modo == "reflexion":
            pregunta = comentarios_iniciales(fuente, cortar_en=("Respuesta", "a) Estimo"))
            partes.append(f"```python\n{pregunta}\n```")
    md = "\n\n".join(partes) + "\n"
    destino = AQUI / f"{nombre}.md"
    destino.write_text(md, encoding="utf-8")
    paginas = markdown_a_pdf(md, destino.with_suffix(".pdf"), AQUI, None)
    print(f"{spec['notebook']} → {destino.with_suffix('.pdf').name} ({paginas} p.)")
    return destino


if __name__ == "__main__":
    for nombre, spec in ACTIVIDADES.items():
        convertir(nombre, spec)
