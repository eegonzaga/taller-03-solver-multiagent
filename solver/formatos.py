"""El entregable en el formato pedido: Markdown, PDF o notebook ejecutado.

- **PDF**: Markdown → HTML (markdown-it) → PDF con PyMuPDF (`Story`), igual que
  `generar_enunciados.py` del kit, sin navegador ni LaTeX. Si pasa del límite de páginas se
  reduce la letra; si aun así no cabe, el redactor tiene que acortar.
- **Notebook**: el texto del redactor trae marcadores `[[CODIGO T1]]` donde va el código
  aprobado de cada subtarea. Cada celda de código entra en la carpeta de su subtarea, corre el
  script tal como lo aprobó el revisor y muestra sus figuras. El notebook se ejecuta con
  nbconvert **con el mismo entorno limpio del sandbox** (REGLA 3) y un tiempo máximo.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import nbformat
import pymupdf
from markdown_it import MarkdownIt

from .sandbox import entorno_limpio

MARCADOR_CODIGO = re.compile(r"\[\[CODIGO\s+([\w\-]+)\]\]")
MARCADOR_FIGURA = re.compile(r"\[\[FIGURA\s+([\w\-]+)\]\]")

CSS = """
body {{ font-family: sans-serif; font-size: {tam}pt; line-height: 1.3; }}
h1 {{ font-size: {h1}pt; margin-bottom: 4pt; }}
h2 {{ font-size: {h2}pt; margin-top: 8pt; margin-bottom: 3pt; }}
h3 {{ font-size: {tam}pt; margin-top: 6pt; margin-bottom: 2pt; }}
p {{ margin-top: 2pt; margin-bottom: 4pt; text-align: justify; }}
table {{ border-collapse: collapse; margin: 4pt 0; }}
th, td {{ border: 0.6pt solid #888; padding: 1pt 3pt; font-size: {tabla}pt; }}
th {{ background-color: #eeeeee; }}
code {{ font-family: monospace; font-size: {tabla}pt; }}
pre {{ white-space: pre-wrap; font-size: {tabla}pt; }}
"""


# ─────────────────────────────────────────────────────────── figuras en Markdown
def insertar_figuras(md: str, figuras: dict[str, list[str]]) -> str:
    """`[[FIGURA T3]]` → las imágenes de la subtarea T3 (rutas relativas a la salida). Las
    figuras que el redactor no colocó van al final de «Resultados» (o al final)."""
    usadas: set[str] = set()

    def reemplazo(m):
        rutas = figuras.get(m.group(1), [])
        usadas.add(m.group(1))
        return "\n\n".join(f"![Figura de {m.group(1)}]({r})" for r in rutas)

    md = MARCADOR_FIGURA.sub(reemplazo, md)
    faltan = [r for sid, rutas in figuras.items() if sid not in usadas for r in rutas]
    if faltan:
        bloque = "\n\n" + "\n\n".join(f"![Figura]({r})" for r in faltan) + "\n\n"
        m = re.search(r"^#+\s*Resultados.*$", md, re.M | re.I)
        if m:
            siguiente = re.search(r"^#+\s", md[m.end():], re.M)
            corte = m.end() + siguiente.start() if siguiente else len(md)
            md = md[:corte].rstrip() + bloque + md[corte:]
        else:
            md = md.rstrip() + bloque
    return md


def quitar_marcadores_codigo(md: str) -> str:
    return MARCADOR_CODIGO.sub("", md)


# ─────────────────────────────────────────────────────────── PDF
def markdown_a_pdf(md: str, destino: Path, base: Path, max_paginas: int | None) -> int:
    """Escribe el PDF y devuelve su número de páginas, bajando la letra si hace falta."""
    html = MarkdownIt("commonmark").enable("table").render(md)
    html = html.replace("<img ", '<img width="330" ')
    paginas = 0
    for tam in (10.5, 10, 9.5, 9, 8.5):
        css = CSS.format(tam=tam, h1=tam + 5, h2=tam + 2, tabla=tam - 1)
        historia = pymupdf.Story(html=html, user_css=css, archive=str(base))
        escritor = pymupdf.DocumentWriter(str(destino))
        a4 = pymupdf.paper_rect("a4")
        caja = a4 + (50, 50, -50, -50)
        mas, paginas = True, 0
        while mas:
            dispositivo = escritor.begin_page(a4)
            mas, _ = historia.place(caja)
            historia.draw(dispositivo)
            escritor.end_page()
            paginas += 1
        escritor.close()
        if not max_paginas or paginas <= max_paginas:
            break
    return paginas


# ─────────────────────────────────────────────────────────── notebook
PREPARACION = '''# Preparación: cada subtarea corre en su carpeta, tal como la aprobó el revisor.
import os
from pathlib import Path
from IPython.display import Image, display

RAIZ = Path.cwd()

def mostrar_figuras(carpeta):
    for png in sorted(Path(carpeta).rglob("*.png")):
        display(Image(filename=str(png)))'''


def construir_notebook(md: str, codigos: dict[str, str], destino: Path) -> list[str]:
    """Convierte el texto con marcadores en un notebook. Devuelve las subtareas con código
    aprobado que el redactor no colocó (se agregan al final, para no perder evidencia)."""
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    celdas = [nbformat.v4.new_code_cell(PREPARACION)]
    colocadas: list[str] = []

    def celda_de_codigo(sid: str):
        codigo = re.sub(r"^from __future__ import .*$", "", codigos[sid], flags=re.M).strip()
        return nbformat.v4.new_code_cell(
            f"# Subtarea {sid}: código aprobado (se ejecuta en una copia: trabajo_nb/{sid}/)\n"
            f"os.chdir(RAIZ / 'trabajo_nb' / '{sid}')\n\n{codigo}\n\n"
            f"os.chdir(RAIZ)\nmostrar_figuras(RAIZ / 'trabajo_nb' / '{sid}')")

    pos = 0
    for m in MARCADOR_CODIGO.finditer(md):
        texto = md[pos:m.start()].strip()
        if texto:
            celdas.append(nbformat.v4.new_markdown_cell(texto))
        if m.group(1) in codigos and m.group(1) not in colocadas:
            celdas.append(celda_de_codigo(m.group(1)))
            colocadas.append(m.group(1))
        pos = m.end()
    if md[pos:].strip():
        celdas.append(nbformat.v4.new_markdown_cell(md[pos:].strip()))
    faltantes = [sid for sid in codigos if sid not in colocadas]
    for sid in faltantes:
        celdas.append(nbformat.v4.new_markdown_cell(f"### Código de la subtarea {sid}"))
        celdas.append(celda_de_codigo(sid))
    nb.cells = celdas
    nbformat.write(nb, destino)
    return faltantes


def ejecutar_notebook(ruta: Path, timeout_s: int, cache_mpl: Path) -> tuple[bool, str]:
    """nbconvert en un proceso aparte, con el entorno limpio del sandbox.

    El notebook corre sobre una COPIA de trabajo/ (trabajo_nb/): si un resultado no es
    determinista (una latencia), la ejecución del notebook no debe sobrescribir la evidencia
    de la que el redactor copió sus cifras (REGLA 5). Medido en la tarea R1 del 2026-10-07."""
    carpeta = ruta.parent
    import shutil
    if (carpeta / "trabajo").exists():
        shutil.rmtree(carpeta / "trabajo_nb", ignore_errors=True)
        shutil.copytree(carpeta / "trabajo", carpeta / "trabajo_nb",
                        ignore=shutil.ignore_patterns(".mpl"))
    entorno = entorno_limpio(carpeta, cache_mpl)
    jupyter = carpeta / ".jupyter"
    entorno.update({"JUPYTER_RUNTIME_DIR": str(jupyter / "runtime"),
                    "JUPYTER_DATA_DIR": str(jupyter / "data"),
                    "JUPYTER_CONFIG_DIR": str(jupyter / "config"),
                    "IPYTHONDIR": str(jupyter / "ipython"),
                    "JUPYTER_PLATFORM_DIRS": "1"})
    try:
        p = subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
                            "--inplace", f"--ExecutePreprocessor.timeout={timeout_s}", ruta.name],
                           cwd=carpeta, env=entorno, capture_output=True, text=True,
                           timeout=timeout_s * 4)
    except subprocess.TimeoutExpired:
        return False, "el notebook superó el tiempo máximo"
    return p.returncode == 0, (p.stderr or p.stdout)[-2500:]


def texto_de_notebook(ruta: Path) -> tuple[str, str]:
    """(markdown, salidas) del notebook: lo que escribió el redactor y lo que imprimió la
    ejecución, por separado."""
    nb = nbformat.read(ruta, as_version=4)
    md, salidas = [], []
    for c in nb.cells:
        if c.cell_type == "markdown":
            md.append(c.source)
        for o in c.get("outputs", []):
            salidas.append(o.get("text", "") + o.get("data", {}).get("text/plain", ""))
    return "\n\n".join(md), "\n".join(salidas)
