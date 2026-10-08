#!/usr/bin/env python3
"""informe/informe.md → informe/informe.pdf

    python tablas.py && python informe/construir_informe.py

Las líneas `<!-- incluir: ruta -->` se reemplazan por el contenido de ese archivo (las tablas
que genera tablas.py desde los CSV, salidas de la Parte 0, CSV del juez), para que ninguna
tabla del informe se copie a mano. Las rutas son relativas a la raíz del repositorio.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from solver.formatos import markdown_a_pdf  # noqa: E402


def csv_a_tabla(ruta: Path) -> str:
    with ruta.open(encoding="utf-8") as f:
        filas = list(csv.reader(f))
    # Encabezados largos de CSV (juez_aprueba_y_codigo_rechaza) partidos para que la tabla quepa.
    lineas = ["| " + " | ".join(c.replace("_", " ") for c in filas[0]) + " |", "|" + "---|" * len(filas[0])]
    return "\n".join(lineas + ["| " + " | ".join(f) + " |" for f in filas[1:]])


def incluir(m: re.Match) -> str:
    ruta = RAIZ / m.group(1).strip()
    if ruta.suffix == ".csv":
        return csv_a_tabla(ruta)
    texto = ruta.read_text(encoding="utf-8")
    if ruta.suffix == ".txt":
        return "```\n" + texto.strip() + "\n```"
    return re.sub(r"^# .*\n", "", texto)          # sin el título del archivo incluido


if __name__ == "__main__":
    md = (RAIZ / "informe" / "informe.md").read_text(encoding="utf-8")
    md = re.sub(r"<!-- incluir: (.+?) -->", incluir, md)
    (RAIZ / "informe" / "informe_completo.md").write_text(md, encoding="utf-8")
    paginas = markdown_a_pdf(md, RAIZ / "informe" / "informe.pdf", RAIZ, None)
    print(f"informe/informe.pdf: {paginas} páginas")
