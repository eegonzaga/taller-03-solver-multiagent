#!/usr/bin/env python3
"""Los enunciados de práctica, de Markdown a PDF.

    python enunciados/generar_pdfs.py

El solver recibe **un PDF**, que es como llega una tarea de verdad. Los enunciados se escriben
en Markdown (se revisan en un diff) y se imprimen con la misma función que usa el redactor
(`solver.formatos.markdown_a_pdf`, PyMuPDF sin navegador).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))

from solver.formatos import markdown_a_pdf  # noqa: E402

if __name__ == "__main__":
    for md in sorted(AQUI.glob("tarea-*.md")):
        pdf = md.with_suffix(".pdf")
        paginas = markdown_a_pdf(md.read_text(encoding="utf-8"), pdf, AQUI, None)
        texto = "".join(p.get_text() for p in pymupdf.open(pdf))
        print(f"{md.name} → {pdf.name} ({paginas} p., {len(texto)} caracteres)")
