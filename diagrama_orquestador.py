#!/usr/bin/env python3
"""El diagrama del orquestador, sacado del grafo de LangGraph compilado (no dibujado a mano).

    python diagrama_orquestador.py    → informe/diagrama_orquestador.png y .mmd (Mermaid)
"""
from pathlib import Path

from solver.orquestador import dibujar_orquestador

if __name__ == "__main__":
    carpeta = Path(__file__).resolve().parent / "informe"
    carpeta.mkdir(exist_ok=True)
    dibujar_orquestador(carpeta / "diagrama_orquestador.png", carpeta / "diagrama_orquestador.mmd")
    print((carpeta / "diagrama_orquestador.mmd").read_text())
