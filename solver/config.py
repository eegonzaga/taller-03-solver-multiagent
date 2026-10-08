"""Configuración del solver: rutas, límites y los interruptores de las versiones recortadas.

Todo lo que un experimento puede querer cambiar está aquí, con su valor por defecto y la
razón. Las variables de entorno solo sobreescriben; ningún valor secreto vive en este archivo.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

RAIZ = Path(__file__).resolve().parents[1]
CORPUS = RAIZ / "corpus"          # los papers del curso (el corpus del Taller 02)
CACHE = RAIZ / ".cache"           # el corpus indexado: se calcula una vez


@dataclass
class Config:
    # FRENO 1 — presupuesto de tokens por corrida, con una reserva que solo puede usar el
    # redactor. La solución de referencia gastó entre 79 000 y 138 000 tokens por tarea.
    presupuesto_tokens: int = int(os.getenv("SOLVER_PRESUPUESTO", "450000"))
    reserva_redactor: int = int(os.getenv("SOLVER_RESERVA_REDACTOR", "60000"))

    # FRENO 2 — máximo de intentos de código por subtarea.
    max_intentos: int = int(os.getenv("SOLVER_MAX_INTENTOS", "3"))

    # FRENO 3 — tiempo máximo de un script en el sandbox (segundos).
    timeout_s: int = int(os.getenv("SOLVER_TIMEOUT_S", "180"))

    # FRENO 4 — quién aprueba un script que quiere usar la red. None = preguntar por consola
    # si hay terminal; si no la hay, se deniega (nadie pudo aprobarlo).
    confirmar_red: Callable[[str, list[str], str], bool] | None = None

    # Rondas de corrección: plan inválido → planificador; entregable inválido → redactor.
    max_rondas_plan: int = 3
    max_rondas_redactor: int = 3

    # Versiones recortadas de la Parte 2.b.
    usar_grafo: bool = True          # False: el investigador solo recibe los k más parecidos
    usar_revisor: bool = True        # False: un intento por script y ninguna revisión
    k_fragmentos: int = 4            # fragmentos por subtarea en la versión «sin grafo»
    k_corpus: int = 3                # fragmentos del corpus de papers por subtarea

    # Embeddings: «h200» (bge-m3 en el Ollama de la H200, como en el Taller 02) o «tfidf»
    # (sin red; para las pruebas con el modelo de guion).
    embeddings: str = os.getenv("SOLVER_EMBEDDINGS", "h200")

    # Llamadas al LLM en paralelo al indexar (entidades de las secciones y del corpus).
    hilos_indexado: int = 8

    extra: dict = field(default_factory=dict)
