"""Solver multiagente con GraphRAG — Taller 03 v2, MMIA 6013.

Agentes: lector, indexador (GraphRAG), planificador, investigador, programador, ejecutor
(sandbox), revisor y redactor, coordinados por un orquestador de LangGraph.

    from solver import Solver
    resultado = Solver().solve("enunciados/tarea-a-generativo-discriminativo.pdf", "corridas/tarea-A")
"""
from .config import Config
from .solver import Solver, SolverSinGrafo, SolverSinRevisor

__all__ = ["Config", "Solver", "SolverSinGrafo", "SolverSinRevisor"]
