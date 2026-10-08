"""La clase pública del solver, con el contrato del Taller 03 v2 (y el del Taller 4).

    resultado = Solver().solve("tarea.pdf", "carpeta_salida/")
        status       "completado" | "parcial" | "fallido"
        entregables  ["carpeta_salida/reporte.md", ...]
        subtareas    [{"id", "tipo", "status", "intentos"}]
        usage        {"tokens_entrada", "tokens_salida", "por_agente": {...}}
        model        el id servido, leído de /v1/models
        trace        la lista de eventos (también en carpeta_salida/traza.jsonl)

    Solver().run("ruta/al/enunciado.pdf") → {"answer", "trace", "status", "model", "usage"}

Versiones recortadas de la Parte 2.b, para el evaluador:
    --solver solver:SolverSinGrafo      el investigador solo recibe los k más parecidos
    --solver solver:SolverSinRevisor    un intento por script y ninguna revisión

¿Quién decide que la tarea está resuelta? (Parte 5, pregunta 1) → `_status`: el código,
no un agente. El planificador propone, el revisor aprueba cada script, pero el status sale de
las subtareas aprobadas y de que el entregable pase `validar_entregable` (REGLA 5 incluida).
"""
from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path

from .config import RAIZ, Config
from .llm import ClienteLLM, Presupuesto, ServidorH200
from .orquestador import Orquestador
from .traza import Traza


class Solver:
    def __init__(self, config: Config | None = None, servidor=None):
        self.config = config or Config()
        self.servidor = servidor           # None → la H200; GuionLLM para los frenos

    def solve(self, ruta_pdf: str, salida: str) -> dict:
        salida_p = Path(salida)
        salida_p.mkdir(parents=True, exist_ok=True)
        traza = Traza(salida_p)                                    # REGLA 6, desde el inicio
        traza.registrar("inicio", entrada=str(ruta_pdf), version=self.version,
                        config={k: v for k, v in vars(self.config).items() if k != "confirmar_red"})
        estado: dict = {}
        modelo, llm = "", None
        try:
            servidor = self.servidor or ServidorH200()
            modelo = servidor.modelo
            llm = ClienteLLM(servidor, traza,
                             Presupuesto(self.config.presupuesto_tokens, self.config.reserva_redactor))
            estado = Orquestador(ruta_pdf, salida_p, llm, traza, self.config).correr()
        except Exception as err:  # noqa: BLE001 — la traza registra la falla y el contrato se cumple
            traza.registrar("error", error=f"{type(err).__name__}: {err}",
                            detalle=traceback.format_exc()[-3000:])
            estado["error"] = f"{type(err).__name__}: {err}"

        subtareas = list((estado.get("subtareas") or {}).values())
        entregable = estado.get("entregable")
        entregables = [entregable] if entregable and Path(entregable).exists() else []
        status = self._status(estado, subtareas, entregables)
        usage = llm.usage if llm else {"tokens_entrada": 0, "tokens_salida": 0, "por_agente": {}}
        if estado.get("plan"):
            plan = {**estado["plan"], "estado_final": subtareas}
            (salida_p / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
        traza.registrar("fin", status=status, entregables=entregables, subtareas=subtareas,
                        usage=usage, faltantes=estado.get("faltantes", []),
                        problemas_entrega=estado.get("problemas_entrega", []))
        resultado = {"status": status, "entregables": entregables, "subtareas": subtareas,
                     "usage": usage, "model": modelo, "trace": traza.eventos,
                     "traza_ruta": str(traza.ruta)}
        if estado.get("error"):
            resultado["error"] = estado["error"]
        return resultado

    @staticmethod
    def _status(estado: dict, subtareas: list[dict], entregables: list[str]) -> str:
        """La decisión final la toma este código (Parte 5, pregunta 1)."""
        if not entregables:
            return "fallido"
        todo_bien = (all(s["status"] in {"completado", "para_redactor"} for s in subtareas)
                     and not estado.get("problemas_entrega")
                     and not estado.get("presupuesto_agotado"))
        return "completado" if todo_bien else "parcial"

    def run(self, pregunta: str) -> dict:
        """Taller 4: la pregunta es la ruta a un enunciado; la respuesta, el entregable."""
        salida = RAIZ / "corridas" / f"run-{datetime.now():%Y%m%d-%H%M%S}"
        r = self.solve(pregunta, str(salida))
        answer = ""
        if r["entregables"]:
            ruta = Path(r["entregables"][0])
            answer = ruta.read_text(encoding="utf-8") if ruta.suffix in {".md", ".ipynb"} else str(ruta)
        return {"answer": answer, "trace": r["trace"], "status": r["status"],
                "model": r["model"], "usage": r["usage"]}

    @property
    def version(self) -> str:
        if not self.config.usar_grafo:
            return "sin_grafo"
        if not self.config.usar_revisor:
            return "sin_revisor"
        return "completo"


class SolverSinGrafo(Solver):
    def __init__(self, config: Config | None = None, servidor=None):
        super().__init__(config or Config(usar_grafo=False), servidor)


class SolverSinRevisor(Solver):
    def __init__(self, config: Config | None = None, servidor=None):
        super().__init__(config or Config(usar_revisor=False, max_intentos=1), servidor)
