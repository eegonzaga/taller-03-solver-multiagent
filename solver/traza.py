"""REGLA 6 — Siempre se guarda la traza, también si algo falla.

Cada evento es una línea de `traza.jsonl` que se escribe y se vacía a disco en el momento:
si el proceso muere a mitad de la corrida, lo ocurrido hasta ese punto ya está guardado.

- Por llamada al LLM: agente, modelo, tokens de entrada y salida, latencia y error.
- Por ejecución: código de salida, duración y archivos creados.
- Además: plan, validaciones, veredictos del revisor y cada freno que salta.

Los mensajes completos de cada llamada van aparte, a `traza_llm/`, para poder reconstruir
qué recibió y qué produjo cada agente (Parte 2.c) sin inflar la traza principal.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path


class Traza:
    def __init__(self, carpeta: Path, reiniciar: bool = True):
        self.carpeta = Path(carpeta)
        self.ruta = self.carpeta / "traza.jsonl"
        self.carpeta_llm = self.carpeta / "traza_llm"
        self.carpeta_llm.mkdir(parents=True, exist_ok=True)
        if reiniciar or not self.ruta.exists():
            self.ruta.write_text("", encoding="utf-8")      # una corrida, una traza
        self.eventos: list[dict] = []
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()

    def registrar(self, tipo: str, **datos) -> dict:
        with self._lock:
            evento = {"n": len(self.eventos) + 1,
                      "t": datetime.now().isoformat(timespec="seconds"),
                      "t_rel_s": round(time.perf_counter() - self._t0, 2),
                      "tipo": tipo, **datos}
            self.eventos.append(evento)
            with self.ruta.open("a", encoding="utf-8") as f:
                f.write(json.dumps(evento, ensure_ascii=False, default=str) + "\n")
        return evento

    def guardar_intercambio(self, n: int, agente: str, mensajes: list[dict], respuesta: dict) -> str:
        """El intercambio completo con el LLM; la traza principal guarda solo la ruta."""
        ruta = self.carpeta_llm / f"{n:04d}-{agente}.json"
        ruta.write_text(json.dumps({"mensajes": mensajes, "respuesta": respuesta},
                                   ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return str(ruta.relative_to(self.carpeta))
