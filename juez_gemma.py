#!/usr/bin/env python3
"""Parte 4.E — Juez de otra familia: gemma3 califica los entregables y se compara con las
comprobaciones de código (acuerdo y kappa de Cohen). Con la H200 (VPN).

    python juez_gemma.py                 # corridas/completo y corridas/sin_grafo
        → resultados/juez_gemma.csv (un veredicto por comprobación)
        → resultados/juez_resumen.csv (acuerdo y kappa por versión y en total)

El juez (gemma3:27b, en el Ollama de la H200) NO es de la familia del solver (GLM). Recibe el
enunciado (su rúbrica), el entregable y la lista de archivos de la salida, y responde, para
cada comprobación del golden set redactada en lenguaje natural (`descripcion`), si se cumple.
No ve la verdad calculada: juzga como lo haría un corrector humano que no ejecuta código.
"""
from __future__ import annotations

import csv
import json
import os
import urllib.request
from pathlib import Path

from evaluar_solver import entregable, texto_de

RAIZ = Path(__file__).resolve().parent
OLLAMA = os.getenv("H200_EMBED_URL", "http://172.28.230.10:11434")
MODELO_JUEZ = os.getenv("JUEZ_MODELO", "gemma3:27b")
# «control_sin_ejecutar»: el reporte de la Parte 0.a (cifras inventadas, bien redactado) como si
# fuera la entrega de la Tarea A. Mide si el juez distingue una cifra medida de una inventada.
VERSIONES = ["completo", "sin_grafo", "control_sin_ejecutar"]

PROMPT = """Eres un profesor que corrige una tarea de maestría. Recibes el enunciado, el
entregable de un estudiante y la lista de archivos que entregó. Para CADA criterio de la
lista decide si el entregable lo cumple (true) o no (false), con una razón breve. No puedes
ejecutar código: juzga lo que ves escrito. Si no puedes verificar una cifra, decide si es
plausible y coherente con el enunciado.
Responde solo JSON: {"veredictos": [{"id": "...", "cumple": true, "razon": "..."}]}"""


def juzgar(enunciado: str, texto: str, archivos: list[str], criterios: list[dict]) -> dict[str, dict]:
    usuario = (f"ENUNCIADO\n{enunciado[:8000]}\n\nARCHIVOS ENTREGADOS\n{archivos[:60]}\n\n"
               f"ENTREGABLE\n{texto[:14000]}\n\nCRITERIOS\n" +
               "\n".join(f"- {c['id']}: {c['descripcion']}" for c in criterios))
    cuerpo = {"model": MODELO_JUEZ, "stream": False, "format": "json",
              "options": {"temperature": 0, "num_ctx": 16384},
              "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": usuario}]}
    pet = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(cuerpo).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(pet, timeout=900) as r:
        contenido = json.loads(r.read())["message"]["content"]
    datos = json.loads(contenido)
    return {str(v.get("id")): v for v in datos.get("veredictos", [])}


def kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def evaluador_de(v: str) -> dict:
    with (RAIZ / "resultados" / f"resultados_solver_{v}.csv").open(encoding="utf-8") as f:
        return {(r["tarea"], r["check"]): int(r["ok"]) for r in csv.DictReader(f)}


def main() -> None:
    import sys
    golden = json.loads((RAIZ / "golden_tareas.json").read_text(encoding="utf-8"))
    filas = []
    if "--solo-resumen" in sys.argv:
        # Los veredictos del juez ya guardados; solo se refresca la columna del evaluador
        # (por si se re-evaluó con una versión corregida de evaluar_solver.py).
        with (RAIZ / "resultados" / "juez_gemma.csv").open(encoding="utf-8") as f:
            filas = list(csv.DictReader(f))
        evaluadores = {v: evaluador_de(v) for v in {f["version"] for f in filas}}
        for f in filas:
            f["evaluador"] = evaluadores[f["version"]].get((f["tarea"], f["check"]), "")
            f["juez"] = int(f["juez"]) if f["juez"] != "" else ""
    for v in ([] if filas else VERSIONES):
        ruta_csv = RAIZ / "resultados" / f"resultados_solver_{v}.csv"
        if not ruta_csv.exists():
            continue
        with ruta_csv.open(encoding="utf-8") as f:
            evaluador = {(r["tarea"], r["check"]): int(r["ok"]) for r in csv.DictReader(f)}
        for t in golden["tareas"]:
            salida = RAIZ / "corridas" / v / f"tarea-{t['id']}"
            if not salida.exists():
                continue
            principal = entregable(salida, t["entregable"]) if salida.exists() else None
            texto = texto_de(principal) if principal else "(no hay entregable)"
            archivos = sorted(str(p.relative_to(salida)) for p in salida.rglob("*")
                              if p.is_file() and not any(x in p.parts for x in ("traza_llm", ".mpl", ".jupyter", "intentos")))
            enunciado = texto_de(RAIZ / t["pdf"])
            veredictos = juzgar(enunciado, texto, archivos, t["checks"])
            for c in t["checks"]:
                ver = veredictos.get(c["id"], {})
                filas.append({"version": v, "tarea": t["id"], "check": c["id"], "tipo": c["tipo"],
                              "evaluador": evaluador.get((t["id"], c["id"]), ""),
                              "juez": int(bool(ver.get("cumple"))) if ver else "",
                              "razon": str(ver.get("razon", "sin veredicto"))[:300]})
            print(f"{v} {t['id']}: juez {sum(f['juez'] == 1 for f in filas if f['version'] == v and f['tarea'] == t['id'])}"
                  f"/{len(t['checks'])} · evaluador {sum(evaluador.get((t['id'], c['id']), 0) for c in t['checks'])}")

    destino = RAIZ / "resultados" / "juez_gemma.csv"
    with destino.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    resumen = []
    for grupo in VERSIONES + ["total"]:  # «total» = todas las filas
        pares = [(f["evaluador"], f["juez"]) for f in filas
                 if (grupo == "total" or f["version"] == grupo) and f["juez"] != "" and f["evaluador"] != ""]
        if not pares:
            continue
        a, b = [p[0] for p in pares], [p[1] for p in pares]
        resumen.append({"grupo": grupo, "n": len(pares),
                        "acuerdo": round(sum(x == y for x, y in pares) / len(pares), 3),
                        "kappa": round(kappa(a, b), 3),
                        "aprueba_evaluador": sum(a), "aprueba_juez": sum(b),
                        "juez_aprueba_y_codigo_rechaza": sum(x == 0 and y == 1 for x, y in pares),
                        "juez_rechaza_y_codigo_aprueba": sum(x == 1 and y == 0 for x, y in pares)})
    with (RAIZ / "resultados" / "juez_resumen.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resumen[0]))
        w.writeheader()
        w.writerows(resumen)
    for r in resumen:
        print(r)


if __name__ == "__main__":
    main()
