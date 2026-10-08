#!/usr/bin/env python3
"""Las tablas del informe (Parte 2.b), solo a partir de los CSV del evaluador.

    python tablas.py          → resultados/tablas.md

Lee, para cada versión (completo y sin_grafo), `resumen_solver_<v>.csv`,
`resultados_solver_<v>.csv` y `tokens_agente_solver_<v>.csv`. Ninguna cifra de estas tablas se
escribe a mano: si un CSV cambia, se vuelve a correr este script.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

RESULTADOS = Path(__file__).resolve().parent / "resultados"
VERSIONES = ["completo", "sin_grafo"]


def leer(nombre: str) -> list[dict]:
    ruta = RESULTADOS / nombre
    if not ruta.exists():
        return []
    with ruta.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tabla(encabezado: list[str], filas: list[list]) -> str:
    lineas = ["| " + " | ".join(encabezado) + " |", "|" + "---|" * len(encabezado)]
    lineas += ["| " + " | ".join(str(c) for c in f) + " |" for f in filas]
    return "\n".join(lineas)


def main() -> None:
    partes = ["# Tablas de la Parte 2.b (generadas por tablas.py desde los CSV)\n"]

    filas, totales = [], []
    for v in VERSIONES:
        resumen = leer(f"resumen_solver_{v}.csv")
        for r in resumen:
            filas.append([r["tarea"], v, r["status"], f"{r['aprobadas']}/{r['total']}", r["procedencia"],
                          f"{r['subtareas_fallidas']}/{r['subtareas']}", r["intentos_codigo"],
                          r["tokens_entrada"], r["tokens_salida"], r["duracion_s"]])
        if resumen:
            totales.append([v, sum(int(r["aprobadas"]) for r in resumen), sum(int(r["total"]) for r in resumen),
                            sum(int(r["subtareas_fallidas"]) for r in resumen),
                            sum(int(r["intentos_codigo"]) for r in resumen),
                            sum(int(r["tokens_entrada"]) for r in resumen),
                            sum(int(r["tokens_salida"]) for r in resumen),
                            round(sum(float(r["duracion_s"] or 0) for r in resumen), 1)])
    filas.sort(key=lambda f: (f[0], VERSIONES.index(f[1])))
    partes.append("## Por tarea y versión\n")
    partes.append(tabla(["Tarea", "Versión", "Status", "Checks", "Proced.", "Fallidas",
                         "Intentos", "Tok. entrada", "Tok. salida", "s"], filas))
    partes.append("\n## Totales por versión\n")
    partes.append(tabla(["Versión", "Aprobadas", "Total", "Subtareas fallidas", "Intentos",
                         "Tokens entrada", "Tokens salida", "Duración (s)"],
                        [[t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]] for t in totales]))

    partes.append("\n## Tokens por agente (suma de las tareas)\n")
    por_agente: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for v in VERSIONES:
        for r in leer(f"tokens_agente_solver_{v}.csv"):
            acc = por_agente[(r["agente"], v)]
            acc[0] += int(r["llamadas"])
            acc[1] += int(r["tokens_entrada"])
            acc[2] += int(r["tokens_salida"])
    agentes = sorted({a for a, _ in por_agente},
                     key=lambda a: -sum(sum(por_agente[(a, v)][1:]) for v in VERSIONES))
    filas = []
    for a in agentes:
        fila = [a]
        for v in VERSIONES:
            ll, ent, sal = por_agente.get((a, v), [0, 0, 0])
            fila += [ll, ent, sal]
        filas.append(fila)
    encabezado = ["Agente"] + [f"{c} ({v})" for v in VERSIONES for c in ("llamadas", "entrada", "salida")]
    partes.append(tabla(encabezado, filas))

    partes.append("\n## Tokens por tarea y agente (entrada / salida)\n")
    por_tarea: dict[tuple[str, str, str], tuple[str, str]] = {}
    for v in VERSIONES:
        for r in leer(f"tokens_agente_solver_{v}.csv"):
            por_tarea[(r["tarea"], r["agente"], v)] = (r["tokens_entrada"], r["tokens_salida"])
    claves = sorted({(t, a) for t, a, _ in por_tarea}, key=lambda k: (k[0], agentes.index(k[1])
                                                                     if k[1] in agentes else 99))
    filas = [[t, a] + [" / ".join(por_tarea.get((t, a, v), ("0", "0"))) for v in VERSIONES]
             for t, a in claves]
    partes.append(tabla(["Tarea", "Agente"] + [f"{v}" for v in VERSIONES], filas))

    partes.append("\n## Comprobaciones: completo frente a sin_grafo\n")
    estado = {v: {(r["tarea"], r["check"]): r for r in leer(f"resultados_solver_{v}.csv")} for v in VERSIONES}
    claves = sorted(set(estado["completo"]) | set(estado["sin_grafo"]))
    filas = [[t, c, (estado["completo"].get((t, c)) or {}).get("tipo", ""),
              "✓" if (estado["completo"].get((t, c)) or {}).get("ok") == "1" else "✗",
              "✓" if (estado["sin_grafo"].get((t, c)) or {}).get("ok") == "1" else "✗",
              (estado["completo"].get((t, c)) or {}).get("detalle", "")[:70]]
             for t, c in claves]
    partes.append(tabla(["Tarea", "Check", "Tipo", "Completo", "Sin grafo", "Detalle (completo)"], filas))

    destino = RESULTADOS / "tablas.md"
    destino.write_text("\n".join(partes) + "\n", encoding="utf-8")
    print(destino.read_text(encoding="utf-8"))
    remediciones()


def remediciones() -> None:
    """Parte 2.c: cada corrección medida antes y después, con el mismo golden set.
    Cada caso es una tarea y una secuencia de etapas (etiqueta, CSV de resumen)."""
    casos = [
        ("C", [("v1: medición inicial", "v1/resumen_solver_{v}.csv"),
               ("revisor: fuga solo en ajustes supervisados", "v1/resumen_solver_{v}_fixC.csv")]),
        ("R1", [("medición oficial", "resumen_solver_{v}.csv"),
                ("notebook sobre una copia (trabajo_nb/)", "fix/resumen_solver_{v}.csv"),
                ("revisor sabe que no hay red; plausibilidad solo en clasificadores", "fix2/resumen_solver_{v}.csv")]),
        ("R2", [("medición oficial", "resumen_solver_{v}.csv"),
                ("presupuesto en los reintentos por max_tokens", "fix/resumen_solver_{v}.csv")]),
    ]
    filas = []
    for tarea, etapas in casos:
        for v in VERSIONES:
            for etiqueta, plantilla in etapas:
                r = next((x for x in leer(plantilla.format(v=v)) if x["tarea"] == tarea), None)
                if r:
                    filas.append([tarea, v, etiqueta, r["status"], f"{r['aprobadas']}/{r['total']}",
                                  r["procedencia"], f"{r['subtareas_fallidas']}/{r['subtareas']}",
                                  r["intentos_codigo"], int(r["tokens_entrada"]) + int(r["tokens_salida"]),
                                  r["duracion_s"]])
    if filas:
        texto = "\n\n" + tabla(["Tarea", "Versión", "Etapa", "Status", "Checks", "Proced.",
                                "Fallidas", "Intentos", "Tokens", "s"], filas) + "\n"
        (RESULTADOS / "remediciones.md").write_text(texto, encoding="utf-8")
        print(texto)

if __name__ == "__main__":
    main()
