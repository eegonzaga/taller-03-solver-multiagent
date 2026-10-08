#!/usr/bin/env python3
"""Parte 2.c — La evidencia de las comprobaciones que fallaron, sacada de la traza.

    python peores_casos.py [version]        # por defecto «completo»
        → resultados/peores_casos_<version>.md

Para cada comprobación fallida junta lo que hace falta para señalar al agente responsable:
qué recibió cada agente (citas del investigador), qué produjo (intentos, veredictos,
cifras medidas) y qué decidió el código (revisiones, validación del entregable, REGLA 5).
Ordena primero las fallas de contenido (cifras, procedencia) y después las de forma.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
GRAVEDAD = {"cifra": 0, "cifra_presente": 0, "procedencia": 1, "codigo_sin": 1, "secciones": 2,
            "ipynb_ejecutado": 2, "archivo": 3, "palabras_max": 3, "paginas_max": 3, "contiene": 3}


def eventos(carpeta: Path) -> list[dict]:
    ruta = carpeta / "traza.jsonl"
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines()] if ruta.exists() else []


def resumen_traza(carpeta: Path) -> list[str]:
    ev = eventos(carpeta)
    lineas = []
    for e in ev:
        t = e["tipo"]
        if t == "validacion_plan":
            lineas.append(f"- **plan** ronda {e['ronda']}: {'válido' if e['valido'] else 'inválido: ' + '; '.join(e['problemas'])}")
        elif t == "investigador":
            citas = "; ".join(f"{f['cita']} ({f['motivo']})" for f in e["fragmentos"])
            lineas.append(f"- **investigador** {e['subtarea']}: {citas[:600]}")
        elif t == "ejecucion":
            lineas.append(f"- **ejecutor** {e['subtarea']} intento {e['intento']}: código {e['codigo_salida']}, "
                          f"{e['duracion_s']} s, creó {e['archivos_creados']}"
                          + (f", stderr: {e['stderr'][-200:]!r}" if e.get("stderr") and e["codigo_salida"] else ""))
        elif t == "revision":
            lineas.append(f"- **revisor** {e['subtarea']} intento {e['intento']}: "
                          f"{'APROBADO' if e['aprobado'] else 'RECHAZADO'} por {e['por']}"
                          + (f" — {'; '.join(e['problemas'])[:400]}" if e["problemas"] else ""))
        elif t.startswith("freno_"):
            lineas.append(f"- **{t}** {json.dumps({k: v for k, v in e.items() if k not in {'n', 't', 't_rel_s', 'tipo'}}, ensure_ascii=False)[:300]}")
        elif t == "verificacion_entrega":
            lineas.append(f"- **verificación del entregable** ronda {e['ronda']}: {e['palabras']} palabras, "
                          f"{e['paginas']} páginas; problemas: {e['problemas'] or 'ninguno'}"[:900])
        elif t == "error":
            lineas.append(f"- **error**: {e['error']}")
    return lineas


def main() -> None:
    version = sys.argv[1] if len(sys.argv) > 1 else "completo"
    with (RAIZ / "resultados" / f"resultados_solver_{version}.csv").open(encoding="utf-8") as f:
        fallidas = [r for r in csv.DictReader(f) if r["ok"] == "0"]
    fallidas.sort(key=lambda r: (GRAVEDAD.get(r["tipo"], 9), r["tarea"], r["check"]))
    golden = {c["id"]: c for t in json.loads((RAIZ / "golden_tareas.json").read_text(encoding="utf-8"))["tareas"]
              for c in t["checks"]}

    partes = [f"# Comprobaciones fallidas — versión {version}\n",
              f"{len(fallidas)} fallidas. Las tres primeras son las peores (contenido antes que forma).\n"]
    vistas: set[str] = set()
    for r in fallidas:
        partes.append(f"## {r['check']} ({r['tipo']}) — {golden.get(r['check'], {}).get('descripcion', '')}\n")
        partes.append(f"Detalle del evaluador: `{r['detalle']}`\n")
        if r["tarea"] not in vistas:
            vistas.add(r["tarea"])
            carpeta = RAIZ / "corridas" / version / f"tarea-{r['tarea']}"
            partes.append(f"### Traza de la tarea {r['tarea']} ({carpeta.relative_to(RAIZ)}/traza.jsonl)\n")
            partes += resumen_traza(carpeta)
            partes.append("")
    destino = RAIZ / "resultados" / f"peores_casos_{version}.md"
    destino.write_text("\n".join(partes) + "\n", encoding="utf-8")
    print(destino.read_text(encoding="utf-8")[:6000])


if __name__ == "__main__":
    main()
