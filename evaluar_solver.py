#!/usr/bin/env python3
"""evaluar_solver.py — corre un solver sobre el golden set y escribe los CSV de la Parte 2.b.

    python evaluar_solver.py --solver solver:Solver --ruta . --corridas corridas/completo \\
        --salida resultados/resultados_solver_completo.csv
    python evaluar_solver.py --solver solver:SolverSinGrafo --ruta . --corridas corridas/sin_grafo \\
        --salida resultados/resultados_solver_sin_grafo.csv
    python evaluar_solver.py --solo-evaluar corridas/completo --salida ...   # sin correr el solver
    python evaluar_solver.py ... --solo A C                                  # algunas tareas
    python evaluar_solver.py ... --reanudar     # no repite las tareas cuya traza ya tiene «fin»

Cada comprobación se decide con código, sin juicio: un archivo existe o no; una cifra coincide
con su `verdad_py` dentro de la tolerancia o no. Este evaluador **no importa nada del
solver**: si compartieran código, un error del solver podría aprobarse a sí mismo.

Escribe tres CSV (las tablas del informe salen de ellos):
  resultados_*.csv     una fila por comprobación
  resumen_*.csv        una fila por tarea: aprobadas, procedencia, subtareas fallidas,
                       intentos de código, tokens y duración
  tokens_agente_*.csv  una fila por tarea y agente: llamadas y tokens de entrada y salida
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import sys
import time
import traceback
import unicodedata
from glob import glob
from pathlib import Path

AQUI = Path(__file__).resolve().parent
NUM = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)(\s?%)?(?![\w])")
# Lo que no escribió una ejecución no respalda ninguna cifra.
NO_ES_EJECUCION = re.compile(r"traza|grafo|plan|enunciado|intentos|\.mpl|\.jupyter|cache")


# ───────────────────────────────────────────────────────── lectura
def texto_pdf(ruta: Path) -> str:
    import pymupdf
    return unicodedata.normalize("NFKC", "\n".join(p.get_text() for p in pymupdf.open(ruta)))


def texto_de(ruta: Path, con_salidas: bool = True) -> str:
    if ruta.suffix == ".pdf":
        return texto_pdf(ruta)
    if ruta.suffix == ".ipynb":
        nb = json.loads(ruta.read_text(encoding="utf-8"))
        partes = []
        for c in nb.get("cells", []):
            if c["cell_type"] == "markdown":
                partes.append("".join(c.get("source", "")))
            elif con_salidas:
                partes += [salida_de(o) for o in c.get("outputs", [])]
        return "\n".join(partes)
    return ruta.read_text(encoding="utf-8", errors="replace")


def salida_de(o: dict) -> str:
    datos = o.get("data", {})
    return "".join(o.get("text", "")) + "".join(datos.get("text/plain", "")) + \
        "".join(datos.get("text/markdown", ""))


def numeros(texto: str) -> list[tuple[float, int]]:
    """(valor, decimales) de cada cifra; «12,5 %» cuenta como 12.5 y como 0.125."""
    salida = []
    for m in NUM.finditer(texto):
        crudo = m.group(1).replace(",", ".")
        v = float(crudo)
        dec = len(crudo.split(".")[1]) if "." in crudo else 0
        if m.group(2):
            salida.append((v / 100, dec + 2))
        salida.append((v, dec))
    return salida


def sin_codigo(texto: str) -> str:
    return re.sub(r"```.*?```", "", texto, flags=re.S)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return re.sub(r"[^a-z0-9]+", " ", "".join(c for c in s if not unicodedata.combining(c))).strip()


def posicion_seccion(texto: str, seccion: str) -> int:
    """La primera línea que ES el encabezado (con o sin «#» o número; un PDF no tiene «#»)."""
    objetivo, pos = _norm(seccion), 0
    for linea in texto.splitlines(keepends=True):
        h = re.sub(r"^\d+ ", "", _norm(linea.lstrip("# ")))
        if h and len(h) <= len(objetivo) + 40 and (h == objetivo or h.startswith(objetivo + " ")):
            return pos
        pos += len(linea)
    return -1


def verdad_de(check: dict, base: Path) -> float:
    codigo = check["verdad_py"]
    codigo = "\n".join(codigo) if isinstance(codigo, list) else codigo
    ambito: dict = {}
    import os
    anterior = os.getcwd()
    os.chdir(base)                      # verdad_py importa verdades.py desde la raíz
    try:
        exec(codigo, ambito)            # código del golden set, versionado con él; no del solver
    finally:
        os.chdir(anterior)
    return float(ambito["verdad"])


def entregable(salida: Path, patron: str) -> Path | None:
    hallados = sorted(salida.glob(patron)) or sorted(salida.rglob(patron))
    return hallados[0] if hallados else None


def evidencia_de_ejecucion(salida: Path, excluir: Path) -> list[float]:
    """Todo lo que escribió una ejecución: JSON, CSV y logs de los scripts, y las salidas de
    los notebooks."""
    valores: list[float] = []
    for ext in ("json", "csv", "txt", "log"):
        for f in salida.rglob(f"*.{ext}"):
            rel = str(f.relative_to(salida))
            if f == excluir or NO_ES_EJECUCION.search(rel) or f.stat().st_size > 5_000_000:
                continue
            valores += [v for v, _ in numeros(f.read_text(encoding="utf-8", errors="replace"))]
    for nb in salida.rglob("*.ipynb"):
        for c in json.loads(nb.read_text(encoding="utf-8")).get("cells", []):
            for o in c.get("outputs", []):
                valores += [v for v, _ in numeros(salida_de(o))]
    return valores


def codigo_generado(salida: Path) -> str:
    """El código que se ejecutó: los scripts aprobados y las celdas de los notebooks, sin
    comentarios. No cuentan los intentos rechazados (`intentos/`, nunca corrieron) ni un
    comentario que solo MENCIONA una URL: la primera versión los contaba y daba por usada
    la red cuando el sandbox la había bloqueado (corrida R1, 2026-10-07)."""
    # Un script cuenta como ejecutado solo si su carpeta tiene el salida.log del ejecutor: el
    # último intento de una subtarea que el sandbox bloqueó queda escrito pero nunca corrió.
    partes = [p.read_text(encoding="utf-8", errors="replace") for p in salida.rglob("*.py")
              if not NO_ES_EJECUCION.search(str(p.relative_to(salida))) and (p.parent / "salida.log").exists()]
    for nb in salida.rglob("*.ipynb"):
        partes += ["".join(c.get("source", "")) for c in json.loads(nb.read_text(encoding="utf-8"))["cells"]
                   if c["cell_type"] == "code"]
    return "\n".join(l for parte in partes for l in parte.splitlines() if not l.lstrip().startswith("#"))


# ───────────────────────────────────────────────────────── comprobaciones
def comprobar(check: dict, salida: Path, tarea: dict, enunciado: str, base: Path) -> tuple[bool, str]:
    tipo = check["tipo"]
    principal = entregable(salida, tarea["entregable"])

    if tipo == "archivo":
        # Los artefactos del propio solver (grafo.png, la traza) no cuentan como entrega.
        n = len([p for p in glob(str(salida / check["patron"]), recursive=True)
                 if not NO_ES_EJECUCION.search(str(Path(p).relative_to(salida)))])
        if n < check.get("minimo", 1) and check.get("o_imagen_en_ipynb") and principal \
                and principal.suffix == ".ipynb" and '"image/png"' in principal.read_text(encoding="utf-8"):
            return True, "imagen embebida en el notebook"
        return n >= check.get("minimo", 1), f"{n} archivo(s) con {check['patron']}"
    if tipo == "codigo_sin":
        codigo = codigo_generado(salida)
        hallados = [p for p in check["patrones"] if re.search(p, codigo)]
        return not hallados, f"encontrados: {hallados}" if hallados else "ninguno"
    if principal is None:
        return False, f"no hay entregable {tarea['entregable']}"

    texto = texto_de(principal)
    if tipo == "secciones":
        posiciones = [posicion_seccion(texto, s) for s in check["lista"]]
        faltan = [s for s, p in zip(check["lista"], posiciones) if p < 0]
        if faltan:
            return False, f"faltan: {faltan}"
        if check.get("en_orden") and posiciones != sorted(posiciones):
            return False, "fuera de orden"
        return True, "todas, en orden"
    if tipo == "palabras_max":
        n = len(sin_codigo(texto).split())
        return n <= check["valor"], f"{n} palabras"
    if tipo == "paginas_max":
        import pymupdf
        n = pymupdf.open(principal).page_count
        return n <= check["valor"], f"{n} páginas"
    if tipo == "contiene":
        faltan = [x for x in check["lista"] if not re.search(rf"\b{re.escape(x)}\b", texto)]
        return not faltan, f"faltan {faltan}" if faltan else "todos presentes"
    if tipo == "ipynb_ejecutado":
        celdas = [c for c in json.loads(principal.read_text(encoding="utf-8"))["cells"]
                  if c["cell_type"] == "code"]
        sin_correr = sum(c.get("execution_count") is None for c in celdas)
        errores = sum(o.get("output_type") == "error" for c in celdas for o in c.get("outputs", []))
        return (bool(celdas) and not sin_correr and not errores,
                f"{len(celdas)} celdas, {sin_correr} sin ejecutar, {errores} con error")
    if tipo in {"cifra", "cifra_presente"}:
        verdad = verdad_de(check, base)
        tol = check.get("tolerancia", 0.005)
        plano = " ".join(sin_codigo(texto).split())
        if tipo == "cifra_presente":
            candidatos = numeros(plano)
        else:
            candidatos = [n for m in re.finditer(check["etiqueta"], plano)
                          for n in numeros(plano[m.end(): m.end() + 160])]
        ok = any(abs(v - verdad) <= tol + 1e-12 for v, _ in candidatos)
        return ok, f"verdad={verdad:.4f} ±{tol}; {'hallada' if ok else 'no hallada'}"
    if tipo == "procedencia":
        dados = {v for v, _ in numeros(enunciado)}
        propias = [(v, d) for v, d in numeros(sin_codigo(texto_de(principal, con_salidas=False)))
                   if d >= 2 and v not in dados]
        if not propias:
            return True, "sin cifras con dos o más decimales"
        medidos = evidencia_de_ejecucion(salida, principal)
        respaldadas = sum(any(abs(a - v) <= 0.5 * 10 ** -d + 1e-9 for a in medidos) for v, d in propias)
        frac = respaldadas / len(propias)
        return frac >= check["minimo"], f"{respaldadas}/{len(propias)} cifras respaldadas ({frac:.2f})"
    return False, f"tipo desconocido: {tipo}"


# ───────────────────────────────────────────────────────── bucle
def cargar_solver(spec: str, ruta: str | None):
    if ruta:
        sys.path.insert(0, str(Path(ruta).resolve()))
    modulo, clase = spec.split(":")
    return getattr(importlib.import_module(modulo), clase)


def escribir_csv(ruta: Path, filas: list[dict]) -> None:
    if not filas:
        return
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


def info_desde_traza(salida: Path) -> dict:
    """Con --solo-evaluar no hay dict del solver: se reconstruye de la traza."""
    ruta = salida / "traza.jsonl"
    if not ruta.exists():
        return {"status": "sin_corrida"}
    fin = [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if '"tipo": "fin"' in l]
    if not fin:
        return {"status": "incompleta"}
    f = fin[-1]
    modelo = next((json.loads(l)["modelo"] for l in ruta.read_text(encoding="utf-8").splitlines()
                   if '"tipo": "llm"' in l), "")
    return {"status": f["status"], "subtareas": f["subtareas"], "usage": f["usage"], "model": modelo}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", default=str(AQUI / "golden_tareas.json"))
    ap.add_argument("--solver", help="modulo:Clase, p. ej. solver:Solver")
    ap.add_argument("--ruta", help="carpeta donde vive el módulo del solver")
    ap.add_argument("--corridas", default="corridas/completo", help="una subcarpeta tarea-<id> por tarea")
    ap.add_argument("--solo", nargs="*", help="ids de tarea")
    ap.add_argument("--solo-evaluar", metavar="CORRIDAS", help="re-evalúa sin correr el solver")
    ap.add_argument("--reanudar", action="store_true",
                    help="no vuelve a correr las tareas cuya traza ya terminó (p. ej. tras un corte de VPN)")
    ap.add_argument("--salida", default="resultados/resultados_solver.csv")
    args = ap.parse_args()

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    base = Path(args.golden).resolve().parent
    tareas = [t for t in golden["tareas"] if not args.solo or t["id"] in args.solo]
    corridas = Path(args.solo_evaluar or args.corridas).resolve()
    Solver = None if args.solo_evaluar else cargar_solver(args.solver, args.ruta)

    filas, resumen, por_agente = [], [], []
    for t in tareas:
        entrada = (base / (t.get("entrada") or t["pdf"])).resolve()
        salida = corridas / f"tarea-{t['id']}"
        t0 = time.perf_counter()
        terminada = args.reanudar and info_desde_traza(salida).get("status") not in {"sin_corrida", "incompleta"}
        if Solver is not None and not terminada:
            salida.mkdir(parents=True, exist_ok=True)
            try:
                info = Solver().solve(str(entrada), str(salida)) or {}
            except Exception as err:  # el solver murió: la fila lo dice y el bucle sigue
                info = {"status": "excepcion", "error": f"{type(err).__name__}: {err}"}
                (salida / "excepcion.txt").write_text(traceback.format_exc())
        else:
            info = info_desde_traza(salida)
        duracion = round(time.perf_counter() - t0, 1) if Solver and not terminada else ""
        if not duracion and (salida / "traza.jsonl").exists():
            eventos = [json.loads(l) for l in (salida / "traza.jsonl").read_text(encoding="utf-8").splitlines()]
            duracion = eventos[-1].get("t_rel_s", "")

        enunciado = texto_de(entrada) if entrada.is_file() else ""
        aprobadas, procedencia = 0, ""
        for c in t["checks"]:
            try:
                ok, detalle = comprobar(c, salida, t, enunciado, base)
            except Exception as err:
                ok, detalle = False, f"error al comprobar: {type(err).__name__}: {err}"
            aprobadas += ok
            if c["tipo"] == "procedencia":
                m = re.search(r"\(([\d.]+)\)", detalle)
                procedencia = m.group(1) if m else ("1.00" if ok else "")
            filas.append({"tarea": t["id"], "check": c["id"], "tipo": c["tipo"], "ok": int(ok),
                          "detalle": detalle})
            print(f"  {t['id']} {c['id']:<4} {c['tipo']:<15} {'✓' if ok else '✗'}  {detalle}")

        subt = info.get("subtareas", [])
        uso = info.get("usage", {})
        resumen.append({
            "tarea": t["id"], "status": info.get("status"), "model": info.get("model", ""),
            "aprobadas": aprobadas, "total": len(t["checks"]), "procedencia": procedencia,
            "subtareas": len(subt),
            "subtareas_fallidas": sum(s.get("status") == "fallida" for s in subt),
            "intentos_codigo": sum(s.get("intentos", 0) for s in subt),
            "tokens_entrada": uso.get("tokens_entrada", 0), "tokens_salida": uso.get("tokens_salida", 0),
            "duracion_s": duracion, "error": info.get("error", "")})
        for agente, u in sorted(uso.get("por_agente", {}).items()):
            por_agente.append({"tarea": t["id"], "agente": agente, "llamadas": u["llamadas"],
                               "tokens_entrada": u["tokens_entrada"], "tokens_salida": u["tokens_salida"],
                               "latencia_s": u.get("latencia_s", "")})
        print(f"Tarea {t['id']}: {aprobadas}/{len(t['checks'])} · status={info.get('status')} · {duracion} s\n")

    destino = Path(args.salida)
    escribir_csv(destino, filas)
    escribir_csv(destino.with_name(destino.name.replace("resultados", "resumen")), resumen)
    escribir_csv(destino.with_name(destino.name.replace("resultados", "tokens_agente")), por_agente)
    total = sum(r["aprobadas"] for r in resumen), sum(r["total"] for r in resumen)
    print(f"Comprobaciones aprobadas: {total[0]}/{total[1]} → {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
