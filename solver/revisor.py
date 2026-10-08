"""Agente REVISOR — aprueba un script o lo devuelve con la corrección.

REGLA 4 — Resultados en un archivo fijo y revisión con código primero. Las revisiones con
código (las de la Parte 0.c y algunas más) corren SIEMPRE antes; el LLM solo se consulta si
todas pasan. Un código no se deja convencer: si el script evalúa sobre los datos con que
entrenó, o si una exactitud vale 1,000, el LLM ni lo ve.

Revisiones con código:
  - terminó a tiempo y con código de salida 0;
  - escribió `resultados.json`, es JSON válido y tiene al menos una cifra;
  - ninguna cifra es NaN ni infinito;
  - si la subtarea pide figura, hay un PNG nuevo y no está vacío;
  - fuga estática (0.c): se predice sobre la misma variable con que se ajustó;
  - plausibilidad (0.c): exactitud/F1 ≥ 0,999 en un problema con ruido.
"""
from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .llm import ClienteLLM
from .programador import ARCHIVO_RESULTADOS, resumir_json
from .sandbox import Ejecucion


@dataclass
class Veredicto:
    aprobado: bool
    por: str                           # «codigo», «llm», «sandbox», «repeticion», «red», «sin_revisor»
    problemas: list[str] = field(default_factory=list)
    correccion: str = ""


# ──────────────────────────────────────────── revisiones con código (las de la Parte 0.c)
def evalua_sobre_entrenamiento(codigo: str) -> list[str]:
    """Variables que aparecen en un ajuste SUPERVISADO `.fit(X, y)` y otra vez en
    `.predict(X)`/`.score(X, …)`.

    Corrección tras la Parte 2.c: la primera versión también contaba `.fit(X)` sin etiquetas,
    y rechazaba como «fuga» el `lda.fit(X)` + `lda.score(X)` que la Tarea C pide (la
    perplejidad sobre el mismo corpus). Un modelo no supervisado no tiene etiquetas que filtrar.
    """
    ajustadas, evaluadas = set(), []
    try:
        arbol = ast.parse(codigo)
    except SyntaxError:
        return []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Attribute) and nodo.args:
            primero = nodo.args[0]
            if not isinstance(primero, ast.Name):
                continue
            supervisado = len(nodo.args) >= 2 or any(k.arg == "y" for k in nodo.keywords)
            if nodo.func.attr in {"fit", "fit_transform"} and supervisado:
                ajustadas.add(primero.id)
            elif nodo.func.attr in {"predict", "predict_proba", "score"}:
                evaluadas.append(primero.id)
    return sorted({v for v in evaluadas if v in ajustadas})


def _hojas(datos, prefijo: str = ""):
    if isinstance(datos, dict):
        for k, v in datos.items():
            yield from _hojas(v, f"{prefijo}{k}.")
    elif isinstance(datos, list):
        for i, v in enumerate(datos):
            yield from _hojas(v, f"{prefijo}{i}.")
    else:
        yield prefijo.rstrip("."), datos


def implausibles(resultados: dict, techo: float = 0.999) -> list[str]:
    """Exactitud o F1 de un CLASIFICADOR ≥ 0,999 en un problema con ruido (Parte 0.c).
    Corrección tras la tarea R1: la primera versión incluía «recall» y rechazaba un
    recall@10 = 1,0 de un índice ANN frente al kNN exacto, que es legítimo."""
    return [f"{k}={v}" for k, v in _hojas(resultados)
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and k.split(".")[-1].lower().startswith(("acc", "f1", "exact"))
            and "train" not in k.lower() and "entren" not in k.lower() and v >= techo]


def revisiones_con_codigo(subtarea: dict, codigo: str, ejecucion: Ejecucion,
                          carpeta: Path) -> list[str]:
    problemas: list[str] = []
    if ejecucion.timeout:
        return ["el script superó el tiempo máximo del sandbox y se terminó"]
    if ejecucion.codigo_salida != 0:
        cola = "\n".join(ejecucion.stderr.strip().splitlines()[-12:])
        return [f"terminó con código de salida {ejecucion.codigo_salida}:\n{cola}"]

    ruta = carpeta / ARCHIVO_RESULTADOS
    resultados = None
    if not ruta.exists():
        problemas.append(f"no escribió {ARCHIVO_RESULTADOS}")
    else:
        try:
            resultados = json.loads(ruta.read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            problemas.append(f"{ARCHIVO_RESULTADOS} no es JSON válido ({err}); ¿escribiste NaN?")
    if resultados is not None:
        numeros = [(k, v) for k, v in _hojas(resultados)
                   if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not numeros:
            problemas.append(f"{ARCHIVO_RESULTADOS} no contiene ninguna cifra")
        malos = [k for k, v in numeros if isinstance(v, float) and (math.isnan(v) or math.isinf(v))]
        if malos:
            problemas.append(f"cifras NaN o infinitas en: {malos[:8]}")
        raras = implausibles(resultados)
        if raras:
            problemas.append(f"métricas implausibles (≥ 0,999) en un problema con ruido: {raras[:5]}; "
                             "¿se evaluó sobre los datos de entrenamiento?")
    if subtarea.get("figura"):
        pngs = [p for p in ejecucion.archivos_creados if p.endswith(".png")]
        if not pngs:
            problemas.append("la subtarea pide una figura y no se creó ningún PNG")
        elif all((carpeta / p).stat().st_size < 2000 for p in pngs):
            problemas.append(f"las figuras {pngs} están vacías o casi vacías")
    fugas = evalua_sobre_entrenamiento(codigo)
    if fugas:
        problemas.append(f"fuga de datos: se predice/evalúa sobre {fugas}, la misma variable con que se ajustó")
    return problemas


# ──────────────────────────────────────────── el revisor
PROMPT = """Eres el REVISOR de un sistema multiagente. Un script ya pasó las revisiones
automáticas (terminó bien, guardó resultados.json sin NaN, sin fugas evidentes). Decide si
cumple la subtarea según el enunciado citado: parámetros (semillas, particiones,
proporciones), métricas pedidas, cálculos correctos, figura si se pide.
Rechaza solo por errores concretos y verificables; no por estilo. Si rechazas, di
exactamente qué cambiar.
El sandbox PROHÍBE la red (incluso localhost por HTTP) y los procesos: si el enunciado pide
conectarse a un servidor, descargar algo o llamar a una API, la versión local o embebida que el
propio enunciado ofrece como alternativa es la correcta. No rechaces un script por omitir un
paso que necesita la red.
Responde solo JSON: {"aprobado": true|false, "problemas": ["..."], "correccion": "..."}"""


class Revisor:
    def __init__(self, llm: ClienteLLM):
        self.llm = llm

    def revisar(self, subtarea: dict, contexto: str, codigo: str, ejecucion: Ejecucion,
                carpeta: Path) -> Veredicto:
        problemas = revisiones_con_codigo(subtarea, codigo, ejecucion, carpeta)
        if problemas:                             # REGLA 4: el LLM no se consulta
            return Veredicto(False, "codigo", problemas,
                             "Corrige estos problemas detectados por las revisiones automáticas:\n- "
                             + "\n- ".join(problemas))
        resultados = json.loads((carpeta / ARCHIVO_RESULTADOS).read_text(encoding="utf-8"))
        usuario = (f"SUBTAREA {subtarea['id']} — {subtarea.get('titulo', '')}\n"
                   f"Objetivo: {subtarea.get('objetivo', '')}\n\n"
                   f"ENUNCIADO (fragmentos citados)\n{contexto[:9000]}\n\n"
                   f"CÓDIGO\n```python\n{codigo}\n```\n\n"
                   f"SALIDA (stdout, final)\n{ejecucion.stdout[-3000:]}\n\n"
                   f"resultados.json\n{resumir_json(resultados, 4000)}\n\n"
                   f"Archivos creados: {ejecucion.archivos_creados}")
        datos = self.llm.pedir_json("revisor", [{"role": "system", "content": PROMPT},
                                                {"role": "user", "content": usuario}],
                                    max_tokens=16000, salida_estimada=2500)
        aprobado = bool(datos.get("aprobado"))
        return Veredicto(aprobado, "llm", [str(p) for p in datos.get("problemas", [])],
                         "" if aprobado else str(datos.get("correccion", "")))
