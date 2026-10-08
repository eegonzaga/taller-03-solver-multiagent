"""REGLA 5 — Origen de cada cifra.

Antes de entregar, el código comprueba que cada cifra del entregable salió de una ejecución
(resultados.json, stdout, salidas del notebook) o está en el enunciado. Las que no, vuelven
al redactor con la lista.

Se leen las cifras igual que `evaluar_solver.py` (misma expresión regular; un porcentaje
cuenta también como fracción): así el solver no se aprueba con un criterio más blando que el
del evaluador. Una cifra con `d` decimales está respaldada si algún valor medido redondea a
ella (|medido − cifra| ≤ 0,5·10^-d).

Excepción declarada: los enteros del 0 al 10 no se exigen. Son numeración y estructura
(«Parte 2», «Figura 1», «las 3 consultas»); exigirlos haría que casi todo fallara sin
señalar ningún dato inventado.
"""
from __future__ import annotations

import re

NUM = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)(\s?%)?(?![\w])")


def numeros(texto: str) -> list[tuple[float, int, str]]:
    """(valor, decimales, texto original) de cada cifra; un porcentaje da dos entradas."""
    salida = []
    for m in NUM.finditer(texto):
        crudo, pct = m.group(1).replace(",", "."), m.group(2)
        try:
            v = float(crudo)
        except ValueError:
            continue
        dec = len(crudo.split(".")[1]) if "." in crudo else 0
        if pct:
            salida.append((v / 100, dec + 2, m.group(0)))
        salida.append((v, dec, m.group(0)))
    return salida


def sin_codigo(texto: str) -> str:
    return re.sub(r"```.*?```", "", texto, flags=re.S)


def cifras_sin_origen(entregable: str, textos_ejecucion: list[str], enunciado: str) -> list[str]:
    dados = {v for v, _, _ in numeros(enunciado)}
    medidos = sorted({v for t in textos_ejecucion for v, _, _ in numeros(t)})
    huerfanas = []
    for v, d, original in numeros(sin_codigo(entregable)):
        if d == 0 and v <= 10:
            continue
        if v in dados:
            continue
        tolerancia = 0.5 * 10 ** -d + 1e-9
        if not any(abs(a - v) <= tolerancia for a in medidos):
            huerfanas.append(original.strip())
    return list(dict.fromkeys(huerfanas))
