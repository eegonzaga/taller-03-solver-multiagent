#!/usr/bin/env python3
"""Parte 0.c — Terminar sin error no es acertar. Sin modelo, sin red.

    python parte0/c_exit_cero.py

Un «programador» de prueba entrega, para la Parte 2 de la Tarea A, un árbol de decisión que
se evalúa **con los mismos datos con que se entrenó**. El script corre en el sandbox del
solver (`solver.sandbox.ejecutar`, sin entorno y con tiempo máximo), termina con código 0 y
guarda su resultado: exactitud 1,000.

1) Un evaluador ingenuo (¿código 0? ¿existe resultados.json?) lo aprueba.
2) Las revisiones con código del revisor del solver (`solver.revisor`) lo rechazan: la
   estática (se predice sobre la misma variable con que se ajustó) y la de plausibilidad
   (una exactitud ≥ 0,999 en un problema con ruido). Son código: no se dejan convencer.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from solver.revisor import evalua_sobre_entrenamiento, implausibles, revisiones_con_codigo  # noqa: E402
from solver.sandbox import ejecutar, revisar_codigo  # noqa: E402

# Corre, no falla y miente.
CODIGO = '''import json
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = load_digits(return_X_y=True)
modelo = DecisionTreeClassifier(random_state=7).fit(X, y)
acc = accuracy_score(y, modelo.predict(X))
print(f"Exactitud: {acc:.3f}")
json.dump({"modelo": "arbol", "accuracy": acc}, open("resultados.json", "w"))
'''


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="parte0c-") as tmp:
        carpeta = Path(tmp)
        (carpeta / "script.py").write_text(CODIGO)
        print("Revisión estática del sandbox (REGLA 3):",
              "permitido" if revisar_codigo(CODIGO).permitido else "bloqueado")
        ej = ejecutar(carpeta / "script.py", timeout_s=60, cache_mpl=carpeta / ".mpl")
        print("stdout del experimento:", ej.stdout.strip())

        ok = ej.codigo_salida == 0 and (carpeta / "resultados.json").exists()
        print(f"\n1) Evaluador ingenuo: {'APROBADO' if ok else 'RECHAZADO'}  "
              f"(código de salida={ej.codigo_salida}, resultados.json={'sí' if ok else 'no'})")

        resultados = json.loads((carpeta / "resultados.json").read_text())
        print("\n2) Revisiones con código del revisor (REGLA 4):")
        print(f"   estática  — se predice sobre lo mismo que se ajustó: {evalua_sobre_entrenamiento(CODIGO) or 'no'}")
        print(f"   plausible — métricas ≥ 0,999 en un problema con ruido: {implausibles(resultados) or 'no'}")
        problemas = revisiones_con_codigo({"id": "T2", "figura": False}, CODIGO, ej, carpeta)
        print(f"   → {'RECHAZADO' if problemas else 'APROBADO'} sin consultar al LLM. "
              "El programador recibe esto como corrección:")
        for p in problemas:
            print(f"     - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
