"""Agente PROGRAMADOR — escribe un script por subtarea de cálculo.

El prompt le dice lo que el sandbox va a exigir (REGLA 3) y dónde dejar las cifras
(REGLA 4: `resultados.json`, nombre fijo). Si el revisor rechazó el intento anterior, recibe
su código y la corrección.
"""
from __future__ import annotations

import json

from .llm import ClienteLLM, extraer_codigo

ARCHIVO_RESULTADOS = "resultados.json"          # REGLA 4: nombre fijo

PROMPT = f"""Eres el PROGRAMADOR de un sistema multiagente que resuelve tareas de maestría.
Escribe UN script de Python 3 completo para la subtarea indicada. Reglas obligatorias:

1. Guarda TODAS las cifras que calcules en «{ARCHIVO_RESULTADOS}» (en la carpeta actual),
   con json.dump, como un diccionario con nombres claros. Valores con toda su precisión
   (no redondees). Convierte tipos de NumPy a float/int/list antes de guardar.
2. Imprime también las cifras principales con print().
3. Figuras: matplotlib, guárdalas como PNG en la carpeta actual con plt.savefig("nombre.png").
4. Solo rutas relativas a la carpeta actual. Datos de entrada: los archivos que se listan;
   resultados de subtareas anteriores: entradas/<id>.json (si se listan).
5. PROHIBIDO: red o descargas, subprocess/multiprocessing, os.system, borrar o renombrar
   archivos, eval/exec, os.environ, rutas absolutas o con «..». Un script así no se ejecuta.
6. Respeta exactamente los parámetros del enunciado (semillas, particiones, proporciones,
   hiperparámetros). Nunca evalúes sobre los datos de entrenamiento; cualquier escalador o
   transformación se ajusta solo con el entrenamiento. No toques el conjunto de prueba
   para ajustar nada.
7. Bibliotecas disponibles: numpy, pandas, scipy, scikit-learn, matplotlib.
Responde solo con el código en un bloque ```python."""


class Programador:
    def __init__(self, llm: ClienteLLM):
        self.llm = llm

    def programar(self, subtarea: dict, contexto: str, archivos_datos: list[str],
                  entradas: dict[str, list[str]], anterior: dict | None = None) -> str:
        partes = [
            f"SUBTAREA {subtarea['id']} — {subtarea.get('titulo', '')}",
            f"Objetivo: {subtarea.get('objetivo', '')}",
            f"Cifras esperadas: {', '.join(subtarea.get('resultados', [])) or '(las que pida el enunciado)'}",
            f"Debe producir una figura PNG: {'sí' if subtarea.get('figura') else 'no'}",
            "Archivos de datos disponibles: " + (", ".join(archivos_datos) or "ninguno (los datos están en el enunciado o en scikit-learn)"),
            "Resultados de subtareas anteriores: " + (
                "; ".join(f"entradas/{d}.json con claves {claves[:15]}" for d, claves in entradas.items())
                or "ninguno"),
            f"\nCONTEXTO (fragmentos citados del enunciado y de las notas del curso)\n\n{contexto}",
        ]
        if anterior:
            partes += [f"\nTU INTENTO ANTERIOR FUE RECHAZADO.\nCódigo anterior:\n```python\n{anterior['codigo']}\n```",
                       f"Problemas y corrección pedida:\n{anterior['correccion']}",
                       "Escribe el script completo corregido."]
        respuesta = self.llm.pedir("programador", [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": "\n".join(partes)}], max_tokens=24000, salida_estimada=5000)
        return extraer_codigo(respuesta)


def claves_de(resultados: dict) -> list[str]:
    return list(resultados)[:30] if isinstance(resultados, dict) else []


def resumir_json(datos, max_car: int = 5000) -> str:
    """Un JSON de resultados recortado para un prompt: listas largas se acortan."""
    def recortar(x):
        if isinstance(x, list) and len(x) > 25:
            return [recortar(v) for v in x[:25]] + [f"... ({len(x)} elementos)"]
        if isinstance(x, dict):
            return {k: recortar(v) for k, v in x.items()}
        return x
    texto = json.dumps(recortar(datos), ensure_ascii=False)
    return texto if len(texto) <= max_car else texto[:max_car] + " …"
