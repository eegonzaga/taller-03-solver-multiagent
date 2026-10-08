```python
# -*- coding: utf-8 -*-
"""
T2 · Parte 2 — Índice (embeddings + kNN) y función recuperar
============================================================
- Carga los chunks producidos por T1 (entradas/T1.json; configuración principal:
  detalle_overlap80; respaldo: detalle_overlap0).
- Ejecuta el código dado del índice: SentenceTransformer
  'paraphrase-multilingual-MiniLM-L12-v2' con embeddings normalizados, o
  fallback TF-IDF si no hay sentence-transformers. Imprime MOTOR y V.shape.
- Mide el techo del modelo: cuenta los tokens del fragmento más largo con el
  tokenizador del mismo modelo, los compara con max_seq_length e indica si cabe
  entero o cuántos tokens se pierden en silencio. Con el fallback TF-IDF
  registra que no hay truncado.
- Ejercicio 2.1: recuperar(pregunta, k=3) -> [(score, idx_chunk), ...] ordenado
  de mayor a menor score, con esa firma exacta (la Parte 3 y el ejercicio 3.1
  la llaman así).

No produce figura PNG (no se pide). Guarda todas las cifras en resultados.json.
Sin red ni descargas: el modelo denso se intenta cargar solo desde caché local
(local_files_only=True); si no está disponible, se usa el fallback TF-IDF.
"""

import json
import os
import inspect

import numpy as np

# ----------------------------------------------------------------------------
# 0) Chunks de T1 (entradas/T1.json)
# ----------------------------------------------------------------------------
RUTA_T1 = os.path.join("entradas", "T1.json")
if not os.path.exists(RUTA_T1):
    raise FileNotFoundError("No se encontró entradas/T1.json (salida de la subtarea T1).")

with open(RUTA_T1, "r", encoding="utf-8") as f:
    t1 = json.load(f)

_CLAVES_TEXTO = ("texto", "text", "chunk", "fragmento", "contenido", "content",
                 "texto_chunk", "chunk_texto", "pasaje")


def _texto_de_item(it):
    """Extrae el texto de un chunk, sea str o dict con una clave de texto."""
    if isinstance(it, str):
        return it
    if isinstance(it, dict):
        for clave in _CLAVES_TEXTO:
            v = it.get(clave)
            if isinstance(v, str):
                return v
        vals = [v for v in it.values() if isinstance(v, str)]
        if vals:
            return " ".join(vals)
        return json.dumps(it, ensure_ascii=False)
    return str(it)


def extraer_chunks(detalle):
    """Saca la lista de textos de chunk de 'detalle_overlapXX' (lista o dict)."""
    if isinstance(detalle, list) and detalle:
        return [_texto_de_item(it) for it in detalle]
    if isinstance(detalle, dict):
        for clave in ("chunks", "fragmentos", "lista_chunks", "textos", "texts",
                      "chunks_texto", "lista_fragmentos"):
            v = detalle.get(clave)
            if isinstance(v, list) and v:
                return [_texto_de_item(it) for it in v]
        if detalle and all(str(k).isdigit() for k in detalle.keys()):
            return [_texto_de_item(detalle[k])
                    for k in sorted(detalle.keys(), key=lambda x: int(x))]
    return []


def corpus_a_textos(corpus):
    """Textos planos del corpus de T1 (dict o lista, con str o dict)."""
    if isinstance(corpus,
