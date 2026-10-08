```python
# -*- coding: utf-8 -*-
"""
T2 · Parte 2 — Índice (embeddings + kNN), techo del modelo y recuperar() (Ejercicio 2.1)

Consume de la Parte 1 (entradas/T1.json) los parámetros del chunking y las métricas de
verificación. Si T1.json guardara el TEXTO de los chunks, se carga directamente; si no,
se reconstruye de forma determinista replicando EXACTAMENTE la chunk_fijo de T1:
bucle `while inicio < len(texto)` con avance paso = tamano - overlap y SIN corte
especial del último ciclo (con tamano=300, overlap=80 esto da 7 chunks / 1411
caracteres, con el chunk final corto, coincidiendo con T1).

Antes de construir el índice se EXIGE que los chunks coincidan con T1 (num_chunks,
caracteres_en_chunks_overlap80 y, si están, las longitudes por chunk y los chunks por
documento); si la verificación falla, el script ABORTA con error en lugar de continuar
