# -*- coding: utf-8 -*-
"""
T1 · Parte 1 — Chunking fijo con solapamiento (Ejercicio 1.1)
MMIA 6013 · Semana 2 (Miércoles) · RAG mínimo en ~60 líneas

Ejercicio 1.1:
  · Implementar chunk_fijo(texto, tamano=300, overlap=80) sobre caracteres
    (devuelve una lista de strings).
  · Construir el corpus fragmentado FUERA de la solución del ejercicio
    (listas `chunks` y `origen` recorriendo DOCS, tal como da el profesor)
    e imprimir cada chunk como '[i] (origen) primeros 70 caracteres...'.
  · Reportar si algún chunk corta una idea (p. ej. la de días transferibles
    "hasta un máximo de 5 días") y repetir la impresión con overlap=0 para
    comparar los cortes.

Cifras guardadas en resultados.json: num_chunks, chunks_por_documento,
idea_cortada_con_overlap_80, num_chunks_con_overlap_0 (+ complementarias).
Esta subtarea NO produce figuras PNG.
"""

# ----------------------------------------------------------------------
# S0 · Imports y corpus (código dado en el enunciado)
# ----------------------------------------------------------------------
import os
import json
import numpy as np  # dado en S0; esta parte aún no lo usa

# Corpus: documentación interna de una empresa ficticia
DOCS = {
    "politica_vacaciones.md": """Política de vacaciones de NimbusSoft.
Los empleados a tiempo completo acumulan 1.5 días de vacaciones por mes trabajado,
hasta un máximo de 18 días por año. Las vacaciones deben solicitarse con al menos
15 días de anticipación a través del portal interno. Los días no utilizados pueden
transferirse al año siguiente hasta un máximo de 5 días. Durante el primer año,
los días solo pueden tomarse después de superar el período de prueba de 3
meses.""",
    "politica_remoto.md": """Política de trabajo remoto de NimbusSoft.
El trabajo remoto está permitido hasta 3 días por semana para todos los roles
excepto soporte de infraestructura on-site. Los días remotos se coordinan con el
líder de equipo. Para trabajar desde el exterior del país se requiere aprobación
de Recursos Humanos con 30 días de anticipación y un máximo de 60 días por año.""",
    "gastos.md": """Política de reembolso de gastos de NimbusSoft.
Los gastos de viaje se reembolsan presentando factura dentro de los 30 días
posteriores al gasto. El límite diario de alimentación en viajes es de 45 USD.
Los pasajes aéreos deben comprarse en clase económica salvo vuelos de más de
8 horas, donde se permite económica premium con aprobación del gerente de área.""",
}

# ----------------------------------------------------------------------
# Ejercicio 1.1 — chunking fijo con solapamiento (SOLUCIÓN)
# ----------------------------------------------------------------------
TAMANO = 300   # tamaño de chunk en caracteres (notebook de hoy: 300/80)
OVERLAP = 80   # solapamiento en caracteres


def _chunk_spans(texto, tamano, overlap):
    """Spans [inicio, fin) de cada chunk sobre el texto original.

    Única fuente de verdad del corte: la ventana avanza
    `tamano - overlap` caracteres por paso, de modo que cada chunk
    repite los últimos `overlap` caracteres del chunk anterior.
    """
    if not (0 <= overlap < tamano):
        raise ValueError("Se requiere 0 <= overlap < tamano")
    paso = tamano - overlap
    spans = []
    inicio = 0
    while inicio < len(texto):
        spans.append((inicio, min(inicio + tamano, len(texto))))
        inicio += paso
    return spans


def chunk_fijo(texto, tamano=300, overlap=80):
    """Ejercicio 1.1: chunking fijo sobre caracteres con solapamiento.

    Devuelve una lista de strings: cada chunk tiene a lo sumo `tamano`
    caracteres y comparte `overlap` caracteres con el chunk anterior, de
    modo que una idea que cruza una frontera aparece completa en al menos
    un chunk (a cambio de inflar el índice: C/(C-V) = 300/220 -> +36.4 %).
    """
    return [texto[a:b] for a, b in _chunk_spans(texto, tamano, overlap)]


# ----------------------------------------------------------------------
# Herramientas de análisis: ¿algún chunk corta una idea?
# ----------------------------------------------------------------------
# La "idea" de los días transferibles es la oración que empieza en
# "Los días no utilizados pueden" y termina en la cifra clave
# "hasta un máximo de 5 días." (la frase que cita el enunciado).
IDEA_DOC = "politica_vacaciones.md"
IDEA_INI = "Los días no utilizados pueden"
IDEA_FIN = "hasta un máximo de 5 días."
FRASE_CLAVE = "hasta un máximo de 5 días"


def _offsets_globales(overlap):
    """Índice global del primer chunk de cada documento en `chunks`."""
    offs, acc = {}, 0
    for nombre, texto in DOCS.items():
        offs[nombre] = acc
        acc += len(chunk_fijo(texto, TAMANO, overlap))
    return offs


def _clasificar_span(nombre_doc, ini, fin, overlap):
    """Clasifica cada chunk del documento frente al span [ini, fin):
      'completa' -> el chunk contiene el span entero,
      'corta'    -> lo interseca pero no lo contiene entero,
      'fuera'    -> no lo toca.
    Los índices de chunk son GLOBALES (los de las listas chunks/origen).
    """
    texto = DOCS[nombre_doc]
    base = _offsets_globales(overlap)[nombre_doc]
    filas = []
    for j, (a, b) in enumerate(_chunk_spans(texto, TAMANO, overlap)):
        interseca = (a < fin) and (ini < b)
        completa = (a <= ini) and (fin <= b)
        estado = "completa" if completa else ("corta" if interseca else "fuera")
        filas.append({"chunk": base + j, "span": (a, b), "estado": estado})
    return filas


def _reporte_idea(overlap):
    """Imprime y devuelve el análisis de corte de la idea para un overlap."""
    texto = DOCS[IDEA_DOC]
    ini = texto.find(IDEA_INI)
    fin = texto.find(IDEA_FIN) + len(IDEA_FIN)
    assert ini != -1 and fin > ini, "No se encontró la idea en el documento"
    idea = texto[ini:fin].replace("\n", " ")
    filas = _clasificar_span(IDEA_DOC, ini, fin, overlap)

    print("")
    print("-" * 72)
    print(f"¿Algún chunk corta la idea? (overlap={overlap})")
    print("-" * 72)
    print(f"Idea buscada en '{IDEA_DOC}', caracteres {ini} a {fin - 1}:")
    print(f'  "{idea}"')
    cortan, completan = [], []
    for f in filas:
        i, (a, b), est = f["chunk"], f["span"], f["estado"]
        trozo = texto[a:b]
        if est == "completa":
            completan.append(i)
            print(f"  [{i}] contiene la idea COMPLETA (chars {a}-{b - 1})")
        elif est == "corta":
            cortan.append(i)
            if a <= ini:
                cola = trozo[-40:].replace("\n", " ")
                print(f'  [{i}] CORTA la idea: termina en "...{cola}"')
            else:
                cabeza = trozo[:40].replace("\n", " ")
                print(f'  [{i}] CORTA la idea: empieza en "{cabeza}..."')
        else:
            print(f"  [{i}] no toca la idea")
    return {"ini": int(ini), "fin": int(fin), "idea": idea,
            "cortan": cortan, "completan": completan}


# ----------------------------------------------------------------------
# Corpus fragmentado — FUERA del ejercicio: las Partes 2 y 3 lo consumen
# (código tal como lo da el profesor)
# ----------------------------------------------------------------------
chunks, origen = [], []
for nombre, texto in DOCS.items():
    for c in chunk_fijo(texto):
        chunks.append(c)
        origen.append(nombre)

# Comprobación de consistencia: los spans reproducen el corpus exacto
for nombre, texto in DOCS.items():
    reconstruido = [texto[a:b] for a, b in _chunk_spans(texto, TAMANO, OVERLAP)]
    real = [c for c, o in zip(chunks, origen) if o == nombre]
    assert reconstruido == real, "Inconsistencia entre spans y corpus"

print("=" * 72)
print("CHUNKS DEL CORPUS — chunk_fijo(texto, tamano=300, overlap=80)")
print("=" * 72)
for i, c in enumerate(chunks):
    print(f"[{i}] ({origen[i]}) {c[:70]}...")

# ----------------------------------------------------------------------
# Reporte: ¿algún chunk corta una idea? (overlap=80)
# ----------------------------------------------------------------------
rep80 = _reporte_idea(OVERLAP)
idea_cortada_80 = len(rep80["cortan"]) > 0

# ----------------------------------------------------------------------
# Comparación: repetir la impresión con overlap=0
# ----------------------------------------------------------------------
chunks0, origen0 = [], []
for nombre, texto in DOCS.items():
    for c in chunk_fijo(texto, overlap=0):
        chunks0.append(c)
        origen0.append(nombre)

print("")
print("=" * 72)
print("CHUNKS DEL CORPUS — chunk_fijo(texto, tamano=300, overlap=0)")
print("=" * 72)
for i, c in enumerate(chunks0):
    print(f"[{i}] ({origen0[i]}) {c[:70]}...")

rep0 = _reporte_idea(0)

# ----------------------------------------------------------------------
# La frase literal del enunciado y el corte exacto con overlap=0
# ----------------------------------------------------------------------
texto_vac = DOCS[IDEA_DOC]
f_ini = texto_vac.find(FRASE_CLAVE)
f_fin = f_ini + len(FRASE_CLAVE)
assert f_ini != -1, "No se encontró la frase clave"
filas_f80 = _clasificar_span(IDEA_DOC, f_ini, f_fin, OVERLAP)
filas_f0 = _clasificar_span(IDEA_DOC, f_ini, f_fin, 0)
frase_comp80 = [f["chunk"] for f in filas_f80 if f["estado"] == "completa"]
frase_cort80 = [f["chunk"] for f in filas_f80 if f["estado"] == "corta"]
frase_comp0 = [f["chunk"] for f in filas_f0 if f["estado"] == "completa"]
frase_cort0 = [f["chunk"] for f in filas_f0 if f["estado"] == "corta"]

izq = texto_vac[TAMANO - 35:TAMANO].replace("\n", " ")
der = texto_vac[TAMANO:TAMANO + 35].replace("\n", " ")

print("")
print("-" * 72)
print(f'Frase literal del enunciado: "{FRASE_CLAVE}" (chars {f_ini}-{f_fin - 1})')
print(f"  · overlap=80: COMPLETA en chunk {frase_comp80}; la corta: "
      f"{frase_cort80 if frase_cort80 else 'ninguno'}")
print(f"  · overlap=0 : COMPLETA en chunk {frase_comp0}; la corta: "
      f"{frase_cort0 if frase_cort0 else 'ninguno'}")
print(f"  · con overlap=0 el corte cae dentro de la oración: ...{izq} | {der}...")
print("    la frase sobrevive, pero la oración pierde su sujeto: la IDEA queda mutilada.")

# ----------------------------------------------------------------------
# Conclusión del ejercicio
# ----------------------------------------------------------------------
print("")
print("=" * 72)
print("CONCLUSIÓN (Ejercicio 1.1)")
print("=" * 72)
if rep80["cortan"]:
    print(f"· overlap=80: la idea SÍ es cortada por el chunk {rep80['cortan']}, "
          f"pero aparece COMPLETA en el chunk {rep80['completan']}: el "
          f"solapamiento de {OVERLAP} caracteres la salva.")
else:
    print("· overlap=80: ningún chunk corta la idea.")
if rep0["completan"]:
    print(f"· overlap=0: la idea aparece completa en el chunk {rep0['completan']}.")
else:
    print(f"· overlap=0: la idea queda mutilada: la cortan los chunks "
          f"{rep0['cortan']} y no aparece completa en ninguno (sin solapamiento, "
          "la idea que cruza la frontera se pierde a medias en cada fragmento).")

pequenos = [{"chunk": int(i), "origen": origen[i], "longitud": int(len(c)),
             "texto": c} for i, c in enumerate(chunks) if len(c) < 50]
if pequenos:
    det = "; ".join(f"[{p['chunk']}] ({p['origen']}) {p['longitud']} chars {p['texto']!r}"
                    for p in pequenos)
    print(f"· Artefacto del paso fijo de {TAMANO - OVERLAP} caracteres: chunk(s) "
          f"muy pequeño(s) al final -> {det}")

# ----------------------------------------------------------------------
# Cifras y resultados.json
# ----------------------------------------------------------------------
num_chunks = len(chunks)
num_chunks_0 = len(chunks0)
chunks_por_doc = {n: int(origen.count(n)) for n in DOCS}
chunks_por_doc_0 = {n: int(origen0.count(n)) for n in DOCS}
long_docs = {n: int(len(t)) for n, t in DOCS.items()}
long_chunks = [int(len(c)) for c in chunks]
long_chunks0 = [int(len(c)) for c in chunks0]
caracteres_corpus = int(sum(long_docs.values()))
caracteres_chunks80 = int(sum(long_chunks))
caracteres_chunks0 = int(sum(long_chunks0))
inflacion_teorica = OVERLAP / (TAMANO - OVERLAP)          # V/(C-V), slide 14
inflacion_medida80 = caracteres_chunks80 / caracteres_corpus - 1.0
inflacion_medida0 = caracteres_chunks0 / caracteres_corpus - 1.0

resultados = {
    # ---- cifras esperadas para esta subtarea ----
    "num_chunks": int(num_chunks),
    "chunks_por_documento": chunks_por_doc,
    "idea_cortada_con_overlap_80": bool(idea_cortada_80),
    "num_chunks_con_overlap_0": int(num_chunks_0),
    # ---- parámetros ----
    "parametros": {"tamano_caracteres": int(TAMANO),
                   "overlap_caracteres": int(OVERLAP)},
    # ---- cifras complementarias ----
    "longitud_documentos_caracteres": long_docs,
    "longitud_chunks_overlap80_caracteres": long_chunks,
    "longitud_chunks_overlap0_caracteres": long_chunks0,
    "chunks_por_documento_overlap_0": chunks_por_doc_0,
    "caracteres_corpus_total": caracteres_corpus,
    "caracteres_en_chunks_overlap80": caracteres_chunks80,
    "caracteres_en_chunks_overlap0": caracteres_chunks0,
    "inflacion_teorica_indice_overlap80_frac": float(inflacion_teorica),
    "inflacion_teorica_indice_overlap80_pct": float(inflacion_teorica * 100.0),
    "inflacion_medida_corpus_overlap80_frac": float(inflacion_medida80),
    "inflacion_medida_corpus_overlap0_frac": float(inflacion_medida0),
    "idea_doc": IDEA_DOC,
    "idea_texto": rep80["idea"],
    "idea_span_caracteres": [rep80["ini"], rep80["fin"]],
    "idea_cortada_con_overlap_0": bool(len(rep0["cortan"]) > 0),
    "idea_completa_en_algun_chunk_overlap_80": bool(len(rep80["completan"]) > 0),
    "idea_completa_en_algun_chunk_overlap_0": bool(len(rep0["completan"]) > 0),
    "idea_cortada_por_chunks_overlap_80": [int(i) for i in rep80["cortan"]],
    "idea_completa_en_chunks_overlap_80": [int(i) for i in rep80["completan"]],
    "idea_cortada_por_chunks_overlap_0": [int(i) for i in rep0["cortan"]],
    "idea_completa_en_chunks_overlap_0": [int(i) for i in rep0["completan"]],
    "frase_clave": FRASE_CLAVE,
    "frase_clave_span_caracteres": [int(f_ini), int(f_fin)],
    "frase_clave_completa_en_overlap_80": [int(i) for i in frase_comp80],
    "frase_clave_cortada_en_overlap_80": [int(i) for i in frase_cort80],
    "frase_clave_completa_en_overlap_0": [int(i) for i in frase_comp0],
    "frase_clave_cortada_en_overlap_0": [int(i) for i in frase_cort0],
    "chunks_menos_de_50_caracteres": pequenos,
    "preview_70chars_chunks_overlap80": [c[:70] for c in chunks],
    "preview_70chars_chunks_overlap0": [c[:70] for c in chunks0],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("")
print("=" * 72)
print("CIFRAS PRINCIPALES (guardadas en resultados.json)")
print("=" * 72)
print(f"num_chunks (overlap=80): {num_chunks}")
print(f"chunks_por_documento: {chunks_por_doc}")
print(f"idea_cortada_con_overlap_80: {idea_cortada_80}")
print(f"num_chunks_con_overlap_0: {num_chunks_0}")
print(f"  · longitudes de documentos (caracteres): {long_docs}")
print(f"  · caracteres: corpus={caracteres_corpus}, en chunks overlap=80: "
      f"{caracteres_chunks80}, en chunks overlap=0: {caracteres_chunks0}")
print(f"  · inflación teórica del índice C/(C-V): +{inflacion_teorica * 100:.4f} % "
      "(slide 14: +36.4 %)")
print(f"  · inflación medida en este corpus: overlap=80: "
      f"+{inflacion_medida80 * 100:.4f} % | overlap=0: "
      f"+{inflacion_medida0 * 100:.4f} %")
print("OK: resultados.json escrito en la carpeta actual.")
