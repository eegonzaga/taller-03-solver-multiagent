# -*- coding: utf-8 -*-
# =============================================================================
# T1 — Configuración y Parte 1 — Chunking fijo con solapamiento
# MMIA 6013 · Semana 2 · Miércoles — "RAG mínimo en ~60 líneas"
# Pipeline completo: chunking → índice → retrieval → prompt con contexto → LLM.
# Esta subtarea monta la base (imports + corpus DOCS) y resuelve el Ejercicio
# 1.1: chunk_fijo(texto, tamano=300, overlap=80) sobre caracteres, construcción
# de chunks/origen FUERA del ejercicio, impresión de chunks con su documento de
# origen y comparación de los cortes entre overlap=80 y overlap=0.
# =============================================================================

# -----------------------------------------------------------------------------
# Configuración base — código dado por el profesor
# -----------------------------------------------------------------------------
import os
import json
import numpy as np

# -----------------------------------------------------------------------------
# Corpus: documentación interna de una empresa ficticia — código dado
# -----------------------------------------------------------------------------
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

TAMANO, OVERLAP = 300, 80  # parámetros del enunciado (valores por defecto)

# -----------------------------------------------------------------------------
# Ejercicio 1.1 — chunking fijo con solapamiento (SOLUCIÓN)
# -----------------------------------------------------------------------------
def chunk_fijo(texto, tamano=300, overlap=80):
    """Fragmenta `texto` en ventanas deslizantes de `tamano` caracteres con
    solapamiento `overlap` (todo medido en caracteres). Devuelve una lista
    de strings.

    - overlap=0 : los fragmentos particionan el texto (sin repetir caracteres).
    - overlap>0 : cada frontera repite los últimos `overlap` caracteres del
      fragmento anterior, de modo que una idea que cruce el corte aparece
      completa en al menos un fragmento (a costa de inflar el índice en
      C/(C-V): con 300/80 -> 300/220 = +36.36%).
    """
    if tamano <= 0:
        raise ValueError("tamano debe ser un entero positivo")
    if overlap < 0 or overlap >= tamano:
        raise ValueError("overlap debe cumplir 0 <= overlap < tamano")
    chunks = []
    inicio = 0
    n = len(texto)
    while inicio < n:
        fin = min(inicio + tamano, n)
        chunks.append(texto[inicio:fin])
        if fin >= n:
            break
        inicio = fin - overlap
    return chunks


# Helper interno de análisis (NO sustituye a chunk_fijo: solo añade las
# posiciones [inicio, fin) de cada chunk dentro de su documento para poder
# decir con exactitud qué idea queda cortada y en qué chunk).
def _chunk_fijo_con_spans(texto, tamano=300, overlap=80):
    spans = []
    inicio = 0
    n = len(texto)
    while inicio < n:
        fin = min(inicio + tamano, n)
        spans.append((texto[inicio:fin], int(inicio), int(fin)))
        if fin >= n:
            break
        inicio = fin - overlap
    return spans


# -----------------------------------------------------------------------------
# El corpus fragmentado. Vive FUERA del ejercicio porque las Partes 2 y 3 lo
# consumen: si estuviera dentro de la solución del 1.1, la versión de estudiante
# no lo tendría. (Código dado, con chunk_fijo ya implementado.)
# -----------------------------------------------------------------------------
chunks, origen = [], []
for nombre, texto in DOCS.items():
    for c in chunk_fijo(texto):  # tamano=300, overlap=80 (por defecto)
        chunks.append(c)
        origen.append(nombre)

for i, c in enumerate(chunks):
    print(f"[{i}] ({origen[i]}) {c[:70]}...")

# --- Repetir la fragmentación con overlap=0 para comparar los cortes --------
chunks0, origen0 = [], []
for nombre, texto in DOCS.items():
    for c in chunk_fijo(texto, tamano=300, overlap=0):
        chunks0.append(c)
        origen0.append(nombre)

print("\n--- Misma fragmentación con overlap=0 (comparación de cortes) ---")
for i, c in enumerate(chunks0):
    print(f"[{i}] ({origen0[i]}) {c[:70]}...")

# -----------------------------------------------------------------------------
# Análisis de los cortes: ¿algún chunk corta una idea?
# Idea de referencia: la de los días transferibles que señala el enunciado
# (la oración completa, tal como queda en el corpus con su salto de línea).
# -----------------------------------------------------------------------------
DOC_IDEA = "politica_vacaciones.md"
IDEA_TRANSFERENCIA = ("Los días no utilizados pueden\n"
                      "transferirse al año siguiente hasta un máximo de 5 días.")
FRASE_CLAVE = "hasta un máximo de 5 días"


def _offsets_globales(overlap):
    """Índice global en el que empiezan los chunks de cada documento."""
    offset, acum = {}, 0
    for nombre, texto in DOCS.items():
        offset[nombre] = acum
        acum += len(chunk_fijo(texto, TAMANO, overlap))
    return offset


def analizar_idea(doc, idea, tamano, overlap, offsets):
    """Clasifica cada chunk del documento respecto a `idea`:
    'completa' si el chunk la contiene entera; 'cortan' si solo la solapa en
    parte (la mutila). Usa las posiciones reales de inicio/fin de cada chunk."""
    texto = DOCS[doc]
    pos_ini = texto.find(idea)
    if pos_ini < 0:
        raise ValueError("La idea no se encontró en el documento")
    pos_fin = pos_ini + len(idea)
    cortan, completas = [], []
    for j, (_c, ini, fin) in enumerate(_chunk_fijo_con_spans(texto, tamano, overlap)):
        g = offsets[doc] + j
        if ini <= pos_ini and pos_fin <= fin:
            completas.append(int(g))
        elif ini < pos_fin and fin > pos_ini:
            cortan.append(int(g))
    return {
        "idea": idea,
        "documento": doc,
        "posicion_en_documento": [int(pos_ini), int(pos_fin)],
        "chunks_que_la_cortan": cortan,
        "chunks_donde_aparece_completa": completas,
        "cortada_en_algun_chunk": bool(cortan),
        "aparece_completa_en_algun_chunk": bool(completas),
    }


offset80 = _offsets_globales(OVERLAP)
offset0 = _offsets_globales(0)

idea80 = analizar_idea(DOC_IDEA, IDEA_TRANSFERENCIA, TAMANO, OVERLAP, offset80)
idea0 = analizar_idea(DOC_IDEA, IDEA_TRANSFERENCIA, TAMANO, 0, offset0)
frase80 = analizar_idea(DOC_IDEA, FRASE_CLAVE, TAMANO, OVERLAP, offset80)
frase0 = analizar_idea(DOC_IDEA, FRASE_CLAVE, TAMANO, 0, offset0)

# --- Conteos -----------------------------------------------------------------
num_chunks = int(len(chunks))
num_chunks0 = int(len(chunks0))
chunks_por_documento = {n: int(sum(1 for o in origen if o == n)) for n in DOCS}
chunks_por_documento0 = {n: int(sum(1 for o in origen0 if o == n)) for n in DOCS}

# --- Inflación del índice (costo del solapamiento) ---------------------------
chars_docs = int(sum(len(t) for t in DOCS.values()))
chars_chunks80 = int(sum(len(c) for c in chunks))
chars_chunks0 = int(sum(len(c) for c in chunks0))
inflacion_teorica_80 = float(TAMANO / (TAMANO - OVERLAP) - 1)  # C/(C-V) - 1
inflacion_teorica_0 = 0.0
inflacion_real_80 = float(chars_chunks80 / chars_docs - 1)
inflacion_real_0 = float(chars_chunks0 / chars_docs - 1)

# --- Detalle de los cortes (borde de cada chunk) ------------------------------
def detalle_cortes(overlap, offsets):
    filas = []
    for nombre, texto in DOCS.items():
        for j, (c, ini, fin) in enumerate(_chunk_fijo_con_spans(texto, TAMANO, overlap)):
            filas.append({
                "chunk": int(offsets[nombre] + j),
                "documento": nombre,
                "inicio_caracter": int(ini),
                "fin_caracter": int(fin),
                "empieza_con": c[:45],
                "termina_con": c[-45:],
            })
    return filas


cortes80 = detalle_cortes(OVERLAP, offset80)
cortes0 = detalle_cortes(0, offset0)

reconstruccion0 = {n: bool("".join(chunk_fijo(t, TAMANO, 0)) == t)
                   for n, t in DOCS.items()}
reconstruccion80 = {n: bool("".join(chunk_fijo(t, TAMANO, OVERLAP)) == t)
                    for n, t in DOCS.items()}

# --- Comparación overlap=80 vs overlap=0 --------------------------------------
def _fmt(xs):
    return "[" + ", ".join(str(x) for x in xs) + "]"


pos_frase = int(frase80["posicion_en_documento"][0])
comparacion_overlap_80_vs_0 = {
    "num_chunks_overlap80": num_chunks,
    "num_chunks_overlap0": num_chunks0,
    "idea_aparece_completa_overlap80": idea80["aparece_completa_en_algun_chunk"],
    "idea_aparece_completa_overlap0": idea0["aparece_completa_en_algun_chunk"],
    "chunks_que_cortan_la_idea_overlap80": idea80["chunks_que_la_cortan"],
    "chunks_que_cortan_la_idea_overlap0": idea0["chunks_que_la_cortan"],
    "inflacion_teorica_indice_overlap80": inflacion_teorica_80,
    "inflacion_teorica_indice_overlap0": inflacion_teorica_0,
    "inflacion_real_corpus_overlap80": inflacion_real_80,
    "inflacion_real_corpus_overlap0": inflacion_real_0,
    "lectura": (
        f"Con overlap=80 la idea de los días transferibles queda cortada en el/los "
        f"chunk(s) {_fmt(idea80['chunks_que_la_cortan'])} pero aparece COMPLETA en el/los "
        f"chunk(s) {_fmt(idea80['chunks_donde_aparece_completa'])}: el solapamiento la salva "
        f"(costo: inflación teórica del índice C/(C-V) = {TAMANO}/{TAMANO - OVERLAP} = "
        f"+{100 * inflacion_teorica_80:.2f}%). Con overlap=0 la misma idea queda cortada en "
        f"{_fmt(idea0['chunks_que_la_cortan'])} y NO aparece completa en ninguno: sin "
        f"solapamiento, una idea que cruza la frontera queda mutilada en los dos fragmentos. "
        f"La subfrase '{FRASE_CLAVE}' empieza en el carácter {pos_frase} del documento, "
        f"después del primer corte (carácter {TAMANO}), así que cae entera en el chunk 1 con "
        f"ambas configuraciones {_fmt(frase80['chunks_donde_aparece_completa'])} vs "
        f"{_fmt(frase0['chunks_donde_aparece_completa'])}; la oración completa de la idea, "
        f"en cambio, solo sobrevive con overlap=80."
    ),
}

# -----------------------------------------------------------------------------
# Cifras principales por pantalla
# -----------------------------------------------------------------------------
print("\n=== CIFRAS PRINCIPALES — T1 · Parte 1 (chunking) ===")
print(f"num_chunks (tamano=300, overlap=80): {num_chunks}")
print(f"chunks_por_documento: {chunks_por_documento}")
print(f"num_chunks_overlap0: {num_chunks0}")
print(f"chunks_por_documento_overlap0: {chunks_por_documento0}")

idea_txt = IDEA_TRANSFERENCIA.replace("\n", " ")
print(f"\nIdea analizada: \"{idea_txt}\"")
print(f"  overlap=80 -> cortada en chunks {idea80['chunks_que_la_cortan']} | "
      f"completa en chunks {idea80['chunks_donde_aparece_completa']}")
print(f"  overlap=0  -> cortada en chunks {idea0['chunks_que_la_cortan']} | "
      f"completa en chunks {idea0['chunks_donde_aparece_completa']}")
print(f"  subfrase '{FRASE_CLAVE}': completa en {frase80['chunks_donde_aparece_completa']} "
      f"(overlap=80) y en {frase0['chunks_donde_aparece_completa']} (overlap=0)")

g0 = offset80[DOC_IDEA]
print("\nObservación del corte en 'politica_vacaciones.md':")
print(f"  overlap=80 · chunk[{g0}] termina con: {chunks[g0][-45:]!r}")
print(f"  overlap=80 · chunk[{g0 + 1}] empieza con: {chunks[g0 + 1][:45]!r}")
print(f"  overlap=0  · chunk[{g0}] termina con: {chunks0[g0][-45:]!r}")
print(f"  overlap=0  · chunk[{g0 + 1}] empieza con: {chunks0[g0 + 1][:45]!r}")

print(f"\nInflación teórica del índice con 300/80: C/(C-V)-1 = "
      f"{inflacion_teorica_80:.6f} (+{100 * inflacion_teorica_80:.2f}%)")
print(f"Inflación real en este corpus: {inflacion_real_80:.6f} (overlap=80) vs "
      f"{inflacion_real_0:.6f} (overlap=0)")
print(f"Reconstrucción exacta del documento con overlap=0: {reconstruccion0}")
print("\nLectura de la comparación overlap=80 vs overlap=0:")
print(comparacion_overlap_80_vs_0["lectura"])

# -----------------------------------------------------------------------------
# Guardar TODAS las cifras en resultados.json (carpeta actual)
# -----------------------------------------------------------------------------
resultados = {
    "subtarea": "T1 — Configuración y Parte 1 — chunking fijo con solapamiento",
    "parametros_chunking": {
        "tamano_caracteres": int(TAMANO),
        "overlap_caracteres": int(OVERLAP),
        "overlap_comparacion": 0,
        "unidad": "caracteres",
    },
    "num_chunks": num_chunks,
    "chunks_por_documento": chunks_por_documento,
    "idea_cortada_overlap80": idea80,
    "idea_cortada_overlap0": idea0,
    "comparacion_overlap_80_vs_0": comparacion_overlap_80_vs_0,
    "idea_aparece_completa_overlap80": bool(idea80["aparece_completa_en_algun_chunk"]),
    "idea_aparece_completa_overlap0": bool(idea0["aparece_completa_en_algun_chunk"]),
    "frase_clave_hasta_max_5_dias_overlap80": frase80,
    "frase_clave_hasta_max_5_dias_overlap0": frase0,
    "num_chunks_overlap0": num_chunks0,
    "chunks_por_documento_overlap0": chunks_por_documento0,
    "longitud_documentos_caracteres": {n: int(len(t)) for n, t in DOCS.items()},
    "longitudes_chunks_overlap80": [int(len(c)) for c in chunks],
    "longitudes_chunks_overlap0": [int(len(c)) for c in chunks0],
    "caracteres_totales_docs": chars_docs,
    "caracteres_totales_chunks_overlap80": chars_chunks80,
    "caracteres_totales_chunks_overlap0": chars_chunks0,
    "inflacion_teorica_overlap80": inflacion_teorica_80,
    "inflacion_teorica_overlap0": inflacion_teorica_0,
    "inflacion_real_corpus_overlap80": inflacion_real_80,
    "inflacion_real_corpus_overlap0": inflacion_real_0,
    "cortes_overlap80": cortes80,
    "cortes_overlap0": cortes0,
    "chunks_overlap80": [{"chunk": int(i), "documento": origen[i], "texto": chunks[i]}
                         for i in range(len(chunks))],
    "chunks_overlap0": [{"chunk": int(i), "documento": origen0[i], "texto": chunks0[i]}
                        for i in range(len(chunks0))],
    "verificacion_reconstruccion_overlap0": reconstruccion0,
    "verificacion_reconstruccion_overlap80": reconstruccion80,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print(f"\n'resultados.json' guardado en la carpeta actual con {len(resultados)} claves.")
