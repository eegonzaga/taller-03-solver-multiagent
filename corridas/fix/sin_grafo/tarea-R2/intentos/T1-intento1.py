# -*- coding: utf-8 -*-
"""
T1 — Parte 1 — Chunking fijo con solapamiento.

Script autocontenido que:
  1) Define el corpus DOCS (3 políticas de NimbusSoft). El corpus vive FUERA de
     chunk_fijo porque las Partes 2 y 3 lo consumen.
  2) Implementa chunk_fijo(texto, tamano=300, overlap=80) sobre caracteres
     (devuelve una lista de strings).
  3) Construye chunks y origen con el bucle dado e imprime cada chunk como
     '[i] (origen) primeros 70 caracteres...'.
  4) Observa si algún chunk corta la idea de los días transferibles
     ('hasta un máximo de 5 días'); verifica que con overlap=80 esa idea
     aparece completa en algún chunk aunque otro la corte; repite la
     inspección con overlap=0 y anota la diferencia.
  5) Guarda en resultados.json las cifras pedidas (num_chunks,
     longitud_max_chunk_caracteres, idea_cortada_overlap_80,
     idea_cortada_overlap_0) más chunks y origen para las subtareas siguientes.

Sin red, sin subprocess, sin rutas absolutas. Esta subtarea no pide figura PNG.
"""

import json

# ---------------------------------------------------------------------------
# Corpus dado: 3 políticas internas de NimbusSoft.
# ---------------------------------------------------------------------------
DOCS = {
    "politica_vacaciones": (
        "POLÍTICA DE VACACIONES DE NIMBUSSOFT. "
        "Todo empleado tiene derecho a 23 días naturales de vacaciones al año. "
        "Se devengan mensualmente y se solicitan en el portal interno con 15 días de antelación. "
        "La aprueba el responsable directo. "
        "Los días no disfrutados se transfieren al año siguiente, "
        "hasta un máximo de 5 días; el exceso se pierde sin compensación económica. "
        "Si un festivo cae dentro del periodo vacacional, no se descuenta del saldo. "
        "En caso de baja médica durante las vacaciones, se interrumpe el cómputo y los días afectados se recuperan. "
        "Al cese del empleado, se liquidan las vacaciones devengadas y no disfrutadas. "
        "Las dudas sobre esta política se canalizan en People Office."
    ),
    "politica_teletrabajo": (
        "POLÍTICA DE TELETRABAJO DE NIMBUSSOFT. "
        "El personal puede trabajar en remoto hasta dos días por semana, previo acuerdo con su responsable. "
        "Los días de teletrabajo se registran en la herramienta de control horario y el empleado debe ser "
        "localizable en su franja habitual. "
        "NimbusSoft proporciona el portátil y una ayuda de 20 euros al mes para la conexión. "
        "El puesto en casa debe cumplir los criterios de ergonomía y seguridad publicados en la intranet. "
        "El equipo en remoto acude a la oficina en las jornadas de equipo marcadas en el calendario anual. "
        "Esta política se revisa cada doce meses."
    ),
    "politica_gastos_viaje": (
        "POLÍTICA DE GASTOS DE VIAJE DE NIMBUSSOFT. "
        "Los gastos de viaje en servicio se reembolsan contra factura dentro de los 30 días siguientes a su fecha. "
        "El transporte en tren se abona íntegro; en coche particular se aplican 0,35 euros por kilómetro. "
        "La pernoctación tiene un límite de 120 euros por noche en ciudad de nivel uno y de 90 euros en el resto. "
        "Las comidas se cubren con dietas de 40 euros por día completo y 25 euros por media jornada. "
        "Los vuelos se reservan en la agencia corporativa con al menos siete días de antelación. "
        "Toda factura debe incluir el NIF de NimbusSoft; si no lo incluye, se rechaza."
    ),
}

# Idea sensible que no debe perderse al fragmentar (los días transferibles):
FRASE = "hasta un máximo de 5 días"
TAMANO = 300    # tamaño de chunk en caracteres (defecto del ejercicio)
OVERLAP = 80    # solapamiento por defecto del ejercicio


# ---------------------------------------------------------------------------
# Ejercicio 1.1 — chunking fijo con solapamiento (ventana deslizante sobre caracteres)
# ---------------------------------------------------------------------------
def chunk_fijo(texto, tamano=300, overlap=80):
    """Divide `texto` en ventanas de `tamano` caracteres; cada ventana nueva
    retrocede `overlap` caracteres respecto del corte anterior, de modo que lo
    cortado en un chunk reaparece en el siguiente. Devuelve una lista de
    strings; el último chunk puede quedar más corto que `tamano`."""
    if tamano <= 0:
        raise ValueError("tamano debe ser un entero positivo")
    if overlap < 0 or overlap >= tamano:
        raise ValueError("overlap debe cumplir 0 <= overlap < tamano")
    chunks = []
    inicio = 0
    n = len(texto)
    while inicio < n:
        fin = inicio + tamano
        chunks.append(texto[inicio:fin])  # el slicing recorta el último chunk
        if fin >= n:
            break
        inicio = fin - overlap            # retrocede `overlap` caracteres
    return chunks


def analizar_corte_idea(docs, frase, tamano, overlap):
    """Localiza en qué chunks (índice global, mismo orden que el bucle dado) la
    frase queda CORTADA y en cuáles aparece COMPLETA.
    Devuelve (completa_en, cortada_en, detalle)."""
    completa_en, cortada_en, detalle = [], [], []
    offset = 0
    for nombre, texto in docs.items():
        cs = chunk_fijo(texto, tamano=tamano, overlap=overlap)
        # inicios de ventana: misma iteración que dentro de chunk_fijo
        starts, inicio, n = [], 0, len(texto)
        while inicio < n:
            starts.append(inicio)
            if inicio + tamano >= n:
                break
            inicio += tamano - overlap
        pos = texto.find(frase)
        for j, c in enumerate(cs):
            g = offset + j
            if pos == -1:
                continue
            s = starts[j]
            e = s + len(c)  # fin real del chunk (el último chunk va recortado)
            interseca = (s < pos + len(frase)) and (e > pos)
            completa = (pos >= s) and (pos + len(frase) <= e)
            if completa:
                completa_en.append(g)
            elif interseca:
                cortada_en.append(g)
                visible = c[max(pos - s, 0): pos + len(frase) - s]
                detalle.append({"chunk": int(g), "origen": nombre,
                                "fragmento_visible": visible})
        offset += len(cs)
    return completa_en, cortada_en, detalle


# ---------------------------------------------------------------------------
# Código dado en el enunciado: chunks/origen viven fuera del ejercicio porque
# las Partes 2 y 3 los consumen.
# ---------------------------------------------------------------------------
chunks, origen = [], []
for nombre, texto in DOCS.items():
    for c in chunk_fijo(texto):
        chunks.append(c)
        origen.append(nombre)

print("=== Chunks con tamano=300, overlap=80 ===")
for i, c in enumerate(chunks):
    print(f"[{i}] ({origen[i]}) {c[:70]}...")

# Repetición de la inspección con overlap=0
chunks0, origen0 = [], []
for nombre, texto in DOCS.items():
    for c in chunk_fijo(texto, overlap=0):
        chunks0.append(c)
        origen0.append(nombre)

print("\n=== Chunks con tamano=300, overlap=0 ===")
for i, c in enumerate(chunks0):
    print(f"[{i}] ({origen0[i]}) {c[:70]}...")

# ---------------------------------------------------------------------------
# Inspección de la idea «hasta un máximo de 5 días»
# ---------------------------------------------------------------------------
comp80, cort80, det80 = analizar_corte_idea(DOCS, FRASE, TAMANO, OVERLAP)
comp0, cort0, det0 = analizar_corte_idea(DOCS, FRASE, TAMANO, 0)

print("\n=== Inspección de la idea «hasta un máximo de 5 días» (días transferibles) ===")
print(f"overlap=80: la idea queda CORTADA en los chunks {cort80} y aparece COMPLETA en los chunks {comp80}")
for d in det80:
    print(f"   chunk [{d['chunk']}] ({d['origen']}) solo llega a mostrar: '{d['fragmento_visible']}'")
if cort80:
    i0 = cort80[0]
    print(f"   final del chunk [{i0}]: '...{chunks[i0][-45:]}'")
if comp80:
    j0 = comp80[0]
    k0 = chunks[j0].find(FRASE)
    print(f"   el chunk [{j0}] la contiene íntegra: '...{chunks[j0][max(k0 - 40, 0):k0 + len(FRASE) + 5]}...'")
print(f"overlap=0 : la idea queda CORTADA en los chunks {cort0} y aparece COMPLETA en los chunks {comp0}")
for d in det0:
    print(f"   chunk [{d['chunk']}] ({d['origen']}) solo llega a mostrar: '{d['fragmento_visible']}'")
print("Diferencia: con overlap=80 la ventana siguiente retrocede 80 caracteres y recupera la idea")
print("entera aunque el chunk anterior la corte; con overlap=0 la parte anterior al corte se pierde")
print("y la idea no aparece completa en ningún chunk (una búsqueda sobre los chunks la daría por perdida).")

# ---------------------------------------------------------------------------
# Cifras y guardado en resultados.json
# ---------------------------------------------------------------------------
num_chunks = len(chunks)
longitud_max_chunk = max((len(c) for c in chunks), default=0)
num_chunks_overlap0 = len(chunks0)
num_por_doc = {nombre: len(chunk_fijo(texto)) for nombre, texto in DOCS.items()}
long_docs = {nombre: len(texto) for nombre, texto in DOCS.items()}

idea_cortada_overlap_80 = len(cort80) > 0
idea_cortada_overlap_0 = len(cort0) > 0
idea_completa_overlap_80 = len(comp80) > 0
idea_completa_overlap_0 = len(comp0) > 0

resultados = {
    "num_chunks": int(num_chunks),
    "longitud_max_chunk_caracteres": int(longitud_max_chunk),
    "idea_cortada_overlap_80": bool(idea_cortada_overlap_80),
    "idea_cortada_overlap_0": bool(idea_cortada_overlap_0),
    "idea_completa_en_algun_chunk_overlap_80": bool(idea_completa_overlap_80),
    "idea_completa_en_algun_chunk_overlap_0": bool(idea_completa_overlap_0),
    "chunks_donde_la_idea_queda_cortada_overlap_80": [int(i) for i in cort80],
    "chunks_donde_la_idea_queda_cortada_overlap_0": [int(i) for i in cort0],
    "chunks_donde_la_idea_aparece_completa_overlap_80": [int(i) for i in comp80],
    "chunks_donde_la_idea_aparece_completa_overlap_0": [int(i) for i in comp0],
    "fragmentos_visibles_de_la_idea_overlap_80": det80,
    "fragmentos_visibles_de_la_idea_overlap_0": det0,
    "idea_buscada": FRASE,
    "tamano_chunk_caracteres": int(TAMANO),
    "overlap_caracteres": int(OVERLAP),
    "overlap_alternativo_caracteres": 0,
    "num_chunks_overlap_0": int(num_chunks_overlap0),
    "num_chunks_por_documento": {k: int(v) for k, v in num_por_doc.items()},
    "longitud_documentos_caracteres": {k: int(v) for k, v in long_docs.items()},
    "chunks": chunks,           # tamano=300, overlap=80 -> lo consumen las Partes 2 y 3
    "origen": origen,
    "chunks_overlap_0": chunks0,
    "origen_overlap_0": origen0,
    "nota": (
        "idea_cortada_overlap_80/0 = True si la frase queda cortada en algún chunk. Con overlap=80 "
        "la idea se corta en un chunk pero aparece COMPLETA en el siguiente (el retroceso de 80 "
        "caracteres la recoge entera). Con overlap=0 se corta y NO aparece completa en ningún chunk: "
        "la parte anterior al corte se pierde. 'chunks' y 'origen' (tamano=300, overlap=80) quedan "
        "guardados aquí para las Partes 2 y 3."
    ),
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("\n=== Cifras guardadas en resultados.json ===")
print(f"num_chunks: {num_chunks}")
print(f"longitud_max_chunk_caracteres: {longitud_max_chunk}")
print(f"idea_cortada_overlap_80: {idea_cortada_overlap_80}")
print(f"idea_cortada_overlap_0: {idea_cortada_overlap_0}")
print(f"idea_completa_en_algun_chunk_overlap_80: {idea_completa_overlap_80}")
print(f"idea_completa_en_algun_chunk_overlap_0: {idea_completa_overlap_0}")
print(f"num_chunks_por_documento: {num_por_doc}")
print(f"num_chunks con overlap=0: {num_chunks_overlap0}")
