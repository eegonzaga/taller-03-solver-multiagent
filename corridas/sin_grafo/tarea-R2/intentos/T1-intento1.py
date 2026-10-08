# -*- coding: utf-8 -*-
"""
T1 — Parte 1 · Chunking fijo con solapamiento (Ejercicio 1.1)

Implementa chunk_fijo(texto, tamano=300, overlap=80) por caracteres, construye
las listas chunks/origen sobre el corpus DOCS con el código dado en el
enunciado, imprime la vista previa '[i] (origen) chunk[:70]...' y observa si
algún chunk corta la idea de los días no utilizados transferibles
("hasta un máximo de 5 días"). Repite la observación con overlap=0 y anota
las diferencias en los cortes y en el número de chunks.

Cifras: num_chunks_overlap80, num_chunks_overlap0,
        idea_cortada_overlap80, idea_cortada_overlap0.
Esta subtarea NO produce figura PNG.
"""

import json

# ---------------------------------------------------------------------------
# Corpus DOCS (dado en el enunciado; no hay archivos de entrada, se define
# aquí). Vive FUERA de la solución del ejercicio porque las Partes 2 y 3 lo
# consumen.
# ---------------------------------------------------------------------------
DOCS = {
    "politica_vacaciones.md": (
        "# Política de vacaciones\n"
        "\n"
        "## Días asignados\n"
        "\n"
        "Todo el personal tiene derecho a 23 días naturales de vacaciones\n"
        "por año natural, más los festivos que fije el calendario laboral.\n"
        "\n"
        "## Días no utilizados\n"
        "\n"
        "Los días de vacaciones no utilizados al cierre del ejercicio se\n"
        "transfieren al año siguiente, hasta un máximo de 5 días.\n"
        "\n"
        "El disfrute de los días transferidos se solicita igual que\n"
        "el ordinario y caduca el 31 de marzo del año siguiente.\n"
        "\n"
        "## Solicitud y aprobación\n"
        "\n"
        "Las vacaciones se solicitan en la herramienta de RRHH con una\n"
        "antelación mínima de 15 días naturales.\n"
        "El responsable responde en un plazo máximo de 5 días hábiles.\n"
        "\n"
        "Las bajas médicas durante las vacaciones no consumen días:\n"
        "se interrumpe el disfrute y se recupera después.\n"
        "\n"
        "Las dudas se consultan con Recursos Humanos.\n"
    ),
    "politica_remoto.md": (
        "# Política de trabajo en remoto\n"
        "\n"
        "## Modalidad híbrida\n"
        "\n"
        "El equipo trabaja en modelo híbrido: tres días en remoto y dos\n"
        "presenciales como referencia. Cada equipo fija sus días presenciales\n"
        "en la ceremonia semanal y los publica en el calendario compartido.\n"
        "\n"
        "## Equipamiento\n"
        "\n"
        "La empresa proporciona portátil, pantalla y silla ergonómica. El\n"
        "material de oficina en remoto se reembolsa según la política de\n"
        "gastos, con un límite mensual de 40 euros por persona.\n"
        "\n"
        "## Conectividad y disponibilidad\n"
        "\n"
        "En horario laboral la persona debe ser localizable y con conexión\n"
        "suficiente para videollamada. Si la conexión doméstica falla, hay\n"
        "que avisar al responsable y buscar alternativa presencial.\n"
        "\n"
        "La empresa no reembolsa la fibra doméstica, pero sí los datos\n"
        "móviles utilizados en viajes de trabajo, según la política de gastos.\n"
    ),
    "gastos.md": (
        "# Política de gastos\n"
        "\n"
        "## Regla general\n"
        "\n"
        "Todo gasto se justifica con ticket y se imputa al proyecto\n"
        "correspondiente. Las tarjetas corporativas son personales e\n"
        "intransferibles: nunca se prestan ni se comparten.\n"
        "\n"
        "## Viajes\n"
        "\n"
        "El tren en clase turista es el medio de transporte por defecto.\n"
        "El avión requiere aprobación previa del responsable cuando exista\n"
        "una alternativa en tren de menos de 4 horas de trayecto.\n"
        "\n"
        "El hotel se reserva por la plataforma corporativa, con un tope de\n"
        "120 euros por noche en capitales y 90 euros en el resto de ciudades.\n"
        "\n"
        "## Comidas y desplazamientos\n"
        "\n"
        "En viaje, la comida se cubre hasta 25 euros al día con ticket.\n"
        "Las comidas de equipo se imputan al proyecto si asisten al menos\n"
        "tres personas y se registra el motivo en la nota del gasto.\n"
        "Los taxis solo se reembolsan con justificación del horario.\n"
    ),
}

# ---------------------------------------------------------------------------
# Ejercicio 1.1 — chunking fijo con solapamiento (solución)
# ---------------------------------------------------------------------------
def chunk_fijo(texto, tamano=300, overlap=80):
    """Divide `texto` en ventanas deslizantes de `tamano` caracteres con
    solapamiento `overlap` (la ventana avanza paso = tamano - overlap).
    Devuelve una lista de strings; el último chunk puede quedar más corto."""
    if tamano <= 0:
        raise ValueError("tamano debe ser un entero positivo")
    if not (0 <= overlap < tamano):
        raise ValueError("overlap debe cumplir 0 <= overlap < tamano")
    paso = tamano - overlap
    return [texto[i:i + tamano] for i in range(0, len(texto), paso)]


# ---------------------------------------------------------------------------
# Parámetros del ejercicio e idea a vigilar
# ---------------------------------------------------------------------------
TAMANO = 300
OVERLAP_DEFECTO = 80
OVERLAP_ALT = 0
FRASE_IDEA = "hasta un máximo de 5 días"   # idea: días no utilizados transferibles
DOC_IDEA = "politica_vacaciones.md"


def construir_chunks(docs, overlap):
    """Código dado en el enunciado, parametrizado con el overlap a probar."""
    chunks, origen = [], []
    for nombre, texto in docs.items():
        for c in chunk_fijo(texto, tamano=TAMANO, overlap=overlap):
            chunks.append(c)
            origen.append(nombre)
    return chunks, origen


def analizar_config(docs, overlap):
    """Fragmenta con (TAMANO, overlap) y analiza el corte de la frase-idea."""
    paso = TAMANO - overlap
    chunks, origen = construir_chunks(docs, overlap)

    # chunks por documento + comprobación con la fórmula teórica ceil(L/paso)
    num_por_doc, inicios_doc_idea = {}, []
    for nombre, texto in docs.items():
        cs = chunk_fijo(texto, tamano=TAMANO, overlap=overlap)
        num_por_doc[nombre] = len(cs)
        esperado = (len(texto) + paso - 1) // paso
        assert len(cs) == esperado, f"desajuste con la fórmula teórica en {nombre}"
        if nombre == DOC_IDEA:
            inicios_doc_idea = [k * paso for k in range(len(cs))]

    # posición global (offset) de cada documento dentro de la lista `chunks`
    offsets, acum = {}, 0
    for nombre in docs:
        offsets[nombre] = acum
        acum += num_por_doc[nombre]

    # localización de la frase-idea dentro de su documento
    texto_idea = docs[DOC_IDEA]
    pos = texto_idea.find(FRASE_IDEA)
    assert pos >= 0, "la frase-idea debe estar en el corpus"
    fin = pos + len(FRASE_IDEA)

    chunks_doc = chunk_fijo(texto_idea, tamano=TAMANO, overlap=overlap)

    # ¿algún chunk contiene la frase COMPLETA?
    con_completa = [i for i, c in enumerate(chunks_doc) if FRASE_IDEA in c]
    # comprobación global (sobre todos los chunks de los 3 documentos)
    assert any(FRASE_IDEA in c for c in chunks) == bool(con_completa)

    # ¿algún chunk la deja CORTADA (toca la frase pero no la contiene entera)?
    que_cortan = []
    k = 0
    while k * paso < len(texto_idea):
        s, e = k * paso, min(k * paso + TAMANO, len(texto_idea))
        toca = (s < fin) and (e > pos)
        entera = (s <= pos) and (fin <= e)
        if toca and not entera:
            que_cortan.append(k)
        k += 1

    extracto = None
    if que_cortan:
        k0 = que_cortan[0]
        extracto = {
            "chunk_que_corta": int(k0),
            "frontera_en_caracter": int(min(k0 * paso + TAMANO, len(texto_idea))),
            "fin_del_chunk_que_corta": chunks_doc[k0][-45:],
            "inicio_del_chunk_siguiente": (chunks_doc[k0 + 1][:45]
                                           if k0 + 1 < len(chunks_doc) else ""),
        }

    return {
        "overlap": int(overlap),
        "paso": int(paso),
        "num_chunks_total": int(len(chunks)),
        "num_chunks_por_doc": {n: int(v) for n, v in num_por_doc.items()},
        "inicio_chunks_doc_idea": [int(x) for x in inicios_doc_idea],
        "posicion_frase_en_doc": int(pos),
        "franja_frase_en_doc": [int(pos), int(fin)],
        "idea_aparece_completa_en_algun_chunk": bool(con_completa),
        "idea_cortada_en_algun_chunk": bool(que_cortan),
        # Criterio principal: la idea queda CORTADA (mutilada) cuando NINGÚN
        # chunk contiene la frase completa -> el corte la hace irrecuperable.
        "idea_cortada": not con_completa,
        "chunks_con_idea_completa_doc": [int(i) for i in con_completa],
        "chunks_con_idea_completa_global": [int(offsets[DOC_IDEA] + i) for i in con_completa],
        "chunks_que_cortan_la_idea_doc": [int(i) for i in que_cortan],
        "extracto_corte": extracto,
    }


def imprimir_observacion(res):
    overlap = res["overlap"]
    print(f"\n--- Observación de los cortes con overlap={overlap} "
          f"(tamano={TAMANO}, paso={res['paso']}) ---")
    print(f"Frase-idea: '{FRASE_IDEA}'")
    print(f"  en {DOC_IDEA}, caracteres "
          f"{res['posicion_frase_en_doc']}..{res['franja_frase_en_doc'][1] - 1}")
    print(f"  chunks totales (3 docs): {res['num_chunks_total']} | "
          f"por doc: {res['num_chunks_por_doc']}")
    if res["idea_cortada_en_algun_chunk"]:
        ex = res["extracto_corte"]
        print(f"  Algún chunk la CORTA: chunk(s) "
              f"{res['chunks_que_cortan_la_idea_doc']} de {DOC_IDEA} "
              f"(frontera en el carácter {ex['frontera_en_caracter']})")
        print(f"    fin del chunk que corta    : ...{ex['fin_del_chunk_que_corta']!r}")
        print(f"    inicio del chunk siguiente : {ex['inicio_del_chunk_siguiente']!r}...")
    else:
        print("  Ningún chunk la corta: la frase no toca ninguna frontera.")
    if res["idea_aparece_completa_en_algun_chunk"]:
        print(f"  La frase SÍ aparece COMPLETA en el/los chunk(s) "
              f"{res['chunks_con_idea_completa_doc']} de {DOC_IDEA} "
              f"(índice global {res['chunks_con_idea_completa_global']}): "
              f"el solapamiento salva la idea.")
    else:
        print("  La frase NO aparece completa en NINGÚN chunk: la idea queda "
              "mutilada e irrecuperable desde el índice.")
    print(f"  => idea_cortada (mutilada, sin recuperación) = {res['idea_cortada']}")


# ---------------------------------------------------------------------------
# (a) overlap = 80 (por defecto): código dado en el enunciado
# ---------------------------------------------------------------------------
chunks, origen = [], []
for nombre, texto in DOCS.items():
    for c in chunk_fijo(texto):
        chunks.append(c)
        origen.append(nombre)

print("=" * 78)
print("VISTA PREVIA — chunk_fijo por defecto (tamano=300, overlap=80)")
print("=" * 78)
for i, c in enumerate(chunks):
    print(f"[{i}] ({origen[i]}) {c[:70]}...")

res80 = analizar_config(DOCS, OVERLAP_DEFECTO)
imprimir_observacion(res80)

# ---------------------------------------------------------------------------
# (b) overlap = 0: se repite la observación
# ---------------------------------------------------------------------------
chunks0, origen0 = [], []
for nombre, texto in DOCS.items():
    for c in chunk_fijo(texto, tamano=TAMANO, overlap=OVERLAP_ALT):
        chunks0.append(c)
        origen0.append(nombre)

print()
print("=" * 78)
print("VISTA PREVIA — chunk_fijo con overlap=0 (tamano=300, overlap=0)")
print("=" * 78)
for i, c in enumerate(chunks0):
    print(f"[{i}] ({origen0[i]}) {c[:70]}...")

res0 = analizar_config(DOCS, OVERLAP_ALT)
imprimir_observacion(res0)

# ---------------------------------------------------------------------------
# Comparación overlap=80 vs overlap=0
# ---------------------------------------------------------------------------
num80 = res80["num_chunks_total"]
num0 = res0["num_chunks_total"]
inflacion_emp = num80 / num0
inflacion_teo = TAMANO / (TAMANO - OVERLAP_DEFECTO)

print()
print("=" * 78)
print("COMPARACIÓN overlap=80 vs overlap=0")
print("=" * 78)
print(f"num_chunks_overlap80 = {num80}  {res80['num_chunks_por_doc']}")
print(f"num_chunks_overlap0  = {num0}  {res0['num_chunks_por_doc']}")
print(f"Inflación del índice (empírica) num80/num0 = {inflacion_emp:.6f} "
      f"(+{(inflacion_emp - 1) * 100:.2f} %)")
print(f"Inflación teórica C/(C-V) = {TAMANO}/{TAMANO - OVERLAP_DEFECTO} = "
      f"{inflacion_teo:.6f} (+{(inflacion_teo - 1) * 100:.2f} %)")
print(f"idea_cortada_overlap80 = {res80['idea_cortada']} "
      f"(¿aparece completa en algún chunk? "
      f"{res80['idea_aparece_completa_en_algun_chunk']})")
print(f"idea_cortada_overlap0  = {res0['idea_cortada']} "
      f"(¿aparece completa en algún chunk? "
      f"{res0['idea_aparece_completa_en_algun_chunk']})")

notas = (
    f"Con overlap=80 (paso {res80['paso']}) la frase-idea '{FRASE_IDEA}' "
    f"(caracteres {res80['posicion_frase_en_doc']}-"
    f"{res80['franja_frase_en_doc'][1] - 1} de {DOC_IDEA}) queda cortada en la "
    f"frontera del carácter {res80['extracto_corte']['frontera_en_caracter']}: "
    f"el chunk {res80['extracto_corte']['chunk_que_corta']} termina en "
    f"'...{res80['extracto_corte']['fin_del_chunk_que_corta']}', pero la ventana "
    f"solapada siguiente la contiene COMPLETA (chunks "
    f"{res80['chunks_con_idea_completa_doc']} del doc), de modo que la idea es "
    f"recuperable: idea_cortada_overlap80=False. Con overlap=0 (paso "
    f"{res0['paso']}) la misma frase cae sobre la frontera del carácter "
    f"{res0['extracto_corte']['frontera_en_caracter']}: el chunk 0 acaba en "
    f"'...{res0['extracto_corte']['fin_del_chunk_que_corta']}' y el chunk 1 "
    f"empieza con '{res0['extracto_corte']['inicio_del_chunk_siguiente']}'; "
    f"ningún chunk la contiene completa y la idea queda mutilada: "
    f"idea_cortada_overlap0=True. Coste del solapamiento: el índice pasa de "
    f"{num0} a {num80} chunks (x{inflacion_emp:.4f}); la fórmula teórica "
    f"C/(C-V) = {TAMANO}/{TAMANO - OVERLAP_DEFECTO} da x{inflacion_teo:.4f} "
    f"(+{(inflacion_teo - 1) * 100:.1f} %)."
)
print("\nNOTA DE OBSERVACIÓN:\n" + notas)

# ---------------------------------------------------------------------------
# Guardado de todas las cifras en resultados.json
# ---------------------------------------------------------------------------
resultados = {
    "tarea": "T1 - Parte 1: chunking fijo con solapamiento (Ejercicio 1.1)",
    "parametros": {
        "tamano_chunk_caracteres": int(TAMANO),
        "overlap_por_defecto": int(OVERLAP_DEFECTO),
        "overlap_alternativo": int(OVERLAP_ALT),
    },
    "corpus": {
        "documentos": list(DOCS.keys()),
        "longitud_docs_caracteres": {n: int(len(t)) for n, t in DOCS.items()},
    },
    "frase_idea": FRASE_IDEA,
    "doc_de_la_idea": DOC_IDEA,
    "detalle_overlap80": res80,
    "detalle_overlap0": res0,
    # ---- cifras pedidas ----
    "num_chunks_overlap80": int(num80),
    "num_chunks_overlap0": int(num0),
    # Criterio: la idea queda CORTADA cuando NINGÚN chunk contiene la frase
    # completa (con overlap=80 otro chunk la salva; con overlap=0 se pierde).
    "idea_cortada_overlap80": bool(res80["idea_cortada"]),
    "idea_cortada_overlap0": bool(res0["idea_cortada"]),
    # ---- lecturas complementarias del mismo análisis ----
    "idea_aparece_completa_en_algun_chunk_overlap80": bool(res80["idea_aparece_completa_en_algun_chunk"]),
    "idea_aparece_completa_en_algun_chunk_overlap0": bool(res0["idea_aparece_completa_en_algun_chunk"]),
    "idea_cortada_en_algun_chunk_overlap80": bool(res80["idea_cortada_en_algun_chunk"]),
    "idea_cortada_en_algun_chunk_overlap0": bool(res0["idea_cortada_en_algun_chunk"]),
    "inflacion_indice_empirica_num80_entre_num0": float(inflacion_emp),
    "inflacion_indice_teorica_C_sobre_C_menos_V": float(inflacion_teo),
    "inflacion_porcentual_teorica": float((inflacion_teo - 1) * 100),
    "notas_observacion": notas,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("\nCIFRAS GUARDADAS EN resultados.json:")
print(f"  num_chunks_overlap80   = {resultados['num_chunks_overlap80']}")
print(f"  num_chunks_overlap0    = {resultados['num_chunks_overlap0']}")
print(f"  idea_cortada_overlap80 = {resultados['idea_cortada_overlap80']}")
print(f"  idea_cortada_overlap0  = {resultados['idea_cortada_overlap0']}")
