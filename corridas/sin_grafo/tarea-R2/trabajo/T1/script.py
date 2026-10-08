# -*- coding: utf-8 -*-
"""
T1 - Parte 1 - Chunking fijo con solapamiento (Ejercicio 1.1)

Implementa chunk_fijo(texto, tamano=300, overlap=80) por caracteres, construye
las listas chunks y origen sobre el corpus DOCS con el codigo dado en el
enunciado e imprime la vista previa con el formato '[i] (origen) chunk[:70]'
seguido de tres puntos. Observa si algun chunk corta la idea de los dias no
utilizados transferibles ('hasta un maximo de 5 dias') y repite la observacion
con overlap=0, anotando las diferencias en los cortes y en el numero de chunks.

Cifras: num_chunks_overlap80, num_chunks_overlap0,
        idea_cortada_overlap80, idea_cortada_overlap0.
Esta subtarea NO produce figura PNG.
"""

import json

# Tres puntos suspensivos para la vista previa, construidos por repeticion
PUNTOS = "." * 3

# ---------------------------------------------------------------------------
# Corpus DOCS (dado en el enunciado). Vive FUERA de la solucion del ejercicio
# porque las Partes 2 y 3 lo consumen.
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
# Ejercicio 1.1 - chunking fijo con solapamiento (solucion)
# ---------------------------------------------------------------------------
def chunk_fijo(texto, tamano=300, overlap=80):
    """Divide texto en ventanas de tamano caracteres que avanzan con paso
    (tamano - overlap); el ultimo chunk puede quedar mas corto."""
    if tamano <= 0:
        raise ValueError("tamano debe ser un entero positivo")
    if not (0 <= overlap < tamano):
        raise ValueError("overlap debe cumplir 0 <= overlap < tamano")
    paso = tamano - overlap
    return [texto[i:i + tamano] for i in range(0, len(texto), paso)]


# ---------------------------------------------------------------------------
# Parametros del ejercicio e idea a vigilar
# ---------------------------------------------------------------------------
TAMANO = 300
OVERLAP_DEFECTO = 80
OVERLAP_ALT = 0
FRASE_IDEA = "hasta un máximo de 5 días"   # idea: dias no utilizados transferibles
DOC_IDEA = "politica_vacaciones.md"


def construir_chunks(docs, overlap):
    """Codigo dado en el enunciado, parametrizado con el overlap a probar."""
    chunks, origen = [], []
    for nombre, texto in docs.items():
        for c in chunk_fijo(texto, tamano=TAMANO, overlap=overlap):
            chunks.append(c)
            origen.append(nombre)
    return chunks, origen


def analizar_config(docs, overlap):
    """Fragmenta con (TAMANO, overlap) y analiza el corte de la frase idea."""
    paso = TAMANO - overlap
    chunks, origen = construir_chunks(docs, overlap)

    # chunks por documento y comprobacion con la formula teorica techo(L/paso)
    num_por_doc = {}
    for nombre, texto in docs.items():
        cs = chunk_fijo(texto, tamano=TAMANO, overlap=overlap)
        num_por_doc[nombre] = len(cs)
        esperado = -(-len(texto) // paso)
        assert len(cs) == esperado, f"desajuste con la formula en {nombre}"

    # offset de cada documento dentro de la lista global chunks
    offsets, acum = {}, 0
    for nombre in docs:
        offsets[nombre] = acum
        acum += num_por_doc[nombre]

    # localizacion de la frase idea dentro de su documento
    texto_idea = docs[DOC_IDEA]
    pos = texto_idea.find(FRASE_IDEA)
    assert pos >= 0, "la frase idea debe estar en el corpus"
    fin = pos + len(FRASE_IDEA)

    chunks_doc = chunk_fijo(texto_idea, tamano=TAMANO, overlap=overlap)

    # chunks que contienen la frase COMPLETA
    con_completa = [i for i, c in enumerate(chunks_doc) if FRASE_IDEA in c]
    assert any(FRASE_IDEA in c for c in chunks) == bool(con_completa)

    # chunks que TOCAN la frase sin contenerla entera (la cortan)
    que_cortan = []
    k = 0
    while k * paso < len(texto_idea):
        s = k * paso
        e = min(k * paso + TAMANO, len(texto_idea))
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
            "final_del_chunk_que_corta": chunks_doc[k0][-45:],
            "inicio_del_chunk_siguiente": (chunks_doc[k0 + 1][:45]
                                           if k0 + 1 < len(chunks_doc) else ""),
        }

    return {
        "overlap": int(overlap),
        "paso": int(paso),
        "num_chunks_total": int(len(chunks)),
        "num_chunks_por_doc": {n: int(v) for n, v in num_por_doc.items()},
        "posicion_frase_en_doc": int(pos),
        "fin_frase_en_doc": int(fin),
        "idea_aparece_completa_en_algun_chunk": bool(con_completa),
        # Criterio: la idea queda CORTADA (mutilada) cuando NINGUN chunk
        # contiene la frase completa, es decir, es irrecuperable del indice.
        "idea_cortada": not con_completa,
        "chunks_con_idea_completa_doc": [int(i) for i in con_completa],
        "chunks_con_idea_completa_global": [int(offsets[DOC_IDEA] + i) for i in con_completa],
        "chunks_que_cortan_la_idea_doc": [int(i) for i in que_cortan],
        "extracto_corte": extracto,
    }


def imprimir_observacion(res):
    overlap = res["overlap"]
    print()
    print(f"--- Observacion de los cortes con overlap={overlap} "
          f"(tamano={TAMANO}, paso={res['paso']}) ---")
    print(f"Frase idea: '{FRASE_IDEA}' en {DOC_IDEA}, "
          f"caracteres {res['posicion_frase_en_doc']} a {res['fin_frase_en_doc'] - 1}")
    print(f"Chunks totales (3 documentos): {res['num_chunks_total']} | "
          f"por documento: {res['num_chunks_por_doc']}")
    if res["chunks_que_cortan_la_idea_doc"]:
        ex = res["extracto_corte"]
        print(f"  Chunk(s) que CORTAN la frase: "
              f"{res['chunks_que_cortan_la_idea_doc']} de {DOC_IDEA} "
              f"(frontera en el caracter {ex['frontera_en_caracter']})")
        print(f"    final del chunk que corta:  {ex['final_del_chunk_que_corta']!r}")
        print(f"    inicio del chunk siguiente: {ex['inicio_del_chunk_siguiente']!r}")
    else:
        print("  Ningun chunk corta la frase: no toca ninguna frontera.")
    if res["idea_aparece_completa_en_algun_chunk"]:
        print(f"  La frase SI aparece COMPLETA en el/los chunk(s) "
              f"{res['chunks_con_idea_completa_doc']} del documento "
              f"(indice global {res['chunks_con_idea_completa_global']}): "
              f"el solapamiento salva la idea.")
    else:
        print("  La frase NO aparece completa en NINGUN chunk: la idea queda "
              "mutilada e irrecuperable desde el indice.")
    print(f"  => idea_cortada (mutilada, sin recuperacion) = {res['idea_cortada']}")


# ---------------------------------------------------------------------------
# (a) overlap = 80 (por defecto): listas chunks y origen con el codigo dado
# ---------------------------------------------------------------------------
chunks, origen = [], []
for nombre, texto in DOCS.items():
    for c in chunk_fijo(texto):
        chunks.append(c)
        origen.append(nombre)

print("=" * 78)
print("VISTA PREVIA - chunk_fijo por defecto (tamano=300, overlap=80)")
print("=" * 78)
for i, c in enumerate(chunks):
    print(f"[{i}] ({origen[i]}) {c[:70]}{PUNTOS}")

res80 = analizar_config(DOCS, OVERLAP_DEFECTO)
imprimir_observacion(res80)

# ---------------------------------------------------------------------------
# (b) overlap = 0: se repite la observacion
# ---------------------------------------------------------------------------
chunks0, origen0 = [], []
for nombre, texto in DOCS.items():
    for c in chunk_fijo(texto, tamano=TAMANO, overlap=OVERLAP_ALT):
        chunks0.append(c)
        origen0.append(nombre)

print()
print("=" * 78)
print("VISTA PREVIA - chunk_fijo con overlap=0 (tamano=300, overlap=0)")
print("=" * 78)
for i, c in enumerate(chunks0):
    print(f"[{i}] ({origen0[i]}) {c[:70]}{PUNTOS}")

res0 = analizar_config(DOCS, OVERLAP_ALT)
imprimir_observacion(res0)

# ---------------------------------------------------------------------------
# Comparacion overlap=80 frente a overlap=0
# ---------------------------------------------------------------------------
num80 = res80["num_chunks_total"]
num0 = res0["num_chunks_total"]
inflacion_emp = num80 / num0
inflacion_teo = TAMANO / (TAMANO - OVERLAP_DEFECTO)

print()
print("=" * 78)
print("COMPARACION overlap=80 frente a overlap=0")
print("=" * 78)
print(f"num_chunks_overlap80 = {num80}  {res80['num_chunks_por_doc']}")
print(f"num_chunks_overlap0  = {num0}  {res0['num_chunks_por_doc']}")
print(f"Inflacion empirica del indice num80/num0 = {inflacion_emp:.6f} "
      f"(mas {(inflacion_emp - 1) * 100:.2f} por ciento)")
print(f"Inflacion teorica C/(C-V) = {TAMANO}/{TAMANO - OVERLAP_DEFECTO} = "
      f"{inflacion_teo:.6f} (mas {(inflacion_teo - 1) * 100:.2f} por ciento)")
print(f"idea_cortada_overlap80 = {res80['idea_cortada']} "
      f"(frase completa en algun chunk: {res80['idea_aparece_completa_en_algun_chunk']})")
print(f"idea_cortada_overlap0  = {res0['idea_cortada']} "
      f"(frase completa en algun chunk: {res0['idea_aparece_completa_en_algun_chunk']})")

ex80 = res80["extracto_corte"]
ex0 = res0["extracto_corte"]
notas = (
    f"Con overlap=80 (paso {res80['paso']}) la frase idea '{FRASE_IDEA}' "
    f"(caracteres {res80['posicion_frase_en_doc']} a {res80['fin_frase_en_doc'] - 1} "
    f"de {DOC_IDEA}) queda cortada en la frontera del caracter "
    f"{ex80['frontera_en_caracter']}: el chunk {ex80['chunk_que_corta']} termina con "
    f"{ex80['final_del_chunk_que_corta']!r}, pero la ventana solapada siguiente la "
    f"contiene COMPLETA (chunk {res80['chunks_con_idea_completa_doc']} del documento), "
    f"de modo que la idea es recuperable: idea_cortada_overlap80=False. "
    f"Con overlap=0 (paso {res0['paso']}) la misma frase cae sobre la frontera del "
    f"caracter {ex0['frontera_en_caracter']}: el chunk {ex0['chunk_que_corta']} termina "
    f"con {ex0['final_del_chunk_que_corta']!r} y el chunk siguiente empieza con "
    f"{ex0['inicio_del_chunk_siguiente']!r}; ningun chunk la contiene completa y la "
    f"idea queda mutilada en los dos fragmentos: idea_cortada_overlap0=True. "
    f"Diferencia en los cortes: en ambos casos algun chunk corta la frase, pero con "
    f"solapamiento el corte se repara en la ventana siguiente, mientras que sin "
    f"solapamiento la frontera parte la frase en dos mitades irrecuperables. "
    f"Coste del solapamiento: el indice pasa de {num0} a {num80} chunks (factor "
    f"{inflacion_emp:.4f}); la formula teorica C/(C-V) = "
    f"{TAMANO}/{TAMANO - OVERLAP_DEFECTO} da un factor {inflacion_teo:.4f} "
    f"(mas {(inflacion_teo - 1) * 100:.1f} por ciento)."
)
print()
print("NOTA DE OBSERVACION:")
print(notas)

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
    "criterio_idea_cortada": ("True si NINGUN chunk contiene la frase completa "
                              "(idea mutilada e irrecuperable desde el indice)"),
    "detalle_overlap80": res80,
    "detalle_overlap0": res0,
    # ---- cifras pedidas ----
    "num_chunks_overlap80": int(num80),
    "num_chunks_overlap0": int(num0),
    "idea_cortada_overlap80": bool(res80["idea_cortada"]),
    "idea_cortada_overlap0": bool(res0["idea_cortada"]),
    # ---- lecturas complementarias del mismo analisis ----
    "idea_aparece_completa_en_algun_chunk_overlap80": bool(res80["idea_aparece_completa_en_algun_chunk"]),
    "idea_aparece_completa_en_algun_chunk_overlap0": bool(res0["idea_aparece_completa_en_algun_chunk"]),
    "algun_chunk_corta_la_frase_overlap80": bool(res80["chunks_que_cortan_la_idea_doc"]),
    "algun_chunk_corta_la_frase_overlap0": bool(res0["chunks_que_cortan_la_idea_doc"]),
    "inflacion_indice_empirica_num80_entre_num0": float(inflacion_emp),
    "inflacion_indice_teorica_C_sobre_C_menos_V": float(inflacion_teo),
    "inflacion_porcentual_teorica": float((inflacion_teo - 1) * 100),
    "notas_observacion": notas,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print()
print("CIFRAS GUARDADAS EN resultados.json:")
print(f"  num_chunks_overlap80   = {resultados['num_chunks_overlap80']}")
print(f"  num_chunks_overlap0    = {resultados['num_chunks_overlap0']}")
print(f"  idea_cortada_overlap80 = {resultados['idea_cortada_overlap80']}")
print(f"  idea_cortada_overlap0  = {resultados['idea_cortada_overlap0']}")
