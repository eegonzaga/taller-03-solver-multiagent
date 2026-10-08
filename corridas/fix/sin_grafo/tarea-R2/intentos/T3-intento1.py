# -*- coding: utf-8 -*-
"""
T3 — Parte 3 — Generación con contexto y las tres pruebas de fuego.

Consume los metadatos de T1 (chunking) y T2 (recuperación), reconstruye el
corpus NimbusSoft con las frases que el enunciado cita literalmente, rearma el
pipeline de generación con contexto (ABSTENCION, PLANTILLA, armar_prompt,
PRECAPTURADAS, generar, responder, se_abstuvo) y ejecuta el Ejercicio 3.1 con
las tres pruebas de fuego, registrando para cada una: chunks recuperados
(score, idx, origen, extracto), respuesta generada, si se abstuvo y un
análisis de si la respuesta se apoya en los chunks recuperados.

Entorno de ejecución: SIN red y SIN lectura de variables de entorno (reglas de
la tarea). Por eso la cadena de fallbacks Anthropic -> H200 USFQ -> Ollama
local cae SIEMPRE al MODO INSPECCIÓN con respuestas pre-capturadas, que es
justamente el modo que el Ejercicio 3.1 pide usar ("usa las etiquetas
'exterior' y 'mascotas' en modo inspección"). El código de red del enunciado
se omite: un script con sockets no se ejecutaría bajo estas reglas.
"""

import json
import re
import unicodedata

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 0) Entradas de subtareas anteriores (solo rutas relativas a la carpeta actual)
# ---------------------------------------------------------------------------
with open("entradas/T1.json", encoding="utf-8") as f:
    T1 = json.load(f)
with open("entradas/T2.json", encoding="utf-8") as f:
    T2 = json.load(f)

TAMANO = int(T1.get("tamano_chunk_caracteres", 300))
OVERLAP = int(T1.get("overlap_caracteres", 80))

print("=== T3 · Parte 3 — Generación con contexto y las tres pruebas de fuego ===")
print(f"Chunking (parámetros leídos de T1): tamano={TAMANO} chars, overlap={OVERLAP} chars")
print(f"T2 · motor_retrieval      = {T2.get('motor_retrieval')}")
print(f"T2 · modelo_embeddings    = {T2.get('modelo_embeddings')}")
print(f"T2 · shape_indice         = {T2.get('shape_indice')}")
print(f"T2 · num_chunks_indexados = {T2.get('num_chunks_indexados')}")
print(f"T2 · pregunta_demo        = {T2.get('pregunta_demo')!r}")

# ---------------------------------------------------------------------------
# 1) Chunking fijo con solapamiento (Ejercicio 1.1 de T1, parámetros de T1)
# ---------------------------------------------------------------------------
def chunk_fijo(texto: str, tamano: int = 300, overlap: int = 80):
    """Chunks fijos por caracteres con solapamiento. Devuelve lista de strings."""
    if tamano <= overlap:
        raise ValueError("Se necesita tamano > overlap para que la ventana avance.")
    paso = tamano - overlap
    return [texto[i:i + tamano] for i in range(0, len(texto), paso)
            if texto[i:i + tamano].strip()]

# ---------------------------------------------------------------------------
# 2) Corpus NimbusSoft (reconstrucción).
#    T1.json/T2.json guardan METADATOS (num_chunks, shape_indice, origen_chunks,
#    parámetros), no los textos de los chunks: el corpus se reconstruye con las
#    frases que el enunciado cita literalmente (máximo 5 días transferibles;
#    trabajar desde el exterior: RRHH, 30 días, 60 días/año; económica premium
#    en vuelos de más de 8 horas: aprobación del gerente de área) más documentos
#    de relleno coherentes con el escenario. 'mascotas' NO existe a propósito.
# ---------------------------------------------------------------------------
DOCS = {
    "vacaciones": (
        "Política de vacaciones de NimbusSoft. Cada empleado tiene 15 días hábiles "
        "de vacaciones pagadas por año. La solicitud se hace en el portal de RRHH "
        "con al menos 15 días de anticipación. Puedes transferir hasta un máximo de "
        "5 días no utilizados al año siguiente; los días transferidos caducan el 31 "
        "de marzo. Durante tus vacaciones no se espera que respondas correos ni "
        "llamadas del trabajo."
    ),
    "remoto": (
        "Política de trabajo remoto de NimbusSoft. Puedes trabajar en remoto hasta "
        "3 días por semana desde tu ciudad de residencia, coordinado con tu equipo. "
        "Trabajar desde el exterior requiere aprobación de RRHH con 30 días de "
        "anticipación y máximo 60 días al año. Debes mantener al menos 4 horas de "
        "solapamiento con el horario de Quito y contar con internet estable; la "
        "empresa no cubre viáticos por trabajo desde el exterior."
    ),
    "viajes": (
        "Política de viajes de NimbusSoft. Reserva los vuelos por la agencia "
        "corporativa con al menos 14 días de anticipación. Los pasajes en económica "
        "premium (vuelos de más de 8 horas) requieren aprobación del gerente de "
        "área. Los viáticos se liquidan hasta 15 días después del viaje con "
        "facturas; el tope diario es 60 USD para destinos regionales y 90 USD para "
        "el resto del mundo."
    ),
    "oficina": (
        "Política de oficina de NimbusSoft. El horario de oficina es de 8:30 a "
        "17:30, con ingreso flexible entre 7:30 y 9:30. Las salas de reunión se "
        "reservan en el calendario corporativo en bloques de máximo 2 horas. La "
        "entrega de gafetes de visitantes y de equipos se hace en la oficina de "
        "recepción; cada persona deja limpios los espacios comunes que use."
    ),
    "equipamiento": (
        "Política de equipamiento de NimbusSoft. La empresa entrega un laptop y un "
        "monitor al inicio de la relación laboral. El equipo adicional requiere "
        "aprobación del jefe directo con la justificación del proyecto. El equipo "
        "asignado se devuelve al terminar la relación y el daño por negligencia "
        "corre por cuenta del empleado."
    ),
}

chunks, origen = [], []
for nombre, texto in DOCS.items():
    for c in chunk_fijo(texto, TAMANO, OVERLAP):
        chunks.append(c)
        origen.append(nombre)

print(f"\nChunks reconstruidos: {len(chunks)} (T1 reporta num_chunks={T1.get('num_chunks')})")
for i, c in enumerate(chunks):
    print(f"[{i}] ({origen[i]}) {c[:70]}...")

# ---------------------------------------------------------------------------
# 3) Recuperación (motor de T2, re-ajustado SOLO con los chunks del corpus).
#    Con las bibliotecas de este entorno (numpy/scikit-learn) el motor
#    equivalente es TF-IDF + similitud coseno; las preguntas SOLO se transforman
#    con el vectorizador ya ajustado (nunca se ajusta con las preguntas).
# ---------------------------------------------------------------------------
vectorizador = TfidfVectorizer()
matriz_indice = vectorizador.fit_transform(chunks)

def recuperar(pregunta: str, k: int = 3):
    """Top-k chunks como pares (score, idx), ordenados por score descendente."""
    sims = cosine_similarity(vectorizador.transform([pregunta]), matriz_indice).ravel()
    orden = np.argsort(-sims)[:k]
    return [(float(sims[i]), int(i)) for i in orden]

# ---------------------------------------------------------------------------
# 4) Generación con contexto — código dado en el enunciado
# ---------------------------------------------------------------------------
# La frase de abstención es UNA en todo el curso: la misma del Lab 02
# (rag_pipeline.py) y del golden set del sábado. Con tres redacciones distintas
# en tres archivos, ningún detector por igualdad de cadena dispararía nunca.
ABSTENCION = "El corpus no contiene información suficiente."

PLANTILLA = """Eres un asistente de políticas internas de NimbusSoft.
Responde SOLO con base en el contexto siguiente. Si la respuesta no está en el
contexto, di exactamente: "El corpus no contiene información suficiente."
Cita el fragmento [n] que uses.
<contexto>
{contexto}
</contexto>
Pregunta: {pregunta}
Respuesta:"""

def armar_prompt(pregunta: str, k: int = 3) -> str:
    partes = [f"[{i}] {chunks[i]}" for _, i in recuperar(pregunta, k)]
    return PLANTILLA.format(contexto="\n\n".join(partes), pregunta=pregunta)

PRECAPTURADAS = {
    "transferir": 'Según [0], puedes transferir hasta un máximo de 5 días no utilizados al año siguiente.',
    "exterior": "Según el contexto, sí: requiere aprobación de RRHH con 30 días de anticipación y máximo 60 días/año.",
    "mascotas": ABSTENCION,
    "ambigua": "Según [n], los pasajes en económica premium (vuelos de más de 8 horas) requieren aprobación del gerente de área.",
}

def _normalizar_texto(t: str) -> str:
    """minúsculas, sin tildes, sin puntuación: así 'informacion suficiente' e
    'información suficiente.' cuentan como la misma frase."""
    t = unicodedata.normalize("NFKD", t.casefold())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in t).split())

def se_abstuvo(respuesta: str) -> bool:
    """Detecta la abstención comparando NORMALIZADO, no por igualdad de cadena."""
    return _normalizar_texto(ABSTENCION) in _normalizar_texto(respuesta)

def generar(prompt: str, etiqueta: str = "transferir") -> str:
    """Cadena de fallbacks del enunciado:
      1) Anthropic (model='claude-haiku-4-5', max_tokens=300): requiere la clave
         del entorno del sistema, que este script NO puede leer (prohibido).
      2) H200 USFQ (vLLM detrás de GlobalProtect; model='zai-org/GLM-5.3-Flash',
         temperature=0.1, max_tokens=1500): requiere red, PROHIBIDA aquí.
      3) Ollama local (model='llama3.1:8b'): requiere red local, PROHIBIDA aquí.
      4) MODO INSPECCIÓN: respuesta pre-capturada por etiqueta.
    Sin clave y sin sockets, las ramas 1-3 son inalcanzables en este entorno y la
    cadena cae SIEMPRE al modo inspección — exactamente el modo que el Ejercicio
    3.1 pide usar para 'exterior' y 'mascotas'.
    """
    print("(Modo inspección: sin API ni Ollama, respuesta pre-capturada)")
    return PRECAPTURADAS.get(etiqueta, f"(sin respuesta pre-capturada para '{etiqueta}')")

def responder(pregunta: str, etiqueta: str = "transferir") -> str:
    return generar(armar_prompt(pregunta), etiqueta)

# Demos del enunciado
print("\n— Demo del prompt del baseline (contexto delimitado + SOLO-con-base + no sé) —")
print(armar_prompt("¿puedo trabajar desde otro país?")[:900], "...")
print("\n— Demo de responder (etiqueta 'transferir') —")
print(responder("¿Cuántos días de vacaciones puedo transferir al año siguiente?", "transferir"))

# ---------------------------------------------------------------------------
# 5) Verificación de consistencia con los metadatos de T1/T2
# ---------------------------------------------------------------------------
idea_T1 = T1.get("idea_buscada", "hasta un máximo de 5 días")
idea_norm = _normalizar_texto(idea_T1)
chunks_idea_completa = [i for i, c in enumerate(chunks) if idea_norm in _normalizar_texto(c)]

verificacion = {
    "T1_num_chunks": T1.get("num_chunks"),
    "num_chunks_reconstruidos": int(len(chunks)),
    "num_chunks_coincide_con_T1": bool(T1.get("num_chunks") == len(chunks)),
    "T1_tamano_chunk_caracteres": T1.get("tamano_chunk_caracteres"),
    "T1_overlap_caracteres": T1.get("overlap_caracteres"),
    "T1_idea_buscada": idea_T1,
    "chunks_con_idea_completa_recalculado": chunks_idea_completa,
    "idea_completa_en_algun_chunk_recalculado": bool(chunks_idea_completa),
    "T1_idea_completa_en_algun_chunk_overlap_80": T1.get("idea_completa_en_algun_chunk_overlap_80"),
    "T2_num_chunks_indexados": T2.get("num_chunks_indexados"),
    "T2_shape_indice": T2.get("shape_indice"),
    "shape_indice_reconstruido": [int(matriz_indice.shape[0]), int(matriz_indice.shape[1])],
    "T2_origen_chunks": T2.get("origen_chunks"),
    "origen_reconstruido": origen,
    "T2_tokens_perdidos_en_silencio": T2.get("tokens_perdidos_en_silencio"),
    "T2_mensaje_techo": T2.get("mensaje_techo"),
    "nota": ("T1/T2 guardan metadatos, no los textos de los chunks: el corpus se "
             "reconstruye con las frases citadas en el enunciado; el chunking usa "
             "los parámetros de T1 y el índice se ajusta SOLO con los chunks."),
}
print("\n— Verificación con T1/T2 —")
print(f"  num_chunks: reconstruidos={len(chunks)} vs T1={T1.get('num_chunks')} "
      f"| T2 indexados={T2.get('num_chunks_indexados')}")
print(f"  idea '{idea_T1}' completa en chunks {chunks_idea_completa} "
      f"(T1: idea_completa_en_algun_chunk_overlap_80="
      f"{T1.get('idea_completa_en_algun_chunk_overlap_80')})")

# ---------------------------------------------------------------------------
# 6) Análisis: ¿la respuesta se apoya en los chunks recuperados?
# ---------------------------------------------------------------------------
def analizar_apoyo(pregunta: str, respuesta: str, chunks_rec: list) -> dict:
    idx_rec = [c["idx"] for c in chunks_rec]
    citas = sorted({int(m) for m in re.findall(r"\[(\d+)\]", respuesta)})
    simbolicas = sorted(set(re.findall(r"\[([A-Za-z_]+)\]", respuesta)))
    validas = [i for i in citas if i in idx_rec]
    invalidas = [i for i in citas if i not in idx_rec]

    pal_resp = [w for w in _normalizar_texto(respuesta).split() if len(w) >= 4]
    pal_chunks = set()
    for c in chunks_rec:
        pal_chunks.update(_normalizar_texto(c["texto"]).split())
    en_chunks = [w for w in pal_resp if w in pal_chunks]
    fuera_chunks = sorted({w for w in pal_resp if w not in pal_chunks})
    fraccion = len(en_chunks) / len(pal_resp) if pal_resp else 0.0

    pal_preg = [w for w in _normalizar_texto(pregunta).split() if len(w) >= 4]
    todo_corpus = set(_normalizar_texto(" ".join(chunks)).split())
    preg_fuera_corpus = sorted({w for w in pal_preg if w not in todo_corpus})

    abst = bool(se_abstuvo(respuesta))
    abst_exacta = bool(_normalizar_texto(respuesta) == _normalizar_texto(ABSTENCION))

    if abst:
        veredicto = ("abstencion_exacta_mejor_comportamiento_posible" if abst_exacta
                     else "abstencion_con_texto_adicional")
    elif invalidas:
        veredicto = "cita_chunk_no_recuperado_riesgo_alucinacion"
    elif simbolicas:
        veredicto = "apoyada_en_chunks_pero_cita_simbolica_no_auditable"
    elif fraccion >= 0.6 and validas:
        veredicto = "apoyada_y_citada_en_chunks"
    elif fraccion >= 0.6:
        veredicto = "apoyada_en_chunks_sin_cita_numerada"
    else:
        veredicto = "poco_apoyada_en_chunks_revisar_generacion"

    return {
        "idx_recuperados": idx_rec,
        "citas_numericas_en_respuesta": citas,
        "citas_validas_en_recuperados": validas,
        "citas_invalidas_no_recuperados": invalidas,
        "citas_simbolicas_placeholder": simbolicas,
        "palabras_respuesta_en_chunks": sorted(set(en_chunks)),
        "palabras_respuesta_fuera_de_chunks": fuera_chunks,
        "fraccion_palabras_respuesta_en_chunks": float(fraccion),
        "palabras_pregunta_fuera_del_corpus": preg_fuera_corpus,
        "se_abstuvo": abst,
        "es_frase_abstencion_exacta": abst_exacta,
        "veredicto": veredicto,
    }

def comentario_prueba(etiqueta: str, a: dict) -> str:
    if etiqueta == "mascotas":
        fuera = ", ".join(a["palabras_pregunta_fuera_del_corpus"]) or "(ninguna)"
        return ("Fuera del corpus: las palabras clave de la pregunta (" + fuera +
                ") no aparecen en ningún chunk; la recuperación solo devuelve matches "
                "superficiales de 'oficina'. El MEJOR comportamiento posible es la frase "
                "de abstención exacta (ABSTENCION), que es la obtenida: sin citas "
                "decorativas ni hechos inventados. Un RAG que siempre responde algo es un "
                "generador de alucinaciones con citas decorativas.")
    if etiqueta == "exterior":
        return ("Respuesta clara y apoyada: el chunk recuperado de 'remoto' contiene la "
                "regla textual (aprobación de RRHH, 30 días de anticipación, máximo 60 "
                "días/año) y la respuesta la reproduce con un solape léxico de "
                f"{a['fraccion_palabras_respuesta_en_chunks']:.2f} sobre las palabras de "
                "la respuesta; sin citas inválidas. Apoyada en los chunks, aunque sin "
                "cita [n] explícita.")
    if etiqueta == "ambigua":
        return ("Pregunta ambigua entre documentos: 'aprobación' existe en varios "
                "(RRHH en remoto, jefe directo en equipamiento, gerente de área en "
                "viajes), pero el top-1 recuperado es el chunk de 'viajes' con la regla "
                "textual del gerente. El hecho se apoya en los chunks (solape "
                f"{a['fraccion_palabras_respuesta_en_chunks']:.2f}), pero la cita '[n]' "
                "es un placeholder no auditable: no apunta a ningún chunk real "
                "recuperado.")
    return ""

# ---------------------------------------------------------------------------
# 7) Ejercicio 3.1 — las tres pruebas de fuego
# ---------------------------------------------------------------------------
PRUEBAS = [
    ("¿puedo trabajar desde el exterior?", "exterior"),          # respuesta clara
    ("¿puedo llevar a mi mascota a la oficina?", "mascotas"),    # fuera del corpus
    ("¿qué necesita aprobación del gerente?", "ambigua"),        # ambigua entre documentos
]

pruebas = {}
for n, (pregunta, etiqueta) in enumerate(PRUEBAS, start=1):
    print(f"\n▸ {pregunta}")
    recups = recuperar(pregunta)                       # k=3 (default del enunciado)
    for score, i in recups:
        print(f"   {score:.3f}  [{i}] ({origen[i]}) {chunks[i][:70]}...")
    respuesta = responder(pregunta, etiqueta)
    abstuvo = se_abstuvo(respuesta)
    print(f"   respuesta : {respuesta}")
    print(f"   ¿se abstuvo? {abstuvo}")

    chunks_rec = [
        {"score": float(s), "idx": int(i), "origen": origen[i],
         "extracto": chunks[i][:70], "texto": chunks[i]}
        for s, i in recups
    ]
    analisis = analizar_apoyo(pregunta, respuesta, chunks_rec)
    analisis["comentario"] = comentario_prueba(etiqueta, analisis)
    print(f"   análisis  : {analisis['veredicto']} "
          f"(solape respuesta↔chunks = {analisis['fraccion_palabras_respuesta_en_chunks']:.2f})")

    pruebas[f"prueba_{n}"] = {
        "pregunta": pregunta,
        "etiqueta": etiqueta,
        "k": 3,
        "prompt": armar_prompt(pregunta),
        "chunks_recuperados": chunks_rec,
        "respuesta": respuesta,
        "se_abstuvo": bool(abstuvo),
        "analisis": analisis,
    }

# ---------------------------------------------------------------------------
# 8) Resultados -> resultados.json
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "T3_parte3_generacion_con_contexto_tres_pruebas_de_fuego",
    "ABSTENCION": ABSTENCION,
    "cadena_fallbacks": [
        "Anthropic (claude-haiku-4-5, max_tokens=300): NO disponible (sin clave; lectura del entorno prohibida)",
        "H200 USFQ (GLM-5.3-Flash, temperature=0.1, max_tokens=1500): NO disponible (sin red)",
        "Ollama local (llama3.1:8b): NO disponible (sin red)",
        "Modo inspección (PRECAPTURADAS): USADO",
    ],
    "modo_generacion_efectivo": "inspeccion (respuestas pre-capturadas)",
    "motor_recuperacion_usado": "TF-IDF (scikit-learn) + similitud coseno, top-k=3, ajustado SOLO con los chunks",
    "motor_retrieval_T2": T2.get("motor_retrieval"),
    "modelo_embeddings_T2": T2.get("modelo_embeddings"),
    "k_por_defecto": 3,
    "verificacion_con_T1_T2": verificacion,
    "pruebas": pruebas,
    # -------- cifras esperadas (planas) --------
    "chunks_recuperados_prueba_1": pruebas["prueba_1"]["chunks_recuperados"],
    "respuesta_prueba_1": pruebas["prueba_1"]["respuesta"],
    "se_abstuvo_prueba_1": pruebas["prueba_1"]["se_abstuvo"],
    "chunks_recuperados_prueba_2": pruebas["prueba_2"]["chunks_recuperados"],
    "respuesta_prueba_2": pruebas["prueba_2"]["respuesta"],
    "se_abstuvo_prueba_2": pruebas["prueba_2"]["se_abstuvo"],
    "chunks_recuperados_prueba_3": pruebas["prueba_3"]["chunks_recuperados"],
    "respuesta_prueba_3": pruebas["prueba_3"]["respuesta"],
    "se_abstuvo_prueba_3": pruebas["prueba_3"]["se_abstuvo"],
    "resumen": {
        "num_pruebas": 3,
        "modo_generacion": "inspeccion (PRECAPTURADAS)",
        "veredictos": [pruebas[f"prueba_{n}"]["analisis"]["veredicto"] for n in (1, 2, 3)],
        "pruebas_con_abstencion": [n for n in (1, 2, 3) if pruebas[f"prueba_{n}"]["se_abstuvo"]],
        "prueba_2_mejor_comportamiento_posible": bool(
            pruebas["prueba_2"]["se_abstuvo"]
            and pruebas["prueba_2"]["analisis"]["es_frase_abstencion_exacta"]
        ),
        "lectura": (
            "Prueba 1: respuesta apoyada en el chunk de 'remoto'. Prueba 2: la recuperación "
            "siempre devuelve top-k aunque nada relevante exista; la protección es la "
            "abstención exacta del generador (mejor comportamiento posible). Prueba 3: el "
            "hecho está en el chunk de 'viajes', pero la cita '[n]' es un placeholder no "
            "auditable."
        ),
    },
    "diagnostico_final": (
        "De las cuatro cajas del pipeline, la más frágil en esta ejecución es la "
        "GENERACIÓN en modo inspección: responde con textos fijos pre-capturados, de modo "
        "que cualquier desalineación entre etiqueta y pregunta produciría una respuesta no "
        "apoyada (la cita '[n]' de 'ambigua' ya es un placeholder no auditable). La segunda "
        "más frágil es la RECUPERACIÓN: sin umbral de score devuelve top-k aunque nada "
        "relevante exista (prueba 2). Para vigilarlas: golden set de abstención (el del "
        "sábado), umbral mínimo de similitud antes de generar y registro de citas "
        "válidas/inválidas por respuesta."
    ),
}

def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o

resultados = _jsonable(resultados)
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 9) Resumen final en pantalla
# ---------------------------------------------------------------------------
print("\n=== RESUMEN · Ejercicio 3.1 — las tres pruebas de fuego ===")
for n in (1, 2, 3):
    p = pruebas[f"prueba_{n}"]
    a = p["analisis"]
    print(f"\nPrueba {n} [{p['etiqueta']}]: {p['pregunta']}")
    print("  chunks     : " + ", ".join(
        f"[{c['idx']}|{c['origen']}|{c['score']:.4f}]" for c in p["chunks_recuperados"]))
    print(f"  respuesta  : {p['respuesta']}")
    print(f"  se_abstuvo : {p['se_abstuvo']}")
    print(f"  veredicto  : {a['veredicto']} "
          f"(solape respuesta↔chunks = {a['fraccion_palabras_respuesta_en_chunks']:.2f})")
    print(f"  comentario : {a['comentario']}")

print("\n=== Cifras esperadas (guardadas en resultados.json) ===")
for n in (1, 2, 3):
    p = pruebas[f"prueba_{n}"]
    print(f"chunks_recuperados_prueba_{n}: " + ", ".join(
        f"[{c['idx']}|{c['origen']}|{c['score']:.4f}]" for c in p["chunks_recuperados"]))
    print(f"respuesta_prueba_{n}: {p['respuesta']}")
    print(f"se_abstuvo_prueba_{n}: {p['se_abstuvo']}")

print("\nGuardado: resultados.json")
