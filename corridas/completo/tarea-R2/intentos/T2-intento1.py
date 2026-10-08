# -*- coding: utf-8 -*-
"""
T2 · Parte 2 — Índice (embeddings + kNN) y recuperar()
MMIA 6013 · Semana 2 · Miércoles — RAG mínimo en ~60 líneas.

Qué hace este script:
  1. Recupera (o reconstruye fielmente) los chunks de la Parte 1 (T1):
     chunk_fijo sobre caracteres, tamano=300 / overlap=80, sobre el corpus
     DOCS del enunciado.
  2. Ejecuta la celda dada del índice: sentence-transformers
     'paraphrase-multilingual-MiniLM-L12-v2' con embeddings normalizados
     (construye V y vec_consulta) o fallback TF-IDF normalizado si el modelo
     no está disponible.
  3. Mide el techo del modelo: cuenta los tokens del fragmento más largo con
     el tokenizador del propio modelo y los compara con max_seq_length,
     reportando si cabe entero o cuántos tokens se pierden en silencio.
  4. Ejercicio 2.1: recuperar(pregunta, k=3) -> [(score, idx_chunk), ...]
     ordenada de mayor a menor score, con esa firma exacta porque la Parte 3
     y el ejercicio 3.1 la llaman.

Salidas: resultados.json (todas las cifras, sin redondear) + print de las
cifras principales. Esta subtarea NO produce figura PNG.
"""

import json
import os

import numpy as np

# ---------------------------------------------------------------------------
# 0. Corpus del enunciado (mismo texto exacto que fragmenta la Parte 1)
# ---------------------------------------------------------------------------
DOCS = {
    "politica_vacaciones.md": (
        "Política de vacaciones de NimbusSoft.\n"
        "Los empleados a tiempo completo acumulan 1.5 días de vacaciones por mes trabajado,\n"
        "hasta un máximo de 18 días por año. Las vacaciones deben solicitarse con al menos\n"
        "15 días de anticipación a través del portal interno. Los días no utilizados pueden\n"
        "transferirse al año siguiente hasta un máximo de 5 días. Durante el primer año,\n"
        "los días solo pueden tomarse después de superar el período de prueba de 3\n"
        "meses."
    ),
    "politica_remoto.md": (
        "Política de trabajo remoto de NimbusSoft.\n"
        "El trabajo remoto está permitido hasta 3 días por semana para todos los roles\n"
        "excepto soporte de infraestructura on-site. Los días remotos se coordinan con el\n"
        "líder de equipo. Para trabajar desde el exterior del país se requiere aprobación\n"
        "de Recursos Humanos con 30 días de anticipación y un máximo de 60 días por año."
    ),
    "gastos.md": (
        "Política de reembolso de gastos de NimbusSoft.\n"
        "Los gastos de viaje se reembolsan presentando factura dentro de los 30 días\n"
        "posteriores al gasto. El límite diario de alimentación en viajes es de 45 USD.\n"
        "Los pasajes aéreos deben comprarse en clase económica salvo vuelos de más de\n"
        "8 horas, donde se permite económica premium con aprobación del gerente de área."
    ),
}

# ---------------------------------------------------------------------------
# 1. Chunks: textos de T1 si están guardados; si no, reconstrucción fiel
# ---------------------------------------------------------------------------
T1_PATH = os.path.join("entradas", "T1.json")
t1 = None
if os.path.exists(T1_PATH):
    with open(T1_PATH, "r", encoding="utf-8") as f:
        t1 = json.load(f)
    print(f"entradas/T1.json leído. Claves: {sorted(t1.keys())}")
else:
    print("(no se encontró entradas/T1.json; se reconstruye todo desde el enunciado)")

# Parámetros de chunking declarados por T1 (por defecto, los del enunciado)
tamano, overlap = 300, 80
if isinstance(t1, dict) and isinstance(t1.get("parametros_chunking"), dict):
    pc = t1["parametros_chunking"]
    for clave in ("tamano", "tamano_caracteres", "chunk_size", "tam"):
        if isinstance(pc.get(clave), (int, float)):
            tamano = int(pc[clave])
            break
    for clave in ("overlap", "overlap_caracteres", "solape"):
        if isinstance(pc.get(clave), (int, float)):
            overlap = int(pc[clave])
            break


def _chunk_fijo(texto, tam, ovl, modo="corte"):
    """chunk_fijo de la Parte 1, sobre caracteres.
    modo='corte': al llegar al final se corta (chunk final parcial, estándar).
    modo='cola': sin corte; puede quedar un chunk final corto (variante)."""
    paso = tam - ovl
    if paso <= 0:
        paso = tam
    chunks, inicio, n = [], 0, len(texto)
    while inicio < n:
        chunks.append(texto[inicio:inicio + tam])
        if modo == "corte" and inicio + tam >= n:
            break
        inicio += paso
    return chunks


def _origen_de(chunk):
    for nombre, texto in DOCS.items():
        if chunk in texto:
            return nombre
    return "desconocido"


chunks, origen, fuente_chunks = None, None, None

if isinstance(t1, dict):
    # (a) lista plana de textos de chunks guardada por T1
    for clave in ("chunks", "textos_chunks", "chunks_overlap80", "chunks_textos"):
        v = t1.get(clave)
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            chunks = list(v)
            origen = [_origen_de(c) for c in chunks]
            fuente_chunks = f"textos leídos de entradas/T1.json ('{clave}')"
            break
    # (b) chunks agrupados por documento como listas de strings
    if chunks is None:
        cpd = t1.get("chunks_por_documento")
        if isinstance(cpd, dict) and cpd:
            valores = list(cpd.values())
            if all(isinstance(x, list) and x and isinstance(x[0], str) for x in valores):
                chunks, origen = [], []
                for nombre, lista in cpd.items():
                    chunks.extend(lista)
                    origen.extend([nombre] * len(lista))
                fuente_chunks = "textos leídos de entradas/T1.json ('chunks_por_documento')"

if chunks is None:
    # (c) reconstrucción con chunk_fijo; se elige la variante que cuadre con T1
    def _construir(modo):
        cs, og = [], []
        for nombre, texto in DOCS.items():
            partes = _chunk_fijo(texto, tamano, overlap, modo)
            cs.extend(partes)
            og.extend([nombre] * len(partes))
        return cs, og

    num_t1 = t1.get("num_chunks") if isinstance(t1, dict) else None
    longs_t1 = t1.get("longitudes_chunks_overlap80") if isinstance(t1, dict) else None

    def _cuadra(cand):
        cs, _ = cand
        if isinstance(num_t1, (int, float)) and len(cs) != int(num_t1):
            return False
        if (isinstance(longs_t1, list) and len(longs_t1) == len(cs)
                and all(isinstance(a, (int, float)) for a in longs_t1)):
            if not all(int(a) == len(c) for a, c in zip(longs_t1, cs)):
                return False
        return True

    cand_corte = _construir("corte")
    cand_cola = _construir("cola")
    if _cuadra(cand_corte):
        chunks, origen = cand_corte
        fuente_chunks = (f"reconstruidos con chunk_fijo(tamano={tamano}, overlap={overlap}) "
                         "sobre el corpus del enunciado (variante estándar, verificada contra T1)")
    elif _cuadra(cand_cola):
        chunks, origen = cand_cola
        fuente_chunks = (f"reconstruidos con chunk_fijo(tamano={tamano}, overlap={overlap}) "
                         "sobre el corpus del enunciado (variante con chunk final corto, verificada contra T1)")
    else:
        chunks, origen = cand_corte
        fuente_chunks = (f"reconstruidos con chunk_fijo(tamano={tamano}, overlap={overlap}) "
                         "sobre el corpus del enunciado (variante estándar; sin coincidencia exacta con T1)")

print(f"\nChunks ({len(chunks)}) — fuente: {fuente_chunks}")
for i, c in enumerate(chunks):
    print(f"[{i}] ({origen[i]}) {len(c)} car. · {c[:70].replace(chr(10), ' ')}...")

# Verificación explícita contra las cifras de T1
verificacion_t1 = {"t1_leido": bool(t1 is not None)}
if isinstance(t1, dict):
    if isinstance(t1.get("num_chunks"), (int, float)):
        verificacion_t1["num_chunks_T1"] = int(t1["num_chunks"])
        verificacion_t1["num_chunks_coincide"] = bool(len(chunks) == int(t1["num_chunks"]))
    lt1 = t1.get("longitudes_chunks_overlap80")
    if isinstance(lt1, list):
        if len(lt1) == len(chunks) and all(isinstance(a, (int, float)) for a in lt1):
            verificacion_t1["longitudes_coinciden"] = bool(
                all(int(a) == len(c) for a, c in zip(lt1, chunks)))
        else:
            verificacion_t1["longitudes_coinciden"] = None
print("Verificación con T1:", verificacion_t1)

# ---------------------------------------------------------------------------
# 2. Celda dada — índice: embeddings normalizados (o fallback TF-IDF)
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    try:
        # local_files_only evita cualquier intento de descarga (si la versión lo soporta)
        _m = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", local_files_only=True)
    except TypeError:
        _m = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def embed(ts):
        return np.asarray(_m.encode(ts, normalize_embeddings=True))

    MOTOR = "denso (sentence-transformers)"
    V = embed(chunks)

    def vec_consulta(q):
        return embed([q])[0]
except Exception:
    # El código dado captura solo ImportError; sin red/caché el modelo puede
    # fallar a cargar por otras razones: misma semántica de fallback.
    from sklearn.feature_extraction.text import TfidfVectorizer
    _tf = TfidfVectorizer().fit(chunks)  # se ajusta SOLO con el corpus indexado
    Vs = _tf.transform(chunks).toarray()
    V = Vs / (np.linalg.norm(Vs, axis=1, keepdims=True) + 1e-9)

    def vec_consulta(q):
        v = _tf.transform([q]).toarray()[0]
        return v / (np.linalg.norm(v) + 1e-9)

    MOTOR = "léxico (TF-IDF fallback)"

print("\nRetrieval:", MOTOR, "| índice:", V.shape)

# ---------------------------------------------------------------------------
# 3. El techo del modelo, medido (celda dada)
# ---------------------------------------------------------------------------
tope = None
tokens_por_chunk = None
idx_mas_largo = None
tokens_mas_largo = None
cabe_entero = None
tokens_perdidos = None
nota_techo = None

if MOTOR.startswith("denso"):
    tope = int(_m.max_seq_length)
    tokens_por_chunk = [int(len(_m.tokenizer(c)["input_ids"])) for c in chunks]
    idx_mas_largo = int(np.argmax(tokens_por_chunk))
    tokens_mas_largo = int(tokens_por_chunk[idx_mas_largo])
    cabe_entero = bool(tokens_mas_largo <= tope)
    tokens_perdidos = int(max(0, tokens_mas_largo - tope))
    print(f"\ntope del modelo: {tope} tokens")
    print(f"fragmento más largo: [{idx_mas_largo}] con {tokens_mas_largo} tokens "
          f"({len(chunks[idx_mas_largo])} caracteres)")
    print("cabe entero" if cabe_entero else
          f"NO cabe: se pierden {tokens_perdidos} tokens en silencio")
    nota_techo = (
        f"Medido con el tokenizador del propio modelo "
        f"(paraphrase-multilingual-MiniLM-L12-v2, max_seq_length={tope}). "
        + ("El fragmento más largo cabe entero: no se pierde ningún token."
           if cabe_entero else
           f"El fragmento más largo NO cabe: {tokens_perdidos} tokens se pierden "
           "en silencio (el modelo trunca sin avisar).")
    )
else:
    cabe_entero = True   # el fallback léxico no trunca: lee el fragmento entero
    tokens_perdidos = 0
    nota_techo = ("Sin sentence-transformers no hay modelo que truncar: el fallback TF-IDF "
                  "lee el fragmento entero (no hay truncado en tokens). El techo del MiniLM "
                  "multilingüe referenciado en el corpus del curso es 128 tokens "
                  "(fila embed_notebook_s2), pero no aplica a este motor léxico y aquí no "
                  "pudo medirse con el tokenizador del modelo.")
    print("\nSin sentence-transformers no hay modelo que truncar: el fallback TF-IDF lee "
          "el fragmento entero.")

# ---------------------------------------------------------------------------
# 4. Ejercicio 2.1 — recuperar(pregunta, k=3) → [(score, idx_chunk), ...]
# ---------------------------------------------------------------------------
def recuperar(pregunta, k=3):
    """Devuelve los k chunks más relevantes como [(score, idx_chunk), ...],
    ordenados de mayor a menor score, usando el índice V y vec_consulta.
    Firma exacta: la Parte 3 y el ejercicio 3.1 la llaman así."""
    q = vec_consulta(pregunta)
    scores = V @ q                        # coseno: filas de V y q normalizadas
    orden = np.argsort(-scores, kind="stable")[:k]
    return [(float(scores[i]), int(i)) for i in orden]


# Sanity check con las preguntas que usan la Parte 3 y el ejercicio 3.1
PREGUNTAS_DEMO = [
    ("¿puedo trabajar desde otro país?", "demo de armar_prompt (Parte 3)"),
    ("¿Cuántos días de vacaciones puedo transferir al año siguiente?", "transferir"),
    ("¿puedo trabajar desde el exterior?", "exterior"),
    ("¿puedo llevar a mi mascota a la oficina?", "mascotas (fuera del corpus)"),
    ("¿qué necesita aprobación del gerente?", "ambigua"),
]
demo_recuperar = {}
print("\nSanity check de recuperar(pregunta, k=3):")
for pregunta, etiqueta in PREGUNTAS_DEMO:
    res = recuperar(pregunta, k=3)
    demo_recuperar[pregunta] = {
        "etiqueta": etiqueta,
        "top_k": [{"score": s, "idx_chunk": i, "origen": origen[i],
                   "extracto": chunks[i][:70]} for s, i in res],
    }
    print(f"\n▸ {pregunta}  [{etiqueta}]")
    for score, i in res:
        print(f"   {score:.3f}  [{i}] ({origen[i]}) {chunks[i][:70].replace(chr(10), ' ')}...")

# ---------------------------------------------------------------------------
# 5. Guardar todas las cifras en resultados.json (sin redondear)
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "T2 · Parte 2 — Índice (embeddings + kNN) y recuperar()",
    "motor_retrieval": MOTOR,
    "forma_indice_V": [int(V.shape[0]), int(V.shape[1])],
    "tope_tokens_modelo": tope,
    "tokens_fragmento_mas_largo": tokens_mas_largo,
    "fragmento_cabe_entero": bool(cabe_entero),
    "tokens_perdidos_fragmento_mas_largo": int(tokens_perdidos) if tokens_perdidos is not None else None,
    "idx_fragmento_mas_largo": idx_mas_largo,
    "longitud_caracteres_fragmento_mas_largo": (int(len(chunks[idx_mas_largo]))
                                                if idx_mas_largo is not None else None),
    "tokens_por_chunk": tokens_por_chunk,
    "nota_techo": nota_techo,
    "num_chunks": int(len(chunks)),
    "origen_chunks": list(origen),
    "longitudes_chunks_caracteres": [int(len(c)) for c in chunks],
    "parametros_chunking_usados": {"tamano": int(tamano), "overlap": int(overlap)},
    "fuente_chunks": fuente_chunks,
    "verificacion_con_T1": verificacion_t1,
    "demo_recuperar_k3": demo_recuperar,
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)
print("\nresultados.json guardado en la carpeta actual.")

# ---------------------------------------------------------------------------
# 6. Cifras principales
# ---------------------------------------------------------------------------
print("\n=== Cifras principales — T2 · Parte 2 ===")
print("motor_retrieval:", MOTOR)
print("forma_indice_V:", [int(V.shape[0]), int(V.shape[1])])
print("tope_tokens_modelo:", tope)
print("tokens_fragmento_mas_largo:", tokens_mas_largo)
print("fragmento_cabe_entero:", cabe_entero)
if tope is not None:
    print("tokens_perdidos_fragmento_mas_largo:", tokens_perdidos)
