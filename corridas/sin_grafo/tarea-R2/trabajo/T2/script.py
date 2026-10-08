# -*- coding: utf-8 -*-
"""
T2 · Parte 2 — Índice (embeddings + kNN) y función recuperar
============================================================

Qué hace este script:
  1) Carga los chunks producidos por T1 (entradas/T1.json). INSPECCIONA la
     estructura real de 'detalle_overlap80' y extrae los textos de chunk
     reales: lee la lista correcta y toma el campo de texto de cada ítem.
     Nunca usa dicts de conteos como chunks y nunca une valores con join.
     VERIFICACIÓN DURA antes de indexar: si len(chunks) !=
     t1['num_chunks_overlap80'] se lanza un error (no se continúa).
  2) Ejecuta el código dado del índice: SentenceTransformer
     'paraphrase-multilingual-MiniLM-L12-v2' con embeddings normalizados,
     o fallback TF-IDF si no hay sentence-transformers o el modelo no está
     en caché local (regla del sistema: sin red ni descargas, por eso el
     modelo denso se carga con local_files_only=True). Imprime MOTOR y V.shape.
  3) Mide el techo del modelo: cuenta los tokens del fragmento más largo con
     el tokenizador del mismo modelo, los compara con max_seq_length e indica
     si cabe entero o cuántos tokens se pierden en silencio. Con el fallback
     TF-IDF registra que no hay truncado.
  4) Ejercicio 2.1: recuperar(pregunta, k=3) -> [(score, idx_chunk), ...]
     ordenada de mayor a menor score, con esa firma exacta (la Parte 3 y el
     ejercicio 3.1 la llaman así).

Salidas:
  - resultados.json con: motor_retrieval, forma_indice, tope_tokens_modelo,
    tokens_fragmento_mas_largo, fragmento_cabe_entero y cifras auxiliares.
  - No se pide ninguna figura PNG.
"""

import json
import sys

import numpy as np

# ----------------------------------------------------------------------------
# 0) Salida UTF-8 robusta (evita UnicodeEncodeError en consolas de Windows)
# ----------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ----------------------------------------------------------------------------
# 1) Cargar T1 e inspeccionar la estructura real de 'detalle_overlap80'
# ----------------------------------------------------------------------------
RUTA_T1 = "entradas/T1.json"

with open(RUTA_T1, "r", encoding="utf-8") as f:
    t1 = json.load(f)

# Claves candidatas para el campo de texto de un chunk y para la lista de chunks
CLAVES_TEXTO = ("texto", "text", "chunk", "fragmento", "contenido", "content",
                "texto_chunk", "chunk_texto", "chunk_text", "pasaje", "passage",
                "cuerpo", "body", "contenido_chunk")
CLAVES_LISTA = ("chunks", "fragmentos", "lista_chunks", "textos", "texts",
                "chunks_texto", "lista_fragmentos", "fragmentos_texto",
                "chunks_lista", "items", "lista")


def resumen_estructura(obj, prof=0):
    """Resumen compacto de una estructura, para mensajes de error/inspección."""
    if isinstance(obj, dict):
        if prof >= 2:
            return f"dict({len(obj)} claves)"
        dentro = ", ".join(f"{k}: {resumen_estructura(v, prof + 1)}"
                           for k, v in list(obj.items())[:10])
        return "{" + dentro + "}"
    if isinstance(obj, list):
        if prof >= 2:
            return f"list[{len(obj)}]"
        item0 = resumen_estructura(obj[0], prof + 1) if obj else "vacía"
        return f"list[{len(obj)}] de {item0}"
    if isinstance(obj, str):
        return f"str[{len(obj)}]"
    return repr(obj)


def describir_estructura(obj, nombre, nivel=0):
    """Imprime la estructura real observada (inspección pedida)."""
    sangria = "  " * nivel
    if isinstance(obj, dict):
        print(f"{sangria}[inspección] {nombre}: dict con {len(obj)} claves "
              f"{list(obj.keys())}")
        if nivel < 1:
            for k, v in list(obj.items())[:15]:
                describir_estructura(v, f"{nombre}.{k}", nivel + 1)
    elif isinstance(obj, list):
        tipos = sorted({type(it).__name__ for it in obj[:5]})
        print(f"{sangria}[inspección] {nombre}: list de {len(obj)} ítems "
              f"(tipos: {tipos})")
        if obj and nivel < 2:
            describir_estructura(obj[0], f"{nombre}[0]", nivel + 1)
    elif isinstance(obj, str):
        vista = obj[:70].replace("\n", " ")
        sufijo = "..." if len(obj) > 70 else ""
        print(f"{sangria}[inspección] {nombre}: str de {len(obj)} chars: "
              f"«{vista}{sufijo}»")
    else:
        print(f"{sangria}[inspección] {nombre}: {type(obj).__name__} = {obj!r}")


def es_dict_de_conteos(d):
    """True si d es un dict no vacío cuyos valores son todos numéricos."""
    if isinstance(d, dict) and d:
        return all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   for v in d.values())
    return False


def texto_de_item(it, contexto=""):
    """Devuelve el TEXTO de un chunk a partir de un ítem de la lista.

    - str  -> se usa tal cual (debe ser no vacío).
    - dict -> se toma el campo de texto: primero una clave conocida de texto;
              si no hay, el único campo str del dict; si hay varios, el más
              largo (el texto es el campo largo, los metadatos son cortos).
    NUNCA une valores con join y NUNCA acepta dicts de conteos: si un dict
    no tiene ningún campo de texto claro, se lanza error.
    """
    if isinstance(it, str):
        if it.strip():
            return it
        raise ValueError(f"Chunk vacío en {contexto}.")
    if isinstance(it, dict):
        if es_dict_de_conteos(it):
            raise ValueError(f"Ítem de {contexto} es un dict de conteos, "
                             f"no un chunk: {it}.")
        for clave in CLAVES_TEXTO:
            v = it.get(clave)
            if isinstance(v, str) and v.strip():
                return v
        campos_str = [v for v in it.values() if isinstance(v, str) and v.strip()]
        if len(campos_str) == 1:
            return campos_str[0]
        if len(campos_str) > 1:
            return max(campos_str, key=len)
        raise ValueError(f"Ningún campo de texto en el ítem de {contexto}; "
                         f"claves del ítem: {list(it.keys())}.")
    raise ValueError(f"Ítem de tipo inesperado ({type(it).__name__}) "
                     f"en {contexto}.")


def extraer_chunks(detalle, nombre):
    """Extrae la lista ORDENADA de textos de chunk desde 'detalle_overlapXX'.

    Acepta:
      A) que la clave sea directamente la lista de chunks (str o dict);
      B) un dict que contiene esa lista bajo una clave conocida de lista;
      C) un dict con cualquier otra clave cuya lista tenga ítems str/dict
         con campo de texto (se prefiere la de textos más largos: los chunks
         son párrafos, no etiquetas ni nombres de documento);
      D) un dict posicional {"0": chunk, "1": chunk, ...}.
    Descarta dicts de conteos y listas de números. Nunca une con join.
    """
    if detalle is None:
        return []
    if es_dict_de_conteos(detalle):
        print(f"[aviso] {nombre} es un dict de conteos; no contiene chunks.")
        return []
    # Caso A: la clave ya es la lista de chunks
    if isinstance(detalle, list):
        return [texto_de_item(it, f"{nombre}[{i}]")
                for i, it in enumerate(detalle)]
    if isinstance(detalle, dict):
        # Caso B: lista bajo una clave conocida
        for clave in CLAVES_LISTA:
            v = detalle.get(clave)
            if (isinstance(v, list) and v
                    and all(isinstance(it, (str, dict)) for it in v)):
                try:
                    return [texto_de_item(it, f"{nombre}.{clave}[{i}]")
                            for i, it in enumerate(v)]
                except ValueError:
                    continue
        # Caso C: cualquier otra clave con lista de str/dict con texto
        candidatas = []
        for clave, v in detalle.items():
            if clave in CLAVES_LISTA:
                continue  # ya probadas en el caso B
            if (isinstance(v, list) and v
                    and all(isinstance(it, (str, dict)) for it in v)):
                try:
                    textos = [texto_de_item(it, f"{nombre}.{clave}[{i}]")
                              for i, it in enumerate(v)]
                except ValueError:
                    continue
                if textos:
                    candidatas.append((clave, textos))
        if candidatas:
            candidatas.sort(key=lambda ct: -sum(len(t) for t in ct[1]) / len(ct[1]))
            elegida = candidatas[0]
            print(f"[inspección] lista de chunks tomada de la clave "
                  f"'{elegida[0]}' de {nombre}.")
            return elegida[1]
        # Caso D: dict posicional {"0": ..., "1": ...}
        claves_num = [k for k in detalle if str(k).isdigit()]
        if claves_num and len(claves_num) == len(detalle):
            return [texto_de_item(detalle[k], f"{nombre}.{k}")
                    for k in sorted(claves_num, key=lambda x: int(x))]
    return []


# --- Inspección explícita de la estructura real (pedida por la corrección) ---
print("[inspección] Estructura real de 'detalle_overlap80' en entradas/T1.json:")
describir_estructura(t1.get("detalle_overlap80"), "detalle_overlap80")

# ----------------------------------------------------------------------------
# 2) Extracción de los chunks reales + VERIFICACIÓN DURA antes de indexar
# ----------------------------------------------------------------------------
chunks = extraer_chunks(t1.get("detalle_overlap80"), "detalle_overlap80")
config_chunks = "overlap80"
clave_num = "num_chunks_overlap80"

if not chunks:
    print("[aviso] 'detalle_overlap80' no rindió chunks; se intenta "
          "'detalle_overlap0'.")
    chunks = extraer_chunks(t1.get("detalle_overlap0"), "detalle_overlap0")
    config_chunks = "overlap0"
    clave_num = "num_chunks_overlap0"

if not chunks:
    raise ValueError(
        "No se pudieron extraer chunks de entradas/T1.json. Estructura de "
        f"'detalle_overlap80': {resumen_estructura(t1.get('detalle_overlap80'))}"
    )

if clave_num not in t1:
    raise ValueError(f"entradas/T1.json no contiene la clave '{clave_num}'.")

num_esperado = int(t1[clave_num])
if len(chunks) != num_esperado:
    # Verificación dura: no se continúa con un número equivocado de chunks
    raise ValueError(
        f"VERIFICACIÓN DURA FALLIDA antes de indexar: se extrajeron "
        f"{len(chunks)} chunks de 'detalle_{config_chunks}' pero T1 reporta "
        f"{clave_num} = {num_esperado}. No se continúa. Estructura observada: "
        f"{resumen_estructura(t1.get('detalle_' + config_chunks))}"
    )
print(f"[verificación dura] len(chunks) = {len(chunks)} == "
      f"{clave_num} = {num_esperado} -> OK")

n_chunks = len(chunks)
dups = n_chunks - len(set(chunks))
if dups:
    print(f"[aviso] hay {dups} chunks repetidos; revisar la extracción.")
cortos = [i for i, c in enumerate(chunks) if len(c) < 20]
if cortos:
    print(f"[aviso] chunks sospechosamente cortos: {cortos}.")

print(f"\nChunks cargados de T1 (configuración {config_chunks}): {n_chunks}")
for i in sorted(set([0, 1, n_chunks - 1])):
    vista = chunks[i][:70].replace("\n", " ")
    print(f"  chunk[{i}] ({len(chunks[i])} chars): «{vista}...»")

# ----------------------------------------------------------------------------
# 3) Índice: código dado (denso con sentence-transformers o fallback TF-IDF)
#    Regla del sistema: sin red ni descargas -> el modelo denso se carga solo
#    desde la caché local (local_files_only=True); si no está, fallback TF-IDF.
# ----------------------------------------------------------------------------
_m = None
embed = None
vec_consulta = None
V = None
MOTOR = None

try:
    from sentence_transformers import SentenceTransformer
    _ST_DISPONIBLE = True
except Exception as _e_imp:
    SentenceTransformer = None
    _ST_DISPONIBLE = False
    print(f"[aviso] sentence-transformers no disponible "
          f"({type(_e_imp).__name__}): se usa el fallback TF-IDF.")

if _ST_DISPONIBLE:
    try:
        _m = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2",
                                 local_files_only=True)
    except TypeError:
        print("[aviso] esta versión de sentence-transformers no acepta "
              "local_files_only; sin red no se puede descargar el modelo:")
        print("        se usa el fallback TF-IDF.")
        _m = None
    except Exception as _e_load:
        print(f"[aviso] el modelo denso no está en la caché local "
              f"({type(_e_load).__name__}); sin red no se descarga:")
        print("        se usa el fallback TF-IDF.")
        _m = None

if _m is not None:
    try:
        def embed(ts):
            return np.asarray(_m.encode(ts, normalize_embeddings=True))

        V = embed(chunks)
        MOTOR = "denso (sentence-transformers)"

        def vec_consulta(q):
            return embed([q])[0]
    except Exception as _e_enc:
        print(f"[aviso] fallo al codificar el corpus con el modelo denso "
              f"({type(_e_enc).__name__}): se usa el fallback TF-IDF.")
        _m = None
        embed = None
        V = None

if V is None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _tf = TfidfVectorizer().fit(chunks)
    Vs = _tf.transform(chunks).toarray()
    V = Vs / (np.linalg.norm(Vs, axis=1, keepdims=True) + 1e-9)
    MOTOR = "léxico (TF-IDF fallback)"

    def vec_consulta(q):
        v = _tf.transform([q]).toarray()[0]
        return v / (np.linalg.norm(v) + 1e-9)

print("Retrieval:", MOTOR, "| índice:", V.shape)

# ----------------------------------------------------------------------------
# 4) El techo del modelo, medido (código dado, con rama TF-IDF del enunciado)
# ----------------------------------------------------------------------------
if MOTOR.startswith("denso") and _m is not None:
    tope = int(_m.max_seq_length)
    tokens = [len(_m.tokenizer(c)["input_ids"]) for c in chunks]
    idx_mas_largo = int(np.argmax(tokens))
    tokens_mas_largo = int(tokens[idx_mas_largo])
    chars_mas_largo = int(len(chunks[idx_mas_largo]))
    cabe_entero = bool(tokens_mas_largo <= tope)
    tokens_perdidos = int(max(0, tokens_mas_largo - tope))
    hay_truncado = not cabe_entero

    print(f"tope del modelo: {tope} tokens")
    print(f"fragmento más largo: [{idx_mas_largo}] con {tokens_mas_largo} tokens "
          f"({chars_mas_largo} caracteres)")
    if cabe_entero:
        print("cabe entero")
        nota_techo = ("El fragmento más largo cabe entero en la ventana del "
                      f"modelo ({tope} tokens): no se pierde ningún token.")
    else:
        print(f"NO cabe: se pierden {tokens_perdidos} tokens en silencio")
        nota_techo = (f"El fragmento más largo NO cabe: el modelo trunca en "
                      f"{tope} tokens y se pierden {tokens_perdidos} tokens "
                      f"en silencio.")
else:
    tope = None
    tokens_mas_largo = None
    tokens_perdidos = 0
    hay_truncado = False
    cabe_entero = True  # sin modelo denso no hay ventana que trunque
    idx_mas_largo = int(np.argmax([len(c) for c in chunks]))
    chars_mas_largo = int(len(chunks[idx_mas_largo]))
    nota_techo = ("Sin sentence-transformers no hay modelo que truncar: el "
                  "fallback TF-IDF lee el fragmento entero (no hay truncado).")
    print("Sin sentence-transformers no hay modelo que truncar: el fallback "
          "TF-IDF lee el fragmento entero.")

# ----------------------------------------------------------------------------
# 5) Ejercicio 2.1 — recuperar(pregunta, k=3) -> [(score, idx_chunk), ...]
# ----------------------------------------------------------------------------
def recuperar(pregunta, k=3):
    """
    Devuelve los k chunks más relevantes para `pregunta` como lista de
    (score, idx_chunk) ordenada de mayor a menor score.

    Usa el índice V (filas normalizadas) y vec_consulta(pregunta), de modo
    que el producto punto V @ q equivale a la similitud coseno. Firma exacta
    pedida: la Parte 3 y el ejercicio 3.1 la llaman así.
    """
    q = vec_consulta(pregunta)
    scores = np.asarray(V @ q, dtype=float)
    orden = np.argsort(-scores, kind="stable")
    return [(float(scores[i]), int(i)) for i in orden[: int(k)]]


# --- Demo / verificación de recuperar (con la frase idea de T1) --------------
pregunta_demo = t1.get("frase_idea")
if not isinstance(pregunta_demo, str) or not pregunta_demo.strip():
    pregunta_demo = t1.get("idea_cortada_overlap80")
if not isinstance(pregunta_demo, str) or not pregunta_demo.strip():
    pregunta_demo = "¿Qué dice el material sobre fragmentación y recuperación?"

top_demo = recuperar(pregunta_demo, k=3)
scores_demo = [s for s, _ in top_demo]
orden_descendente = all(scores_demo[i] >= scores_demo[i + 1]
                        for i in range(len(scores_demo) - 1))
longitud_ok = len(top_demo) == min(3, n_chunks)

print("\nDemo de recuperar(pregunta, k=3) — pregunta: frase idea de T1")
print("pregunta:", pregunta_demo)
for s, i in top_demo:
    vista = chunks[i][:70].replace("\n", " ")
    print(f"  score={s:.4f}  chunk[{i}]  «{vista}...»")

# ----------------------------------------------------------------------------
# 6) Cifras auxiliares y guardado en resultados.json
# ----------------------------------------------------------------------------
normas = np.linalg.norm(np.asarray(V, dtype=float), axis=1)
norma_media = float(np.mean(normas))

longs_chars = [len(c) for c in chunks]
chars_min = int(np.min(longs_chars))
chars_media = float(np.mean(longs_chars))
chars_max = int(np.max(longs_chars))

resultados = {
    "tarea": "T2 · Parte 2 — Índice (embeddings + kNN) y función recuperar",
    "parametros": {
        "modelo_embeddings": "paraphrase-multilingual-MiniLM-L12-v2",
        "embeddings_normalizados": True,
        "config_chunks_usada": config_chunks,
        "fuente_chunks": f"entradas/T1.json::detalle_{config_chunks}",
        "k_defecto_recuperar": 3,
        "similitud": "coseno (producto punto sobre vectores normalizados)",
        "sin_red": True,
    },
    "motor_retrieval": MOTOR,
    "forma_indice": [int(V.shape[0]), int(V.shape[1])],
    "num_chunks_indexados": int(V.shape[0]),
    "dimension_embedding": int(V.shape[1]),
    "norma_media_filas_V": norma_media,
    "tope_tokens_modelo": tope,
    "tokens_fragmento_mas_largo": tokens_mas_largo,
    "fragmento_cabe_entero": bool(cabe_entero),
    "idx_fragmento_mas_largo": idx_mas_largo,
    "caracteres_fragmento_mas_largo": chars_mas_largo,
    "tokens_perdidos_en_silencio": int(tokens_perdidos),
    "hay_truncado": bool(hay_truncado),
    "nota_techo": nota_techo,
    "verificacion_num_chunks": {
        "clave_t1": clave_num,
        "esperado": num_esperado,
        "extraidos": int(n_chunks),
        "ok": bool(n_chunks == num_esperado),
    },
    "estructura_detalle_overlap80": resumen_estructura(t1.get("detalle_overlap80")),
    "chars_chunk_min": chars_min,
    "chars_chunk_media": chars_media,
    "chars_chunk_max": chars_max,
    "chunks_duplicados": int(dups),
    "preview_chunks": [chunks[i][:60] for i in range(n_chunks)],
    "recuperar_orden_descendente": bool(orden_descendente),
    "recuperar_longitud_ok": bool(longitud_ok),
    "demo_recuperar": {
        "pregunta": pregunta_demo,
        "k": 3,
        "top": [{"score": float(s), "idx_chunk": int(i),
                 "vista_previa": chunks[i][:100]} for s, i in top_demo],
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# 7) Resumen de las cifras principales
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Cifras principales (guardadas en resultados.json):")
print("  motor_retrieval:", MOTOR)
print("  forma_indice:", resultados["forma_indice"])
print("  tope_tokens_modelo:", tope)
print("  tokens_fragmento_mas_largo:", tokens_mas_largo)
print("  fragmento_cabe_entero:", cabe_entero)
print("  hay_truncado:", hay_truncado)
print("  tokens_perdidos_en_silencio:", tokens_perdidos)
print("  verificación dura de chunks:", n_chunks, "==", num_esperado, "-> OK")
print("  recuperar orden descendente:", orden_descendente,
      "| longitud ok:", longitud_ok)
print("=" * 70)
print("resultados.json guardado en la carpeta actual.")
