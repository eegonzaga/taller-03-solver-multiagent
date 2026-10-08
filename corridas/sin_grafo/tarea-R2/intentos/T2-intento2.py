# -*- coding: utf-8 -*-
"""
T2 · Parte 2 — Índice (embeddings + kNN) y función recuperar
============================================================

Qué hace este script:
  1) Carga los chunks producidos por T1 (entradas/T1.json; configuración
     principal: detalle_overlap80; respaldo: detalle_overlap0).
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
# 1) Chunks de T1 (entradas/T1.json)
# ----------------------------------------------------------------------------
RUTA_T1 = "entradas/T1.json"

with open(RUTA_T1, "r", encoding="utf-8") as f:
    t1 = json.load(f)

_CLAVES_TEXTO = ("texto", "text", "chunk", "fragmento", "contenido", "content",
                 "texto_chunk", "chunk_texto", "pasaje", "passage")


def _texto_de_item(it):
    """Extrae el texto de un chunk, sea str o dict con una clave de texto."""
    if isinstance(it, str):
        return it
    if isinstance(it, dict):
        for clave in _CLAVES_TEXTO:
            v = it.get(clave)
            if isinstance(v, str) and v.strip():
                return v
        vals = [v for v in it.values() if isinstance(v, str) and v.strip()]
        if vals:
            return " ".join(vals)
        return json.dumps(it, ensure_ascii=False)
    return str(it)


def extraer_chunks(detalle):
    """Saca la lista ordenada de textos de chunk de 'detalle_overlapXX'."""
    if isinstance(detalle, list) and detalle:
        return [_texto_de_item(it) for it in detalle]
    if isinstance(detalle, dict):
        for clave in ("chunks", "fragmentos", "lista_chunks", "textos", "texts",
                      "chunks_texto", "lista_fragmentos", "items"):
            v = detalle.get(clave)
            if isinstance(v, list) and v:
                return [_texto_de_item(it) for it in v]
        if detalle and all(str(k).isdigit() for k in detalle.keys()):
            return [_texto_de_item(detalle[k])
                    for k in sorted(detalle.keys(), key=lambda x: int(x))]
        pares = [(k, v) for k, v in detalle.items() if isinstance(v, (str, dict))]
        if pares:
            pares.sort(key=lambda kv: str(kv[0]))
            return [_texto_de_item(v) for _, v in pares]
    return []


def _limpiar(lista):
    return [c for c in lista if isinstance(c, str) and c.strip()]


chunks = _limpiar(extraer_chunks(t1.get("detalle_overlap80")))
config_chunks = "overlap80"
if not chunks:
    chunks = _limpiar(extraer_chunks(t1.get("detalle_overlap0")))
    config_chunks = "overlap0"
if not chunks:
    raise ValueError("No se pudieron extraer los chunks de entradas/T1.json "
                     "(claves detalle_overlap80 / detalle_overlap0).")

n_chunks = len(chunks)
print(f"Chunks cargados de T1 (configuración {config_chunks}): {n_chunks}")

# ----------------------------------------------------------------------------
# 2) Índice: código dado (denso con sentence-transformers o fallback TF-IDF)
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
# 3) El techo del modelo, medido (código dado, con rama TF-IDF del enunciado)
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
# 4) Ejercicio 2.1 — recuperar(pregunta, k=3) -> [(score, idx_chunk), ...]
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
# 5) Cifras auxiliares y guardado en resultados.json
# ----------------------------------------------------------------------------
normas = np.linalg.norm(np.asarray(V, dtype=float), axis=1)
norma_media = float(np.mean(normas))

longs_chars = [len(c) for c in chunks]
chars_min = int(np.min(longs_chars))
chars_media = float(np.mean(longs_chars))
chars_max = int(np.max(longs_chars))

clave_num = "num_chunks_overlap80" if config_chunks == "overlap80" else "num_chunks_overlap0"
num_t1 = t1.get(clave_num)
try:
    num_t1_int = int(num_t1)
    coincide_num = bool(num_t1_int == n_chunks)
except (TypeError, ValueError):
    num_t1_int = None
    coincide_num = None

resultados = {
    "tarea": "T2 · Parte 2 — Índice (embeddings + kNN) y función recuperar",
    "parametros": {
        "modelo_embeddings": "paraphrase-multilingual-MiniLM-L12-v2",
        "embeddings_normalizados": True,
        "config_chunks_usada": config_chunks,
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
    "num_chunks_t1_reportado": num_t1_int,
    "coincide_con_num_chunks_t1": coincide_num,
    "chars_chunk_min": chars_min,
    "chars_chunk_media": chars_media,
    "chars_chunk_max": chars_max,
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
# 6) Resumen de las cifras principales
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
print("  recuperar orden descendente:", orden_descendente,
      "| longitud ok:", longitud_ok)
print("=" * 70)
print("resultados.json guardado en la carpeta actual.")
