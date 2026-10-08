# -*- coding: utf-8 -*-
"""
T2 — Parte 2 — Índice (embeddings + kNN) y techo del modelo.

Pasos:
  1. Cargar los chunks y su origen desde la subtarea T1 (entradas/T1.json).
  2. Celda del índice: SentenceTransformer paraphrase-multilingual-MiniLM-L12-v2
     con embeddings normalizados; fallback TF-IDF (vectores normalizados) si
     sentence-transformers no está disponible o el modelo no puede cargarse.
     Imprime MOTOR y V.shape.
  3. Celda del techo: max_seq_length del modelo frente a los tokens del
     fragmento más largo medidos con SU tokenizador; con fallback TF-IDF,
     mensaje de que no hay modelo que trunque.
  4. Ejercicio 2.1: recuperar(pregunta, k=3) -> [(score, idx_chunk), ...]
     ordenada de mayor a menor score (similitud coseno sobre el índice V),
     con esa firma exacta porque la Parte 3 y el ejercicio 3.1 la llaman.
  5. Guardar todas las cifras en resultados.json (carpeta actual).

La subtarea no pide ninguna figura PNG.
"""

import json
import os

import numpy as np

# ----------------------------------------------------------------------
# 1. Cargar chunks y origen desde T1
# ----------------------------------------------------------------------
RUTA_T1 = os.path.join("entradas", "T1.json")

t1 = {}
if os.path.exists(RUTA_T1):
    try:
        with open(RUTA_T1, "r", encoding="utf-8") as f:
            t1 = json.load(f)
        print(f"[T1] cargado {RUTA_T1} ({len(t1)} claves)")
    except Exception as e:
        print(f"[T1] AVISO: no se pudo leer {RUTA_T1} ({type(e).__name__}: {e})")
else:
    print(f"[T1] AVISO: no existe {RUTA_T1}; se usan parámetros por defecto.")

# Parámetros del chunking registrados por T1
tam = int(t1.get("tamano_chunk_caracteres") or 300)
ovl = int(t1.get("overlap_caracteres") or 80)
objetivo_n = int(t1.get("num_chunks") or 12)


def _a_texto(x):
    """Aplana un elemento de la lista de chunks a texto plano."""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for clave in ("texto", "text", "chunk", "contenido", "content"):
            if clave in x:
                return str(x[clave])
    return str(x)


chunks = None
origen_chunks = None
for clave in ("chunks", "fragmentos", "lista_chunks", "chunks_texto",
              "textos", "corpus_chunks", "chunks_overlap"):
    valor = t1.get(clave) if isinstance(t1, dict) else None
    if isinstance(valor, (list, tuple)) and len(valor) > 0:
        chunks = [_a_texto(c) for c in valor]
        origen_chunks = f"entradas/T1.json (clave '{clave}')"
        break

if chunks is None:
    # T1.json guarda las métricas del chunking pero no el texto de los chunks:
    # reconstrucción determinista con los parámetros de T1 sobre un corpus base
    # que incluye la idea buscada en T1.
    parrafos_base = []
    idea = t1.get("idea_buscada")
    if idea:
        parrafos_base.append(str(idea))
    parrafos_base.extend([
        "La búsqueda vectorial representa cada fragmento de texto como un vector denso de "
        "números, llamado embedding, de modo que dos pasajes con significado parecido quedan "
        "cerca en ese espacio aunque no compartan palabras exactas; recuperar es, entonces, "
        "un problema de vecinos más cercanos sobre la colección indexada.",
        "El índice se construye calculando un embedding por fragmento y apilando los vectores "
        "en una matriz; la consulta se proyecta al mismo espacio y se ordenan los fragmentos "
        "por similitud coseno, que con vectores normalizados se reduce a un producto punto.",
        "El chunking por caracteres es cómodo pero engañoso: el modelo de embeddings no lee "
        "caracteres sino tokens, y trunca en silencio todo lo que exceda su max_seq_length, "
        "sin lanzar excepciones ni avisos; la cola del fragmento simplemente desaparece.",
        "Por eso el techo real del sistema no lo fija el tamaño de fragmento elegido en el "
        "notebook sino el tope de tokens del modelo, y comparar ambas unidades exige medir "
        "con el tokenizador del propio modelo en lugar de suponer que trescientos caben.",
        "Un solape entre fragmentos consecutivos reduce la probabilidad de que una idea quede "
        "cortada justo en el borde de un chunk, al precio de indexar texto repetido y de "
        "inflar el tamaño del índice y el costo de la búsqueda.",
        "Si el fragmento es demasiado grande, el truncado deja parte del texto fuera del "
        "alcance de la búsqueda; si es demasiado pequeño, cada vector describe un significado "
        "incompleto y la recuperación se degrada por falta de contexto.",
        "La normalización de los embeddings convierte la similitud coseno en producto punto y "
        "permite implementar el kNN por fuerza bruta con una multiplicación de matrices sobre "
        "toda la colección, suficiente para corpus pequeños como el de este taller.",
        "La evaluación honesta de un recuperador necesita preguntas con respuesta conocida, "
        "un valor de k y una métrica de acierto; nada debe ajustarse mirando el conjunto de "
        "prueba, porque en cuanto se mira para ajustar deja de ser prueba.",
    ])

    paso = max(1, tam - ovl)
    necesarios = tam + max(0, objetivo_n - 1) * paso
    texto = ""
    i = 0
    while len(texto) < necesarios:
        texto = (texto + " " + parrafos_base[i % len(parrafos_base)]).strip()
        i += 1
    texto = texto[:necesarios]

    chunks = []
    inicio = 0
    while inicio < len(texto):
        chunks.append(texto[inicio:inicio + tam])
        if inicio + tam >= len(texto):
            break
        inicio += paso
    origen_chunks = (f"reconstruido con parámetros de T1 (tamano_chunk_caracteres={tam}, "
                     f"overlap_caracteres={ovl}, num_chunks={objetivo_n}); "
                     f"T1.json no incluye el texto de los chunks")

print(f"[chunks] {len(chunks)} fragmentos | origen: {origen_chunks}")

# ----------------------------------------------------------------------
# 2. Celda del índice: embeddings densos normalizados o fallback TF-IDF
# ----------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer

    _m = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def embed(ts):
        return np.asarray(_m.encode(ts, normalize_embeddings=True))

    MOTOR = "denso (sentence-transformers)"
    V = embed(chunks)

    def vec_consulta(q):
        return embed([q])[0]
except Exception as e:
    # ImportError (paquete no instalado) o fallo al obtener/cargar el modelo:
    # se usa el fallback léxico TF-IDF con vectores normalizados.
    print(f"[índice] sentence-transformers no disponible "
          f"({type(e).__name__}: {e}); uso el fallback TF-IDF.")
    from sklearn.feature_extraction.text import TfidfVectorizer

    _tf = TfidfVectorizer().fit(chunks)
    Vs = _tf.transform(chunks).toarray()
    V = Vs / (np.linalg.norm(Vs, axis=1, keepdims=True) + 1e-9)

    def vec_consulta(q):
        v = _tf.transform([q]).toarray()[0]
        return v / (np.linalg.norm(v) + 1e-9)

    MOTOR = "léxico (TF-IDF fallback)"

print("Retrieval:", MOTOR, "| índice:", V.shape)

# ----------------------------------------------------------------------
# 3. Celda del techo del modelo, medido
# ----------------------------------------------------------------------
if MOTOR.startswith("denso"):
    tope = int(_m.max_seq_length)
    tokens = [len(_m.tokenizer(c)["input_ids"]) for c in chunks]
    mas_largo = int(np.argmax(tokens))
    tokens_mas_largo = int(tokens[mas_largo])
    chars_mas_largo = int(len(chunks[mas_largo]))
    cabe_entero = bool(tokens_mas_largo <= tope)
    tokens_perdidos = int(max(0, tokens_mas_largo - tope))

    print(f"tope del modelo: {tope} tokens")
    print(f"fragmento más largo: [{mas_largo}] con {tokens_mas_largo} tokens "
          f"({chars_mas_largo} caracteres)")
    if cabe_entero:
        mensaje_techo = "cabe entero"
        print("cabe entero")
    else:
        mensaje_techo = f"NO cabe: se pierden {tokens_perdidos} tokens en silencio"
        print(mensaje_techo)
else:
    tope = None
    tokens_mas_largo = None
    mas_largo = None
    chars_mas_largo = int(max(len(c) for c in chunks))
    cabe_entero = None
    tokens_perdidos = None
    mensaje_techo = ("Sin sentence-transformers no hay modelo que trunque: "
                     "el fallback TF-IDF lee el fragmento entero.")
    print(mensaje_techo)

# ----------------------------------------------------------------------
# 4. Ejercicio 2.1 — recuperar(pregunta, k=3)
# ----------------------------------------------------------------------
def recuperar(pregunta, k=3):
    """Devuelve [(score, idx_chunk), ...] ordenada de mayor a menor score.

    Similitud coseno entre el vector de la consulta (vec_consulta) y cada fila
    del índice V. Como las filas de V y la consulta están normalizadas, el
    producto punto es exactamente la similitud coseno. Firma exacta exigida:
    la Parte 3 y el ejercicio 3.1 la llaman así.
    """
    vq = np.asarray(vec_consulta(pregunta), dtype=float)
    scores = V @ vq
    orden = np.argsort(-scores, kind="stable")[: int(k)]
    return [(float(scores[i]), int(i)) for i in orden]


# Demo de la función con la idea buscada en T1 como consulta
pregunta_demo = str(t1.get("idea_buscada") or
                    "¿Qué ocurre cuando un fragmento supera el tope de tokens del modelo?")
top_demo = recuperar(pregunta_demo, k=3)
scores_demo = [s for s, _ in top_demo]
assert all(scores_demo[j] >= scores_demo[j + 1] for j in range(len(scores_demo) - 1)), \
    "los scores deben estar ordenados de mayor a menor"
print("\nDemo de recuperar(pregunta, k=3) — consulta: idea_buscada de T1")
for s, i in top_demo:
    print(f"  score={s:.6f}  idx_chunk={i}  :: {chunks[i][:70]}...")

# ----------------------------------------------------------------------
# 5. Guardar todas las cifras en resultados.json
# ----------------------------------------------------------------------
resultados = {
    "motor_retrieval": MOTOR,
    "shape_indice": [int(V.shape[0]), int(V.shape[1])],
    "tope_tokens_modelo": tope,
    "tokens_fragmento_mas_largo": tokens_mas_largo,
    "cabe_entero": cabe_entero,
    "idx_fragmento_mas_largo": mas_largo,
    "caracteres_fragmento_mas_largo": chars_mas_largo,
    "tokens_perdidos_en_silencio": tokens_perdidos,
    "mensaje_techo": mensaje_techo,
    "modelo_embeddings": "paraphrase-multilingual-MiniLM-L12-v2",
    "num_chunks_indexados": int(len(chunks)),
    "origen_chunks": origen_chunks,
    "tamano_chunk_caracteres": tam,
    "overlap_caracteres": ovl,
    "pregunta_demo": pregunta_demo,
    "demo_top3": [{"score": s, "idx_chunk": i} for s, i in top_demo],
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("\n=== Cifras principales (guardadas en resultados.json) ===")
print("motor_retrieval:", resultados["motor_retrieval"])
print("shape_indice:", resultados["shape_indice"])
print("tope_tokens_modelo:", resultados["tope_tokens_modelo"])
print("tokens_fragmento_mas_largo:", resultados["tokens_fragmento_mas_largo"])
print("cabe_entero:", resultados["cabe_entero"])
print("tokens_perdidos_en_silencio:", resultados["tokens_perdidos_en_silencio"])
