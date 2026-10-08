# -*- coding: utf-8 -*-
"""
T2 · Parte 2 — Índice (embeddings + kNN), techo del modelo y recuperar() (Ejercicio 2.1)

Consume de la Parte 1 (entradas/T1.json) los parámetros del chunking y las métricas de
verificación; el TEXTO de los chunks se reconstruye de forma determinista con la misma
chunk_fijo(tamano=300, overlap=80) sobre el corpus DOCS del enunciado (T1.json guarda
métricas, no el texto de los chunks).

Ejecuta:
  1) la celda dada del índice: sentence-transformers 'paraphrase-multilingual-MiniLM-L12-v2'
     con embeddings normalizados, o fallback TF-IDF si no está disponible; imprime MOTOR
     y la forma del índice V.
  2) la celda dada del techo del modelo: max_seq_length del MiniLM frente a los tokens del
     fragmento más largo según su propio tokenizador (con fallback TF-IDF: no hay modelo
     que trunque).
  3) el Ejercicio 2.1: recuperar(pregunta, k=3) -> [(score, idx_chunk), ...] de mayor a menor
     score, con esa firma exacta (la Parte 3 y el ejercicio 3.1 la llaman así).
  4) un ejemplo de recuperación para verificar el orden.

Salidas: resultados.json (todas las cifras, precisión completa) + prints. Sin figuras PNG.
Sin red, sin subprocess, sin rutas absolutas: solo rutas relativas a la carpeta actual.
"""

import json
import numpy as np

# ------------------------------------------------------------------------------------------
# 0) Entrada de la subtarea anterior (T1): parámetros del chunking y métricas de verificación
# ------------------------------------------------------------------------------------------
try:
    with open("entradas/T1.json", "r", encoding="utf-8") as f:
        t1 = json.load(f)
    if not isinstance(t1, dict):
        t1 = {}
except FileNotFoundError:
    t1 = {}
    print("(aviso: no se encontró entradas/T1.json; se usan los parámetros del enunciado)")


def _buscar_param(dic, nombres):
    """Busca un valor numérico por varios nombres posibles de clave (un nivel anidado)."""
    if not isinstance(dic, dict):
        return None
    for n in nombres:
        if n in dic and isinstance(dic[n], (int, float)) and not isinstance(dic[n], bool):
            return dic[n]
    for v in dic.values():
        if isinstance(v, dict):
            r = _buscar_param(v, nombres)
            if r is not None:
                return r
    return None


_tam = _buscar_param(t1.get("parametros"),
                     ["tamano", "tamano_chunk", "chunk_size", "tamano_caracteres"])
_ovl = _buscar_param(t1.get("parametros"),
                     ["overlap", "overlap_caracteres", "solape"])
TAMANO = int(_tam) if _tam is not None else 300
OVERLAP = int(_ovl) if _ovl is not None else 80
print(f"Parámetros de chunking (T1/enunciado): tamano={TAMANO} caracteres, "
      f"overlap={OVERLAP} caracteres")

# ------------------------------------------------------------------------------------------
# Corpus del enunciado: documentación interna de NimbusSoft (texto exacto del cuaderno)
# ------------------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------------------
# 1) Chunks de la Parte 1: misma función y mismos parámetros con los que T1 los construyó
# ------------------------------------------------------------------------------------------
def chunk_fijo(texto, tamano=300, overlap=80):
    """Ejercicio 1.1: ventanas de `tamano` caracteres con solapamiento `overlap`."""
    paso = tamano - overlap
    if paso <= 0:
        raise ValueError("overlap debe ser menor que tamano")
    trozos = []
    inicio = 0
    while inicio < len(texto):
        trozos.append(texto[inicio:inicio + tamano])
        if inicio + tamano >= len(texto):
            break
        inicio += paso
    return trozos


chunks, origen = [], []
for nombre, texto in DOCS.items():
    for c in chunk_fijo(texto, TAMANO, OVERLAP):
        chunks.append(c)
        origen.append(nombre)

print(f"\nChunks ({len(chunks)}) reconstruidos con los parámetros de la Parte 1:")
for i, c in enumerate(chunks):
    print(f"[{i}] ({origen[i]}) {c[:70]}...")

# Verificación contra las métricas guardadas por T1
lens = [len(c) for c in chunks]
num_chunks_t1 = t1.get("num_chunks")
car_chunks_t1 = t1.get("caracteres_en_chunks_overlap80")
coincide_num = True if num_chunks_t1 is None else (int(num_chunks_t1) == len(chunks))
coincide_car = True if car_chunks_t1 is None else (int(car_chunks_t1) == int(sum(lens)))
print(f"\nVerificación con T1 · num_chunks: T1={num_chunks_t1} vs "
      f"reconstruidos={len(chunks)} -> {'coincide' if coincide_num else 'NO coincide'}")
print(f"Verificación con T1 · caracteres en chunks (overlap 80): T1={car_chunks_t1} vs "
      f"reconstruidos={int(sum(lens))} -> {'coincide' if coincide_car else 'NO coincide'}")

# ------------------------------------------------------------------------------------------
# 2) Celda dada — índice: embeddings densos normalizados o fallback TF-IDF
# ------------------------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    _m = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def embed(ts):
        return np.asarray(_m.encode(ts, normalize_embeddings=True))

    MOTOR = "denso (sentence-transformers)"
    V = embed(chunks)

    def vec_consulta(q):
        return embed([q])[0]
except Exception:  # ImportError o cualquier fallo al cargar el modelo -> fallback del curso
    from sklearn.feature_extraction.text import TfidfVectorizer
    _tf = TfidfVectorizer().fit(chunks)
    Vs = _tf.transform(chunks).toarray()
    V = Vs / (np.linalg.norm(Vs, axis=1, keepdims=True) + 1e-9)

    def vec_consulta(q):
        v = _tf.transform([q]).toarray()[0]
        return v / (np.linalg.norm(v) + 1e-9)

    MOTOR = "léxico (TF-IDF fallback)"

assert V.shape[0] == len(chunks), "el índice V debe tener una fila por chunk"
print("\nRetrieval:", MOTOR, "| índice:", V.shape)

# ------------------------------------------------------------------------------------------
# 3) Celda dada — el techo del modelo, medido
# ------------------------------------------------------------------------------------------
idx_max_car = int(np.argmax(lens))  # fragmento más largo en caracteres (siempre medible)
if MOTOR.startswith("denso"):
    tope = int(_m.max_seq_length)
    tokens_por_chunk = [len(_m.tokenizer(c)["input_ids"]) for c in chunks]
    idx_max_tok = int(np.argmax(tokens_por_chunk))
    tokens_mas_largo = int(tokens_por_chunk[idx_max_tok])
    cabe = bool(tokens_mas_largo <= tope)
    perdidos = int(max(0, tokens_mas_largo - tope))
    cabe_entero_o_tokens_perdidos = ("cabe entero" if cabe
                                     else f"NO cabe: se pierden {perdidos} tokens en silencio")
    print(f"\ntope del modelo: {tope} tokens")
    print(f"fragmento más largo: [{idx_max_tok}] con {tokens_mas_largo} tokens "
          f"({len(chunks[idx_max_tok])} caracteres)")
    print("cabe entero" if cabe else
          f"NO cabe: se pierden {perdidos} tokens en silencio")
else:
    tope = None
    idx_max_tok = None
    tokens_mas_largo = None
    cabe = None
    perdidos = None
    tokens_por_chunk = None
    cabe_entero_o_tokens_perdidos = ("fallback TF-IDF: no hay modelo que trunque; "
                                     "el fragmento entero se lee")
    print("\nSin sentence-transformers no hay modelo que truncar: el fallback TF-IDF lee "
          "el fragmento entero.")
    print(f"(info) fragmento más largo en caracteres: [{idx_max_car}] con "
          f"{lens[idx_max_car]} caracteres; sin el tokenizador del modelo no se pueden "
          f"contar sus tokens.")

# ------------------------------------------------------------------------------------------
# 4) Ejercicio 2.1 — recuperar(pregunta, k=3) -> [(score, idx_chunk), ...]
#    Firma exacta: la Parte 3 (armar_prompt) y el ejercicio 3.1 la llaman así.
# ------------------------------------------------------------------------------------------
def recuperar(pregunta, k=3):
    """Devuelve los k chunks más relevantes para la pregunta, de MAYOR a MENOR score.

    Usa el índice V (filas normalizadas) y vec_consulta(pregunta) (vector normalizado):
    el producto punto equivale a la similitud coseno.
    """
    v = vec_consulta(pregunta)
    scores = V @ v
    orden = np.argsort(-scores)  # de mayor a menor score
    return [(float(scores[i]), int(i)) for i in orden[:int(k)]]


# Ejemplo de recuperación para verificar el orden
pregunta_ejemplo = "¿Cuántos días de vacaciones puedo transferir al año siguiente?"
top3 = recuperar(pregunta_ejemplo, k=3)
scores_ej = [s for s, _ in top3]
orden_ok = bool(all(scores_ej[j] >= scores_ej[j + 1] for j in range(len(scores_ej) - 1)))

print("\nEjemplo de recuperación (Ejercicio 2.1)")
print(f"pregunta: {pregunta_ejemplo}")
print("top-3 (score, idx_chunk), de mayor a menor:")
for score, i in top3:
    print(f"   {score:.4f}  [{i}] ({origen[i]}) {chunks[i][:70]}...")
print("¿orden de mayor a menor verificado?", orden_ok)

print(f"\nranking completo ({len(chunks)} chunks) para la misma pregunta:")
for score, i in recuperar(pregunta_ejemplo, k=len(chunks)):
    print(f"   {score:.4f}  [{i}] ({origen[i]}) {chunks[i][:60]}...")

# ------------------------------------------------------------------------------------------
# 5) Guardar TODAS las cifras en resultados.json (precisión completa, sin redondear)
# ------------------------------------------------------------------------------------------
resultados = {
    "subtarea": "T2_parte2",
    "motor_retrieval": MOTOR,
    "forma_indice": [int(V.shape[0]), int(V.shape[1])],
    "num_chunks_indexados": int(V.shape[0]),
    "dimension_indice": int(V.shape[1]),
    "tope_tokens_modelo": tope,
    "tokens_fragmento_mas_largo": tokens_mas_largo,
    "idx_fragmento_mas_largo_por_tokens": idx_max_tok,
    "idx_fragmento_mas_largo_por_caracteres": idx_max_car,
    "caracteres_fragmento_mas_largo": int(lens[idx_max_car]),
    "cabe_entero_o_tokens_perdidos": cabe_entero_o_tokens_perdidos,
    "cabe_entero": cabe,
    "tokens_perdidos_fragmento_mas_largo": perdidos,
    "tokens_por_chunk": ([int(t) for t in tokens_por_chunk] if tokens_por_chunk is not None else None),
    "caracteres_por_chunk": [int(l) for l in lens],
    "pregunta_ejemplo": pregunta_ejemplo,
    "top3_ejemplo_scores": [float(s) for s in scores_ej],
    "top3_ejemplo_idx": [int(i) for _, i in top3],
    "top3_ejemplo": [[float(s), int(i)] for s, i in top3],
    "orden_descendente_verificado": orden_ok,
    "parametros_chunking": {"tamano": int(TAMANO), "overlap": int(OVERLAP)},
    "num_chunks": int(len(chunks)),
    "chunks_por_documento": {n: int(origen.count(n)) for n in DOCS},
    "caracteres_en_chunks": int(sum(lens)),
    "chunks": chunks,
    "origen": origen,
    "verificacion_con_T1": {
        "num_chunks_T1": (int(num_chunks_t1) if num_chunks_t1 is not None else None),
        "coincide_num_chunks": bool(coincide_num),
        "caracteres_en_chunks_overlap80_T1": (int(car_chunks_t1) if car_chunks_t1 is not None else None),
        "coincide_caracteres_en_chunks": bool(coincide_car),
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("\n=== Cifras principales (guardadas en resultados.json) ===")
print("motor_retrieval:", resultados["motor_retrieval"])
print("forma_indice:", resultados["forma_indice"])
print("tope_tokens_modelo:", resultados["tope_tokens_modelo"])
print("tokens_fragmento_mas_largo:", resultados["tokens_fragmento_mas_largo"])
print("cabe_entero_o_tokens_perdidos:", resultados["cabe_entero_o_tokens_perdidos"])
print("top3_ejemplo_scores:", resultados["top3_ejemplo_scores"])
print("top3_ejemplo_idx:", resultados["top3_ejemplo_idx"])
print("orden_descendente_verificado:", resultados["orden_descendente_verificado"])
