# -*- coding: utf-8 -*-
"""
T3 — Parte 3: Qdrant embebido con filtros de metadata.

MECÁNICA documentada (colección → upsert(vector+payload) → query con filtro):
  1) COLECCIÓN   create_collection('faqs', size=64, Distance.COSINE):
                 declara la DIMENSIÓN de los vectores y la MÉTRICA (coseno).
                 Es el "esquema" de la base vectorial.
  2) UPSERT      upsert('faqs', points=[PointStruct(id, vector, payload)]):
                 cada punto es vector + payload; el payload ({texto, tema}) es
                 la metadata filtrable (sin payload solo se recuperarían ids).
  3) QUERY       query_points(query=embed(pregunta), query_filter=..., limit=2):
                 la búsqueda vectorial se ACOTA con un filtro de metadata
                 (tema == 'pagos'); el filtro no reemplaza la búsqueda, la
                 restringe.

Conexión: se intenta primero un servidor Qdrant en http://localhost:6333
(timeout=3 s); si no responde, se cae a modo embebido QdrantClient(':memory:').
Si qdrant-client no está instalado, la parte se salta con el mensaje previsto y
se ejecuta una réplica NumPy FIEL de la misma mecánica (mismos embed_demo con
semilla hashlib.sha256, similitud coseno, filtro tema='pagos', limit=2) para
entregar igualmente las cifras hits_qdrant_filtrados y num_puntos_coleccion.
"""

import json
import hashlib
import numpy as np

# ----------------------------------------------------------------- parámetros
DIM = 64                 # dimensión de los vectores (enunciado)
DISTANCIA = "COSINE"     # métrica de la colección (enunciado)
CONSULTA = "¿cuándo me devuelven el dinero?"
FILTRO_TEMA = "pagos"
LIMIT = 2
NOMBRE_COLECCION = "faqs"

faqs = [
    ("Para resetear tu contraseña entra a Configuración > Seguridad.", "cuenta"),
    ("Los reembolsos se procesan en 5 a 7 días hábiles.", "pagos"),
    ("Puedes exportar tus datos en formato CSV desde el panel.", "cuenta"),
    ("La factura electrónica se emite al confirmar el pago.", "pagos"),
    ("El soporte atiende de lunes a viernes de 9h a 18h.", "soporte"),
    ("La API permite 100 solicitudes por minuto en el plan básico.", "api"),
]

# ---------------------------------------------------- embedding del enunciado
def embed_demo(texto):
    """Embedding sintético determinista. Semilla vía hashlib.sha256 (NO hash(),
    que cambia entre procesos: con un servidor persistente los vectores deben
    ser iguales en cada corrida). Devuelve vector float32 normalizado (64-d)."""
    semilla = int.from_bytes(hashlib.sha256(texto.encode()).digest()[:4], "little")
    r = np.random.default_rng(semilla)
    v = r.normal(size=DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()

# ------------------------------------------------------------- resultados.json
def cargar_resultados():
    try:
        with open("resultados.json", "r", encoding="utf-8") as f:
            prev = json.load(f)
        return prev if isinstance(prev, dict) else {}
    except Exception:
        return {}

def guardar_resultados(res):
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

# ------------------------------------- réplica NumPy fiel de la mecánica Qdrant
def replica_numpy():
    """Réplica sin qdrant-client de: colección → upsert(vector+payload) → query
    con filtro. Los vectores son EXACTAMENTE los que upsertaría Qdrant
    (embed_demo) y, al estar normalizados, producto punto = similitud coseno =
    score que Qdrant devuelve con Distance.COSINE."""
    # "colección" + "upsert": matriz (6, 64) de vectores unitarios + payloads
    vectores = np.array([embed_demo(t) for t, _ in faqs], dtype=np.float32)
    num_puntos = int(vectores.shape[0])
    # "query": embedding de la pregunta
    q = np.array(embed_demo(CONSULTA), dtype=np.float32)
    sims = vectores @ q  # vectores unitarios → similitud coseno
    scores_todos = [
        {"id": int(i), "score": float(sims[i]), "tema": tema, "texto": t}
        for i, (t, tema) in enumerate(faqs)
    ]
    # "query_filter": solo tema == 'pagos' (el filtro acota, no reemplaza)
    idx = [i for i, (_, tema) in enumerate(faqs) if tema == FILTRO_TEMA]
    sims_f = sims[idx]
    orden = np.argsort(-sims_f)[:LIMIT]
    hits = [
        {"pos": int(j + 1), "score": float(sims_f[i]),
         "tema": faqs[idx[i]][1], "texto": faqs[idx[i]][0]}
        for j, i in enumerate(orden)
    ]
    return num_puntos, hits, scores_todos

# =============================================================== ejecución ===
resultados = cargar_resultados()

qdrant_ok = True
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (Distance, VectorParams, PointStruct,
                                      Filter, FieldCondition, MatchValue)
except Exception as e:
    qdrant_ok = False
    print("qdrant-client no instalado — `pip install qdrant-client` para esta parte.")
    print(f"  (detalle: {type(e).__name__}: {e})")
    print("  Parte saltada; se ejecuta una réplica NumPy fiel de la misma mecánica")
    print("  (mismos embed_demo, similitud coseno, filtro tema='pagos', limit=2).")

num_puntos_coleccion = None
hits_qdrant_filtrados = None
scores_todos = None
modo = None

if qdrant_ok:
    try:
        # ---- conexión: servidor local primero, embebido como respaldo ----
        try:
            cliente = QdrantClient(url="http://localhost:6333", timeout=3)
            cliente.get_collections()  # prueba real de que el servidor responde
            modo = "servidor_http_localhost_6333"
            print("Conectado al servidor Qdrant en http://localhost:6333")
        except Exception:
            cliente = QdrantClient(":memory:")
            modo = "embebido_memory"
            print("Servidor no disponible — usando Qdrant embebido (:memory:)")

        # ---- 1) COLECCIÓN desde cero (el servidor persiste entre corridas) ----
        try:
            existe = cliente.collection_exists(NOMBRE_COLECCION)
        except AttributeError:
            existe = any(c.name == NOMBRE_COLECCION
                         for c in cliente.get_collections().collections)
        if existe:
            cliente.delete_collection(NOMBRE_COLECCION)
        cliente.create_collection(
            NOMBRE_COLECCION,
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        )

        # ---- 2) UPSERT: puntos = id + vector + payload {texto, tema} ----
        cliente.upsert(NOMBRE_COLECCION, points=[
            PointStruct(id=i, vector=embed_demo(t),
                        payload={"texto": t, "tema": tema})
            for i, (t, tema) in enumerate(faqs)
        ])
        num_puntos_coleccion = int(cliente.count(NOMBRE_COLECCION).count)
        print("Colección creada con", num_puntos_coleccion, "puntos")

        q_vec = embed_demo(CONSULTA)
        filtro = Filter(must=[FieldCondition(key="tema",
                                             match=MatchValue(value=FILTRO_TEMA))])

        # ---- 3) QUERY con filtro de metadata (tema='pagos'), limit=2 ----
        try:
            hits = cliente.query_points(NOMBRE_COLECCION, query=q_vec,
                                        query_filter=filtro, limit=LIMIT).points
        except (AttributeError, TypeError):  # compatibilidad clientes antiguos
            hits = cliente.search(collection_name=NOMBRE_COLECCION,
                                  query_vector=q_vec, query_filter=filtro,
                                  limit=LIMIT)
        hits_qdrant_filtrados = [
            {"pos": int(j + 1), "score": float(h.score),
             "tema": str(h.payload["tema"]), "texto": str(h.payload["texto"])}
            for j, h in enumerate(hits)
        ]

        # extra documental: ranking SIN filtro (para ver qué acota el filtro)
        try:
            todos = cliente.query_points(NOMBRE_COLECCION, query=q_vec,
                                         limit=len(faqs)).points
        except (AttributeError, TypeError):
            todos = cliente.search(collection_name=NOMBRE_COLECCION,
                                   query_vector=q_vec, limit=len(faqs))
        scores_todos = [
            {"id": int(h.id), "score": float(h.score),
             "tema": str(h.payload["tema"]), "texto": str(h.payload["texto"])}
            for h in todos
        ]
    except Exception as e:
        print(f"Aviso: el camino Qdrant falló ({type(e).__name__}: {e}).")
        print("Se usa la réplica NumPy fiel para no dejar las cifras vacías.")
        modo = "replica_numpy_fallo_qdrant"
        num_puntos_coleccion, hits_qdrant_filtrados, scores_todos = replica_numpy()
else:
    modo = "replica_numpy_sin_qdrant_client"
    num_puntos_coleccion, hits_qdrant_filtrados, scores_todos = replica_numpy()

# --------------------------------------------------------- salida en consola
print("\nConsulta:", CONSULTA)
print(f"Filtro de metadata: tema='{FILTRO_TEMA}'   limit={LIMIT}")
print("Hits filtrados (score | [tema] | texto):")
for h in hits_qdrant_filtrados:
    print(f"  {h['score']:.3f} [{h['tema']}] {h['texto']}")

print("\nMECÁNICA documentada: colección → upsert(vector+payload) → query con filtro")
print("  1) create_collection('faqs', size=64, Distance.COSINE): declara dimensión y métrica.")
print("  2) upsert(PointStruct(id, vector, payload)): cada punto es vector + payload;")
print("     el payload {texto, tema} es la metadata filtrable.")
print("  3) query_points(query=embed(pregunta), query_filter=tema=='pagos', limit=2):")
print("     el filtro no reemplaza la búsqueda vectorial: la acota.")

print("\n(Nota: con embeddings sintéticos el score no es semántico —")
print(" el objetivo aquí es la MECÁNICA de colección/upsert/query/filtro.)")

# ----------------------------------------------------------- resultados.json
resultados.update({
    "qdrant_client_disponible": bool(qdrant_ok),
    "qdrant_modo_ejecucion": modo,                      # servidor | :memory: | réplica
    "qdrant_coleccion": NOMBRE_COLECCION,
    "qdrant_dimension_vectores": int(DIM),
    "qdrant_distancia": DISTANCIA,
    "qdrant_num_documentos_upsert": int(len(faqs)),
    "qdrant_consulta": CONSULTA,
    "qdrant_filtro_tema": FILTRO_TEMA,
    "qdrant_limit": int(LIMIT),
    "num_puntos_coleccion": int(num_puntos_coleccion),
    "hits_qdrant_filtrados": hits_qdrant_filtrados,
    "qdrant_scores_todos_sin_filtro": scores_todos,
    "qdrant_mecanica": ("colección (create_collection: dimensión 64 + Distance.COSINE) "
                        "→ upsert (PointStruct: id + vector + payload {texto, tema}) "
                        "→ query con filtro (query_points: query=embed(pregunta), "
                        "query_filter tema=='pagos', limit=2; el filtro acota la "
                        "búsqueda vectorial, no la reemplaza)"),
})
guardar_resultados(resultados)

print("\nCifras guardadas en resultados.json:")
print("  num_puntos_coleccion =", num_puntos_coleccion)
print("  hits_qdrant_filtrados =")
print(json.dumps(hits_qdrant_filtrados, indent=2, ensure_ascii=False))
print("  modo de ejecución =", modo)

# Reflexión (comentario del enunciado): con N documentos del proyecto, si N es
# pequeño (miles), kNN exacto basta (búsqueda lineal ~N*64 operaciones); un
# índice ANN (HNSW) se justifica a partir de ~10^5–10^6 puntos. Operativamente,
# Qdrant embebido sirve para prototipar y el contenedor/persistente para producción.
