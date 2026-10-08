# -*- coding: utf-8 -*-
"""
T3 — Parte 3 — Qdrant embebido: colección, upsert y query con filtro.

CORRECCIÓN del intento anterior (rechazado por intentar usar la red):
este entorno de ejecución PROHÍBE toda comunicación de red. Por eso este
script corregido NO abre ninguna conexión (ni siquiera a un servidor local
en el puerto 6333 con timeout 3 s): ese paso del código dado se SALTA y se
DOCUMENTA, aplicando directamente el fallback local que el propio enunciado
prevé (modo embebido, sin Docker).

Plan (todo local: sin red, sin subprocess, sin Docker):
  1) Comprueba si qdrant-client está instalado (import local, sin red).
     - Si lo está: usa QdrantClient(":memory:") — implementación 100 % en
       proceso, cero red. Recrea la colección 'faqs' (64 dims,
       Distance.COSINE) desde cero, upsert de los 6 FAQs con payload
       {texto, tema} y query_points con query_filter tema='pagos', limit=2.
     - Si no lo está (caso de este entorno, cuyas bibliotecas son solo
       numpy/pandas/scipy/scikit-learn/matplotlib): instalarlo con pip es
       IMPOSIBLE aquí (pip necesita red y subprocess, prohibidos) -> se
       DOCUMENTA EL SALTO de la parte Qdrant y se replica la mecánica
       EXACTA con numpy:
         * mismos embeddings deterministas via hashlib.sha256 (embed_demo),
         * misma colección de 6 puntos (id, vector, payload {texto, tema}),
         * misma búsqueda por similitud coseno — con 6 puntos Qdrant
           ":memory:" hace kNN exacto, así que la réplica da las MISMAS
           cifras,
         * mismo query_filter tema='pagos' y limit=2.

Cifras esperadas (se guardan en resultados.json y se imprimen):
  - num_puntos_coleccion      (puntos en la colección tras el upsert: 6)
  - hits_filtro_pagos_scores  (scores de los 2 hits con filtro tema='pagos')

Nota del enunciado: con embeddings sintéticos el score NO es semántico; el
objetivo es la MECÁNICA de colección/upsert/query/filtro.

No se genera ninguna figura PNG (la subtarea no lo pide).
"""

import json
import hashlib

import numpy as np

# ----------------------------------------------------------------- parámetros
COLECCION = "faqs"
DIM = 64
DISTANCIA = "COSINE"
FILTRO_TEMA = "pagos"
LIMIT = 2
CONSULTA = "¿cuándo me devuelven el dinero?"

# ------------------------------------------------- datos dados en el enunciado
faqs = [
    ("Para resetear tu contraseña entra a Configuración > Seguridad.", "cuenta"),
    ("Los reembolsos se procesan en 5 a 7 días hábiles.", "pagos"),
    ("Puedes exportar tus datos en formato CSV desde el panel.", "cuenta"),
    ("La factura electrónica se emite al confirmar el pago.", "pagos"),
    ("El soporte atiende de lunes a viernes de 9h a 18h.", "soporte"),
    ("La API permite 100 solicitudes por minuto en el plan básico.", "api"),
]


# ------------------------------------------ embeddings sintéticos deterministas
def embed_demo(texto):
    """Embedding determinista del código dado (hashlib.sha256, NO hash()).

    hash() cambia entre procesos; sha256 da la misma semilla en cada corrida,
    imprescindible si el almacenamiento fuera persistente.
    """
    semilla = int.from_bytes(hashlib.sha256(texto.encode()).digest()[:4], "little")
    r = np.random.default_rng(semilla)
    v = r.normal(size=DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


# ------------------------------------------------------- camino qdrant-client
def flujo_qdrant_embebido():
    """Ejecuta la mecánica con qdrant-client, SOLO en modo embebido ":memory:".

    Sin red: no se abre ninguna conexión. El paso del código dado 'intentar
    conexión al servidor Qdrant local (puerto 6333, timeout 3 s)' se salta y
    se documenta; el enunciado ya prevé caer al modo embebido si el servidor
    no está disponible, y aquí se aplica ese fallback directamente.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import (Distance, VectorParams, PointStruct,
                                      Filter, FieldCondition, MatchValue)

    # Modo embebido: misma API, almacenamiento en proceso, cero red, cero Docker.
    cliente = QdrantClient(":memory:")

    # recrear la colección desde cero (un servidor persiste entre corridas)
    if cliente.collection_exists(COLECCION):
        cliente.delete_collection(COLECCION)
    cliente.create_collection(
        COLECCION,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    )

    # upsert: punto = id + vector + payload {texto, tema}
    cliente.upsert(COLECCION, points=[
        PointStruct(id=i, vector=embed_demo(t), payload={"texto": t, "tema": tema})
        for i, (t, tema) in enumerate(faqs)
    ])
    num_puntos = int(cliente.count(COLECCION).count)
    print("Colección creada con", num_puntos, "puntos")

    # consulta con filtro de metadata: solo tema="pagos"
    filtro = Filter(must=[FieldCondition(key="tema",
                                         match=MatchValue(value=FILTRO_TEMA))])
    if hasattr(cliente, "query_points"):      # API actual (qdrant-client >= 1.10)
        hits = cliente.query_points(
            COLECCION,
            query=embed_demo(CONSULTA),
            query_filter=filtro,
            limit=LIMIT,
        ).points
    else:                                     # API antigua, por robustez
        hits = cliente.search(
            collection_name=COLECCION,
            query_vector=embed_demo(CONSULTA),
            query_filter=filtro,
            limit=LIMIT,
        )

    hits_out = [
        {"id": int(h.id), "score": float(h.score),
         "tema": str(h.payload["tema"]), "texto": str(h.payload["texto"])}
        for h in hits
    ]
    return num_puntos, hits_out, "qdrant_embebido_memory"


# --------------------- réplica numpy (solo si qdrant-client no está disponible)
def flujo_replica_numpy():
    """Réplica EXACTA de la mecánica Qdrant con numpy (mismas cifras).

    Correspondencia con el código dado:
      create_collection("faqs", 64, COSINE)     -> vectores de 64 dims, coseno
      upsert(PointStruct(id, vector, payload))  -> lista `puntos`
      count("faqs").count                       -> len(puntos)
      query_points(query=..., query_filter=..., limit=2)
                                                -> filtro por payload + kNN coseno
    Con 6 puntos la búsqueda de Qdrant ":memory:" es kNN exacto con distancia
    coseno, así que esta réplica produce las MISMAS cifras.
    """
    # "colección": punto = id + vector + payload (equivalente a PointStruct)
    puntos = [
        {"id": i, "vector": embed_demo(t), "payload": {"texto": t, "tema": tema}}
        for i, (t, tema) in enumerate(faqs)
    ]
    num_puntos = int(len(puntos))                  # count("faqs").count
    print("Colección creada con", num_puntos, "puntos")

    # vector de consulta
    q = np.asarray(embed_demo(CONSULTA), dtype=np.float64)

    # query_filter: Filter(must=[FieldCondition(key="tema", match=MatchValue("pagos"))])
    candidatos = [p for p in puntos if p["payload"]["tema"] == FILTRO_TEMA]

    # score = similitud coseno (Distance.COSINE), como hace Qdrant
    for p in candidatos:
        v = np.asarray(p["vector"], dtype=np.float64)
        p["score"] = float(np.dot(v, q) / (np.linalg.norm(v) * np.linalg.norm(q)))

    # limit=2, orden descendente por score (desempate por id, como Qdrant)
    top = sorted(candidatos, key=lambda p: (-p["score"], p["id"]))[:LIMIT]

    hits_out = [
        {"id": int(p["id"]), "score": float(p["score"]),
         "tema": p["payload"]["tema"], "texto": p["payload"]["texto"]}
        for p in top
    ]
    return num_puntos, hits_out, "replica_numpy_sin_qdrant_client"


# ------------------------------------------------------------------------ main
print("=" * 72)
print("T3 · Parte 3 — Qdrant embebido: colección, upsert y query con filtro")
print("Entorno SIN red: no se intenta ningún servidor (paso de conexión saltado)")
print("=" * 72)

# 1) ¿Está qdrant-client? (import local; no hay ninguna comunicación de red)
import_ok, error_import = True, None
try:
    from qdrant_client import QdrantClient  # noqa: F401
except Exception as e:  # ImportError y cualquier fallo de import -> no disponible
    import_ok, error_import = False, f"{type(e).__name__}: {e}"

num_puntos, hits_out, modo = None, None, None
error_qdrant = None
flujo_qdrant_ok = False

if import_ok:
    try:
        num_puntos, hits_out, modo = flujo_qdrant_embebido()
        flujo_qdrant_ok = True
    except Exception as e:  # cualquier fallo -> réplica numpy local
        error_qdrant = f"{type(e).__name__}: {e}"
        print("Aviso: el camino qdrant-client falló ->", error_qdrant)

if num_puntos is None:
    if not import_ok:
        print("-" * 72)
        print("SALTO DOCUMENTADO de la parte Qdrant:")
        print(" * qdrant-client no está instalado en este entorno"
              + (f" ({error_import})" if error_import else ""))
        print(" * instalarlo con pip NO es posible aquí: pip necesita red y")
        print("   subprocess, ambos prohibidos en este entorno de ejecución.")
        print(" * el paso 'intentar conexión al servidor Qdrant local (puerto")
        print("   6333, timeout 3 s)' también se salta: la red está prohibida;")
        print("   se aplica el fallback local que el enunciado prevé.")
        print(" * para conservar las cifras esperadas se replica la mecánica")
        print("   EXACTA con numpy (embeddings sha256, colección de 6 puntos,")
        print("   coseno, filtro tema='pagos', limit=2; con 6 puntos el kNN")
        print("   exacto de Qdrant ':memory:' da las mismas cifras).")
        print("-" * 72)
    num_puntos, hits_out, modo = flujo_replica_numpy()

# ------------------------------------------------- reporte (formato del dado)
print()
print(f"Colección '{COLECCION}': {DIM} dims, Distance.{DISTANCIA}, "
      f"{num_puntos} puntos (upsert de {len(faqs)} FAQs)")
print(f"Consulta: {CONSULTA}")
print(f"Filtro: tema='{FILTRO_TEMA}'   limit={LIMIT}")
for h in hits_out:
    print(f"  {h['score']:.3f} [{h['tema']}] {h['texto']}")
print()
print("(Nota: con embeddings sintéticos el score no es semántico —")
print(" el objetivo aquí es la MECÁNICA de colección/upsert/query/filtro.)")

# ------------------------------------------------------------- resultados.json
resultados = {
    "subtarea": "T3_parte3_qdrant_embebido",
    "modo_ejecucion": modo,
    "red_utilizada": False,
    "conexion_servidor_intentada": False,
    "motivo_salto_conexion_servidor": (
        "Este entorno prohíbe toda comunicación de red; el paso del código dado "
        "'intentar conexión al servidor Qdrant local (puerto 6333, timeout 3 s)' "
        "se salta y se documenta, aplicando directamente el fallback local "
        "(modo embebido) que el propio enunciado prevé."
    ),
    "qdrant_client_importado": bool(import_ok),
    "flujo_qdrant_ejecutado": bool(flujo_qdrant_ok),
    "error_import_qdrant_client": error_import,
    "error_flujo_qdrant": error_qdrant,
    "instalacion_pip_intentada": False,
    "nota_instalacion_pip": (
        "El enunciado permite instalar qdrant-client con pip si falta, pero pip "
        "requiere red y subprocess, prohibidos en este entorno: no es posible "
        "instalarlo, se documenta el salto de la parte y se replica la mecánica "
        "con numpy (kNN coseno exacto + filtro de payload), que con 6 puntos "
        "coincide con Qdrant ':memory:'."
    ),
    "coleccion": COLECCION,
    "dimension_vectores": int(DIM),
    "distancia": DISTANCIA,
    "num_faqs_indexadas": int(len(faqs)),
    "consulta": CONSULTA,
    "filtro": {"tema": FILTRO_TEMA},
    "limit": int(LIMIT),
    "num_puntos_coleccion": int(num_puntos),
    "hits_filtro_pagos_scores": [float(h["score"]) for h in hits_out],
    "hits_filtro_pagos_ids": [int(h["id"]) for h in hits_out],
    "hits_filtro_pagos_temas": [h["tema"] for h in hits_out],
    "hits_filtro_pagos_textos": [h["texto"] for h in hits_out],
    "hits_filtro_pagos_detalle": [
        {"id": int(h["id"]), "score": float(h["score"]),
         "tema": h["tema"], "texto": h["texto"]}
        for h in hits_out
    ],
    "nota_score": (
        "Con embeddings sintéticos (sha256 -> normal(0,1) -> normalizado) el "
        "score NO es semántico; el objetivo es la mecánica "
        "colección/upsert/query/filtro."
    ),
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print()
print("Cifras guardadas en resultados.json:")
print("  num_puntos_coleccion     =", resultados["num_puntos_coleccion"])
print("  hits_filtro_pagos_scores =", resultados["hits_filtro_pagos_scores"])
print("  modo_ejecucion           =", resultados["modo_ejecucion"])
