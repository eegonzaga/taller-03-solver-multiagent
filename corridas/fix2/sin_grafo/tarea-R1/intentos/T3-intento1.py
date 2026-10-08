# -*- coding: utf-8 -*-
"""
T3 — Parte 3 — Qdrant embebido: colección, upsert y query con filtro.

Qué hace:
  1) Intenta conectar con un servidor Qdrant en http://localhost:6333 (timeout 3 s);
     si no responde, cae al modo embebido QdrantClient(":memory:") (sin Docker).
  2) Si qdrant-client no está instalado: este entorno prohíbe red y subprocess, por
     lo que NO es posible instalarlo con pip desde el script; se DOCUMENTA EL SALTO
     de la parte Qdrant y se replica la mecánica EXACTA con numpy (mismos embeddings
     deterministas sha256, misma similitud coseno, mismo filtro de payload), que
     coincide con Qdrant :memory: porque con 6 puntos la búsqueda es exacta.
  3) Crea la colección 'faqs' desde cero (64 dims, Distance.COSINE), hace upsert de
     los 6 FAQs (payload {texto, tema}) y consulta '¿cuándo me devuelven el dinero?'
     con query_filter tema='pagos' y limit=2.
  4) Guarda en resultados.json: num_puntos_coleccion y hits_filtro_pagos_scores
     (más el detalle de cada hit: score, tema, texto).

Nota: con embeddings sintéticos (sha256 -> normal(0,1) -> normalizado) el score NO
es semántico; el objetivo es la MECÁNICA de colección/upsert/query/filtro.
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
URL_SERVIDOR = "http://localhost:6333"
TIMEOUT_S = 3

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
    """Embedding determinista: sha256(texto) -> semilla -> normal(64) -> unitario.

    Se usa hashlib.sha256 y NO hash(): hash() cambia entre procesos y, con un
    servidor persistente, los vectores deben ser iguales en cada corrida.
    """
    semilla = int.from_bytes(hashlib.sha256(texto.encode()).digest()[:4], "little")
    r = np.random.default_rng(semilla)
    v = r.normal(size=DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


# ------------------------------------------------------- flujo Qdrant (código dado)
def flujo_qdrant():
    from qdrant_client import QdrantClient
    from qdrant_client.models import (Distance, VectorParams, PointStruct,
                                      Filter, FieldCondition, MatchValue)

    # Servidor Qdrant local (Docker en :6333); si no responde, modo embebido.
    try:
        cliente = QdrantClient(url=URL_SERVIDOR, timeout=TIMEOUT_S)
        cliente.get_collections()
        modo = "servidor_local_" + URL_SERVIDOR
        print("Conectado al servidor Qdrant en " + URL_SERVIDOR)
    except Exception as e:
        cliente = QdrantClient(":memory:")
        modo = "embebido_memory"
        print("Servidor no disponible — usando Qdrant embebido (:memory:)")
        print(f"  (motivo: {type(e).__name__})")

    # el servidor persiste entre corridas: recrear la colección desde cero
    try:
        existe = bool(cliente.collection_exists(COLECCION))
    except AttributeError:  # qdrant-client muy antiguo, sin collection_exists
        existe = any(c.name == COLECCION for c in cliente.get_collections().collections)
    if existe:
        cliente.delete_collection(COLECCION)

    cliente.create_collection(
        COLECCION,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    )
    cliente.upsert(COLECCION, points=[
        PointStruct(id=i, vector=embed_demo(t), payload={"texto": t, "tema": tema})
        for i, (t, tema) in enumerate(faqs)
    ])
    num_puntos = int(cliente.count(COLECCION).count)
    print("Colección creada con", num_puntos, "puntos")

    # consulta con filtro de metadata: solo tema="pagos"
    filtro = Filter(must=[FieldCondition(key="tema", match=MatchValue(value=FILTRO_TEMA))])
    if hasattr(cliente, "query_points"):        # API actual (qdrant-client >= 1.10)
        hits = cliente.query_points(
            COLECCION,
            query=embed_demo(CONSULTA),
            query_filter=filtro,
            limit=LIMIT,
        ).points
    else:                                       # API antigua (por robustez)
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
    return num_puntos, hits_out, modo


# --------------------- réplica numpy (solo si qdrant-client no está disponible)
def flujo_replica_numpy():
    """Réplica exacta de la mecánica con numpy: kNN coseno exacto + filtro payload.

    Con 6 puntos, Qdrant :memory: hace búsqueda exacta con distancia coseno, así
    que esta réplica produce las mismas cifras (salvo redondeo float32).
    """
    modo = "replica_numpy_sin_qdrant_client"
    V = np.array([embed_demo(t) for t, _ in faqs], dtype=np.float32)      # (6, 64)
    q = np.asarray(embed_demo(CONSULTA), dtype=np.float32)                # (64,)
    sims = (V @ q) / (np.linalg.norm(V, axis=1) * np.linalg.norm(q))      # coseno
    candidatos = [i for i, (_, tema) in enumerate(faqs) if tema == FILTRO_TEMA]
    top = sorted(candidatos, key=lambda i: (-float(sims[i]), i))[:LIMIT]
    hits_out = [
        {"id": int(i), "score": float(sims[i]),
         "tema": faqs[i][1], "texto": faqs[i][0]}
        for i in top
    ]
    return len(faqs), hits_out, modo


# --------------------------------------------------------------------- main
print("=" * 72)
print("T3 · Parte 3 — Qdrant embebido: colección, upsert y query con filtro")
print("=" * 72)

tiene_qdrant = True
error_import = None
try:
    from qdrant_client import QdrantClient  # noqa: F401  (chequeo de disponibilidad)
    from qdrant_client.models import Distance  # noqa: F401
except ImportError as e:
    tiene_qdrant = False
    error_import = f"{type(e).__name__}: {e}"

num_puntos, hits_out, modo, error_qdrant = None, None, None, None

if tiene_qdrant:
    try:
        num_puntos, hits_out, modo = flujo_qdrant()
    except Exception as e:
        error_qdrant = f"{type(e).__name__}: {e}"
        print("Aviso: el flujo Qdrant falló ->", error_qdrant)
        print("Se replica la mecánica con numpy para conservar las cifras esperadas.")

if num_puntos is None:
    if not tiene_qdrant:
        print("-" * 72)
        print("SALTO DOCUMENTADO — qdrant-client no está instalado y NO es posible")
        print("instalarlo con pip desde este script (pip requiere red y subprocess,")
        print("prohibidos en este entorno). Para no perder las cifras esperadas se")
        print("replica la mecánica EXACTA con numpy:")
        print("  * mismos embeddings deterministas (hashlib.sha256 -> normal(64)),")
        print("  * misma colección de 6 puntos con payload {texto, tema},")
        print("  * misma similitud coseno, filtro tema='pagos' y limit=2.")
        print("  (Con 6 puntos Qdrant :memory: hace kNN exacto: la réplica coincide.)")
        print("-" * 72)
    num_puntos, hits_out, modo = flujo_replica_numpy()

# ------------------------------------------------ reporte de hits (formato dado)
print()
print(f"Consulta: {CONSULTA}")
print(f"Filtro: tema='{FILTRO_TEMA}'   limit={LIMIT}")
for h in hits_out:
    print(f"  {h['score']:.3f} [{h['tema']}] {h['texto']}")
print()
print("(Nota: con embeddings sintéticos el score no es semántico —")
print(" el objetivo aquí es la MECÁNICA de colección/upsert/query/filtro.)")

# ------------------------------------------------------------- resultados.json
hits_scores = [float(h["score"]) for h in hits_out]
resultados = {
    "subtarea": "T3_parte3_qdrant_embebido",
    "modo_ejecucion": modo,
    "qdrant_client_disponible": bool(tiene_qdrant),
    "error_import_qdrant_client": error_import,
    "error_flujo_qdrant": error_qdrant,
    "instalacion_pip_intentada": False,
    "nota_instalacion": (
        "El enunciado permite instalar qdrant-client con pip si falta, pero este "
        "entorno prohíbe red y subprocess, por lo que la instalación automática no "
        "es posible; si faltaba el paquete se documenta el salto de la parte Qdrant "
        "y se replica la mecánica con numpy (kNN coseno exacto + filtro de payload), "
        "equivalente a Qdrant :memory: con 6 puntos."
    ),
    "coleccion": COLECCION,
    "dimension_vectores": int(DIM),
    "distancia": DISTANCIA,
    "num_faqs_indexadas": int(len(faqs)),
    "consulta": CONSULTA,
    "filtro": {"tema": FILTRO_TEMA},
    "limit": int(LIMIT),
    "num_puntos_coleccion": int(num_puntos),
    "hits_filtro_pagos_scores": hits_scores,
    "hits_filtro_pagos_temas": [h["tema"] for h in hits_out],
    "hits_filtro_pagos_textos": [h["texto"] for h in hits_out],
    "hits_filtro_pagos_detalle": [
        {"id": int(h["id"]), "score": float(h["score"]),
         "tema": h["tema"], "texto": h["texto"]}
        for h in hits_out
    ],
    "nota_score": (
        "Con embeddings sintéticos (sha256 -> normal(0,1) -> normalizado) el score "
        "NO es semántico; el objetivo es la mecánica colección/upsert/query/filtro."
    ),
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print()
print("Cifras guardadas en resultados.json:")
print("  num_puntos_coleccion     =", resultados["num_puntos_coleccion"])
print("  hits_filtro_pagos_scores =", resultados["hits_filtro_pagos_scores"])
print("  modo_ejecucion           =", resultados["modo_ejecucion"])
