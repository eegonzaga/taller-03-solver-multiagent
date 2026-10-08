# -*- coding: utf-8 -*-
"""
T3 — Parte 3: Qdrant embebido con filtros de metadata
=====================================================
Ejecuta la MECÁNICA del código dado por el profesor:

  1) Cliente Qdrant en modo EMBEBIDO QdrantClient(':memory:').
     CORRECCIÓN respecto al intento anterior: el código original intentaba
     primero conectarse a http://localhost:6333 (timeout 3). Ese intento de
     conexión se OMITE por completo porque el entorno de ejecución prohíbe
     cualquier uso de red. Se aplica directamente el fallback que el propio
     código dado documenta: el modo embebido ':memory:', que es 100% local,
     en proceso, y no abre ningún socket.
  2) Colección 'faqs' creada desde cero: 64 dimensiones, Distance.COSINE.
  3) Upsert de los 6 FAQs con embed_demo (semilla vía hashlib.sha256, NUNCA
     hash(), que cambia entre procesos) y payload {texto, tema}.
  4) Consulta '¿cuándo me devuelven el dinero?' con query_filter tema='pagos'
     y limit=2; se registran los hits (score, tema, texto) y el conteo.

Nota: con embeddings sintéticos el score NO es semántico; el objetivo es la
mecánica colección/upsert/query/filtro.

Si qdrant-client no está instalado (o falla al ejecutarse), se replica la
misma mecánica con un mini almacén vectorial local en memoria (misma métrica
COSINE, mismos vectores deterministas de embed_demo, mismo filtro), usando
solo los datos del enunciado y numpy, de modo que las cifras esperadas
(num_puntos_coleccion, hits_consulta_filtro_pagos) queden siempre registradas
en resultados.json, indicando qué motor las produjo.

Salidas:
  - resultados.json (carpeta actual, rutas relativas)
  - print() de las cifras principales
  - Esta parte NO produce figura PNG.
"""

import json
import hashlib
import numpy as np

# ----------------------------------------------------------------------
# Datos del enunciado: las 6 FAQs (texto, tema)
# ----------------------------------------------------------------------
faqs = [
    ("Para resetear tu contraseña entra a Configuración > Seguridad.", "cuenta"),
    ("Los reembolsos se procesan en 5 a 7 días hábiles.", "pagos"),
    ("Puedes exportar tus datos en formato CSV desde el panel.", "cuenta"),
    ("La factura electrónica se emite al confirmar el pago.", "pagos"),
    ("El soporte atiende de lunes a viernes de 9h a 18h.", "soporte"),
    ("La API permite 100 solicitudes por minuto en el plan básico.", "api"),
]

CONSULTA = "¿cuándo me devuelven el dinero?"
DIM = 64                    # dimensión declarada de la colección 'faqs'
NOMBRE_COLECCION = "faqs"
TEMA_FILTRO = "pagos"
LIMIT = 2


def embed_demo(texto):
    """Embedding sintético determinista por documento (en el lab real:
    sentence-transformers). Semilla con hashlib.sha256 y NO hash():
    hash() cambia entre procesos y, con un servidor persistente, los
    vectores deben ser iguales en cada corrida."""
    semilla = int.from_bytes(hashlib.sha256(texto.encode()).digest()[:4], "little")
    r = np.random.default_rng(semilla)
    v = r.normal(size=DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


# ----------------------------------------------------------------------
# Estructura de resultados (se guarda SIEMPRE en resultados.json)
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T3 - Parte 3: Qdrant embebido con filtros de metadata",
    "qdrant_client_disponible": False,
    "parte_ejecutada": False,
    "motor": None,
    "conexion_servidor": (
        "intento a http://localhost:6333 (timeout 3) OMITIDO: el entorno de "
        "ejecución prohíbe uso de red; se aplica directamente el fallback "
        "documentado del código dado, el modo embebido ':memory:' (local)"
    ),
    "coleccion": NOMBRE_COLECCION,
    "dimension_vectores": DIM,
    "distancia": "COSINE",
    "metodo_embedding": "embed_demo (semilla hashlib.sha256, no hash())",
    "consulta": CONSULTA,
    "filtro": {"tema": TEMA_FILTRO},
    "limit": LIMIT,
    "num_puntos_coleccion": None,
    "hits_consulta_filtro_pagos": [],
}


def registrar(motor, api, num_puntos, hits):
    """Guarda las cifras de la parte en el diccionario de resultados."""
    resultados.update(
        {
            "parte_ejecutada": True,
            "motor": motor,
            "api_consulta": api,
            "num_faqs_upsert": len(faqs),
            "num_puntos_coleccion": int(num_puntos),
            "hits_consulta_filtro_pagos": hits,
        }
    )


def imprimir_hits(hits):
    """Imprime los hits con el mismo formato del código dado."""
    for h in hits:
        print(f"  {h['score']:.3f} [{h['tema']}] {h['texto']}")
    print()
    print("(Nota: con embeddings sintéticos el score no es semántico —")
    print(" el objetivo aquí es la MECÁNICA de colección/upsert/query/filtro.)")


# ----------------------------------------------------------------------
# Import de qdrant-client: si no está, la mecánica se replica en local
# ----------------------------------------------------------------------
qdrant_ok = False
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
    )
    qdrant_ok = True
except ImportError:
    print(
        "qdrant-client no instalado — se replica la mecánica en local para "
        "registrar las cifras (instalable con `pip install qdrant-client`)."
    )

if qdrant_ok:
    resultados["qdrant_client_disponible"] = True
    try:
        # --------------------------------------------------------------
        # 1) Modo embebido ':memory:': 100% local, sin sockets ni red.
        #    (El intento al servidor de localhost se omite por política
        #    del entorno; queda documentado en resultados.json.)
        # --------------------------------------------------------------
        cliente = QdrantClient(":memory:")
        print(
            "Usando Qdrant embebido (:memory:) — el intento de conexión a "
            "http://localhost:6333 se omite (entorno sin red)."
        )

        # --------------------------------------------------------------
        # 2) Colección 'faqs' desde cero (el servidor persiste entre
        #    corridas: recrear la colección; en :memory: nunca existe)
        # --------------------------------------------------------------
        if cliente.collection_exists(NOMBRE_COLECCION):
            cliente.delete_collection(NOMBRE_COLECCION)
        cliente.create_collection(
            NOMBRE_COLECCION,
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        )

        # --------------------------------------------------------------
        # 3) Upsert de los 6 puntos: vector + payload {texto, tema}
        # --------------------------------------------------------------
        cliente.upsert(
            NOMBRE_COLECCION,
            points=[
                PointStruct(
                    id=i,
                    vector=embed_demo(t),
                    payload={"texto": t, "tema": tema},
                )
                for i, (t, tema) in enumerate(faqs)
            ],
        )
        num_puntos = int(cliente.count(NOMBRE_COLECCION).count)
        print("Colección creada con", num_puntos, "puntos")

        # --------------------------------------------------------------
        # 4) Consulta con filtro de metadata: solo tema="pagos", limit=2
        #    (API moderna query_points; fallback a search en versiones
        #    antiguas de qdrant-client, con los mismos parámetros)
        # --------------------------------------------------------------
        filtro_pagos = Filter(
            must=[FieldCondition(key="tema", match=MatchValue(value=TEMA_FILTRO))]
        )
        qvec = embed_demo(CONSULTA)
        if hasattr(cliente, "query_points"):
            hits_q = cliente.query_points(
                NOMBRE_COLECCION,
                query=qvec,
                query_filter=filtro_pagos,
                limit=LIMIT,
            ).points
            api_usada = "query_points"
        else:
            hits_q = cliente.search(
                collection_name=NOMBRE_COLECCION,
                query_vector=qvec,
                query_filter=filtro_pagos,
                limit=LIMIT,
            )
            api_usada = "search (compatibilidad con versiones antiguas)"

        print()
        hits = [
            {
                "id": int(h.id),
                "score": float(h.score),  # precisión completa, sin redondear
                "tema": str(h.payload["tema"]),
                "texto": str(h.payload["texto"]),
            }
            for h in hits_q
        ]
        imprimir_hits(hits)
        registrar("qdrant-client (modo embebido ':memory:')", api_usada, num_puntos, hits)

    except Exception as e:
        # Red de seguridad: un fallo de la librería no debe impedir registrar
        # las cifras; se replica la mecánica en local como respaldo.
        resultados["error_qdrant"] = repr(e)
        print("ERROR en qdrant-client:", repr(e))
        print("→ se replica la mecánica en local para registrar las cifras.")

# ----------------------------------------------------------------------
# Respaldo local (solo si qdrant-client no está disponible o falló):
# réplica fiel de la mecánica colección → upsert(vector+payload) →
# query con filtro, con la misma métrica COSINE y los mismos vectores
# deterministas de embed_demo. Usa únicamente datos del enunciado.
# ----------------------------------------------------------------------
if not resultados["parte_ejecutada"]:
    print()
    print("Réplica local en memoria de la mecánica Qdrant:")
    # "colección desde cero": 64 dims, métrica COSINE (aquí, lista vacía)
    coleccion = []
    for i, (t, tema) in enumerate(faqs):  # upsert: punto = vector + payload
        coleccion.append(
            {"id": i, "vector": embed_demo(t), "payload": {"texto": t, "tema": tema}}
        )
    num_puntos = len(coleccion)
    print("Colección creada con", num_puntos, "puntos")

    # query con filtro de metadata: solo tema="pagos"
    candidatos = [p for p in coleccion if p["payload"]["tema"] == TEMA_FILTRO]
    q = np.asarray(embed_demo(CONSULTA), dtype=np.float32)
    scored = []
    for p in candidatos:
        v = np.asarray(p["vector"], dtype=np.float32)
        # score COSINE de Qdrant = similitud coseno (1 - distancia coseno)
        score = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v)))
        scored.append((score, p))
    scored.sort(key=lambda sp: sp[0], reverse=True)
    hits = [
        {
            "id": int(p["id"]),
            "score": s,  # precisión completa, sin redondear
            "tema": str(p["payload"]["tema"]),
            "texto": str(p["payload"]["texto"]),
        }
        for s, p in scored[:LIMIT]
    ]
    print()
    imprimir_hits(hits)
    registrar(
        "réplica local en memoria (qdrant-client no disponible; misma métrica "
        "COSINE y mismos vectores embed_demo sobre los datos del enunciado)",
        "filtro de metadata + similitud coseno en numpy",
        num_puntos,
        hits,
    )

# ----------------------------------------------------------------------
# Guardar TODAS las cifras en resultados.json (carpeta actual)
# ----------------------------------------------------------------------
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# Resumen de cifras principales
# ----------------------------------------------------------------------
print()
print("=" * 70)
print("Cifras guardadas en resultados.json")
print("=" * 70)
print("  motor                      =", resultados["motor"])
print("  num_puntos_coleccion       =", resultados["num_puntos_coleccion"])
print("  hits_consulta_filtro_pagos =")
for h in resultados["hits_consulta_filtro_pagos"]:
    print(f"    id={h['id']}  score={h['score']!r}  [{h['tema']}]  {h['texto']}")
if not resultados["hits_consulta_filtro_pagos"]:
    print("    (sin hits — ver resultados.json)")
