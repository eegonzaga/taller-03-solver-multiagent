# -*- coding: utf-8 -*-
"""
T3 · Parte 3 — Qdrant embebido con filtros de metadata
=======================================================
Ejecuta el código dado de la Parte 3 del notebook S2·MAR (MMIA 6013 · Semana 2):

  1. Intenta conectar a un servidor Qdrant en http://localhost:6333 (timeout 3 s);
     si no responde, cae a modo embebido QdrantClient(":memory:").
  2. Crea la colección 'faqs' (64 dimensiones, distancia COSINE).
  3. Upsert de las 6 FAQs con payload {texto, tema}, con embeddings sintéticos
     deterministas de embed_demo (semilla vía hashlib.sha256 — nunca hash(),
     que cambia entre procesos).
  4. Consulta "¿cuándo me devuelven el dinero?" con filtro de metadata
     tema='pagos' y limit=2.
  5. Si qdrant-client no está instalado, la parte se salta sola
     (try/except ImportError) sin romper la ejecución.

Cifras registradas en resultados.json:
  - modo_qdrant          : 'servidor_local_6333' | 'embebido_memory' | 'no_instalado'
  - num_puntos_coleccion : número de puntos en la colección 'faqs'
  - hits_filtro_pagos    : lista de hits [{id, score, tema, texto}, ...]

Figura PNG: esta parte no requiere figura.
"""

import json
import hashlib

import numpy as np

# ----------------------------------------------------------------------
# Datos: las 6 FAQs del enunciado (texto, tema)
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
DIM = 64  # dimensión de los vectores de la colección

# Diccionario de resultados (valores por defecto si la parte se salta)
resultados = {
    "modo_qdrant": "no_instalado",
    "num_puntos_coleccion": None,
    "hits_filtro_pagos": [],
    "consulta": CONSULTA,
    "filtro_tema": "pagos",
    "limit": 2,
    "dims_coleccion": DIM,
    "distancia": "COSINE",
    "num_faqs_upsert": len(faqs),
    "qdrant_ejecutado": False,
}

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
    import qdrant_client as _qc

    resultados["qdrant_client_version"] = str(getattr(_qc, "__version__", "desconocida"))

    # ------------------------------------------------------------------
    # Embeddings sintéticos por documento (en el lab: sentence-transformers).
    # Semilla con hashlib y no hash(): hash() cambia entre procesos, y con un
    # servidor persistente los vectores deben ser iguales en cada corrida.
    # ------------------------------------------------------------------
    def embed_demo(texto):
        semilla = int.from_bytes(hashlib.sha256(texto.encode()).digest()[:4], "little")
        r = np.random.default_rng(semilla)
        v = r.normal(size=DIM).astype(np.float32)
        return (v / np.linalg.norm(v)).tolist()

    # ------------------------------------------------------------------
    # Servidor Qdrant local (Docker en :6333); si no responde, modo embebido.
    # ------------------------------------------------------------------
    try:
        cliente = QdrantClient(url="http://localhost:6333", timeout=3)
        cliente.get_collections()
        resultados["modo_qdrant"] = "servidor_local_6333"
        print("Conectado al servidor Qdrant en http://localhost:6333")
    except Exception:
        cliente = QdrantClient(":memory:")
        resultados["modo_qdrant"] = "embebido_memory"
        print("Servidor no disponible — usando Qdrant embebido (:memory:)")

    # el servidor persiste entre corridas: recrear la colección desde cero
    if cliente.collection_exists("faqs"):
        cliente.delete_collection("faqs")

    cliente.create_collection(
        "faqs",
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    )

    cliente.upsert(
        "faqs",
        points=[
            PointStruct(
                id=i,
                vector=embed_demo(t),
                payload={"texto": t, "tema": tema},
            )
            for i, (t, tema) in enumerate(faqs)
        ],
        wait=True,  # asegurar que los puntos quedan antes de contar/consultar
    )

    num_puntos = int(cliente.count("faqs").count)
    resultados["num_puntos_coleccion"] = num_puntos
    print("Colección creada con", num_puntos, "puntos")

    # ------------------------------------------------------------------
    # Consulta con filtro de metadata: solo tema="pagos"
    # ------------------------------------------------------------------
    filtro_pagos = Filter(
        must=[FieldCondition(key="tema", match=MatchValue(value="pagos"))]
    )
    vector_consulta = embed_demo(CONSULTA)

    if hasattr(cliente, "query_points"):  # API moderna (qdrant-client >= 1.10)
        hits = cliente.query_points(
            "faqs",
            query=vector_consulta,
            query_filter=filtro_pagos,
            limit=2,
        ).points
    else:  # compatibilidad con versiones antiguas del cliente
        hits = cliente.search(
            collection_name="faqs",
            query_vector=vector_consulta,
            query_filter=filtro_pagos,
            limit=2,
        )

    hits_reg = [
        {
            "id": int(h.id),
            "score": float(h.score),
            "tema": str(h.payload["tema"]),
            "texto": str(h.payload["texto"]),
        }
        for h in hits
    ]
    resultados["hits_filtro_pagos"] = hits_reg
    resultados["qdrant_ejecutado"] = True

    for h in hits_reg:
        print(f"  {h['score']:.3f} [{h['tema']}] {h['texto']}")

    print()
    print("(Nota: con embeddings sintéticos el score no es semántico —")
    print(" el objetivo aquí es la MECÁNICA de colección/upsert/query/filtro.)")

except ImportError as e:
    # qdrant-client no instalado: la parte se salta sola, sin romper el notebook
    print("qdrant-client no instalado — `pip install qdrant-client` para esta parte.")
    print(f"  (detalle: {e})")
    resultados["modo_qdrant"] = "no_instalado"
except Exception as e:
    # cualquier otro fallo de la parte Qdrant tampoco debe romper la ejecución
    print(f"Aviso: la parte Qdrant falló ({e!r}); se registran los valores parciales.")
    resultados["error"] = repr(e)

# ----------------------------------------------------------------------
# Guardar resultados (siempre, haya o no qdrant-client)
# ----------------------------------------------------------------------
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("\n=== Cifras principales (T3 · Parte 3) ===")
print("modo_qdrant:", resultados["modo_qdrant"])
print("num_puntos_coleccion:", resultados["num_puntos_coleccion"])
print("hits_filtro_pagos:")
if resultados["hits_filtro_pagos"]:
    for h in resultados["hits_filtro_pagos"]:
        print(f"  score={h['score']:.6f} | tema={h['tema']} | texto={h['texto']}")
else:
    print("  (sin hits — parte omitida o qdrant-client no disponible)")
print("Resultados guardados en resultados.json")
