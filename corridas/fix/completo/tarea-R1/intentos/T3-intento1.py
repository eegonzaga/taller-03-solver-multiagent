# -*- coding: utf-8 -*-
"""
T3 — Parte 3: Qdrant embebido con filtros de metadata
=====================================================
Ejecuta el código dado por el profesor:

  1) Intenta conectar a un servidor Qdrant en http://localhost:6333 (timeout 3 s);
     si no responde, cae al modo embebido QdrantClient(':memory:').
  2) Crea la colección 'faqs' DESDE CERO (64 dims, Distance.COSINE).
  3) Upsert de los 6 FAQs con embeddings sintéticos embed_demo
     (semilla vía hashlib.sha256 — NUNCA hash(), que cambia entre procesos)
     y payload {texto, tema}.
  4) Consulta '¿cuándo me devuelven el dinero?' con query_filter tema='pagos'
     y limit=2; registra los hits (score, tema, texto) y el conteo de puntos.

Nota: con embeddings sintéticos el score NO es semántico; el objetivo es la
MECÁNICA de colección/upsert/query/filtro. La parte se salta sola si
qdrant-client no está instalado.

Salidas:
  - resultados.json : num_puntos_coleccion, hits_consulta_filtro_pagos, etc.
  - print() de las cifras principales.
  - Esta parte NO requiere figura PNG.
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
DIM = 64  # dimensión de los vectores de la colección 'faqs'


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
    "num_puntos_coleccion": None,
    "hits_consulta_filtro_pagos": [],
}

# ----------------------------------------------------------------------
# Import de qdrant-client: si no está instalado, la parte se salta sola
# ----------------------------------------------------------------------
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
    qdrant_ok = False
    print("qdrant-client no instalado — `pip install qdrant-client` para esta parte.")

if qdrant_ok:
    try:
        # --------------------------------------------------------------
        # 1) Servidor Qdrant local (Docker en :6333); si no responde,
        #    modo embebido :memory:
        # --------------------------------------------------------------
        try:
            cliente = QdrantClient(url="http://localhost:6333", timeout=3)
            cliente.get_collections()
            modo = "servidor_local (http://localhost:6333)"
            print("Conectado al servidor Qdrant en http://localhost:6333")
        except Exception:
            cliente = QdrantClient(":memory:")
            modo = "embebido (:memory:)"
            print("Servidor no disponible — usando Qdrant embebido (:memory:)")

        # --------------------------------------------------------------
        # 2) Colección 'faqs' desde cero (el servidor persiste entre
        #    corridas: recrear la colección)
        # --------------------------------------------------------------
        if cliente.collection_exists("faqs"):
            cliente.delete_collection("faqs")
        cliente.create_collection(
            "faqs",
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        )

        # --------------------------------------------------------------
        # 3) Upsert de los 6 puntos: vector + payload {texto, tema}
        # --------------------------------------------------------------
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
        )
        num_puntos = int(cliente.count("faqs").count)
        print("Colección creada con", num_puntos, "puntos")

        # --------------------------------------------------------------
        # 4) Consulta con filtro de metadata: solo tema="pagos", limit=2
        #    (API moderna query_points; fallback a search en versiones
        #    antiguas de qdrant-client, con los mismos parámetros)
        # --------------------------------------------------------------
        filtro_pagos = Filter(
            must=[FieldCondition(key="tema", match=MatchValue(value="pagos"))]
        )
        if hasattr(cliente, "query_points"):
            hits = cliente.query_points(
                "faqs",
                query=embed_demo(CONSULTA),
                query_filter=filtro_pagos,
                limit=2,
            ).points
            api_usada = "query_points"
        else:
            hits = cliente.search(
                collection_name="faqs",
                query_vector=embed_demo(CONSULTA),
                query_filter=filtro_pagos,
                limit=2,
            )
            api_usada = "search (compatibilidad versiones antiguas)"

        print()
        for h in hits:
            print(f"  {h.score:.3f} [{h.payload['tema']}] {h.payload['texto']}")
        print()
        print("(Nota: con embeddings sintéticos el score no es semántico —")
        print(" el objetivo aquí es la MECÁNICA de colección/upsert/query/filtro.)")

        # Registrar los hits con precisión completa (sin redondear)
        hits_reg = [
            {
                "id": int(h.id),
                "score": float(h.score),
                "tema": str(h.payload["tema"]),
                "texto": str(h.payload["texto"]),
            }
            for h in hits
        ]

        resultados.update(
            {
                "qdrant_client_disponible": True,
                "parte_ejecutada": True,
                "modo_conexion": modo,
                "coleccion": "faqs",
                "dimension_vectores": DIM,
                "distancia": "COSINE",
                "num_faqs_upsert": len(faqs),
                "metodo_embedding": "embed_demo (semilla hashlib.sha256, no hash())",
                "api_consulta": api_usada,
                "consulta": CONSULTA,
                "filtro": {"tema": "pagos"},
                "limit": 2,
                "num_puntos_coleccion": num_puntos,
                "hits_consulta_filtro_pagos": hits_reg,
            }
        )
    except Exception as e:
        # Red de seguridad: cualquier fallo de la librería no debe impedir
        # guardar resultados.json; la parte queda marcada como no ejecutada.
        resultados["error_ejecucion"] = repr(e)
        print("ERROR durante la ejecución de Qdrant:", repr(e))

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
print("  num_puntos_coleccion       =", resultados["num_puntos_coleccion"])
print("  hits_consulta_filtro_pagos =")
for h in resultados["hits_consulta_filtro_pagos"]:
    print(
        f"    id={h['id']}  score={h['score']:.6f}  [{h['tema']}]  {h['texto']}"
    )
if not resultados["hits_consulta_filtro_pagos"]:
    print("    (sin hits — parte omitida o con error; ver resultados.json)")
