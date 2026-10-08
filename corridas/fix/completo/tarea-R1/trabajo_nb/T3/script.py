# -*- coding: utf-8 -*-
"""
T3 — Parte 3: Qdrant embebido con filtros de metadata
=====================================================
Ejecuta el código dado por el profesor tal cual:

  1) Intento de conexión a http://localhost:6333 (timeout 3); si el servidor
     no responde, fallback a QdrantClient(':memory:') — modo embebido, 100%
     local y en proceso. La excepción de conexión se captura, así que el
     bloque es seguro incluso sin red: produce exactamente las mismas cifras
     (6 puntos, mismos hits) que el modo embebido directo.
  2) Colección 'faqs' creada desde cero: 64 dimensiones, Distance.COSINE.
  3) Upsert de los 6 FAQs con embed_demo (semilla vía hashlib.sha256, NUNCA
     hash(), que cambia entre procesos) y payload {texto, tema}.
  4) Consulta '¿cuándo me devuelven el dinero?' con query_filter tema='pagos'
     y limit=2; se registran los hits (score, tema, texto) y el conteo.

Nota: con embeddings sintéticos el score NO es semántico; el objetivo es la
mecánica colección/upsert/query/filtro.

La parte se salta sola si qdrant-client no está instalado: se imprime el
mensaje de salto y se registra parte_ejecutada=false con el motivo en
resultados.json.

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
    "motivo_salto": None,
    "error": None,
    "motor": None,
    "conexion_servidor": None,
    "coleccion": NOMBRE_COLECCION,
    "dimension_vectores": DIM,
    "distancia": "COSINE",
    "metodo_embedding": "embed_demo (semilla hashlib.sha256, no hash())",
    "consulta": CONSULTA,
    "filtro": {"tema": TEMA_FILTRO},
    "limit": LIMIT,
    "num_faqs_upsert": len(faqs),
    "num_puntos_coleccion": None,
    "hits_consulta_filtro_pagos": [],
}

# ----------------------------------------------------------------------
# Import de qdrant-client: si no está, la parte se salta sola
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

if not qdrant_ok:
    # Salto especificado: qdrant-client no instalado
    resultados["motivo_salto"] = (
        "qdrant-client no instalado — `pip install qdrant-client` para esta parte."
    )
    print(resultados["motivo_salto"])
else:
    resultados["qdrant_client_disponible"] = True
    try:
        # ----------------------------------------------------------
        # 1) Servidor Qdrant local (Docker en :6333); si no responde,
        #    modo embebido. Bloque de conexión del código dado, tal cual.
        # ----------------------------------------------------------
        try:
            cliente = QdrantClient(url="http://localhost:6333", timeout=3)
            cliente.get_collections()
            print("Conectado al servidor Qdrant en http://localhost:6333")
            resultados["conexion_servidor"] = "http://localhost:6333"
        except Exception:
            cliente = QdrantClient(":memory:")
            print("Servidor no disponible — usando Qdrant embebido (:memory:)")
            resultados["conexion_servidor"] = "Qdrant embebido (:memory:)"

        # ----------------------------------------------------------
        # 2) Colección 'faqs' desde cero (el servidor persiste entre
        #    corridas: recrear la colección)
        # ----------------------------------------------------------
        if cliente.collection_exists(NOMBRE_COLECCION):
            cliente.delete_collection(NOMBRE_COLECCION)
        cliente.create_collection(
            NOMBRE_COLECCION,
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        )

        # ----------------------------------------------------------
        # 3) Upsert de los 6 puntos: vector + payload {texto, tema}
        # ----------------------------------------------------------
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

        # ----------------------------------------------------------
        # 4) Consulta con filtro de metadata: solo tema="pagos", limit=2
        #    (API moderna query_points; fallback a search en versiones
        #    antiguas de qdrant-client, con los mismos parámetros)
        # ----------------------------------------------------------
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
        for h in hits:
            print(f"  {h['score']:.3f} [{h['tema']}] {h['texto']}")
        print()
        print("(Nota: con embeddings sintéticos el score no es semántico —")
        print(" el objetivo aquí es la MECÁNICA de colección/upsert/query/filtro.)")

        resultados.update(
            {
                "parte_ejecutada": True,
                "motor": "qdrant-client",
                "api_consulta": api_usada,
                "num_puntos_coleccion": int(num_puntos),
                "hits_consulta_filtro_pagos": hits,
            }
        )
    except Exception as e:
        # Un fallo de la librería no debe impedir guardar resultados.json
        resultados["error"] = repr(e)
        print("ERROR en la ejecución de qdrant-client:", repr(e))

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
print("  conexion_servidor          =", resultados["conexion_servidor"])
print("  parte_ejecutada            =", resultados["parte_ejecutada"])
if resultados["motivo_salto"]:
    print("  motivo_salto               =", resultados["motivo_salto"])
if resultados["error"]:
    print("  error                      =", resultados["error"])
print("  num_puntos_coleccion       =", resultados["num_puntos_coleccion"])
print("  hits_consulta_filtro_pagos =")
for h in resultados["hits_consulta_filtro_pagos"]:
    print(f"    id={h['id']}  score={h['score']!r}  [{h['tema']}]  {h['texto']}")
if not resultados["hits_consulta_filtro_pagos"]:
    print("    (sin hits — ver resultados.json)")
