# -*- coding: utf-8 -*-
"""
T3 · Parte 3 — Qdrant embebido con filtros de metadata.

Ejecuta el código dado de la sesión S2·MAR (Parte 3):
  · intenta un servidor Qdrant en http://localhost:6333 (timeout 3 s) y,
    si no responde, cae al modo embebido QdrantClient(':memory:');
  · recrea la colección 'faqs' (64 dims, Distance.COSINE) desde cero;
  · hace upsert de las 6 FAQs con payload {texto, tema} y embeddings
    sintéticos deterministas (semilla vía hashlib.sha256, vectores norma 1);
  · consulta '¿cuándo me devuelven el dinero?' con query_filter tema='pagos'
    y limit=2, imprimiendo score, tema y texto de los hits;
  · verifica que el filtro de metadata efectivamente restringe resultados.

Si qdrant-client no está instalado, la parte se salta sola (ImportError).
No se requiere figura PNG. Cifras -> resultados.json (carpeta actual).
"""

import json
import hashlib

import numpy as np

# ---------------------------------------------------------------------------
# Datos del enunciado: 6 FAQs (texto, tema)
# ---------------------------------------------------------------------------
faqs = [
    ("Para resetear tu contraseña entra a Configuración > Seguridad.", "cuenta"),
    ("Los reembolsos se procesan en 5 a 7 días hábiles.", "pagos"),
    ("Puedes exportar tus datos en formato CSV desde el panel.", "cuenta"),
    ("La factura electrónica se emite al confirmar el pago.", "pagos"),
    ("El soporte atiende de lunes a viernes de 9h a 18h.", "soporte"),
    ("La API permite 100 solicitudes por minuto en el plan básico.", "api"),
]

NOMBRE_COLECCION = "faqs"
DIM = 64
CONSULTA = "¿cuándo me devuelven el dinero?"


# ---------------------------------------------------------------------------
# Embeddings sintéticos deterministas (en el lab real: sentence-transformers).
# Semilla con hashlib.sha256 y NO hash(): hash() cambia entre procesos y, con
# un servidor persistente, los vectores deben ser iguales en cada corrida.
# ---------------------------------------------------------------------------
def embed_demo(texto):
    semilla = int.from_bytes(hashlib.sha256(texto.encode()).digest()[:4], "little")
    r = np.random.default_rng(semilla)
    v = r.normal(size=DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


# ---------------------------------------------------------------------------
# resultados.json: se escribe SIEMPRE, también si qdrant-client falta.
# ---------------------------------------------------------------------------
resultados = {
    "qdrant_client_instalado": False,
    "modo_qdrant_usado": None,
    "num_puntos_coleccion": None,
    "hits_con_filtro_tema_pagos": [],
}

# ---------------------------------------------------------------------------
# Import opcional de qdrant-client (la parte se salta sola si no está).
# ---------------------------------------------------------------------------
qdrant_importado = False
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
    qdrant_importado = True
except ImportError:
    print("qdrant-client no instalado — `pip install qdrant-client` para esta parte.")

if qdrant_importado:
    resultados["qdrant_client_instalado"] = True

    def ejecutar_query(cliente, vector, filtro, limit):
        """query_points (API actual); fallback a search() en versiones antiguas."""
        try:
            resp = cliente.query_points(
                NOMBRE_COLECCION,
                query=vector,
                query_filter=filtro,
                limit=limit,
                with_payload=True,
            )
            return list(resp.points)
        except (AttributeError, TypeError):
            kwargs = dict(
                collection_name=NOMBRE_COLECCION,
                query_vector=vector,
                limit=limit,
                with_payload=True,
            )
            if filtro is not None:
                kwargs["query_filter"] = filtro
            return list(cliente.search(**kwargs))

    try:
        # -- 1) servidor local (timeout 3 s); si no responde, modo embebido ---
        servidor_disponible = False
        try:
            cliente = QdrantClient(url="http://localhost:6333", timeout=3)
            cliente.get_collections()
            servidor_disponible = True
            modo = "servidor_local (http://localhost:6333)"
            print("Conectado al servidor Qdrant en http://localhost:6333")
        except Exception:
            cliente = QdrantClient(":memory:")
            modo = "embebido (:memory:)"
            print("Servidor no disponible — usando Qdrant embebido (:memory:)")
        resultados["modo_qdrant_usado"] = modo
        resultados["servidor_local_disponible"] = servidor_disponible

        # -- 2) recrear la colección desde cero (el servidor persiste) --------
        try:
            existe = cliente.collection_exists(NOMBRE_COLECCION)
        except AttributeError:
            existe = any(
                c.name == NOMBRE_COLECCION
                for c in cliente.get_collections().collections
            )
        if existe:
            cliente.delete_collection(NOMBRE_COLECCION)
        cliente.create_collection(
            NOMBRE_COLECCION,
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        )

        # -- 3) upsert: id + vector (norma 1) + payload {texto, tema} ---------
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
        resultados["num_puntos_coleccion"] = num_puntos
        resultados["normas_vectores_faqs"] = [
            float(np.linalg.norm(np.asarray(embed_demo(t), dtype=np.float64)))
            for t, _ in faqs
        ]
        print("Colección creada con", num_puntos, "puntos")

        # -- 4) consulta con filtro de metadata: solo tema="pagos" ------------
        qvec = embed_demo(CONSULTA)
        filtro_pagos = Filter(
            must=[FieldCondition(key="tema", match=MatchValue(value="pagos"))]
        )
        hits = ejecutar_query(cliente, qvec, filtro_pagos, limit=2)

        print("\nConsulta:", CONSULTA)
        print("Filtro: tema='pagos' | limit=2 -> hits:")
        for h in hits:
            print(f"  {h.score:.3f} [{h.payload['tema']}] {h.payload['texto']}")

        resultados["hits_con_filtro_tema_pagos"] = [
            {
                "pos": i,
                "id": int(h.id),
                "score": float(h.score),
                "tema": str(h.payload["tema"]),
                "texto": str(h.payload["texto"]),
            }
            for i, h in enumerate(hits)
        ]
        resultados["norma_vector_consulta"] = float(
            np.linalg.norm(np.asarray(qvec, dtype=np.float64))
        )

        # -- 5) verificación: ¿el filtro restringe de verdad? -----------------
        # (a) sin filtro y limit=6 se devuelven los 6 puntos (todos los temas):
        hits_sin_filtro = ejecutar_query(cliente, qvec, None, limit=len(faqs))
        temas_sin_filtro = [str(h.payload["tema"]) for h in hits_sin_filtro]
        puntos_por_tema = {}
        for t in temas_sin_filtro:
            puntos_por_tema[t] = puntos_por_tema.get(t, 0) + 1
        otros_temas = sorted({t for t in temas_sin_filtro if t != "pagos"})

        # (b) con filtro solo 'pagos' y exactamente los ids de tema 'pagos':
        ids_hits = sorted(int(h.id) for h in hits)
        ids_pagos_esperados = sorted(
            i for i, (_, tema) in enumerate(faqs) if tema == "pagos"
        )
        todos_pagos = bool(hits) and all(h.payload["tema"] == "pagos" for h in hits)
        ids_correctos = ids_hits == ids_pagos_esperados

        # (c) contra-prueba: filtro tema='cuenta' -> solo hits de 'cuenta':
        filtro_cuenta = Filter(
            must=[FieldCondition(key="tema", match=MatchValue(value="cuenta"))]
        )
        hits_cuenta = ejecutar_query(cliente, qvec, filtro_cuenta, limit=2)
        todos_cuenta = bool(hits_cuenta) and all(
            h.payload["tema"] == "cuenta" for h in hits_cuenta
        )

        filtro_restringe = bool(
            todos_pagos and ids_correctos and otros_temas and todos_cuenta
        )

        print("\nVerificación del filtro de metadata:")
        print("  temas SIN filtro (limit=6):", temas_sin_filtro)
        print("  temas CON filtro tema='pagos':", [h.payload["tema"] for h in hits])
        print("  ids CON filtro:", ids_hits,
              "| ids 'pagos' esperados:", ids_pagos_esperados)
        print("  contra-prueba filtro tema='cuenta':",
              [h.payload["tema"] for h in hits_cuenta])
        print("  ¿el filtro restringe los resultados?:", filtro_restringe)

        resultados["verificacion_filtro"] = {
            "filtro_restringe_resultados": filtro_restringe,
            "todos_los_hits_son_tema_pagos": bool(todos_pagos),
            "ids_hits_coinciden_con_puntos_pagos": bool(ids_correctos),
            "num_hits_filtro_pagos": len(hits),
            "ids_hits_filtro_pagos": ids_hits,
            "ids_esperados_tema_pagos": ids_pagos_esperados,
            "num_puntos_por_tema": puntos_por_tema,
            "temas_en_hits_sin_filtro_limit6": temas_sin_filtro,
            "otros_temas_visibles_sin_filtro": otros_temas,
            "contraprueba_filtro_tema_cuenta_todos_son_cuenta": bool(todos_cuenta),
            "hits_con_filtro_tema_cuenta": [
                {
                    "id": int(h.id),
                    "score": float(h.score),
                    "tema": str(h.payload["tema"]),
                }
                for h in hits_cuenta
            ],
        }

        print("\n(Nota: con embeddings sintéticos el score no es semántico —")
        print(" el objetivo aquí es la MECÁNICA de colección/upsert/query/filtro.)")

    except Exception as exc:
        resultados["error_ejecucion"] = f"{type(exc).__name__}: {exc}"
        print("Error en la parte Qdrant:", resultados["error_ejecucion"])
else:
    resultados["modo_qdrant_usado"] = "no_ejecutado (qdrant-client no instalado)"

# ---------------------------------------------------------------------------
# Guardar resultados.json (siempre) e imprimir cifras principales
# ---------------------------------------------------------------------------
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print("\n=== Cifras principales (T3 · Parte 3: Qdrant embebido + filtros) ===")
print("modo_qdrant_usado:", resultados["modo_qdrant_usado"])
print("num_puntos_coleccion:", resultados["num_puntos_coleccion"])
print("hits_con_filtro_tema_pagos:")
for h in resultados["hits_con_filtro_tema_pagos"]:
    print(f"  {h['score']:.3f} [{h['tema']}] {h['texto']}")
if "verificacion_filtro" in resultados:
    print("filtro_restringe_resultados:",
          resultados["verificacion_filtro"]["filtro_restringe_resultados"])
print("Resultados guardados en resultados.json")
