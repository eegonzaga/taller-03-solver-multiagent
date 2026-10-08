# -*- coding: utf-8 -*-
"""
T3 · Parte 3 — Qdrant embebido con filtros de metadata.

Reproduce la mecánica de la Parte 3 de S2·MAR (colección -> upsert(vector+payload)
-> query con filtro de metadata) con las 6 FAQs del enunciado:

  · colección 'faqs' con 64 dimensiones y Distance.COSINE, recreada desde cero;
  · embeddings sintéticos deterministas embed_demo() (semilla vía
    hashlib.sha256 del texto, vectores con norma 1);
  · consulta '¿cuándo me devuelven el dinero?' con query_filter tema='pagos'
    y limit=2, imprimiendo score, tema y texto de cada hit;
  · verificación de que el filtro de metadata restringe de verdad los
    resultados (sin filtro se ven todos los temas; con filtro solo 'pagos';
    contra-prueba con filtro tema='cuenta').

CORRECCIÓN respecto al intento anterior: NO se intenta conectar a ningún
servidor (la URL http://localhost:6333 del código dado queda descartada:
este entorno prohíbe conexiones de red). Se usa directamente el modo
embebido local QdrantClient(':memory:'), que corre dentro del propio
proceso sin abrir ningún socket; ese es precisamente el fallback previsto
por el enunciado. Si qdrant-client no está instalado (ImportError) se
imprime el mensaje dado y la misma mecánica se replica localmente con
numpy (determinista, sin red) para que las cifras queden siempre guardadas.

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
DIM = 64                      # dimensión de los vectores
CONSULTA = "¿cuándo me devuelven el dinero?"
TEMA_FILTRO = "pagos"
LIMIT = 2


# ---------------------------------------------------------------------------
# Embeddings sintéticos deterministas (en el lab real: sentence-transformers).
# Semilla con hashlib.sha256 y NO hash(): hash() cambia entre procesos y, con
# una base persistente, los vectores deben ser iguales en cada corrida.
# ---------------------------------------------------------------------------
def embed_demo(texto):
    semilla = int.from_bytes(hashlib.sha256(texto.encode()).digest()[:4], "little")
    r = np.random.default_rng(semilla)
    v = r.normal(size=DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


# ---------------------------------------------------------------------------
# resultados.json: diccionario con las cifras de la parte
# ---------------------------------------------------------------------------
resultados = {
    "qdrant_client_instalado": False,
    "modo_qdrant_usado": None,
    "num_puntos_coleccion": None,
    "hits_con_filtro_tema_pagos": [],
}


def a_dict_hit(h):
    """ScoredPoint de qdrant -> dict simple serializable."""
    return {
        "id": int(h.id),
        "score": float(h.score),
        "tema": str(h.payload["tema"]),
        "texto": str(h.payload["texto"]),
    }


def registrar_hits(hits_dicts):
    resultados["hits_con_filtro_tema_pagos"] = [
        {
            "pos": i,
            "id": int(h["id"]),
            "score": float(h["score"]),
            "tema": str(h["tema"]),
            "texto": str(h["texto"]),
        }
        for i, h in enumerate(hits_dicts)
    ]


def verificar_filtro(hits_filtro, hits_sin_filtro, hits_cuenta):
    """Comprueba que el filtro de metadata restringe los resultados:
    (a) sin filtro (limit=6) se recuperan los 6 puntos, de varios temas;
    (b) con filtro tema='pagos' los hits son EXACTAMENTE los puntos 'pagos';
    (c) contra-prueba: filtro tema='cuenta' -> solo hits de 'cuenta'."""
    temas_sin = [h["tema"] for h in hits_sin_filtro]
    por_tema = {}
    for t in temas_sin:
        por_tema[t] = por_tema.get(t, 0) + 1
    otros_temas = sorted({t for t in temas_sin if t != TEMA_FILTRO})

    ids_hits = sorted(int(h["id"]) for h in hits_filtro)
    ids_esperados = sorted(i for i, (_, tema) in enumerate(faqs) if tema == TEMA_FILTRO)
    todos_pagos = bool(hits_filtro) and all(h["tema"] == TEMA_FILTRO for h in hits_filtro)
    ids_ok = ids_hits == ids_esperados
    todos_cuenta = bool(hits_cuenta) and all(h["tema"] == "cuenta" for h in hits_cuenta)
    restringe = bool(todos_pagos and ids_ok and otros_temas and todos_cuenta)

    print("\nVerificación del filtro de metadata:")
    print(f"  temas SIN filtro (limit={len(faqs)}):", temas_sin)
    print("  temas CON filtro tema='pagos':", [h["tema"] for h in hits_filtro])
    print("  ids CON filtro:", ids_hits, "| ids 'pagos' esperados:", ids_esperados)
    print("  contra-prueba filtro tema='cuenta':", [h["tema"] for h in hits_cuenta])
    print("  ¿el filtro restringe los resultados?:", restringe)

    return {
        "filtro_restringe_resultados": restringe,
        "todos_los_hits_son_tema_pagos": bool(todos_pagos),
        "ids_hits_coinciden_con_puntos_pagos": bool(ids_ok),
        "num_hits_filtro_pagos": len(hits_filtro),
        "ids_hits_filtro_pagos": ids_hits,
        "ids_esperados_tema_pagos": ids_esperados,
        "num_puntos_por_tema": por_tema,
        "temas_en_hits_sin_filtro_limit6": temas_sin,
        "otros_temas_visibles_sin_filtro": otros_temas,
        "contraprueba_filtro_tema_cuenta_todos_son_cuenta": bool(todos_cuenta),
        "hits_con_filtro_tema_cuenta": [
            {"id": int(h["id"]), "score": float(h["score"]), "tema": str(h["tema"])}
            for h in hits_cuenta
        ],
    }


def imprimir_nota():
    print("\n(Nota: con embeddings sintéticos el score no es semántico —")
    print(" el objetivo aquí es la MECÁNICA de colección/upsert/query/filtro.)")


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

ejecutado_con_qdrant = False

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
        # -- 1) modo embebido local, SIN red ---------------------------------
        # El código dado intenta antes un servidor en http://localhost:6333;
        # ese intento se omite (entorno sin red) y se usa directamente el
        # fallback previsto por el enunciado: Qdrant embebido ':memory:',
        # que corre en el propio proceso sin abrir ninguna conexión.
        cliente = QdrantClient(":memory:")
        resultados["modo_qdrant_usado"] = "embebido (:memory:)"
        print("Usando Qdrant embebido (:memory:) — proceso local, sin red.")

        # -- 2) recrear la colección desde cero (64 dims, COSINE) ------------
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

        # -- 3) upsert: id + vector (norma 1) + payload {texto, tema} --------
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
        print("Colección creada con", num_puntos, "puntos")

        # -- 4) consulta con filtro de metadata: solo tema="pagos" -----------
        qvec = embed_demo(CONSULTA)
        filtro_pagos = Filter(
            must=[FieldCondition(key="tema", match=MatchValue(value=TEMA_FILTRO))]
        )
        hits_filtro = [
            a_dict_hit(h) for h in ejecutar_query(cliente, qvec, filtro_pagos, LIMIT)
        ]

        print("\nConsulta:", CONSULTA)
        print("Filtro: tema='pagos' | limit=2 -> hits:")
        for h in hits_filtro:
            print(f"  {h['score']:.3f} [{h['tema']}] {h['texto']}")
        registrar_hits(hits_filtro)

        # -- 5) verificación de que el filtro restringe ----------------------
        hits_sin = [
            a_dict_hit(h) for h in ejecutar_query(cliente, qvec, None, len(faqs))
        ]
        filtro_cuenta = Filter(
            must=[FieldCondition(key="tema", match=MatchValue(value="cuenta"))]
        )
        hits_cuenta = [
            a_dict_hit(h) for h in ejecutar_query(cliente, qvec, filtro_cuenta, LIMIT)
        ]
        resultados["verificacion_filtro"] = verificar_filtro(
            hits_filtro, hits_sin, hits_cuenta
        )
        resultados["norma_vector_consulta"] = float(
            np.linalg.norm(np.asarray(qvec, dtype=np.float64))
        )

        imprimir_nota()
        ejecutado_con_qdrant = True

    except Exception as exc:
        resultados["error_qdrant"] = f"{type(exc).__name__}: {exc}"
        print("Aviso: qdrant-client falló en ejecución —", resultados["error_qdrant"])

# ---------------------------------------------------------------------------
# Fallback local (sin red, determinista): replica la MISMA mecánica
# colección/upsert/query/filtro con numpy. Se usa solo si qdrant-client
# no está instalado (ImportError ya avisado) o falló en ejecución; así las
# cifras de la parte quedan siempre medidas y guardadas en resultados.json.
# ---------------------------------------------------------------------------
if not ejecutado_con_qdrant:
    if qdrant_importado:
        resultados["modo_qdrant_usado"] = "emulacion_numpy_local (qdrant-client falló)"
    else:
        resultados["modo_qdrant_usado"] = (
            "emulacion_numpy_local (qdrant-client no instalado)"
        )
    print("\nReplicando la misma mecánica localmente con numpy (sin red, determinista).")

    # "colección" en memoria: upsert de puntos = id + vector norma 1 + payload
    coleccion = [
        {
            "id": i,
            "vector": np.asarray(embed_demo(t), dtype=np.float64),
            "payload": {"texto": t, "tema": tema},
        }
        for i, (t, tema) in enumerate(faqs)
    ]
    num_puntos = len(coleccion)
    resultados["num_puntos_coleccion"] = int(num_puntos)
    print("Colección creada con", num_puntos, "puntos")

    qvec = np.asarray(embed_demo(CONSULTA), dtype=np.float64)

    def query_emulada(filtro_tema, limit):
        """Filtro de metadata (must: tema == filtro_tema) + kNN coseno exacto.
        Con vectores norma 1 el coseno es el producto punto; se escribe la
        fórmula completa por claridad."""
        candidatos = [
            p for p in coleccion
            if filtro_tema is None or p["payload"]["tema"] == filtro_tema
        ]
        scored = []
        for p in candidatos:
            score = float(
                np.dot(qvec, p["vector"])
                / (np.linalg.norm(qvec) * np.linalg.norm(p["vector"]))
            )
            scored.append(
                {
                    "id": int(p["id"]),
                    "score": score,
                    "tema": str(p["payload"]["tema"]),
                    "texto": str(p["payload"]["texto"]),
                }
            )
        scored.sort(key=lambda d: (-d["score"], d["id"]))
        return scored[:limit]

    hits_filtro = query_emulada(TEMA_FILTRO, LIMIT)
    print("\nConsulta:", CONSULTA)
    print("Filtro: tema='pagos' | limit=2 -> hits:")
    for h in hits_filtro:
        print(f"  {h['score']:.3f} [{h['tema']}] {h['texto']}")
    registrar_hits(hits_filtro)

    hits_sin = query_emulada(None, len(faqs))
    hits_cuenta = query_emulada("cuenta", LIMIT)
    resultados["verificacion_filtro"] = verificar_filtro(
        hits_filtro, hits_sin, hits_cuenta
    )
    resultados["norma_vector_consulta"] = float(np.linalg.norm(qvec))

    imprimir_nota()

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
