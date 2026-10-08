# -*- coding: utf-8 -*-
"""
T1 · Parte 3 — Qdrant embebido: colección, upsert y query con filtro
MMIA 6013 · Semana 2 · Martes (S2·MAR)

CORRECCIÓN respecto al intento anterior (rechazado):
  Este entorno PROHÍBE conexiones de red. Por eso NO se intenta conectar a un
  servidor Qdrant local (puerto 6333), ni siquiera en localhost: cualquier
  intento de conexión, aunque sea local, usa sockets y fue rechazado.
  Comportamiento corregido:
    1) Si qdrant-client está instalado -> se usa DIRECTAMENTE
       QdrantClient(":memory:") (modo embebido, 100% en proceso, sin sockets),
       que es exactamente el modo que el enunciado prevé cuando el servidor no
       responde. Se registra modo_qdrant="embebido" y se anota que la prueba
       del servidor se omitió por la política sin-red del entorno.
    2) Si qdrant-client no está instalado -> 'pip install qdrant-client' NO
       puede intentarse (red y subprocess prohibidos), así que se registra
       modo_qdrant="omitido" (la parte "se salta sola", previsto por el
       enunciado) y, para no dejar la subtarea sin cifras, se calcula el
       resultado DETERMINISTA equivalente con numpy: embed_demo (semilla vía
       hashlib.sha256, no hash()) + similitud coseno sobre los FAQs de tema
       'pagos', top-2. Con Distance.COSINE y vectores ya normalizados, el
       score de Qdrant ES el producto punto, así que el fallback es exacto.

Datos: los 6 FAQs vienen del enunciado (no hay archivos de entrada).
Salidas: resultados.json en la carpeta actual. Esta subtarea NO produce PNG.
"""

import json
import hashlib

import numpy as np

# ---------------------------------------------------------------------------
# Datos del enunciado: los 6 FAQs (texto, tema)
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
DIM = 64                  # dimensión declarada en la colección
DISTANCIA = "COSINE"      # Distance.COSINE
CONSULTA = "¿cuándo me devuelven el dinero?"
FILTRO_TEMA = "pagos"
LIMITE = 2


# ---------------------------------------------------------------------------
# embed_demo (código dado): semilla con hashlib.sha256 y NO hash(), porque
# hash() cambia entre procesos y los vectores deben ser iguales en cada
# corrida. En el lab real: sentence-transformers.
# ---------------------------------------------------------------------------
def semilla_de(texto):
    return int.from_bytes(hashlib.sha256(texto.encode()).digest()[:4], "little")


def embed_demo(texto):
    r = np.random.default_rng(semilla_de(texto))
    v = r.normal(size=DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def busqueda_coseno_con_filtro(texto_consulta, tema_filtro, limite):
    """Réplica determinista en numpy de la query de Qdrant con Distance.COSINE:
    los vectores de embed_demo ya tienen norma 1, así que el producto punto ES
    la similitud coseno (el score que Qdrant reporta para COSINE)."""
    q = np.asarray(embed_demo(texto_consulta), dtype=np.float32)
    pares = []
    for i, (t, tema) in enumerate(faqs):
        if tema != tema_filtro:          # query_filter: must tema == 'pagos'
            continue
        v = np.asarray(embed_demo(t), dtype=np.float32)
        pares.append({"id": int(i), "score": float(np.dot(q, v)),
                      "tema": tema, "texto": t})
    pares.sort(key=lambda d: -d["score"])
    return pares[:limite]


def main():
    modo_qdrant = "omitido"           # embebido | omitido  (servidor: no se intenta)
    motivo_omision = None
    fuente_scores = "numpy_fallback"  # qdrant | numpy_fallback
    scores_hits = []
    n_puntos = 0
    qdrant_disponible = False
    nota_servidor = (
        "Prueba de conexión al servidor Qdrant local (puerto 6333) OMITIDA: este "
        "entorno de ejecución prohíbe toda conexión de red, incluida localhost. "
        "Se aplica directamente el modo embebido (:memory:), que es el "
        "comportamiento previsto por el enunciado cuando el servidor no responde."
    )

    # Embeddings deterministas (idénticos en cualquier rama que los use)
    embeddings_faqs = [embed_demo(t) for t, _ in faqs]
    embedding_consulta = embed_demo(CONSULTA)

    # -- 1) ¿Está qdrant-client? (solo import local, sin red) ----------------
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (Distance, VectorParams, PointStruct,
                                          Filter, FieldCondition, MatchValue)
        qdrant_disponible = True
    except ImportError as e:
        motivo_omision = (
            "qdrant-client no está instalado y 'pip install qdrant-client' no puede "
            "intentarse en este entorno (red y subprocess prohibidos). La Parte 3 se "
            "omite sola, comportamiento previsto por el enunciado. "
            f"Detalle: ImportError: {e}"
        )
        print("qdrant-client no instalado — `pip install qdrant-client` para esta parte.")
        print("Aquí no se puede instalar (sin red ni subprocess): la Parte 3 se registra")
        print("como 'omitida' (previsto por el enunciado) y se deja el resultado")
        print("determinista equivalente calculado con numpy (embed_demo + coseno).")

    # -- 2) Ejecutar la Parte 3 con Qdrant en modo EMBEBIDO (sin sockets) ----
    if qdrant_disponible:
        try:
            # SIN intento de conexión a servidor: modo embebido directo.
            cliente = QdrantClient(":memory:")
            modo_qdrant = "embebido"
            print("Servidor no intentado (sin red en este entorno) — usando "
                  "Qdrant embebido (:memory:)")

            # recrear la colección desde cero (compatible con APIs viejas)
            try:
                if cliente.collection_exists(NOMBRE_COLECCION):
                    cliente.delete_collection(NOMBRE_COLECCION)
            except AttributeError:
                try:
                    cliente.delete_collection(NOMBRE_COLECCION)
                except Exception:
                    pass
            cliente.create_collection(
                NOMBRE_COLECCION,
                vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
            )
            cliente.upsert(NOMBRE_COLECCION, points=[
                PointStruct(id=i, vector=embeddings_faqs[i],
                            payload={"texto": t, "tema": tema})
                for i, (t, tema) in enumerate(faqs)
            ])
            n_puntos = int(cliente.count(NOMBRE_COLECCION).count)
            print("Colección creada con", n_puntos, "puntos")

            # consulta con filtro de metadata: solo tema="pagos"
            filtro = Filter(must=[FieldCondition(key="tema",
                                                 match=MatchValue(value=FILTRO_TEMA))])
            try:
                hits = cliente.query_points(
                    NOMBRE_COLECCION,
                    query=embedding_consulta,
                    query_filter=filtro,
                    limit=LIMITE,
                ).points
            except AttributeError:
                # qdrant-client antiguo (pre-1.10): misma consulta con la API search
                hits = cliente.search(
                    collection_name=NOMBRE_COLECCION,
                    query_vector=embedding_consulta,
                    query_filter=filtro,
                    limit=LIMITE,
                )
            scores_hits = [
                {"id": int(h.id), "score": float(h.score),
                 "tema": str(h.payload["tema"]), "texto": str(h.payload["texto"])}
                for h in hits
            ]
            fuente_scores = "qdrant"
        except Exception as e:
            modo_qdrant = "omitido"
            motivo_omision = (
                "qdrant-client se importó pero la ejecución embebida falló; se omite "
                "y se usa el resultado determinista equivalente con numpy. "
                f"Detalle: {type(e).__name__}: {e}"
            )
            scores_hits = []
            fuente_scores = "numpy_fallback"
            print("Aviso: la ejecución con qdrant-client falló ->", motivo_omision)

    # -- 3) Referencia determinista con numpy (fallback o verificación) ------
    hits_numpy = busqueda_coseno_con_filtro(CONSULTA, FILTRO_TEMA, LIMITE)
    if fuente_scores != "qdrant":
        scores_hits = hits_numpy

    max_abs_diff = None
    if fuente_scores == "qdrant" and len(scores_hits) == len(hits_numpy):
        max_abs_diff = max(abs(a["score"] - b["score"])
                           for a, b in zip(scores_hits, hits_numpy))

    # -- 4) Hits en el formato del código dado -------------------------------
    print()
    print(f"Consulta: {CONSULTA}")
    print(f'Filtro: tema="{FILTRO_TEMA}"   limit={LIMITE}')
    print(f"Fuente de los hits: {fuente_scores}")
    for h in scores_hits:
        print(f"  {h['score']:.3f} [{h['tema']}] {h['texto']}")
    print()
    print("(Nota: con embeddings sintéticos el score no es semántico —")
    print(" el objetivo aquí es la MECÁNICA de colección/upsert/query/filtro.)")

    # -- 5) Guardar TODAS las cifras en resultados.json ----------------------
    semillas = {t: int(semilla_de(t)) for t, _ in faqs}
    semillas[CONSULTA] = int(semilla_de(CONSULTA))

    resultados = {
        "subtarea": "T1 · Parte 3 — Qdrant embebido: colección, upsert y query con filtro",
        "modo_qdrant": modo_qdrant,                     # embebido | omitido
        "qdrant_client_disponible": bool(qdrant_disponible),
        "servidor_local_intentado": False,
        "motivo_no_servidor": nota_servidor,
        "motivo_omision": motivo_omision,
        "pip_install_intentado": False,
        "pip_install_no_posible": ("red y subprocess prohibidos en este entorno de "
                                   "ejecución; la parte se omite sola, como prevé "
                                   "el enunciado"),
        "coleccion": NOMBRE_COLECCION,
        "dim_vector": int(DIM),
        "distancia": DISTANCIA,
        "n_faqs": int(len(faqs)),
        "n_puntos_coleccion_faqs": int(n_puntos),
        "consulta": CONSULTA,
        "filtro_tema": FILTRO_TEMA,
        "limite": int(LIMITE),
        "fuente_scores": fuente_scores,
        "scores_hits_tema_pagos": scores_hits,          # [{id, score, tema, texto}, ...]
        "scores_hits_tema_pagos_referencia_numpy": hits_numpy,
        "max_abs_diff_scores_qdrant_vs_numpy": max_abs_diff,
        "semillas_sha256_por_texto": semillas,
        "embeddings_faqs_64d": [
            {"id": int(i), "tema": tema, "texto": t, "vector": embeddings_faqs[i]}
            for i, (t, tema) in enumerate(faqs)
        ],
        "embedding_consulta_64d": embedding_consulta,
        "figura_png_generada": False,
        "nota": ("Embeddings sintéticos: el score no es semántico; el objetivo es la "
                 "mecánica de colección/upsert/query con filtro. Semilla vía "
                 "hashlib.sha256 (no hash()). Sin conexiones de red: modo embebido "
                 ":memory: o fallback numpy determinista. Esta subtarea no produce "
                 "figura PNG."),
    }

    # resultados.json en la carpeta actual; se conserva lo de otras subtareas
    try:
        with open("resultados.json", "r", encoding="utf-8") as f:
            previo = json.load(f)
        if not isinstance(previo, dict):
            previo = {}
    except (FileNotFoundError, ValueError, OSError):
        previo = {}
    previo.update(resultados)
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(previo, f, ensure_ascii=False, indent=2)

    # -- 6) Resumen de cifras principales ------------------------------------
    print()
    print("--- Cifras principales (guardadas en resultados.json) ---")
    print("modo_qdrant:", modo_qdrant)
    print("qdrant_client_disponible:", qdrant_disponible)
    print("servidor_local_intentado: False (sin red en este entorno)")
    if motivo_omision:
        print("motivo_omision:", motivo_omision)
    print("n_puntos_coleccion_faqs:", n_puntos)
    print("scores_hits_tema_pagos (precisión completa):")
    for h in scores_hits:
        print(f"  id={h['id']}  score={h['score']!r}  tema={h['tema']!r}")
    print("resultados.json guardado en la carpeta actual.")


if __name__ == "__main__":
    main()
