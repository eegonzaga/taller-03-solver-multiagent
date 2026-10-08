# -*- coding: utf-8 -*-
"""
T1 · Parte 3 — Qdrant embebido: colección, upsert y query con filtro
MMIA 6013 · Semana 2 · Martes (S2·MAR)

Ejecuta la Parte 3 con el código dado por el profesor:
  1) Intenta conectar a un servidor Qdrant en http://localhost:6333 (timeout 3 s).
  2) Si no responde, usa QdrantClient(':memory:')  ->  modo "embebido".
  3) Crea la colección 'faqs' (64 dims, Distance.COSINE), hace upsert de los
     6 FAQs con payload {texto, tema} usando embed_demo (semilla vía
     hashlib.sha256, NO hash()), y consulta "¿cuándo me devuelven el dinero?"
     con query_filter tema='pagos' y limit=2.
  4) Registra el modo usado (servidor/embebido/omitido) y los hits devueltos
     (score, tema, texto) en resultados.json.

Sobre 'pip install qdrant-client': el enunciado prevé que la parte "se salta
sola" si la librería no está instalada. Este entorno de ejecución prohíbe red
y subprocess, por lo que pip install no puede intentarse; en ese caso se
registra modo_qdrant="omitido" con su motivo y, para no dejar la subtarea sin
cifras, se calcula con numpy el resultado DETERMINISTA equivalente
(embed_demo + similitud coseno sobre los FAQs de tema 'pagos', top-2), que es
exactamente lo que Qdrant devolvería con Distance.COSINE sobre vectores ya
normalizados. La procedencia queda registrada en 'fuente_scores'
('qdrant' o 'numpy_fallback').

Salidas: resultados.json (carpeta actual). Esta subtarea NO produce figura PNG.
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
# embed_demo (código dado): semilla con hashlib.sha256 y no hash(), porque
# hash() cambia entre procesos y con un servidor persistente los vectores
# deben ser iguales en cada corrida. En el lab real: sentence-transformers.
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
        pares.append({"id": i, "score": float(np.dot(q, v)), "tema": tema, "texto": t})
    pares.sort(key=lambda d: -d["score"])
    return pares[:limite]


def main():
    modo_qdrant = "omitido"           # servidor | embebido | omitido
    motivo_omision = None
    fuente_scores = "numpy_fallback"  # qdrant | numpy_fallback
    scores_hits = []
    n_puntos = 0
    qdrant_disponible = False

    # Embeddings deterministas (idénticos en cualquier rama que los use)
    embeddings_faqs = [embed_demo(t) for t, _ in faqs]
    embedding_consulta = embed_demo(CONSULTA)

    # -- 1) ¿Está qdrant-client? --------------------------------------------
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

    # -- 2) Ejecutar la Parte 3 con Qdrant (código dado) ---------------------
    if qdrant_disponible:
        try:
            # Servidor Qdrant local (Docker en :6333); si no responde, modo embebido.
            # Conexión LOCAL (localhost) exigida por el enunciado: sin tráfico externo.
            try:
                cliente = QdrantClient(url="http://localhost:6333", timeout=3)
                cliente.get_collections()
                modo_qdrant = "servidor"
                print("Conectado al servidor Qdrant en http://localhost:6333")
            except Exception:
                cliente = QdrantClient(":memory:")
                modo_qdrant = "embebido"
                print("Servidor no disponible — usando Qdrant embebido (:memory:)")

            # el servidor persiste entre corridas: recrear la colección desde cero
            if cliente.collection_exists(NOMBRE_COLECCION):
                cliente.delete_collection(NOMBRE_COLECCION)
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
                "qdrant-client se importó pero la ejecución de la Parte 3 falló; se "
                "omite y se usa el resultado determinista equivalente con numpy. "
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
    semillas = {t: semilla_de(t) for t, _ in faqs}
    semillas[CONSULTA] = semilla_de(CONSULTA)

    resultados = {
        "subtarea": "T1 · Parte 3 — Qdrant embebido: colección, upsert y query con filtro",
        "modo_qdrant": modo_qdrant,                     # servidor | embebido | omitido
        "qdrant_client_disponible": bool(qdrant_disponible),
        "motivo_omision": motivo_omision,
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
            {"id": i, "tema": tema, "texto": t, "vector": embeddings_faqs[i]}
            for i, (t, tema) in enumerate(faqs)
        ],
        "embedding_consulta_64d": embedding_consulta,
        "figura_png_generada": False,
        "nota": ("Embeddings sintéticos: el score no es semántico; el objetivo es la "
                 "mecánica de colección/upsert/query con filtro. Semilla vía "
                 "hashlib.sha256 (no hash()). Esta subtarea no produce figura PNG."),
    }
    if motivo_omision is not None:
        resultados["pip_install_intentado"] = False
        resultados["pip_install_no_posible"] = ("red y subprocess prohibidos en este "
                                                "entorno de ejecución")

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
    if motivo_omision:
        print("motivo_omision:", motivo_omision)
    print("n_puntos_coleccion_faqs:", n_puntos)
    print("scores_hits_tema_pagos (precisión completa):")
    for h in scores_hits:
        print(f"  id={h['id']}  score={h['score']!r}  tema={h['tema']!r}")
    print("resultados.json guardado en la carpeta actual.")


if __name__ == "__main__":
    main()
