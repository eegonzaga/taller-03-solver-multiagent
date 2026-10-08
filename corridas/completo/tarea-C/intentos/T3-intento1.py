# -*- coding: utf-8 -*-
"""
T3 - Parte 3 - Pureza: ¿recupera LDA los temas reales?

Con la distribución documento-tema del modelo LDA de tres temas ajustado en la
Parte 1 (guardada en entradas/T1.json):
  1) Asigna a cada documento (d01-d12) su tema dominante (mayor probabilidad).
  2) Calcula la pureza: para cada tema de LDA se cuentan los documentos de la
     etiqueta real más frecuente entre los asignados a ese tema; se suman esas
     cuentas y se dividen entre 12.
  3) Identifica los documentos mal agrupados: aquellos cuya etiqueta real
     difiere de la etiqueta mayoritaria del tema LDA al que fueron asignados.

Salidas:
  - resultados.json con las claves: asignacion_por_documento, pureza,
    documentos_mal_agrupados (más detalles adicionales).
  - Impresión de la tabla de asignación, el resumen por tema y la pureza.
  - Esta subtarea NO requiere figura PNG.
"""

import json
from collections import Counter

import numpy as np

RUTA_ENTRADA = "entradas/T1.json"
RUTA_RESULTADOS = "resultados.json"

# ----------------------------------------------------------------------
# Corpus y palabras vacías (copia fiel del enunciado). Solo se usan como
# respaldo, por si la distribución documento-tema no pudiera leerse de
# T1.json; el reajuste es determinista (random_state=0, método batch).
# ----------------------------------------------------------------------
TEXTOS_CORPUS = {
    "d01": "El delantero marcó dos goles en el partido de fútbol y el equipo ganó la liga.",
    "d02": "El entrenador del equipo preparó la defensa para el partido final de la liga.",
    "d03": "La tenista ganó el torneo después de un partido largo contra la campeona.",
    "d04": "El equipo de baloncesto perdió el partido por un punto en el último segundo.",
    "d05": "La receta lleva harina, huevos y azúcar; la masa se hornea durante treinta minutos.",
    "d06": "Para la salsa se sofríe cebolla y ajo en aceite y se añade tomate.",
    "d07": "El pan se hornea con harina, agua, sal y levadura después de amasar la masa.",
    "d08": "La sopa de verduras lleva cebolla, zanahoria, ajo y sal, y se cocina a fuego lento.",
    "d09": "El telescopio observó una galaxia lejana y varias estrellas de la nebulosa.",
    "d10": "Los planetas giran alrededor de la estrella; la órbita de la Tierra dura un año.",
    "d11": "La nebulosa es una nube de gas donde nacen estrellas nuevas en la galaxia.",
    "d12": "El telescopio espacial fotografió planetas y la órbita de una luna de Júpiter.",
}

STOP_WORDS = ["el", "la", "los", "las", "de", "del", "en", "y", "a", "un", "una",
              "por", "para", "con", "se", "su", "al", "es", "después", "durante",
              "contra", "donde", "alrededor"]


def _texto_de(doc_id):
    """Devuelve el texto del documento con normalización simple del id."""
    if doc_id in TEXTOS_CORPUS:
        return TEXTOS_CORPUS[doc_id]
    try:
        alt = "d{:02d}".format(int(str(doc_id).lstrip("dD")))
    except ValueError:
        alt = None
    if alt is not None and alt in TEXTOS_CORPUS:
        return TEXTOS_CORPUS[alt]
    raise KeyError("No hay texto para el documento '{}'".format(doc_id))


def reajustar_lda_parte1(ids):
    """Respaldo: reajusta el modelo exacto de la Parte 1 (CountVectorizer con
    las palabras vacías del enunciado + LDA(n_components=3,
    learning_method='batch', max_iter=50, random_state=0)) y devuelve la
    distribución documento-tema. Con la semilla fijada el resultado es
    idéntico al de la Parte 1."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    textos = [_texto_de(i) for i in ids]
    vectorizador = CountVectorizer(stop_words=STOP_WORDS)
    X = vectorizador.fit_transform(textos)
    lda = LatentDirichletAllocation(n_components=3, learning_method="batch",
                                    max_iter=50, random_state=0)
    return lda.fit_transform(X)


def extraer_doc_topic(t1, ids):
    """Obtiene la matriz (n_docs x n_temas) desde T1.json.
    Devuelve (matriz, descripción_de_la_fuente)."""
    n = len(ids)
    dist = t1.get("distribucion_documento_tema")
    try:
        if isinstance(dist, dict):
            # (a) diccionario por documento: {"d01": [p0, p1, p2], ...}
            if all((i in dist) or (str(i) in dist) for i in ids):
                arr = np.asarray(
                    [dist[i] if i in dist else dist[str(i)] for i in ids],
                    dtype=float)
                if arr.ndim == 2 and arr.shape[0] == n and arr.shape[1] >= 2:
                    return arr, "T1.json (diccionario por documento)"
            # (b) diccionario por tema: {"tema_0": [...], ...}
            claves = sorted(dist.keys())
            arr = np.asarray([dist[k] for k in claves], dtype=float)
            if arr.ndim == 2:
                if arr.shape[0] == n and arr.shape[1] != n:
                    return arr, "T1.json (diccionario por tema)"
                if arr.shape[1] == n and arr.shape[0] != n:
                    return arr.T, "T1.json (diccionario por tema, transpuesto)"
        elif dist is not None:
            arr = np.asarray(dist, dtype=float)
            if arr.ndim == 2:
                if arr.shape[0] == n:
                    return arr, "T1.json (lista: una fila por documento)"
                if arr.shape[1] == n:
                    return arr.T, "T1.json (lista transpuesta: una columna por documento)"
    except (TypeError, ValueError, KeyError, IndexError):
        pass
    # Respaldo determinista
    return (np.asarray(reajustar_lda_parte1(ids), dtype=float),
            "reajuste del modelo de la Parte 1 (respaldo determinista)")


def main():
    # ------------------------------------------------------------------
    # 1) Cargar los resultados de la Parte 1
    # ------------------------------------------------------------------
    with open(RUTA_ENTRADA, "r", encoding="utf-8") as f:
        t1 = json.load(f)

    ids = list(t1.get("ids_documentos") or ["d{:02d}".format(i) for i in range(1, 13)])
    n_docs = len(ids)

    et_raw = t1.get("etiquetas_reales")
    if et_raw is None:
        raise KeyError("T1.json no contiene 'etiquetas_reales'.")
    if isinstance(et_raw, dict):
        etiquetas = [et_raw[i] if i in et_raw else et_raw[str(i)] for i in ids]
    else:
        etiquetas = list(et_raw)
    if len(etiquetas) != n_docs:
        raise ValueError("El número de etiquetas reales no coincide con el número de documentos.")

    # ------------------------------------------------------------------
    # 2) Distribución documento-tema del modelo de la Parte 1
    # ------------------------------------------------------------------
    doc_topic, fuente = extraer_doc_topic(t1, ids)
    n_temas = int(doc_topic.shape[1])

    # ------------------------------------------------------------------
    # 3) Tema dominante de cada documento (argmax de la fila)
    # ------------------------------------------------------------------
    temas_dominantes = np.argmax(doc_topic, axis=1)

    # ------------------------------------------------------------------
    # 4) Pureza: por cada tema LDA, cuenta de la etiqueta real más
    #    frecuente entre sus documentos; suma / 12
    # ------------------------------------------------------------------
    resumen_por_tema = {}
    etiqueta_mayoritaria = {}
    suma_mayorias = 0
    for t in range(n_temas):
        indices = [i for i in range(n_docs) if int(temas_dominantes[i]) == t]
        etiquetas_en_tema = [etiquetas[i] for i in indices]
        if etiquetas_en_tema:
            conteo = Counter(etiquetas_en_tema)
            et_maj, n_maj = conteo.most_common(1)[0]
        else:
            conteo = Counter()
            et_maj, n_maj = None, 0
        etiqueta_mayoritaria[t] = et_maj
        suma_mayorias += int(n_maj)
        resumen_por_tema[str(t)] = {
            "n_documentos": len(indices),
            "ids": [ids[i] for i in indices],
            "etiqueta_mayoritaria": et_maj,
            "cuenta_mayoritaria": int(n_maj),
            "desglose_etiquetas": {k: int(v) for k, v in conteo.items()},
        }

    pureza = suma_mayorias / float(n_docs)  # n_docs = 12

    # ------------------------------------------------------------------
    # 5) Asignación por documento y documentos mal agrupados
    # ------------------------------------------------------------------
    asignacion = []
    mal_agrupados = []
    detalle_mal = []
    for i in range(n_docs):
        t = int(temas_dominantes[i])
        et_real = etiquetas[i]
        et_maj = etiqueta_mayoritaria[t]
        bien = (et_real == et_maj)
        asignacion.append({
            "id": ids[i],
            "etiqueta_real": et_real,
            "tema_lda": t,
            "probabilidad_tema_dominante": float(np.max(doc_topic[i])),
            "distribucion_temas": [float(p) for p in doc_topic[i]],
            "etiqueta_mayoritaria_del_tema": et_maj,
            "bien_agrupado": bool(bien),
        })
        if not bien:
            mal_agrupados.append(ids[i])
            detalle_mal.append({
                "id": ids[i],
                "etiqueta_real": et_real,
                "tema_lda": t,
                "etiqueta_mayoritaria_del_tema": et_maj,
                "razon": ("su etiqueta real '{}' difiere de la etiqueta "
                          "mayoritaria '{}' del tema LDA {} al que fue asignado"
                          .format(et_real, et_maj, t)),
            })

    # ------------------------------------------------------------------
    # 6) Guardar resultados.json (tipos nativos, sin redondear)
    # ------------------------------------------------------------------
    resultados = {
        "asignacion_por_documento": asignacion,
        "pureza": float(pureza),
        "documentos_mal_agrupados": mal_agrupados,
        "detalle_documentos_mal_agrupados": detalle_mal,
        "resumen_por_tema": resumen_por_tema,
        "mapeo_tema_a_etiqueta": {str(t): etiqueta_mayoritaria[t]
                                  for t in range(n_temas)},
        "suma_cuentas_mayoritarias": int(suma_mayorias),
        "n_documentos": int(n_docs),
        "n_temas": int(n_temas),
        "fuente_distribucion_documento_tema": fuente,
        "configuracion_lda_parte1": t1.get("configuracion", {}),
    }
    with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 7) Impresión de las cifras principales
    # ------------------------------------------------------------------
    print("=" * 80)
    print("T3 - Parte 3 - Pureza: ¿recupera LDA los temas reales?")
    print("=" * 80)
    print("Fuente de la distribución documento-tema: {}".format(fuente))
    print()
    print("Tabla de asignación por documento:")
    enc = "{:<6} {:<14} {:<9} {:<10} {:<19} {:<13}".format(
        "id", "etiqueta_real", "tema_LDA", "p(tema)", "etiqueta_del_tema", "estado")
    print(enc)
    print("-" * len(enc))
    for a in asignacion:
        estado = "OK" if a["bien_agrupado"] else "MAL AGRUPADO"
        print("{:<6} {:<14} {:<9} {:<10.3f} {:<19} {:<13}".format(
            a["id"], str(a["etiqueta_real"]), a["tema_lda"],
            a["probabilidad_tema_dominante"],
            str(a["etiqueta_mayoritaria_del_tema"]), estado))
    print()
    print("Resumen por tema LDA:")
    for t in range(n_temas):
        r = resumen_por_tema[str(t)]
        print("  Tema {}: {} documentos {} -> etiqueta mayoritaria '{}' "
              "({}/{}); desglose {}".format(
                  t, r["n_documentos"], r["ids"], r["etiqueta_mayoritaria"],
                  r["cuenta_mayoritaria"], r["n_documentos"],
                  r["desglose_etiquetas"]))
    print()
    print("Pureza = suma de cuentas mayoritarias ({}) / {} = {}".format(
        suma_mayorias, n_docs, pureza))
    if mal_agrupados:
        print("Documentos mal agrupados: {}".format(", ".join(mal_agrupados)))
        for d in detalle_mal:
            print("  - {}".format(d["razon"]))
    else:
        print("Documentos mal agrupados: ninguno "
              "(LDA recupera exactamente los tres temas reales).")
    print()
    print("Resultados guardados en {}".format(RUTA_RESULTADOS))


if __name__ == "__main__":
    main()
