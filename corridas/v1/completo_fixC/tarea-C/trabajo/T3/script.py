# -*- coding: utf-8 -*-
"""
T3 - Parte 3 - ¿Recupera LDA los temas reales?

Usa la distribución documento-tema del modelo LDA de tres temas de la Parte 1
(entradas/T1.json). Si ese archivo no está disponible, reajusta el modelo
exacto de la Parte 1: CountVectorizer con la lista de palabras vacías del
enunciado + LatentDirichletAllocation(n_components=3, learning_method="batch",
max_iter=50, random_state=0).

Pasos:
  1) Asignar a cada documento su tema dominante (argmax de p(tema|doc)).
  2) Tabla de asignación por documento junto con la etiqueta real.
  3) Pureza: para cada tema de LDA, contar los documentos de la etiqueta real
     más frecuente entre los asignados a él; sumar esas cuentas y dividir
     por 12.
  4) Identificar los documentos mal agrupados.

Salidas:
  - resultados.json con las claves 'asignacion_por_documento', 'pureza' y
    'documentos_mal_agrupados' (más cifras de apoyo).
  - Impresión por consola de la tabla y de las cifras principales.
  - Esta subtarea NO genera ninguna figura PNG.
"""

import json
import os
from collections import Counter

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1) Corpus, etiquetas reales y palabras vacías (exactamente los del enunciado)
# ---------------------------------------------------------------------------
IDS = ["d01", "d02", "d03", "d04", "d05", "d06",
       "d07", "d08", "d09", "d10", "d11", "d12"]

TEXTOS = [
    "El delantero marcó dos goles en el partido de fútbol y el equipo ganó la liga.",
    "El entrenador del equipo preparó la defensa para el partido final de la liga.",
    "La tenista ganó el torneo después de un partido largo contra la campeona.",
    "El equipo de baloncesto perdió el partido por un punto en el último segundo.",
    "La receta lleva harina, huevos y azúcar; la masa se hornea durante treinta minutos.",
    "Para la salsa se sofríe cebolla y ajo en aceite y se añade tomate.",
    "El pan se hornea con harina, agua, sal y levadura después de amasar la masa.",
    "La sopa de verduras lleva cebolla, zanahoria, ajo y sal, y se cocina a fuego lento.",
    "El telescopio observó una galaxia lejana y varias estrellas de la nebulosa.",
    "Los planetas giran alrededor de la estrella; la órbita de la Tierra dura un año.",
    "La nebulosa es una nube de gas donde nacen estrellas nuevas en la galaxia.",
    "El telescopio espacial fotografió planetas y la órbita de una luna de Júpiter.",
]

ETIQUETAS_ENUNCIADO = ["deporte"] * 4 + ["cocina"] * 4 + ["astronomía"] * 4

STOP_WORDS = [
    "el", "la", "los", "las", "de", "del", "en", "y", "a", "un", "una", "por",
    "para", "con", "se", "su", "al", "es", "después", "durante", "contra",
    "donde", "alrededor",
]

N_DOCS = 12
N_TEMAS = 3


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------
def a_native(obj):
    """Convierte tipos de NumPy a tipos nativos de Python (para json.dump)."""
    if isinstance(obj, dict):
        return {str(k): a_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [a_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return a_native(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def cargar_t1():
    """Lee entradas/T1.json si existe; devuelve None en caso contrario."""
    ruta = os.path.join("entradas", "T1.json")
    if os.path.exists(ruta):
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                datos = json.load(f)
            print("[info] Leído %s (resultados de la Parte 1)." % ruta)
            return datos
        except Exception as exc:
            print("[aviso] No se pudo leer %s (%s); se reajustará el modelo."
                  % (ruta, exc))
    else:
        print("[aviso] No existe %s; se reajustará el modelo de la Parte 1."
              % ruta)
    return None


def extraer_distribucion(bruto):
    """Convierte 'distribucion_documento_tema' de T1.json en un array 12x3.

    Acepta lista de listas (12x3 o 3x12) o diccionario indexado por id de
    documento, por índice entero o como cadena '0'..'11'. Devuelve None si el
    formato no es utilizable.
    """
    if bruto is None:
        return None
    dist = None
    if isinstance(bruto, dict):
        for claves in (IDS,
                       [str(i) for i in range(N_DOCS)],
                       list(range(N_DOCS))):
            try:
                dist = np.array([bruto[c] for c in claves], dtype=float)
                break
            except Exception:
                dist = None
        if dist is None and len(bruto) == N_TEMAS:
            # Posible formato {tema: [valor por documento]} (temas en filas)
            try:
                arr = np.array([bruto[k] for k in bruto], dtype=float)
                if arr.shape == (N_TEMAS, N_DOCS):
                    dist = arr.T
            except Exception:
                dist = None
    elif isinstance(bruto, (list, tuple)):
        try:
            dist = np.array(bruto, dtype=float)
        except Exception:
            dist = None
    if dist is None or dist.ndim != 2:
        return None
    if dist.shape == (N_TEMAS, N_DOCS) and dist.shape != (N_DOCS, N_TEMAS):
        dist = dist.T
    if dist.shape != (N_DOCS, N_TEMAS):
        return None
    if not np.all(np.isfinite(dist)):
        return None
    sumas = dist.sum(axis=1, keepdims=True)
    if np.all(sumas > 0):
        dist = dist / sumas
    return dist


def obtener_distribucion_doc_tema(t1):
    """Devuelve (dist 12x3 de p(tema|doc), descripción de la fuente)."""
    if t1 is not None and "distribucion_documento_tema" in t1:
        dist = extraer_distribucion(t1["distribucion_documento_tema"])
        if dist is not None:
            return dist, "entradas/T1.json ('distribucion_documento_tema' de la Parte 1)"
        print("[aviso] 'distribucion_documento_tema' de T1.json no es utilizable.")
    # Plan B: reajuste determinista del modelo exacto de la Parte 1
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    print("[info] Reajuste del modelo de la Parte 1: CountVectorizer(stop_words="
          "lista del enunciado) + LatentDirichletAllocation(n_components=3, "
          "learning_method='batch', max_iter=50, random_state=0).")
    vectorizer = CountVectorizer(stop_words=STOP_WORDS)
    X = vectorizer.fit_transform(TEXTOS)
    lda = LatentDirichletAllocation(n_components=N_TEMAS,
                                    learning_method="batch",
                                    max_iter=50,
                                    random_state=0)
    lda.fit(X)
    dist = lda.transform(X)
    dist = dist / dist.sum(axis=1, keepdims=True)
    return dist, "reajuste local del modelo de la Parte 1 (random_state=0)"


def obtener_etiquetas_reales(t1):
    """Etiqueta real por documento: enunciado; T1.json solo si coincide."""
    etiquetas = list(ETIQUETAS_ENUNCIADO)
    origen = "enunciado (columna 'tema' del corpus)"
    if t1 is not None and "temas_reales" in t1:
        tr = t1["temas_reales"]
        cand = None
        if isinstance(tr, dict):
            for claves in (IDS,
                           [str(i) for i in range(N_DOCS)],
                           list(range(N_DOCS))):
                try:
                    cand = [str(tr[c]) for c in claves]
                    break
                except Exception:
                    cand = None
        elif isinstance(tr, (list, tuple)) and len(tr) == N_DOCS:
            cand = [str(x) for x in tr]
        if cand is not None:
            if cand == etiquetas:
                origen = "entradas/T1.json ('temas_reales'; coincide con el enunciado)"
            else:
                print("[aviso] 'temas_reales' de T1.json difiere del enunciado; "
                      "se usa la etiqueta real del enunciado.")
    return etiquetas, origen


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    t1 = cargar_t1()
    dist, fuente_dist = obtener_distribucion_doc_tema(t1)
    etiquetas, fuente_etiq = obtener_etiquetas_reales(t1)

    # 1) Tema dominante de cada documento (argmax de la distribución doc-tema)
    temas_dominantes = np.argmax(dist, axis=1)

    # 2) Etiqueta real mayoritaria y cuentas por tema LDA (base de la pureza)
    resumen_temas = {}
    for t in range(N_TEMAS):
        idx = [i for i in range(N_DOCS) if int(temas_dominantes[i]) == t]
        contador = Counter(etiquetas[i] for i in idx)
        if len(contador) > 0:
            etiqueta_may, cuenta_may = contador.most_common(1)[0]
        else:
            etiqueta_may, cuenta_may = None, 0
        resumen_temas[str(t)] = {
            "tema_lda": int(t),
            "documentos_asignados": [IDS[i] for i in idx],
            "n_documentos": int(len(idx)),
            "cuentas_por_etiqueta_real": {k: int(v) for k, v in contador.items()},
            "etiqueta_real_mayoritaria": etiqueta_may,
            "cuenta_mayoritaria": int(cuenta_may),
        }

    aciertos = int(sum(resumen_temas[str(t)]["cuenta_mayoritaria"]
                       for t in range(N_TEMAS)))
    pureza = float(aciertos) / float(N_DOCS)

    # 3) Tabla de asignación por documento
    filas = []
    mal_agrupados = []
    for i in range(N_DOCS):
        t = int(temas_dominantes[i])
        et_real = etiquetas[i]
        et_may = resumen_temas[str(t)]["etiqueta_real_mayoritaria"]
        bien = bool(et_real == et_may)
        fila = {
            "id": IDS[i],
            "texto": TEXTOS[i],
            "etiqueta_real": et_real,
            "tema_lda_dominante": t,
            "etiqueta_mayoritaria_del_tema": et_may,
            "bien_agrupado": bien,
            "p_tema_0": float(dist[i, 0]),
            "p_tema_1": float(dist[i, 1]),
            "p_tema_2": float(dist[i, 2]),
        }
        filas.append(fila)
        if not bien:
            mal_agrupados.append({
                "id": IDS[i],
                "etiqueta_real": et_real,
                "tema_lda_dominante": t,
                "etiqueta_mayoritaria_del_tema": et_may,
                "p_tema_dominante": float(dist[i, t]),
            })

    # 4) Impresión de resultados
    df = pd.DataFrame(filas)
    columnas = ["id", "etiqueta_real", "tema_lda_dominante",
                "etiqueta_mayoritaria_del_tema", "bien_agrupado",
                "p_tema_0", "p_tema_1", "p_tema_2"]
    print("")
    print("=== Asignación por documento (tema dominante de LDA vs etiqueta real) ===")
    print("Fuente de la distribución documento-tema: %s" % fuente_dist)
    print("Fuente de las etiquetas reales: %s" % fuente_etiq)
    print(df[columnas].to_string(index=False))

    print("")
    print("=== Resumen por tema LDA ===")
    for t in range(N_TEMAS):
        r = resumen_temas[str(t)]
        docs_txt = ",".join(r["documentos_asignados"]) if r["documentos_asignados"] else "-"
        print("Tema %d: docs=[%s] | cuentas por etiqueta real=%s | mayoría='%s' (%d docs)"
              % (t, docs_txt, r["cuentas_por_etiqueta_real"],
                 r["etiqueta_real_mayoritaria"], r["cuenta_mayoritaria"]))

    print("")
    print("Pureza = %d / %d = %.6f" % (aciertos, N_DOCS, pureza))

    if mal_agrupados:
        print("")
        print("Documentos mal agrupados:")
        for m in mal_agrupados:
            print("  %s: etiqueta real='%s' -> tema %d (mayoría del tema='%s', "
                  "p(tema dominante)=%.4f)"
                  % (m["id"], m["etiqueta_real"], m["tema_lda_dominante"],
                     m["etiqueta_mayoritaria_del_tema"], m["p_tema_dominante"]))
    else:
        print("")
        print("No hay documentos mal agrupados: LDA recupera exactamente los "
              "tres temas reales.")

    # Palabras top por tema (de la Parte 1), útiles para interpretar los temas
    palabras_top = None
    if t1 is not None and "palabras_top_por_tema" in t1:
        palabras_top = t1["palabras_top_por_tema"]
        print("")
        print("Palabras top por tema (Parte 1):")
        try:
            if isinstance(palabras_top, dict):
                for k in sorted(palabras_top.keys(), key=lambda s: str(s)):
                    print("  Tema %s: %s" % (k, palabras_top[k]))
            else:
                for j in range(len(palabras_top)):
                    print("  Tema %d: %s" % (j, palabras_top[j]))
        except Exception:
            pass

    # 5) Guardado de todas las cifras en resultados.json
    resultados = {
        "asignacion_por_documento": filas,
        "pureza": pureza,
        "documentos_mal_agrupados": mal_agrupados,
        "aciertos_pureza": aciertos,
        "n_documentos": N_DOCS,
        "n_temas": N_TEMAS,
        "resumen_por_tema_lda": resumen_temas,
        "temas_dominantes": [int(x) for x in temas_dominantes],
        "etiquetas_reales": etiquetas,
        "fuente_distribucion_documento_tema": fuente_dist,
        "fuente_etiquetas_reales": fuente_etiq,
        "palabras_top_por_tema": palabras_top,
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(a_native(resultados), f, ensure_ascii=False, indent=2)
    print("")
    print("[ok] Resultados guardados en resultados.json")


if __name__ == "__main__":
    main()
