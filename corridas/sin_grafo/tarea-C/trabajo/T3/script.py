# -*- coding: utf-8 -*-
"""
T3 - Parte 3 — ¿Recupera LDA los temas reales?

Usa la distribución documento-tema del modelo LDA de 3 temas ajustado en la
Parte 1 (guardada en entradas/T1.json) para:
  1) asignar a cada uno de los 12 documentos su tema dominante
     (mayor componente de su distribución documento-tema),
  2) calcular la pureza: para cada tema LDA se cuentan los documentos de la
     etiqueta real más frecuente entre los asignados a él, se suman esas
     cuentas y se dividen entre 12,
  3) construir la tabla de asignación (id, etiqueta real, tema LDA) y
  4) identificar los documentos mal agrupados (aquellos cuya etiqueta real
     difiere de la etiqueta real mayoritaria del tema LDA asignado).

Salidas:
  - resultados.json (asignacion_por_documento, pureza, documentos_mal_agrupados, ...)
  - impresión de la tabla y de las cifras principales.
  - Sin figuras PNG (no se requieren en esta subtarea).
"""

import json
import re
from collections import Counter

import numpy as np

RUTA_ENTRADA = "entradas/T1.json"
RUTA_RESULTADOS = "resultados.json"


# ----------------------------------------------------------------------
# Utilidades de parseo robusto del JSON de la Parte 1
# ----------------------------------------------------------------------
def _orden_clave(k):
    """Clave de ordenación numérica tolerante a prefijos tipo 'doc_0'/'tema_2'."""
    s = str(k)
    m = re.search(r"\d+", s)
    if m:
        return (0, int(m.group()), s)
    return (1, 0, s)


def _claves_ordenadas(d):
    return sorted(d.keys(), key=_orden_clave)


def _a_float(x):
    """Convierte un valor del JSON a float, tolerando dicts/listas envolventes."""
    if isinstance(x, dict):
        for clave in ("probabilidad", "prob", "valor", "value", "p", "peso"):
            if clave in x:
                return float(x[clave])
        if len(x) == 1:
            return _a_float(list(x.values())[0])
        raise ValueError("No se pudo convertir a float: %r" % (x,))
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            return _a_float(x[0])
        raise ValueError("No se pudo convertir a float: %r" % (x,))
    return float(x)


def _extraer_matriz(bruta):
    """Convierte 'distribucion_documento_tema' en un ndarray 2D de floats.

    Acepta: lista de listas, lista de dicts, dict de listas y dict de dicts
    (p. ej. {"0": {"0": 0.1, "1": 0.8, "2": 0.1}, ...}).
    """
    if isinstance(bruta, dict):
        filas = [bruta[c] for c in _claves_ordenadas(bruta)]
    elif isinstance(bruta, (list, tuple)):
        filas = list(bruta)
    else:
        raise ValueError("Formato no reconocido para 'distribucion_documento_tema'.")

    filas_num = []
    for fila in filas:
        if isinstance(fila, dict):
            filas_num.append([_a_float(fila[c]) for c in _claves_ordenadas(fila)])
        elif isinstance(fila, (list, tuple)):
            filas_num.append([_a_float(v) for v in fila])
        else:
            filas_num.append([_a_float(fila)])
    return np.array(filas_num, dtype=float)


def _orientar_matriz(M, n_docs_ref, n_temas_ref):
    """Devuelve la matriz orientada como (n_documentos, n_temas)."""
    if M.ndim == 1:
        if n_docs_ref and n_temas_ref and M.size == n_docs_ref * n_temas_ref:
            return M.reshape(n_docs_ref, n_temas_ref)
        raise ValueError("La distribución documento-tema no tiene una forma reconocible.")
    r, c = M.shape
    if n_docs_ref and n_temas_ref:
        if (r, c) == (n_docs_ref, n_temas_ref):
            return M
        if (r, c) == (n_temas_ref, n_docs_ref):
            return M.T
    if n_temas_ref is not None and r == n_temas_ref and c != n_temas_ref:
        return M.T
    if n_docs_ref is not None and c == n_docs_ref and r != n_docs_ref:
        return M.T
    if r < c:
        return M.T
    return M


def _palabra_de_elemento(p):
    """Extrae la palabra de un elemento de una lista de palabras top."""
    if isinstance(p, str):
        return p
    if isinstance(p, dict):
        for clave in ("palabra", "word", "termino", "término", "token"):
            if clave in p:
                return str(p[clave])
        if len(p) == 1:
            return str(list(p.keys())[0])
        return json.dumps(p, ensure_ascii=False)
    if isinstance(p, (list, tuple)):
        if len(p) > 0:
            return _palabra_de_elemento(p[0])
        return ""
    return str(p)


def _fila_top_a_texto(x):
    """Convierte la entrada de un tema (lista/dict de palabras top) en texto."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for clave in ("palabras", "words", "top", "terminos", "términos"):
            if clave in x and isinstance(x[clave], (list, tuple)):
                return ", ".join(_palabra_de_elemento(p) for p in x[clave])
        try:
            pares = [(str(k), _a_float(v)) for k, v in x.items()]
            pares.sort(key=lambda t: -t[1])
            return ", ".join(p for p, _ in pares)
        except Exception:
            return ", ".join(str(c) for c in _claves_ordenadas(x))
    if isinstance(x, (list, tuple)):
        return ", ".join(_palabra_de_elemento(p) for p in x)
    return str(x)


def _extraer_tops(top_bruto, n_temas):
    """Normaliza 'palabras_top_por_tema' a una lista de cadenas (una por tema)."""
    if top_bruto is None:
        return [""] * n_temas
    if isinstance(top_bruto, dict):
        items = [top_bruto[c] for c in _claves_ordenadas(top_bruto)]
    elif isinstance(top_bruto, (list, tuple)):
        items = list(top_bruto)
    else:
        items = [top_bruto]

    if (n_temas > 0 and len(items) != n_temas and len(items) == n_temas * 5
            and all(isinstance(x, str) for x in items)):
        return [", ".join(items[i * 5:(i + 1) * 5]) for i in range(n_temas)]

    salida = [_fila_top_a_texto(x) for x in items]
    while len(salida) < n_temas:
        salida.append("")
    return salida[:n_temas]


# ----------------------------------------------------------------------
# 1) Cargar los resultados de la Parte 1
# ----------------------------------------------------------------------
with open(RUTA_ENTRADA, "r", encoding="utf-8") as f:
    t1 = json.load(f)

if "distribucion_documento_tema" not in t1:
    raise KeyError("'entradas/T1.json' no contiene la clave 'distribucion_documento_tema'.")

n_docs_ref = t1.get("n_documentos")
n_temas_ref = t1.get("n_componentes")

doc_topic = _extraer_matriz(t1["distribucion_documento_tema"])
doc_topic = _orientar_matriz(doc_topic, n_docs_ref, n_temas_ref)

# --- etiquetas reales ---
etiq = t1.get("etiquetas_reales")
if isinstance(etiq, dict):
    etiquetas = [str(etiq[c]) for c in _claves_ordenadas(etiq)]
elif isinstance(etiq, (list, tuple)):
    etiquetas = [str(e) for e in etiq]
else:
    raise ValueError("Formato no reconocido para 'etiquetas_reales'.")

# Si las etiquetas cuadran con la otra dimensión, la matriz estaba transpuesta.
if len(etiquetas) != doc_topic.shape[0]:
    if len(etiquetas) == doc_topic.shape[1]:
        doc_topic = doc_topic.T
    else:
        raise ValueError(
            "El número de etiquetas reales (%d) no coincide con ninguna dimensión "
            "de la distribución documento-tema %s." % (len(etiquetas), doc_topic.shape)
        )

n_docs, n_temas = doc_topic.shape
if n_docs != 12:
    print("Aviso: se esperaban 12 documentos y hay %d; la pureza se dividira entre %d."
          % (n_docs, n_docs))

# --- palabras top por tema (solo para interpretar cada tema LDA) ---
tops_por_tema = _extraer_tops(t1.get("palabras_top_por_tema"), n_temas)

# ----------------------------------------------------------------------
# 2) Tema dominante de cada documento
# ----------------------------------------------------------------------
tema_dominante = np.argmax(doc_topic, axis=1)

# ----------------------------------------------------------------------
# 3) Pureza
# ----------------------------------------------------------------------
resumen_temas = []
tabla_cruce = {}
etiqueta_mayoritaria_por_tema = {}
numerador_pureza = 0

for k in range(n_temas):
    idx = [i for i in range(n_docs) if int(tema_dominante[i]) == k]
    contador = Counter(etiquetas[i] for i in idx)
    if contador:
        etiqueta_mayor, cuenta_mayor = contador.most_common(1)[0]
    else:
        etiqueta_mayor, cuenta_mayor = None, 0
    numerador_pureza += cuenta_mayor
    etiqueta_mayoritaria_por_tema[k] = etiqueta_mayor
    tabla_cruce[str(k)] = dict(contador)
    resumen_temas.append({
        "tema_lda": int(k),
        "palabras_top": tops_por_tema[k] if k < len(tops_por_tema) else "",
        "n_documentos_asignados": int(len(idx)),
        "ids_documentos": [int(i) for i in idx],
        "etiquetas_reales_presentes": dict(contador),
        "etiqueta_real_mayoritaria": etiqueta_mayor,
        "cuenta_etiqueta_mayoritaria": int(cuenta_mayor),
    })

pureza = float(numerador_pureza) / float(n_docs)

# ----------------------------------------------------------------------
# 4) Tabla de asignación y documentos mal agrupados
# ----------------------------------------------------------------------
asignacion = []
mal_agrupados = []

for i in range(n_docs):
    k = int(tema_dominante[i])
    etiq_real = etiquetas[i]
    etiq_mayor = etiqueta_mayoritaria_por_tema[k]
    bien = (etiq_real == etiq_mayor)
    probs = [float(p) for p in doc_topic[i]]
    asignacion.append({
        "id": int(i),
        "etiqueta_real": etiq_real,
        "tema_lda": int(k),
        "probabilidad_tema_dominante": float(max(probs)),
        "distribucion_documento_tema": probs,
        "etiqueta_mayoritaria_del_tema": etiq_mayor,
        "bien_agrupado": bool(bien),
    })
    if not bien:
        mal_agrupados.append({
            "id": int(i),
            "etiqueta_real": etiq_real,
            "tema_lda": int(k),
            "etiqueta_mayoritaria_del_tema": etiq_mayor,
            "probabilidad_tema_dominante": float(max(probs)),
        })

# ----------------------------------------------------------------------
# 5) Guardar todas las cifras en resultados.json
# ----------------------------------------------------------------------
resultados = {
    "asignacion_por_documento": asignacion,
    "pureza": pureza,
    "documentos_mal_agrupados": mal_agrupados,
    "n_documentos_mal_agrupados": int(len(mal_agrupados)),
    "numerador_pureza": int(numerador_pureza),
    "n_documentos": int(n_docs),
    "n_componentes": int(n_temas),
    "tema_dominante_por_documento": [int(t) for t in tema_dominante],
    "distribucion_documento_tema": [[float(p) for p in fila] for fila in doc_topic],
    "resumen_por_tema_lda": resumen_temas,
    "tabla_cruce_tema_lda_vs_etiqueta_real": tabla_cruce,
}

with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 6) Impresión de las cifras principales
# ----------------------------------------------------------------------
print("=" * 104)
print("T3 - Parte 3: ¿Recupera LDA los temas reales?")
print("=" * 104)

print("\nTabla de asignación por documento:")
print("%3s | %-30s | %8s | %12s | %-30s | %s"
      % ("id", "etiqueta real", "tema LDA", "p(tema dom.)",
         "etiqueta mayoritaria del tema", "resultado"))
print("-" * 104)
for a in asignacion:
    res = "bien" if a["bien_agrupado"] else "MAL AGRUPADO"
    print("%3d | %-30s | %8d | %12.6f | %-30s | %s"
          % (a["id"], a["etiqueta_real"], a["tema_lda"],
             a["probabilidad_tema_dominante"],
             str(a["etiqueta_mayoritaria_del_tema"]), res))

print("\nResumen por tema LDA:")
for r in resumen_temas:
    print("  Tema LDA %d: palabras top = [%s]" % (r["tema_lda"], r["palabras_top"]))
    print("      documentos asignados: %s" % (r["ids_documentos"],))
    print("      etiquetas reales presentes: %s" % (r["etiquetas_reales_presentes"],))
    print("      etiqueta mayoritaria: %s (%d documentos)"
          % (r["etiqueta_real_mayoritaria"], r["cuenta_etiqueta_mayoritaria"]))

print("\nCruce tema LDA vs etiqueta real:")
for k in sorted(tabla_cruce, key=_orden_clave):
    print("  Tema LDA %s: %s" % (k, tabla_cruce[k]))

print("\nCifras principales:")
print("  Pureza = %d/%d = %s" % (numerador_pureza, n_docs, pureza))
print("  Número de documentos mal agrupados: %d" % len(mal_agrupados))
if mal_agrupados:
    print("  Documentos mal agrupados:")
    for m in mal_agrupados:
        print("    - id %d: etiqueta real '%s' -> tema LDA %d "
              "(mayoria del tema: '%s', p(tema dom.) = %.6f)"
              % (m["id"], m["etiqueta_real"], m["tema_lda"],
                 m["etiqueta_mayoritaria_del_tema"],
                 m["probabilidad_tema_dominante"]))
else:
    print("  No hay documentos mal agrupados.")

print("\nResultados guardados en '%s'." % RUTA_RESULTADOS)
