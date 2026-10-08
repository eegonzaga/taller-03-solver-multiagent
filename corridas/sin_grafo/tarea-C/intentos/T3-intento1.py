# -*- coding: utf-8 -*-
"""
T3 - Parte 3 — ¿Recupera LDA los temas reales?

Usa la distribución documento-tema del modelo LDA de 3 temas ajustado en la
Parte 1 (guardada en entradas/T1.json) para:
  1) asignar a cada uno de los 12 documentos su tema dominante
     (mayor componente de su distribución documento-tema),
  2) calcular la pureza: para cada tema LDA se cuentan los documentos de la
     etiqueta real más frecuente entre los asignados a él, se suman esas
     cuentas y se divide entre 12,
  3) construir la tabla de asignación (id, etiqueta real, tema LDA) y
  4) identificar los documentos mal agrupados (aquellos cuya etiqueta real
     difiere de la etiqueta mayoritaria del tema LDA al que fueron asignados).

Salidas:
  - resultados.json (asignacion_por_documento, pureza, documentos_mal_agrupados, ...)
  - impresión de la tabla y de las cifras principales.
  - Sin figuras PNG (no se requieren en esta subtarea).
"""

import json
from collections import Counter

import numpy as np

RUTA_ENTRADA = "entradas/T1.json"
RUTA_RESULTADOS = "resultados.json"


def _claves_ordenadas(d):
    """Ordena claves de un dict numéricamente si es posible."""
    try:
        return sorted(d.keys(), key=lambda x: int(x))
    except (TypeError, ValueError):
        return sorted(d.keys())


def _extraer_tops(top_bruto, n_temas):
    """Normaliza 'palabras_top_por_tema' a una lista de cadenas (una por tema)."""
    if top_bruto is None:
        return [""] * n_temas
    if isinstance(top_bruto, dict):
        top_bruto = [top_bruto[c] for c in _claves_ordenadas(top_bruto)]
    if isinstance(top_bruto, (list, tuple)):
        if len(top_bruto) == n_temas:
            salida = []
            for x in top_bruto:
                if isinstance(x, (list, tuple)):
                    salida.append(", ".join(str(p) for p in x))
                else:
                    salida.append(str(x))
            return salida
        if len(top_bruto) == n_temas * 5:
            return [", ".join(str(p) for p in top_bruto[i * 5:(i + 1) * 5])
                    for i in range(n_temas)]
        texto = ", ".join(str(p) for p in top_bruto)
        return [texto] + [""] * (n_temas - 1)
    return [str(top_bruto)] + [""] * (n_temas - 1)


# ----------------------------------------------------------------------
# 1) Cargar los resultados de la Parte 1
# ----------------------------------------------------------------------
with open(RUTA_ENTRADA, "r", encoding="utf-8") as f:
    t1 = json.load(f)

# --- distribución documento-tema (12 x 3) ---
bruta = t1["distribucion_documento_tema"]
if isinstance(bruta, dict):
    doc_topic = np.array([bruta[c] for c in _claves_ordenadas(bruta)], dtype=float)
else:
    doc_topic = np.array(bruta, dtype=float)

n_docs_ref = t1.get("n_documentos")
n_temas_ref = t1.get("n_componentes")

# Corrección defensiva por si la matriz se hubiera guardado transpuesta
if (n_docs_ref is not None and doc_topic.ndim == 2
        and doc_topic.shape[0] != n_docs_ref and doc_topic.shape[1] == n_docs_ref):
    doc_topic = doc_topic.T

n_docs, n_temas = doc_topic.shape
if n_docs != 12:
    print(f"Aviso: se esperaban 12 documentos y hay {n_docs}; "
          f"la pureza se dividirá entre {n_docs}.")

# --- etiquetas reales ---
etiq = t1["etiquetas_reales"]
if isinstance(etiq, dict):
    etiquetas = [str(etiq[c]) for c in _claves_ordenadas(etiq)]
else:
    etiquetas = [str(e) for e in etiq]

if len(etiquetas) != n_docs:
    raise ValueError("El número de etiquetas reales no coincide con el número "
                     "de documentos de la distribución documento-tema.")

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
    etiquetas_en_tema = [etiquetas[i] for i in idx]
    contador = Counter(etiquetas_en_tema)
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
    "resumen_por_tema_lda": resumen_temas,
    "tabla_cruce_tema_lda_vs_etiqueta_real": tabla_cruce,
}

with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 6) Impresión de las cifras principales
# ----------------------------------------------------------------------
print("=" * 100)
print("T3 - Parte 3: ¿Recupera LDA los temas reales?")
print("=" * 100)

print("\nTabla de asignación por documento:")
print(f"{'id':>3} | {'etiqueta real':<28} | {'tema LDA':>8} | "
      f"{'p(tema dom.)':>12} | {'etiqueta mayoritaria del tema':<30} | resultado")
print("-" * 110)
for a in asignacion:
    res = "bien" if a["bien_agrupado"] else "MAL AGRUPADO"
    print(f"{a['id']:>3} | {a['etiqueta_real']:<28} | {a['tema_lda']:>8} | "
          f"{a['probabilidad_tema_dominante']:>12.6f} | "
          f"{str(a['etiqueta_mayoritaria_del_tema']):<30} | {res}")

print("\nResumen por tema LDA:")
for r in resumen_temas:
    print(f"  Tema LDA {r['tema_lda']}: palabras top = [{r['palabras_top']}]")
    print(f"      documentos asignados: {r['ids_documentos']}")
    print(f"      etiquetas reales presentes: {r['etiquetas_reales_presentes']}")
    print(f"      etiqueta mayoritaria: {r['etiqueta_real_mayoritaria']} "
          f"({r['cuenta_etiqueta_mayoritaria']} documentos)")

print("\nCruce tema LDA vs etiqueta real:")
for k in sorted(tabla_cruce, key=lambda x: int(x)):
    print(f"  Tema LDA {k}: {tabla_cruce[k]}")

print("\nCifras principales:")
print(f"  Pureza = {numerador_pureza}/{n_docs} = {pureza}")
print(f"  Número de documentos mal agrupados: {len(mal_agrupados)}")
if mal_agrupados:
    print("  Documentos mal agrupados:")
    for m in mal_agrupados:
        print(f"    - id {m['id']}: etiqueta real '{m['etiqueta_real']}' -> "
              f"tema LDA {m['tema_lda']} (mayoría '{m['etiqueta_mayoritaria_del_tema']}')")
else:
    print("  No hay documentos mal agrupados.")

print(f"\nResultados guardados en '{RUTA_RESULTADOS}'.")
