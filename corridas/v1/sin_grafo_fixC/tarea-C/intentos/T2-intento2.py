# -*- coding: utf-8 -*-
"""
T2 - Parte 2 - Perplejidad de LDA con 2, 3 y 4 temas.

Correcciones aplicadas respecto al intento anterior:
  1) Se inspeccionan las claves de entradas/T1.json y se extrae la lista real
     de las 23 palabras vacias de la Parte 1 (esta guardada bajo una clave
     cuyo nombre no la identifica; si solo se conservara el conteo, se usaria
     la lista literal del enunciado, transcrita en PALABRAS_VACIAS_ENUNCIADO).
  2) Los 12 textos originales se toman de entradas/T1.json si los conserva, o
     de los archivos de datos del enunciado si existieran. En ningun caso se
     reconstruye el corpus troceando fragmentos del PDF: si los textos no
     estan disponibles, el script aborta con un error explicito en lugar de
     inventar datos.
  3) Se ajusta un UNICO CountVectorizer con esas palabras vacias sobre los 12
     textos reales y se reejecuta LDA (learning_method='batch', max_iter=50,
     random_state=0) con n_components = 2, 3 y 4.
  4) Verificacion de consistencia: la perplejidad con k=3 debe coincidir con
     la de la Parte 1 (137.7877) antes de dar por valida la tabla; si no
     coincide, el script se detiene sin escribir resultados.json.

Esta subtarea NO genera figura PNG. Salida: resultados.json.
"""

import json
import os
import sys

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------------------------------------------------------
# Parametros fijos (identicos a los de la Parte 1)
# ---------------------------------------------------------------------------
N_TEMAS = (2, 3, 4)
LEARNING_METHOD = "batch"
MAX_ITER = 50
RANDOM_STATE = 0

RUTA_T1 = os.path.join("entradas", "T1.json")
PERPLEJIDAD_K3_PARTE1 = 137.7877   # perplejidad con k=3 de la Parte 1
TOLERANCIA_K3 = 0.05               # tolerancia absoluta de la verificacion

# Lista literal de las 23 palabras vacias del enunciado de la practica
# (transcripcion de la Parte 1). Solo se emplea si entradas/T1.json no
# conserva la lista real bajo ninguna clave.
PALABRAS_VACIAS_ENUNCIADO = [
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on",
    "at", "by", "for", "with", "from", "as", "is", "are", "was", "were",
    "be", "this",
]

# Nombres de campo que podria tener cada documento si T1.json guardara los
# textos como lista de diccionarios.
CAMPOS_TEXTO = ("texto", "text", "documento", "contenido", "cuerpo",
                "content", "body", "raw")


# ---------------------------------------------------------------------------
# Utilidades de inspeccion de la estructura JSON de la Parte 1
# ---------------------------------------------------------------------------
def recorrer(obj, ruta="t1"):
    """Genera (ruta, valor) para cada nodo de la estructura JSON."""
    if isinstance(obj, dict):
        for clave, valor in obj.items():
            yield from recorrer(valor, f"{ruta}.{clave}")
    elif isinstance(obj, list):
        yield ruta, obj
        for i, valor in enumerate(obj):
            yield from recorrer(valor, f"{ruta}[{i}]")
    else:
        yield ruta, obj


def es_lista_de_cadenas(valor):
    return (
        isinstance(valor, list)
        and len(valor) > 0
        and all(isinstance(x, str) for x in valor)
    )


def clave_raiz(ruta):
    """Clave de primer nivel de una ruta: 't1.a[0].b' -> 'a'."""
    partes = ruta.split(".")
    return partes[1].split("[")[0] if len(partes) > 1 else ruta


def _prioridad_ruta(ruta):
    """Preferencia entre candidatas a ser la lista de palabras vacias."""
    nombre = ruta.lower()
    if "vacia" in nombre or "stop" in nombre:
        return 0
    if "vocabulario" in nombre or "top5" in nombre:
        return 2
    return 1


def extraer_palabras_vacias(t1, n_pv, tam_vocabulario):
    """Extrae la lista real de las palabras vacias de la Parte 1.

    Se busca en TODAS las claves de T1.json (la lista esta guardada bajo un
    nombre que no la identifica) una lista de exactamente n_pv cadenas. Si
    solo se conserva el conteo, se transcribe la lista literal del enunciado.
    """
    candidatas = []
    for ruta, valor in recorrer(t1):
        if es_lista_de_cadenas(valor) and len(valor) == n_pv:
            if (
                clave_raiz(ruta) == "vocabulario"
                and tam_vocabulario is not None
                and len(valor) == tam_vocabulario
            ):
                continue  # es la lista del vocabulario, no las palabras vacias
            candidatas.append((ruta, list(valor)))
    if candidatas:
        candidatas.sort(key=lambda par: _prioridad_ruta(par[0]))
        if len(candidatas) > 1:
            print(
                f"[aviso] Varias listas de {n_pv} cadenas en T1.json: "
                f"{', '.join(r for r, _ in candidatas)}; "
                f"se usa '{candidatas[0][0]}'."
            )
        ruta, lista = candidatas[0]
        palabras = [str(w).strip().lower() for w in lista]
        return palabras, (
            f"lista real de la Parte 1, extraida de entradas/T1.json "
            f"(ruta '{ruta}')"
        )

    # La lista podria estar guardada como una unica cadena separada por comas.
    for ruta, valor in recorrer(t1):
        if isinstance(valor, str) and valor.count(",") >= n_pv - 1:
            partes = [p.strip().lower() for p in valor.split(",") if p.strip()]
            if len(partes) == n_pv and all(p.isalpha() for p in partes):
                return partes, (
                    f"lista de la Parte 1, extraida de entradas/T1.json "
                    f"(ruta '{ruta}', cadena separada por comas)"
                )

    # Ultimo recurso: transcripcion literal de la lista del enunciado.
    return list(PALABRAS_VACIAS_ENUNCIADO), (
        "lista literal transcrita del enunciado (T1.json solo conserva el "
        "conteo n_palabras_vacias)"
    )


def candidatos_textos_en_t1(t1, n_docs):
    """Genera (ruta, [textos]) para cada estructura de T1.json que podria
    contener los n_docs textos originales."""
    # a) Listas de exactamente n_docs cadenas.
    for ruta, valor in recorrer(t1):
        if es_lista_de_cadenas(valor) and len(valor) == n_docs:
            yield ruta, [str(s) for s in valor]
    # b) Diccionarios con n_docs valores de tipo cadena (id -> texto).
    for ruta, valor in recorrer(t1):
        if isinstance(valor, dict) and len(valor) == n_docs:
            valores = list(valor.values())
            if all(isinstance(v, str) and v.strip() for v in valores):
                yield f"{ruta} (dict id->texto)", [str(v) for v in valores]
    # c) Listas de n_docs diccionarios con un campo que contenga el texto.
    for ruta, valor in recorrer(t1):
        if (
            isinstance(valor, list)
            and len(valor) == n_docs
            and all(isinstance(d, dict) for d in valor)
        ):
            extraidos = []
            campo = None
            for d in valor:
                actual = next(
                    (k for k in CAMPOS_TEXTO if k in d and isinstance(d[k], str)),
                    None,
                )
                if actual is None:
                    break
                campo = actual
                extraidos.append(d[actual])
            if len(extraidos) == n_docs:
                yield f"{ruta} (campo '{campo}')", extraidos
    # d) Una unica cadena con los textos separados por lineas en blanco.
    for ruta, valor in recorrer(t1):
        if isinstance(valor, str) and "\n\n" in valor:
            partes = [p.strip() for p in valor.split("\n\n") if p.strip()]
            if len(partes) == n_docs:
                yield f"{ruta} (cadena multilinea)", partes


def textos_desde_t1(t1, n_docs):
    """Devuelve (textos, ruta, longitud_media) o (None, None, mejor_media)."""
    mejor = None
    for ruta, textos in candidatos_textos_en_t1(t1, n_docs):
        media = sum(len(s) for s in textos) / float(len(textos))
        if mejor is None or media > mejor[0]:
            mejor = (media, ruta, textos)
    if mejor is None:
        return None, None, None
    media, ruta, textos = mejor
    if media < 40.0:
        # Demasiado cortas para ser documentos: parecen identificadores.
        return None, None, media
    return textos, ruta, media


def textos_desde_archivos(t1, n_docs):
    """Intenta leer los textos desde archivos de datos del enunciado, cuando
    ids_documentos son nombres de archivo .txt existentes en la carpeta
    actual. Devuelve (textos, descripcion) o (None, None)."""
    ids = t1.get("ids_documentos")
    if not (
        isinstance(ids, list)
        and len(ids) == n_docs
        and all(isinstance(s, str) and s.strip() for s in ids)
    ):
        return None, None
    if not all(
        s.endswith(".txt")
        and os.path.basename(s) == s
        and ".." not in s
        for s in ids
    ):
        return None, None
    if not all(os.path.exists(s) for s in ids):
        return None, None
    textos = []
    try:
        for nombre in ids:
            with open(nombre, "r", encoding="utf-8") as fh:
                textos.append(fh.read())
    except OSError:
        return None, None
    return textos, (
        "archivos de datos del enunciado "
        "(nombres tomados de ids_documentos de T1.json)"
    )


# ---------------------------------------------------------------------------
# 1) Carga e inspeccion de entradas/T1.json
# ---------------------------------------------------------------------------
if not os.path.exists(RUTA_T1):
    sys.exit(
        "[ERROR] No se encuentra 'entradas/T1.json' (resultados de la Parte 1).\n"
        "Sin ese archivo no se pueden recuperar las palabras vacias ni los\n"
        "textos originales, y esta subtarea no dispone de archivos de datos."
    )

with open(RUTA_T1, "r", encoding="utf-8") as fh:
    t1 = json.load(fh)

if not isinstance(t1, dict):
    sys.exit("[ERROR] 'entradas/T1.json' no contiene un diccionario de resultados.")

print("=== [1] Inspeccion de entradas/T1.json ===")
print(f"Numero de claves: {len(t1)}")
for clave in t1:
    valor = t1[clave]
    if isinstance(valor, list):
        if valor and es_lista_de_cadenas(valor):
            detalle = f"lista de {len(valor)} cadenas (p. ej. {valor[0][:48]!r})"
        elif valor and isinstance(valor[0], list):
            detalle = (
                f"lista de {len(valor)} listas "
                f"(primera de longitud {len(valor[0])})"
            )
        elif valor and isinstance(valor[0], dict):
            detalle = (
                f"lista de {len(valor)} diccionarios "
                f"(claves: {sorted(valor[0].keys())[:6]})"
            )
        elif valor and isinstance(valor[0], (int, float)):
            detalle = f"lista de {len(valor)} numeros"
        else:
            detalle = f"lista de {len(valor)} elementos"
    elif isinstance(valor, dict):
        detalle = f"diccionario con claves {sorted(valor.keys())[:6]}"
    else:
        detalle = repr(valor)
    print(f"  - {clave}: {detalle}")

n_documentos = int(t1.get("n_documentos", 12))
n_pv = int(t1.get("n_palabras_vacias", len(PALABRAS_VACIAS_ENUNCIADO)))
tam_vocab_t1 = t1.get("tamano_vocabulario")
tam_vocab_t1 = int(tam_vocab_t1) if isinstance(tam_vocab_t1, (int, float)) else None

# ---------------------------------------------------------------------------
# 2) Palabras vacias reales de la Parte 1 y textos originales
# ---------------------------------------------------------------------------
palabras_vacias, fuente_pv = extraer_palabras_vacias(t1, n_pv, tam_vocab_t1)

print()
print("=== [2] Origen de los datos ===")
print(f"Palabras vacias ({len(palabras_vacias)}): {fuente_pv}")
print(f"  lista: {palabras_vacias}")
if len(palabras_vacias) != n_pv:
    print(
        f"[aviso] La lista recuperada tiene {len(palabras_vacias)} palabras "
        f"y T1.json indica n_palabras_vacias = {n_pv}."
    )

textos = None
ruta_textos = None
media_textos = None
fuente_textos = None

t1_textos, t1_ruta, t1_media = textos_desde_t1(t1, n_documentos)
if t1_textos is not None:
    textos, ruta_textos, media_textos = t1_textos, t1_ruta, t1_media
    fuente_textos = f"entradas/T1.json (ruta '{t1_ruta}')"
else:
    arch_textos, arch_desc = textos_desde_archivos(t1, n_documentos)
    if arch_textos is not None:
        media_arch = sum(len(s) for s in arch_textos) / float(len(arch_textos))
        if media_arch >= 40.0:
            textos = arch_textos
            ruta_textos = arch_desc
            media_textos = media_arch
            fuente_textos = arch_desc

if textos is None:
    pista = (
        f"mejor lista candidata con longitud media {media_textos:.1f} "
        "caracteres (demasiado corta para ser documentos)"
        if media_textos is not None
        else "ninguna estructura con 12 cadenas de texto"
    )
    print(f"Textos originales: NO disponibles ({pista}).")
    sys.exit(
        "\n[ERROR] No se han podido obtener los 12 textos originales de la practica.\n"
        f"  - '{RUTA_T1}' no los conserva (solo guarda metadatos de la Parte 1).\n"
        "  - Esta subtarea no dispone de archivos de datos con los textos.\n"
        "Segun la correccion recibida, NO se reconstruye el corpus troceando\n"
        "fragmentos del PDF ni se inventan datos: se aborta la ejecucion aqui.\n"
        "Para resolver la subtarea hay que regenerar T1.json conservando los\n"
        "textos o aportar los archivos de datos del enunciado."
    )

print(
    f"Textos originales: {len(textos)} documentos "
    f"(longitud media {media_textos:.0f} caracteres), recuperados de: {ruta_textos}"
)
for i, texto in enumerate(textos):
    print(f"  [{i:2d}] {len(texto):5d} caracteres | {texto[:64]!r}")

if len(textos) != n_documentos:
    sys.exit(
        f"[ERROR] Se esperaban {n_documentos} textos y se han recuperado "
        f"{len(textos)}."
    )

# ---------------------------------------------------------------------------
# 3) Un unico CountVectorizer y LDA con k = 2, 3 y 4
# ---------------------------------------------------------------------------
print()
print("=== [3] CountVectorizer unico y LDA (batch, max_iter=50, random_state=0) ===")
vectorizador = CountVectorizer(stop_words=palabras_vacias)
X = vectorizador.fit_transform(textos)
obtener_nombres = getattr(vectorizador, "get_feature_names_out", None)
if obtener_nombres is not None:
    vocabulario = [str(w) for w in obtener_nombres()]
else:
    vocabulario = [str(w) for w in vectorizador.get_feature_names()]
tokens_totales = int(X.sum())
print(
    f"Matriz documento-termino: {X.shape[0]} documentos x {X.shape[1]} terminos "
    f"({tokens_totales} tokens)"
)
if tam_vocab_t1 is not None:
    print(
        f"Tamano del vocabulario en la Parte 1: {tam_vocab_t1} -> "
        f"{'coincide' if X.shape[1] == tam_vocab_t1 else 'NO coincide'}"
    )

perplejidades = {}
for k in N_TEMAS:
    lda = LatentDirichletAllocation(
        n_components=k,
        learning_method=LEARNING_METHOD,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )
    lda.fit(X)
    perplejidades[k] = float(lda.perplexity(X))
    print(f"  k={k}: perplejidad = {perplejidades[k]:.6f}")

# ---------------------------------------------------------------------------
# 4) Verificacion de consistencia con la Parte 1 (k = 3)
# ---------------------------------------------------------------------------
ref_t1 = t1.get("perplejidad_k3")
referencia_k3 = float(ref_t1) if isinstance(ref_t1, (int, float)) else PERPLEJIDAD_K3_PARTE1
if ref_t1 is not None and abs(referencia_k3 - PERPLEJIDAD_K3_PARTE1) > TOLERANCIA_K3:
    print(
        f"[aviso] T1.json indica perplejidad_k3 = {referencia_k3:.4f}, que difiere "
        f"de la referencia del enunciado ({PERPLEJIDAD_K3_PARTE1})."
    )
diferencia_k3 = abs(perplejidades[3] - referencia_k3)
coincide_k3 = diferencia_k3 <= TOLERANCIA_K3

print()
print("=== [4] Verificacion de consistencia (k = 3 frente a la Parte 1) ===")
print(f"  Perplejidad k=3 de la Parte 1 : {referencia_k3:.4f}")
print(f"  Perplejidad k=3 recalculada   : {perplejidades[3]:.4f}")
print(
    f"  Diferencia absoluta           : {diferencia_k3:.6f} "
    f"(tolerancia {TOLERANCIA_K3})"
)

if not coincide_k3:
    sys.exit(
        "\n[ERROR] Verificacion de consistencia fallida: la perplejidad con k=3\n"
        f"recalculada ({perplejidades[3]:.4f}) no coincide con la de la Parte 1\n"
        f"({referencia_k3:.4f}; referencia del enunciado: {PERPLEJIDAD_K3_PARTE1}).\n"
        "La tabla NO se da por valida y no se escribe resultados.json. Detenerse\n"
        "y revisar: (a) los 12 textos, (b) la lista de palabras vacias, (c) el\n"
        "CountVectorizer y (d) los parametros de LDA.\n"
        f"Diagnostico: textos = {len(textos)} ({ruta_textos}); "
        f"palabras vacias = {len(palabras_vacias)}; "
        f"vocabulario = {X.shape[1]} terminos (Parte 1: {tam_vocab_t1}); "
        f"tokens = {tokens_totales}."
    )
print("  [ok] La perplejidad con k=3 coincide con la Parte 1: tabla valida.")

# ---------------------------------------------------------------------------
# 5) Tabla comparativa, guardado de resultados y resumen
# ---------------------------------------------------------------------------
tabla_perplejidades = [
    {
        "n_temas": int(k),
        "n_componentes": int(k),
        "perplejidad": float(perplejidades[k]),
    }
    for k in N_TEMAS
]
mejor_k = min(N_TEMAS, key=lambda k: perplejidades[k])

salida = {
    "subtarea": "T2 - Parte 2 - Perplejidad con 2, 3 y 4 temas",
    "n_documentos": int(len(textos)),
    "textos": [str(t) for t in textos],
    "ids_documentos": t1.get("ids_documentos"),
    "etiquetas_reales": t1.get("etiquetas_reales"),
    "n_palabras_vacias": int(len(palabras_vacias)),
    "palabras_vacias": [str(w) for w in palabras_vacias],
    "fuente_palabras_vacias": fuente_pv,
    "fuente_textos": fuente_textos,
    "learning_method": LEARNING_METHOD,
    "max_iter": int(MAX_ITER),
    "random_state": int(RANDOM_STATE),
    "n_componentes_probados": [int(k) for k in N_TEMAS],
    "tamano_vocabulario": int(X.shape[1]),
    "vocabulario": vocabulario,
    "tokens_totales": int(tokens_totales),
    "perplejidad_k2": float(perplejidades[2]),
    "perplejidad_k3": float(perplejidades[3]),
    "perplejidad_k4": float(perplejidades[4]),
    "perplejidades_por_tema": {str(k): float(perplejidades[k]) for k in N_TEMAS},
    "tabla_perplejidades": tabla_perplejidades,
    "mejor_n_componentes": int(mejor_k),
    "verificacion_k3": {
        "referencia_parte1": float(referencia_k3),
        "referencia_enunciado": float(PERPLEJIDAD_K3_PARTE1),
        "perplejidad_recalculada": float(perplejidades[3]),
        "diferencia_absoluta": float(diferencia_k3),
        "tolerancia": float(TOLERANCIA_K3),
        "coincide": bool(coincide_k3),
    },
    "genera_figura_png": False,
}
if tam_vocab_t1 is not None:
    salida["tamano_vocabulario_parte1"] = int(tam_vocab_t1)
    salida["coincide_tamano_vocabulario"] = bool(X.shape[1] == tam_vocab_t1)

with open("resultados.json", "w", encoding="utf-8") as fh:
    json.dump(salida, fh, ensure_ascii=False, indent=2)

print()
print("=== [5] Tabla comparativa de perplejidad (sobre el corpus) ===")
print(f"{'n_temas':>8} | {'perplejidad':>14}")
print("-" * 27)
for fila in tabla_perplejidades:
    print(f"{fila['n_temas']:>8} | {fila['perplejidad']:>14.6f}")
print("-" * 27)
print(f"Menor perplejidad: n_components = {mejor_k}")
print()
print("[ok] Cifras guardadas en resultados.json (esta subtarea no genera figura PNG).")
