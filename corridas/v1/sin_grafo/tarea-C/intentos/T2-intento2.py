# -*- coding: utf-8 -*-
"""
T2 - Parte 2 - Numero de temas (2, 3 y 4)
=========================================

Con el mismo CountVectorizer de la Parte 1 (misma lista de palabras vacias y
el resto de parametros por defecto) y los mismos parametros de LDA
(learning_method='batch', max_iter=50, random_state=0), se ajustan modelos
con n_components = 2, 3 y 4 sobre los doce textos de la practica y se
reporta la perplejidad de cada modelo en una sola tabla comparativa.

Correcciones aplicadas tras la revision del intento anterior:
  * Se ELIMINA el corpus inventado. Los doce textos se toman EXACTAMENTE de
    la misma fuente que uso T1: se cargan de entradas/T1.json (claves
    'textos', 'corpus', 'documentos', ...), de otro JSON de la carpeta de
    trabajo (p. ej. corpus.json) o de los doce archivos de texto (.txt/.md)
    presentes en la carpeta de trabajo o en una subcarpeta tipica ('corpus',
    'textos', 'datos', ...). Si no se pueden cargar, se aborta con error: no
    se sustituyen por textos inventados.
  * Verificacion obligatoria ANTES de reportar cualquier cifra:
      (1) el vocabulario obtenido debe coincidir con el guardado en T1.json
          (Jaccard = 1.0 / listas identicas), y
      (2) la perplejidad con k=3 debe reproducir perplejidad_k3 de T1
          (137.78770156685167) dentro de una tolerancia relativa de 1e-6.
    Si alguna verificacion falla, se aborta con error y no se genera la
    tabla comparativa ni se guardan cifras en resultados.json.

Entradas:
  * entradas/T1.json : resultados de la Parte 1 (imprescindible para verificar).
  * Los doce textos de la practica (misma fuente que T1).

Salidas:
  * resultados.json : perplejidad_k2, perplejidad_k3, perplejidad_k4, la
    tabla comparativa y cifras auxiliares, con precision completa.
  * Impresion por pantalla de las verificaciones y de la tabla.

Nota: esta subtarea NO requiere ninguna figura PNG.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# El aviso de sklearn sobre palabras vacias de un solo caracter es inofensivo:
# el tokenizador por defecto nunca genera tokens de un solo caracter.
warnings.filterwarnings("ignore", message="Your stop_words may be inconsistent")

RUTA_T1 = os.path.join("entradas", "T1.json")
RUTA_RESULTADOS = "resultados.json"

VALORES_K = (2, 3, 4)
N_PALABRAS_TOP = 5
N_DOCUMENTOS_ESPERADOS = 12
TOLERANCIA_RELATIVA = 1e-6
# Valor de perplejidad_k3 registrado por T1 (referencia cruzada indicada en la
# revision; la referencia primaria es siempre la que figure en entradas/T1.json).
PERPLEJIDAD_K3_REFERENCIA = 137.78770156685167

# Nombres de clave bajo los que T1.json podria guardar los doce textos.
CLAVES_TEXTOS_T1 = (
    "textos", "corpus", "documentos", "textos_corpus", "textos_practica",
    "textos_originales", "corpus_textos", "documentos_originales",
    "doce_textos", "corpus_completo",
)

# Directorios tipicos (rutas relativas) donde pueden estar los doce textos.
DIRECTORIOS_CANDIDATOS = (".", "corpus", "textos", "datos", "data", "docs",
                          "documentos", "entradas")

# Misma lista de palabras vacias que aplica el CountVectorizer de la Parte 1
PALABRAS_VACIAS = [
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "aqui", "aquí",
    "asi", "así", "como", "con", "contra", "cual", "cuales", "cuando",
    "cuanto", "de", "del", "desde", "donde", "durante", "e", "el", "ella",
    "ellas", "ello", "ellos", "en", "entre", "era", "eran", "eres", "es",
    "esa", "esas", "ese", "eso", "esos", "esta", "estaba", "estan", "está",
    "están", "estar", "estas", "este", "esto", "estos", "estoy", "fue",
    "fueron", "ha", "habia", "había", "han", "hasta", "hay", "la", "las",
    "le", "les", "lo", "los", "mas", "más", "me", "mi", "mis", "mucho",
    "muy", "nada", "ni", "no", "nos", "nosotras", "nosotros", "nuestra",
    "nuestras", "nuestro", "nuestros", "o", "otra", "otras", "otro",
    "otros", "para", "pero", "poco", "por", "porque", "que", "qué",
    "quien", "quienes", "se", "segun", "según", "ser", "si", "sí", "sin",
    "sobre", "soy", "su", "sus", "tal", "tambien", "también", "tan",
    "tanto", "te", "tiene", "tienen", "toda", "todas", "todo", "todos",
    "tras", "tu", "tus", "u", "un", "una", "unas", "unos", "usted",
    "ustedes", "va", "van", "vamos", "ya", "yo", "y",
]


# ----------------------------------------------------------------------
# Funciones auxiliares
# ----------------------------------------------------------------------
def abortar(mensaje):
    """Aborta con error: no se reporta ninguna cifra no verificada."""
    print("\n[ERROR] " + mensaje)
    try:
        with open(RUTA_RESULTADOS, "w", encoding="utf-8") as manejador:
            json.dump(
                {
                    "subtarea": "T2 - Parte 2 - Numero de temas (2, 3 y 4)",
                    "verificacion_superada": False,
                    "error": mensaje,
                },
                manejador,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[ERROR] {RUTA_RESULTADOS} contiene unicamente la marca de error; "
              "no se reportan resultados no verificados.")
    except OSError as error:
        print(f"[ERROR] No se pudo escribir la marca de error: {error}")
    raise SystemExit(1)


def a_nativo(objeto):
    """Convierte tipos de NumPy a tipos nativos de Python (para json.dump)."""
    if isinstance(objeto, np.integer):
        return int(objeto)
    if isinstance(objeto, np.floating):
        return float(objeto)
    if isinstance(objeto, np.bool_):
        return bool(objeto)
    if isinstance(objeto, np.ndarray):
        return objeto.tolist()
    return str(objeto)


def cargar_t1():
    """Carga entradas/T1.json y comprueba que contiene lo necesario."""
    if not os.path.exists(RUTA_T1):
        abortar(f"No se encuentra {RUTA_T1}. La verificacion contra la Parte 1 "
                "es obligatoria, por lo que no se puede continuar.")
    try:
        with open(RUTA_T1, "r", encoding="utf-8") as manejador:
            t1 = json.load(manejador)
    except (OSError, ValueError) as error:
        abortar(f"No se pudo leer {RUTA_T1}: {error}")
    if not isinstance(t1, dict):
        abortar(f"{RUTA_T1} no contiene un objeto JSON con los resultados de T1.")
    for clave in ("vocabulario", "perplejidad_k3"):
        if clave not in t1:
            abortar(f"{RUTA_T1} no contiene la clave '{clave}', necesaria para "
                    "verificar que se usan el mismo corpus y protocolo que en T1.")
    print(f"[info] Resultados de T1 cargados desde {RUTA_T1}.")
    return t1


def normalizar_vocabulario(valor):
    """Devuelve la lista de palabras del vocabulario guardado por T1."""
    if isinstance(valor, dict):
        try:
            pares = sorted(valor.items(), key=lambda par: int(par[1]))
            return [str(palabra) for palabra, _ in pares]
        except (TypeError, ValueError):
            return sorted(str(palabra) for palabra in valor)
    if isinstance(valor, list):
        return [str(palabra) for palabra in valor]
    return None


def normalizar_lista_textos(valor, n_esperado):
    """Devuelve una lista de n_esperado textos (str) o None si no es valida."""
    if not isinstance(valor, list) or len(valor) != n_esperado:
        return None
    textos = []
    for elemento in valor:
        if isinstance(elemento, str):
            texto = elemento
        elif isinstance(elemento, list) and all(isinstance(t, str) for t in elemento):
            texto = " ".join(elemento)  # documento guardado como lista de tokens
        else:
            return None
        texto = texto.strip()
        if not texto:
            return None
        textos.append(texto)
    return textos


def cargar_textos_desde_t1(t1, n_esperado):
    """Carga los doce textos desde entradas/T1.json, si estan guardados ahi."""
    for clave in CLAVES_TEXTOS_T1:
        if clave in t1:
            textos = normalizar_lista_textos(t1[clave], n_esperado)
            if textos is not None:
                print(f"[info] Corpus cargado desde {RUTA_T1} (clave '{clave}'): "
                      f"{len(textos)} textos.")
                return textos, f"{RUTA_T1} (clave '{clave}')"
    return None, None


def cargar_textos_desde_json_sueltos(n_esperado):
    """Busca listas de textos en otros JSON de la carpeta de trabajo o de
    'entradas' (p. ej. corpus.json)."""
    for directorio in (".", "entradas"):
        if not os.path.isdir(directorio):
            continue
        for nombre in sorted(os.listdir(directorio)):
            if not nombre.lower().endswith(".json"):
                continue
            if nombre in ("resultados.json", os.path.basename(RUTA_T1)):
                continue
            ruta = os.path.join(directorio, nombre)
            if not os.path.isfile(ruta):
                continue
            try:
                with open(ruta, "r", encoding="utf-8") as manejador:
                    datos = json.load(manejador)
            except (OSError, ValueError):
                continue
            if isinstance(datos, dict):
                candidatos = [datos[c] for c in CLAVES_TEXTOS_T1 if c in datos]
            elif isinstance(datos, list):
                candidatos = [datos]
            else:
                candidatos = []
            for candidato in candidatos:
                textos = normalizar_lista_textos(candidato, n_esperado)
                if textos is not None:
                    print(f"[info] Corpus cargado desde {ruta}: {len(textos)} textos.")
                    return textos, ruta
    return None, None


def cargar_textos_desde_archivos(n_esperado):
    """Busca exactamente n_esperado archivos de texto (.txt/.text/.md) en los
    directorios tipicos de la carpeta de trabajo (solo rutas relativas)."""
    excluidos = ("readme", "requirements")
    for directorio in DIRECTORIOS_CANDIDATOS:
        if not os.path.isdir(directorio):
            continue
        rutas = []
        for nombre in sorted(os.listdir(directorio)):
            if not nombre.lower().endswith((".txt", ".text", ".md")):
                continue
            if nombre.lower().startswith(excluidos):
                continue
            ruta = os.path.join(directorio, nombre)
            if os.path.isfile(ruta):
                rutas.append(ruta)
        if len(rutas) != n_esperado:
            continue
        textos = []
        valido = True
        for ruta in rutas:
            try:
                with open(ruta, "r", encoding="utf-8") as manejador:
                    texto = manejador.read().strip()
            except (OSError, UnicodeDecodeError):
                valido = False
                break
            if not texto:
                valido = False
                break
            textos.append(texto)
        if not valido:
            continue
        print(f"[info] Corpus cargado desde {len(rutas)} archivos de texto en "
              f"'{os.path.normpath(directorio)}'.")
        return textos, f"{os.path.normpath(directorio)} ({len(rutas)} archivos de texto)"
    return None, None


# ----------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------
def main():
    linea = "=" * 72
    print(linea)
    print("T2 - Parte 2 - Numero de temas (k = 2, 3 y 4)")
    print("Protocolo: mismo corpus/CountVectorizer/LDA que en la Parte 1, con")
    print("verificacion obligatoria contra entradas/T1.json antes de reportar.")
    print(linea)

    # ------------------------------------------------------------------
    # 1) Resultados de la Parte 1 (referencia obligatoria para verificar)
    # ------------------------------------------------------------------
    t1 = cargar_t1()

    vocab_t1 = normalizar_vocabulario(t1["vocabulario"])
    if vocab_t1 is None:
        abortar("La clave 'vocabulario' de T1.json no es una lista ni un "
                "diccionario utilizable.")
    try:
        perp_k3_t1 = float(t1["perplejidad_k3"])
    except (TypeError, ValueError):
        abortar("La clave 'perplejidad_k3' de T1.json no es un numero utilizable.")

    if abs(perp_k3_t1 - PERPLEJIDAD_K3_REFERENCIA) > TOLERANCIA_RELATIVA * abs(PERPLEJIDAD_K3_REFERENCIA):
        print(f"[aviso] perplejidad_k3 en T1.json ({perp_k3_t1!r}) difiere del valor "
              f"de referencia {PERPLEJIDAD_K3_REFERENCIA!r}; se usa el valor de "
              "T1.json como referencia de verificacion.")

    n_esperado = N_DOCUMENTOS_ESPERADOS
    if t1.get("n_documentos") is not None:
        try:
            n_esperado = int(t1["n_documentos"])
        except (TypeError, ValueError):
            pass

    # ------------------------------------------------------------------
    # 2) Carga de los doce textos reales (misma fuente que T1; nada inventado)
    # ------------------------------------------------------------------
    textos, fuente_corpus = cargar_textos_desde_t1(t1, n_esperado)
    if textos is None:
        textos, fuente_corpus = cargar_textos_desde_json_sueltos(n_esperado)
    if textos is None:
        textos, fuente_corpus = cargar_textos_desde_archivos(n_esperado)
    if textos is None:
        abortar(
            "No se han podido cargar los doce textos reales de la practica. "
            "Esta subtarea exige usar EXACTAMENTE el mismo corpus que T1 y "
            "verificarlo contra entradas/T1.json; no se admite ningun corpus "
            "inventado. Incluya los textos en entradas/T1.json (p. ej. clave "
            "'textos', lista de 12 cadenas), en un corpus.json, o como 12 "
            "archivos .txt en la carpeta de trabajo o en una subcarpeta 'corpus'."
        )

    # ------------------------------------------------------------------
    # 3) Vectorizacion: mismo CountVectorizer que en la Parte 1
    # ------------------------------------------------------------------
    vectorizador = CountVectorizer(stop_words=PALABRAS_VACIAS)
    X = vectorizador.fit_transform(textos)

    try:
        vocabulario = [str(v) for v in vectorizador.get_feature_names_out()]
    except AttributeError:  # compatibilidad con scikit-learn antiguo
        vocabulario = [str(v) for v in vectorizador.get_feature_names()]

    n_documentos = int(X.shape[0])
    tamano_vocabulario = int(X.shape[1])
    total_palabras = int(X.sum())
    print(f"[info] Matriz documento-termino: {n_documentos} documentos x "
          f"{tamano_vocabulario} terminos ({total_palabras} palabras en total).")

    if n_documentos == 0 or tamano_vocabulario == 0:
        abortar("La matriz documento-termino esta vacia; el corpus cargado no es "
                "valido.")
    if n_documentos != n_esperado:
        abortar(f"El corpus cargado tiene {n_documentos} documentos; se esperaban "
                f"{n_esperado} (los doce textos de la practica).")
    tam_vocab_t1 = t1.get("tamano_vocabulario")
    if tam_vocab_t1 is not None and int(tam_vocab_t1) != tamano_vocabulario:
        abortar(f"El tamano del vocabulario obtenido ({tamano_vocabulario}) no "
                f"coincide con el registrado en T1 ({int(tam_vocab_t1)}).")

    # ------------------------------------------------------------------
    # 4) Verificacion 1: el vocabulario debe coincidir con el de T1
    # ------------------------------------------------------------------
    conjunto_obtenido = set(vocabulario)
    conjunto_t1 = set(vocab_t1)
    interseccion = conjunto_obtenido & conjunto_t1
    union = conjunto_obtenido | conjunto_t1
    jaccard = (len(interseccion) / len(union)) if union else 0.0
    listas_identicas = (vocabulario == vocab_t1)
    print(f"[verif 1] Vocabulario: {tamano_vocabulario} terminos obtenidos vs "
          f"{len(vocab_t1)} en T1 | Jaccard = {jaccard:.6f} | "
          f"listas identicas = {listas_identicas}")
    if jaccard < 1.0:
        faltan = sorted(conjunto_t1 - conjunto_obtenido)
        sobran = sorted(conjunto_obtenido - conjunto_t1)
        print(f"[verif 1] Terminos en T1 ausentes aqui ({len(faltan)}): "
              + ", ".join(faltan[:25]) + (" ..." if len(faltan) > 25 else ""))
        print(f"[verif 1] Terminos aqui ausentes en T1 ({len(sobran)}): "
              + ", ".join(sobran[:25]) + (" ..." if len(sobran) > 25 else ""))
        abortar("Verificacion 1 FALLIDA: el vocabulario obtenido no coincide con el "
                "guardado en T1.json (Jaccard != 1.0). El corpus cargado (o la "
                "lista de palabras vacias) no es identico al de la Parte 1.")

    # ------------------------------------------------------------------
    # 5) Verificacion 2: la perplejidad con k=3 debe reproducir la de T1
    # ------------------------------------------------------------------
    modelos = {}
    perplejidades = {}

    lda3 = LatentDirichletAllocation(
        n_components=3, learning_method="batch", max_iter=50, random_state=0,
    )
    lda3.fit(X)
    modelos[3] = lda3
    perplejidades[3] = float(lda3.perplexity(X))
    diferencia_relativa = (abs(perplejidades[3] - perp_k3_t1) / abs(perp_k3_t1)
                           if perp_k3_t1 != 0 else float("inf"))
    print(f"[verif 2] Perplejidad k=3: T2 = {perplejidades[3]!r} | "
          f"T1 = {perp_k3_t1!r} | diferencia relativa = {diferencia_relativa:.3e} "
          f"(tolerancia {TOLERANCIA_RELATIVA:.0e})")
    if diferencia_relativa > TOLERANCIA_RELATIVA:
        abortar("Verificacion 2 FALLIDA: la perplejidad con k=3 no reproduce "
                f"perplejidad_k3 de T1 ({perp_k3_t1!r}); se obtuvo "
                f"{perplejidades[3]!r}. El corpus o el protocolo no coinciden "
                "con los de la Parte 1.")
    n_iter_t1 = t1.get("n_iteracion_final")
    if n_iter_t1 is not None:
        try:
            if int(n_iter_t1) != int(lda3.n_iter_):
                print(f"[aviso] n_iter_ con k=3 ({int(lda3.n_iter_)}) difiere del "
                      f"registrado en T1 ({int(n_iter_t1)}); la perplejidad si "
                      "coincide, por lo que se continua.")
        except (TypeError, ValueError):
            pass
    print("[verif] Ambas verificaciones superadas: corpus y protocolo identicos "
          "a los de la Parte 1. Se genera la tabla comparativa.")

    # ------------------------------------------------------------------
    # 6) Ajuste de LDA con k = 2 y k = 4 (k = 3 ya ajustado y verificado)
    # ------------------------------------------------------------------
    for k in (2, 4):
        lda = LatentDirichletAllocation(
            n_components=k, learning_method="batch", max_iter=50, random_state=0,
        )
        lda.fit(X)
        modelos[k] = lda
        perplejidades[k] = float(lda.perplexity(X))
        print(f"[info] LDA con k={k}: perplejidad = {perplejidades[k]!r} "
              f"(n_iter_ = {int(lda.n_iter_)})")

    # Palabras top por tema (auxiliar, mismo criterio que en la Parte 1)
    palabras_top = {}
    palabras_top_con_pesos = {}
    for k in VALORES_K:
        componentes = modelos[k].components_
        probs = componentes / componentes.sum(axis=1, keepdims=True)
        indices_top = [np.argsort(componentes[t])[::-1][:N_PALABRAS_TOP]
                       for t in range(k)]
        palabras_top[k] = [[vocabulario[i] for i in idx] for idx in indices_top]
        palabras_top_con_pesos[k] = [
            [(vocabulario[i], float(probs[t, i])) for i in idx]
            for t, idx in enumerate(indices_top)
        ]

    # ------------------------------------------------------------------
    # 7) Tabla comparativa de perplejidades
    # ------------------------------------------------------------------
    tabla_comparativa = [
        {
            "n_temas": int(k),
            "perplejidad": perplejidades[k],
            "n_iteracion_final": int(modelos[k].n_iter_),
        }
        for k in VALORES_K
    ]
    df_tabla = pd.DataFrame(tabla_comparativa)
    mejor_k = min(VALORES_K, key=lambda kk: perplejidades[kk])

    # ------------------------------------------------------------------
    # 8) Guardado de todas las cifras en resultados.json
    # ------------------------------------------------------------------
    resultados = {
        "subtarea": "T2 - Parte 2 - Numero de temas (2, 3 y 4)",
        "parametros_comunes": {
            "vectorizador": "CountVectorizer (mismas palabras vacias que T1, resto por defecto)",
            "learning_method": "batch",
            "max_iter": 50,
            "random_state": 0,
            "valores_n_components": [2, 3, 4],
        },
        "fuente_corpus": fuente_corpus,
        "n_documentos": n_documentos,
        "tamano_vocabulario": tamano_vocabulario,
        "total_palabras_corpus": total_palabras,
        "perplejidad_k2": perplejidades[2],
        "perplejidad_k3": perplejidades[3],
        "perplejidad_k4": perplejidades[4],
        "n_iteracion_final": {f"k{k}": int(modelos[k].n_iter_) for k in VALORES_K},
        "log_verosimilitud_total": {f"k{k}": float(modelos[k].score(X))
                                    for k in VALORES_K},
        "tabla_comparativa": tabla_comparativa,
        "mejor_n_temas_por_perplejidad": int(mejor_k),
        "palabras_top_por_tema": {f"k{k}": palabras_top[k] for k in VALORES_K},
        "palabras_top_por_tema_con_pesos": {
            f"k{k}": palabras_top_con_pesos[k] for k in VALORES_K
        },
        "vocabulario": vocabulario,
        "verificacion_con_T1": {
            "fuente_corpus": fuente_corpus,
            "vocabulario_jaccard": float(jaccard),
            "vocabulario_listas_identicas": bool(listas_identicas),
            "tamano_vocabulario_T1": (int(tam_vocab_t1)
                                      if tam_vocab_t1 is not None else None),
            "perplejidad_k3_T1": perp_k3_t1,
            "perplejidad_k3_T2": perplejidades[3],
            "diferencia_relativa_k3": float(diferencia_relativa),
            "tolerancia_relativa": TOLERANCIA_RELATIVA,
            "valor_referencia_revision": PERPLEJIDAD_K3_REFERENCIA,
            "verificacion_superada": True,
        },
        "nota_metodologica": (
            "Los doce textos son exactamente los mismos que los usados en la "
            "Parte 1 (misma fuente que T1), lo que se verifica antes de reportar: "
            "(1) el vocabulario coincide con el de T1.json (Jaccard = 1.0) y "
            "(2) la perplejidad con k=3 reproduce perplejidad_k3 de T1 dentro de "
            "una tolerancia relativa de 1e-6. La perplejidad se evalua sobre la "
            "matriz documento-termino del corpus con el mismo protocolo que en "
            "la Parte 1 (el enunciado pide la perplejidad 'sobre el corpus' y no "
            "define conjunto de retencion), de modo que los valores de k = 2, 3 "
            "y 4 son comparables entre si y con perplejidad_k3 de T1."
        ),
    }

    with open(RUTA_RESULTADOS, "w", encoding="utf-8") as manejador:
        json.dump(resultados, manejador, ensure_ascii=False, indent=2,
                  default=a_nativo)
    print(f"\n[ok] Cifras guardadas en {RUTA_RESULTADOS}")

    # ------------------------------------------------------------------
    # 9) Impresion de la tabla comparativa y de las cifras principales
    # ------------------------------------------------------------------
    print("\n" + linea)
    print("T2 - Parte 2 - Perplejidad segun el numero de temas (k = 2, 3, 4)")
    print("Mismo CountVectorizer y mismos parametros de LDA que en la Parte 1;")
    print("corpus verificado contra T1 (vocabulario y perplejidad con k=3).")
    print(linea)
    print(df_tabla.to_string(index=False, float_format=lambda valor: f"{valor:.6f}"))
    print(linea)
    print("Cifras principales (precision completa):")
    print(f"  perplejidad_k2 = {perplejidades[2]!r}")
    print(f"  perplejidad_k3 = {perplejidades[3]!r}")
    print(f"  perplejidad_k4 = {perplejidades[4]!r}")
    print(f"  Menor perplejidad (mejor ajuste): k = {mejor_k} "
          f"({perplejidades[mejor_k]:.6f})")
    print(f"  Verificacion k=3 vs T1: diferencia relativa = "
          f"{diferencia_relativa:.3e} (<= {TOLERANCIA_RELATIVA:.0e}) -> OK")


if __name__ == "__main__":
    main()
