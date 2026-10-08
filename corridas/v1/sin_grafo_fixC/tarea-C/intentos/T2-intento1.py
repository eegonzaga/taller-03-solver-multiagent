# -*- coding: utf-8 -*-
"""
T2 - Parte 2 - Perplejidad con 2, 3 y 4 temas.

Usando el mismo CountVectorizer (mismas palabras vacias) de la Parte 1 y los
mismos parametros de LDA (learning_method='batch', max_iter=50,
random_state=0), se ajusta LatentDirichletAllocation con n_components = 2, 3
y 4, y se reporta la perplejidad de cada modelo sobre el corpus en una unica
tabla comparativa.

Entradas:
  * entradas/T1.json: resultados de la Parte 1 (vocabulario, palabras vacias
    y parametros del modelo). Si conserva los textos crudos se reutilizan.
  * No hay archivos de datos: si T1.json no conserva los textos, los doce
    textos se reconstruyen a partir de los fragmentos del corpus citados en
    el enunciado (Blei, Ng y Jordan, 2003, "Latent Dirichlet Allocation",
    Journal of Machine Learning Research 3 (2003) 993-1022).

Salidas:
  * resultados.json con perplejidad_k2, perplejidad_k3, perplejidad_k4 y
    tabla_perplejidades. Esta subtarea no requiere figura PNG.
"""

import json
import math
import os

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# ---------------------------------------------------------------------------
# 0. Parametros fijos (identicos a los de la Parte 1)
# ---------------------------------------------------------------------------
N_TEMAS = (2, 3, 4)
LEARNING_METHOD = "batch"
MAX_ITER = 50
RANDOM_STATE = 0
N_DOCUMENTOS = 12

# ---------------------------------------------------------------------------
# 1. Resultados de la Parte 1 (entradas/T1.json)
# ---------------------------------------------------------------------------
ruta_t1 = os.path.join("entradas", "T1.json")
t1 = {}
if os.path.exists(ruta_t1):
    try:
        with open(ruta_t1, "r", encoding="utf-8") as fh:
            t1 = json.load(fh)
        print(f"[info] Leido {ruta_t1} ({len(t1)} claves).")
    except (OSError, ValueError) as exc:
        print(f"[aviso] No se pudo leer {ruta_t1}: {exc}")
else:
    print(f"[aviso] No existe {ruta_t1}; se usan los parametros del enunciado.")

# ---------------------------------------------------------------------------
# 2. Corpus de doce textos
# ---------------------------------------------------------------------------
# Fragmentos del corpus citados en el enunciado de la practica. No hay
# archivos de datos disponibles, de modo que estos fragmentos sirven para
# reconstruir los doce textos cuando T1.json no conserva los originales.
FRAGMENTO_PAG_1 = (
    "Journal of Machine Learning Research 3 (2003) 993-1022 Submitted 2/02; "
    "Published 1/03 Latent Dirichlet Allocation David M. Blei "
    "BLEI@CS.BERKELEY.EDU Computer Science Division University of California "
    "Berkeley, CA 94720, USA Andrew Y. Ng ANG@CS.STANFORD.EDU Computer "
    "Science Department Stanford University Stanford, CA 94305, USA Michael "
    "I. Jordan JORDAN@CS.BERKELEY.EDU Computer Science Division and "
    "Department of Statistics University of California Berkeley, CA 94720, "
    "USA Editor: John Lafferty Abstract We describe latent Dirichlet "
    "allocation (LDA), a generative probabilistic model for collections of "
    "discrete data such as text corpora. LDA is a three-level hierarchical "
    "Bayesian model, in which each item of a collection is modeled as a "
    "finite mixture over an underlying set of topics. Each topic is, in "
    "turn, modeled as an infinite mixture over an underlying set of topic "
    "probabilities. In the context of text modeling, the topic probabilities "
    "provide an explicit representation of a document. We present efficient "
    "approximate inference techniques based on variational methods and an EM "
    "algorithm for empirical Bayes parameter estimation. We report results "
    "in document modeling, text classification, and collaborative filtering, "
    "comparing to a mixture of unigrams model and the probabilistic LSI "
    "model. 1. Introduction In this paper we consider the problem of "
    "modeling text corpora and other collections of discrete data. The goal "
    "is to find short descriptions of the members of a collection that "
    "enable efficient processing of large collections while preserving the "
    "essential statistical relationships that are useful for basic tasks "
    "such as classification, novelty detection, summarization, and "
    "similarity and relevance judgments. Significant progress has been made "
    "on this problem by research"
)

FRAGMENTO_PAG_4 = (
    "BLEI, NG, AND JORDAN We wish to find a probabilistic model of a corpus "
    "that not only assigns high probability to members of the corpus, but "
    "also assigns high probability to other similar documents. 3. Latent "
    "Dirichlet allocation Latent Dirichlet allocation (LDA) is a generative "
    "probabilistic model of a corpus. The basic idea is that documents are "
    "represented as random mixtures over latent topics, where each topic is "
    "characterized by a distribution over words. LDA assumes the following "
    "generative process for each document w in a corpus D: 1. Choose N from "
    "a Poisson distribution. 2. Choose theta from a Dirichlet distribution "
    "Dir(alpha). 3. For each of the N words wn: (a) Choose a topic zn from a "
    "Multinomial distribution with parameter theta. (b) Choose a word wn "
    "from the multinomial probability conditioned on the topic zn. Several "
    "simplifying assumptions are made in this basic model, some of which we "
    "remove in subsequent sections. First, the dimensionality k of the "
    "Dirichlet distribution (and thus the dimensionality of the topic "
    "variable z) is assumed known and fixed. Second, the word probabilities "
    "are parameterized by a k by V matrix beta where beta_ij = p(wj = 1 "
    "given zi = 1), which for now we treat as a fixed quantity that is to be "
    "estimated. Finally, the Poisson assumption is not critical to anything "
    "that follows and more realistic document length distributions can be "
    "used as needed. Furthermore, note that N is independent of all the "
    "other data generating variables (theta and z). It is thus an ancillary "
    "variable and we will generally ignore its randomness in the subsequent "
    "development. A k-dimensional Dirichlet random variable theta can take "
    "values in the (k minus 1)-simplex (a k-vector theta lies in the "
    "(k minus 1)-simplex if theta_i >= 0 and its components sum to one), and "
    "has the following probability density on the simplex"
)

CLAVES_TEXTO = (
    "textos", "documentos", "corpus", "textos_crudos",
    "documentos_texto", "raw_documents", "textos_documentos",
)


def textos_desde_t1(t1):
    """Devuelve los textos crudos si la Parte 1 los guardo en su JSON."""
    for clave in CLAVES_TEXTO:
        valor = t1.get(clave)
        if (
            isinstance(valor, list)
            and len(valor) >= 2
            and all(isinstance(s, str) and s.strip() for s in valor)
        ):
            return list(valor), clave
    return None, None


def reconstruir_doce_textos():
    """Divide los fragmentos del corpus del enunciado en doce documentos."""
    palabras = " ".join([FRAGMENTO_PAG_1, FRAGMENTO_PAG_4]).split()
    tam = int(math.ceil(len(palabras) / float(N_DOCUMENTOS)))
    textos = [
        " ".join(palabras[i * tam:(i + 1) * tam]).strip()
        for i in range(N_DOCUMENTOS)
    ]
    return [t for t in textos if t]


textos, clave_textos = textos_desde_t1(t1)
if textos is not None:
    fuente_corpus = f"textos crudos leidos de entradas/T1.json (clave '{clave_textos}')"
else:
    textos = reconstruir_doce_textos()
    fuente_corpus = (
        "reconstruccion del corpus a partir de los fragmentos citados en el "
        "enunciado (Blei, Ng y Jordan, 2003), divididos en doce documentos"
    )

# ---------------------------------------------------------------------------
# 3. Palabras vacias y vocabulario (los mismos de la Parte 1)
# ---------------------------------------------------------------------------
CLAVES_PV = (
    "palabras_vacias", "stop_words", "stopwords", "lista_palabras_vacias",
)


def palabras_vacias_desde_t1(t1):
    for clave in CLAVES_PV:
        valor = t1.get(clave)
        if (
            isinstance(valor, list)
            and len(valor) > 0
            and all(isinstance(s, str) for s in valor)
        ):
            return list(valor), f"lista de palabras vacias de la Parte 1 (clave '{clave}')"
    return sorted(ENGLISH_STOP_WORDS), "lista estandar de scikit-learn (ENGLISH_STOP_WORDS)"


palabras_vacias, fuente_pv = palabras_vacias_desde_t1(t1)

vocabulario_t1 = t1.get("vocabulario")
if isinstance(vocabulario_t1, dict):
    vocabulario_t1 = [str(w) for w in sorted(vocabulario_t1.keys())]
if (
    isinstance(vocabulario_t1, list)
    and len(vocabulario_t1) > 0
    and all(isinstance(w, str) for w in vocabulario_t1)
):
    vocabulario_t1 = list(vocabulario_t1)
else:
    vocabulario_t1 = None


def ajustar_vectorizador(fijar_vocabulario_parte1):
    """Ajusta el CountVectorizer y devuelve (vectorizador, X, descripcion)."""
    if fijar_vocabulario_parte1:
        vec = CountVectorizer(vocabulary=vocabulario_t1, stop_words=palabras_vacias)
        desc = (
            "CountVectorizer de la Parte 1: vocabulario y palabras vacias "
            "tomados de entradas/T1.json"
        )
    else:
        vec = CountVectorizer(stop_words=palabras_vacias)
        desc = (
            "CountVectorizer con las mismas palabras vacias y vocabulario "
            "ajustado sobre el corpus, como en la Parte 1"
        )
    X = vec.fit_transform(textos)
    return vec, X, desc


vec_corpus, X_corpus, desc_corpus = ajustar_vectorizador(False)
tokens_disponibles = int(X_corpus.sum())

cobertura = None
if vocabulario_t1 is not None:
    vec_p1, X_p1, desc_p1 = ajustar_vectorizador(True)
    cobertura = float(X_p1.sum()) / tokens_disponibles if tokens_disponibles > 0 else 0.0
    if cobertura >= 0.30:
        vectorizador, X, fuente_vocabulario = vec_p1, X_p1, desc_p1
    else:
        print(
            f"[aviso] El vocabulario de la Parte 1 solo cubre el {cobertura:.1%} de "
            "los tokens del corpus reconstruido; se ajusta el vocabulario sobre el "
            "corpus (mismas palabras vacias)."
        )
        vectorizador, X, fuente_vocabulario = vec_corpus, X_corpus, desc_corpus
else:
    vectorizador, X, fuente_vocabulario = vec_corpus, X_corpus, desc_corpus

# ---------------------------------------------------------------------------
# 4. LDA con 2, 3 y 4 temas y perplejidad sobre el corpus
# ---------------------------------------------------------------------------
def perplejidades_lda(X, n_temas):
    """Ajusta LDA con los parametros de la Parte 1 y devuelve {k: perplejidad}."""
    resultados = {}
    for k in n_temas:
        lda = LatentDirichletAllocation(
            n_components=k,
            learning_method=LEARNING_METHOD,
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE,
        )
        lda.fit(X)
        resultados[k] = float(lda.perplexity(X))
    return resultados


perplejidades = perplejidades_lda(X, N_TEMAS)

if (
    not all(np.isfinite(v) for v in perplejidades.values())
    and vocabulario_t1 is not None
):
    print(
        "[aviso] Perplejidades no finitas con el vocabulario de la Parte 1; "
        "se reajusta el vocabulario sobre el corpus."
    )
    vectorizador, X, fuente_vocabulario = vec_corpus, X_corpus, desc_corpus
    perplejidades = perplejidades_lda(X, N_TEMAS)

# ---------------------------------------------------------------------------
# 5. Tabla comparativa y guardado de resultados
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
    "tamano_vocabulario": int(X.shape[1]),
    "n_palabras_vacias": int(len(palabras_vacias)),
    "learning_method": LEARNING_METHOD,
    "max_iter": int(MAX_ITER),
    "random_state": int(RANDOM_STATE),
    "perplejidad_k2": float(perplejidades[2]),
    "perplejidad_k3": float(perplejidades[3]),
    "perplejidad_k4": float(perplejidades[4]),
    "tabla_perplejidades": tabla_perplejidades,
    "perplejidades_por_tema": {str(k): float(perplejidades[k]) for k in N_TEMAS},
    "mejor_n_componentes": int(mejor_k),
    "tokens_totales": int(X.sum()),
    "fuente_corpus": fuente_corpus,
    "fuente_vocabulario": fuente_vocabulario,
    "fuente_palabras_vacias": fuente_pv,
    "nota_metodologica": (
        "La perplejidad se calcula con lda.perplexity sobre la matriz "
        "documento-termino del corpus, tal como pide el enunciado. Al no "
        "haber archivos de datos, el corpus se reconstruye a partir de los "
        "fragmentos del enunciado si T1.json no conserva los textos crudos."
    ),
}
if cobertura is not None:
    salida["cobertura_vocabulario_parte1"] = float(cobertura)
if "n_documentos" in t1:
    salida["n_documentos_parte1"] = int(t1["n_documentos"])
if "n_palabras_vacias" in t1:
    salida["n_palabras_vacias_parte1"] = int(t1["n_palabras_vacias"])
if "perplejidad_k3" in t1:
    salida["perplejidad_k3_parte1"] = float(t1["perplejidad_k3"])
    salida["diferencia_perplejidad_k3_vs_parte1"] = float(
        abs(perplejidades[3] - float(t1["perplejidad_k3"]))
    )

with open("resultados.json", "w", encoding="utf-8") as fh:
    json.dump(salida, fh, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 6. Resumen por pantalla
# ---------------------------------------------------------------------------
print()
print("=== T2 - Parte 2: perplejidad de LDA segun el numero de temas ===")
print(
    f"Corpus: {len(textos)} documentos | vocabulario: {X.shape[1]} terminos | "
    f"palabras vacias: {len(palabras_vacias)}"
)
print(f"Tokens totales en la matriz documento-termino: {int(X.sum())}")
print(
    f"LDA: learning_method='{LEARNING_METHOD}', max_iter={MAX_ITER}, "
    f"random_state={RANDOM_STATE}"
)
print()
print("Tabla comparativa de perplejidad (sobre el corpus):")
print(f"{'n_temas':>8} | {'perplejidad':>14}")
print("-" * 26)
for fila in tabla_perplejidades:
    print(f"{fila['n_temas']:>8} | {fila['perplejidad']:>14.6f}")
print("-" * 26)
print(f"Menor perplejidad: n_components = {mejor_k}")
if "perplejidad_k3_parte1" in salida:
    print(
        f"Referencia de la Parte 1 (k=3): {salida['perplejidad_k3_parte1']:.6f} | "
        f"diferencia: {salida['diferencia_perplejidad_k3_vs_parte1']:.6f}"
    )
print()
print("[ok] Cifras guardadas en resultados.json")
