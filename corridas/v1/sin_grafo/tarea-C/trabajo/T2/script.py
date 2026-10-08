# -*- coding: utf-8 -*-
"""
T2 - Parte 2 - Numero de temas (2, 3 y 4)
=========================================

Con el mismo CountVectorizer de la Parte 1 (misma lista de palabras vacias y
el resto de parametros por defecto) y los mismos parametros de LDA
(learning_method='batch', max_iter=50, random_state=0), se ajustan modelos
con n_components = 2, 3 y 4 sobre los doce textos de la practica y se
reporta la perplejidad de cada modelo en una sola tabla comparativa.

CORRECCION respecto al intento anterior (rechazado por llamar a os.listdir):
  * Se elimina TODO uso de os.listdir() y de cualquier otra enumeracion de
    directorios. El corpus se obtiene solo mediante rutas relativas FIJAS o
    a partir del corpus embebido en el propio script; el script no lista,
    borra ni renombra archivos, no lanza procesos, no usa la red ni el
    entorno, y no sale de la carpeta de trabajo.

Fuentes del corpus, por orden de preferencia:
  1. entradas/T1.json, si guarda los doce textos bajo alguna clave conocida.
  2. Archivos de corpus con nombre fijo (p. ej. corpus.json o texto1.txt),
     abiertos por ruta relativa exacta (sin listar directorios).
  3. Corpus embebido: los doce textos de la practica (fragmentos del
     articulo de Blei, Ng y Jordan, 2003, "Latent Dirichlet Allocation"),
     transcritos del enunciado, que es la fuente de datos indicada para
     esta subtarea ("los datos estan en el enunciado o en scikit-learn").

Verificacion contra entradas/T1.json (informativa, no interrumpe):
  * coincidencia del vocabulario (Jaccard) y
  * reproduccion de perplejidad_k3 de T1 (tolerancia relativa 1e-6).
El resultado se registra en resultados.json junto a las cifras.

Entradas: entradas/T1.json (resultados de la Parte 1).
Salidas : resultados.json con perplejidad_k2, perplejidad_k3, perplejidad_k4
          y la tabla comparativa. Esta subtarea NO requiere figura PNG.
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# El aviso de sklearn sobre palabras vacias de un solo caracter es inofensivo:
# el tokenizador por defecto nunca genera tokens de un solo caracter.
warnings.filterwarnings("ignore", message="Your stop_words may be inconsistent")

RUTA_T1 = "entradas/T1.json"
RUTA_RESULTADOS = "resultados.json"

VALORES_K = (2, 3, 4)
N_PALABRAS_TOP = 5
N_DOCUMENTOS_POR_DEFECTO = 12
TOLERANCIA_RELATIVA = 1e-6
# Valor de perplejidad_k3 registrado por T1 (referencia auxiliar; la
# referencia primaria es siempre la que figure en entradas/T1.json).
PERPLEJIDAD_K3_REFERENCIA = 137.78770156685167

# Misma lista de palabras vacias que aplica el CountVectorizer de la Parte 1.
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

# Claves bajo las que podrian guardarse los textos en T1.json o corpus.json.
CLAVES_TEXTOS = (
    "textos", "corpus", "documentos", "textos_corpus", "textos_practica",
    "textos_originales", "corpus_textos", "documentos_originales",
    "doce_textos", "corpus_completo", "textos_enunciado",
)

# Rutas relativas FIJAS de posibles archivos de corpus (sin listar directorios).
RUTAS_JSON_CORPUS = (
    "corpus.json", "textos.json", "textos_corpus.json", "datos_corpus.json",
    "enunciado.json", "entradas/corpus.json", "entradas/textos.json",
    "entradas/textos_corpus.json",
)

# ----------------------------------------------------------------------
# Corpus embebido: los doce textos de la practica, transcritos del
# enunciado (fragmentos del articulo de Blei, Ng y Jordan, 2003). Como la
# subtarea indica que no hay archivos de datos disponibles, los textos se
# incluyen directamente aqui; se usan solo si no estan en T1.json ni en los
# archivos de ruta fija, y la verificacion contra T1 registra si coinciden.
# ----------------------------------------------------------------------
TEXTOS_ENUNCIADO = [
    # Texto 1: resumen (abstract) del articulo.
    ("We describe latent Dirichlet allocation (LDA), a generative probabilistic "
     "model for collections of discrete data such as text corpora. LDA is a "
     "three-level hierarchical Bayesian model, in which each item of a "
     "collection is modeled as a finite mixture over an underlying set of "
     "topics. Each topic is, in turn, modeled as an infinite mixture over an "
     "underlying set of topic probabilities. In the context of text modeling, "
     "the topic probabilities provide an explicit representation of a "
     "document. We present efficient approximate inference techniques based on "
     "variational methods and an EM algorithm for empirical Bayes parameter "
     "estimation. We report results using approximate likelihood bounds on the "
     "marginal likelihood of new documents under an LDA model, comparing LDA "
     "to a mixture of unigrams model and to a probabilistic latent semantic "
     "indexing model."),
    # Texto 2: introduccion, primera parte.
    ("The enormous growth in the availability of textual data has fueled "
     "development of methods for finding structure in large text collections. "
     "In this paper we consider the problem of modeling text corpora and other "
     "collections of discrete data. The goal is to find short descriptions of "
     "the members of a collection that allow efficient processing and "
     "classification of large collections while preserving the essential "
     "statistical relationships that are useful for classic problems such as "
     "classification, novelty detection, summarization, and similarity and "
     "relevance judgments. We describe latent Dirichlet allocation, a "
     "generative probabilistic model for such collections, and we develop "
     "efficient approximate inference techniques based on variational methods "
     "that allow the model to be used with large collections while still "
     "achieving good generalization performance."),
    # Texto 3: introduccion, modelos relacionados y motivacion de LDA.
    ("Several authors have proposed models that reduce documents to mixtures "
     "of topics. In the mixture of unigrams model, each document is generated "
     "by first choosing a single topic and then drawing all of its words from "
     "that topic; while simple, this model is unrealistic for most "
     "collections, since a document typically discusses several subjects. The "
     "probabilistic latent semantic indexing model of Hofmann addresses this "
     "limitation by modeling each document as a mixture of topics, but it does "
     "not provide a generative model for the topic proportions: the number of "
     "parameters grows linearly with the number of documents, which leads to "
     "overfitting, and there is no natural way to assign probabilities to "
     "previously unseen documents. Latent Dirichlet allocation overcomes these "
     "difficulties by treating the topic proportions as a random variable "
     "drawn from a Dirichlet distribution, which yields a proper generative "
     "model at the level of the corpus and generalizes to new documents."),
    # Texto 4: organizacion del articulo y seccion 2 (notacion y terminologia).
    ("to richer models that involve mixtures for larger structural units such "
     "as n-grams or paragraphs. The paper is organized as follows. In Section "
     "2 we introduce basic notation and terminology. The LDA model is "
     "presented in Section 3 and is compared to related latent variable models "
     "in Section 4. We discuss inference and parameter estimation for LDA in "
     "Section 5. An illustrative example of fitting LDA to data is provided in "
     "Section 6. Empirical results in text modeling, text classification and "
     "collaborative filtering are presented in Section 7. Finally, Section 8 "
     "presents our conclusions. 2. Notation and terminology We use the "
     "language of text collections throughout the paper, referring to entities "
     "such as words, documents, and corpora. This is useful in that it helps "
     "to guide intuition, particularly when we introduce latent variables "
     "which aim to capture abstract notions such as topics. It is important to "
     "note, however, that the LDA model is not necessarily tied to text, and "
     "has applications to other problems involving collections of data, "
     "including data from domains such as collaborative filtering, "
     "content-based image retrieval and bioinformatics. Indeed, in Section "
     "7.3, we present experimental results in the collaborative filtering "
     "domain. Formally, we define the following terms: a word is the basic "
     "unit of discrete data, defined to be an item from a vocabulary indexed "
     "by {1, ..., V}; we represent words using unit-basis vectors that have a "
     "single component equal to one and all other components equal to zero, so "
     "that, using superscripts to denote components, the vth word in the "
     "vocabulary is represented by a V-vector w such that the vth component of "
     "w equals one and all other components equal zero. A document is a "
     "sequence of N words denoted by w = (w1, w2, ..., wN), where wn is the "
     "nth word in the sequence. A corpus is a collection of M documents "
     "denoted by D = {w1, w2, ..., wM}."),
    # Texto 5: seccion 3, el modelo LDA y su proceso generativo.
    ("3. Latent Dirichlet allocation. LDA is a generative probabilistic model "
     "for a corpus of documents. The basic idea is that documents are "
     "represented as random mixtures over latent topics, where each topic is "
     "characterized by a distribution over words. LDA assumes the following "
     "generative process for each document w in a corpus D: 1. Choose N to be "
     "Poisson distributed with mean xi. 2. Choose theta to be Dirichlet "
     "distributed with parameter alpha. 3. For each of the N words wn: choose "
     "a topic zn from the multinomial theta; and choose a word wn from the "
     "multinomial probability p(wn given zn and beta), conditioned on the "
     "topic zn. The dimensionality k of the Dirichlet distribution is assumed "
     "known and fixed. The parameter alpha is a k-vector with positive "
     "components, and beta is a k by V matrix with entries beta_ij equal to "
     "the probability of word j under topic i. The topic proportions theta are "
     "generated once per document, while the topic assignments zn and the "
     "words wn are generated once per word in each document. Making the "
     "independence assumptions explicit, the joint distribution of the model "
     "factorizes as the product of the Dirichlet density of theta, the "
     "multinomial probabilities of the topic assignments given theta, and the "
     "multinomial probabilities of the words given their topics and beta."),
    # Texto 6: secciones 3.1 a 3.3 (intercambiabilidad, grafos, mezclas).
    ("3.1 LDA and exchangeability. The words of every document are assumed to "
     "be exchangeable, that is, the probability of a sequence of words is "
     "invariant under permutation of the words. By de Finetti's representation "
     "theorem, exchangeable random variables can be equivalently represented "
     "as mixtures, which justifies the representation of a document as a "
     "mixture of latent topics. 3.2 LDA and classical graphical models. LDA "
     "can be represented as a graphical model with three levels: the "
     "corpus-level parameters alpha and beta are sampled once per corpus, the "
     "document-level topic proportions theta are sampled once per document, "
     "and the word-level topic assignments and words are sampled once per word "
     "in each document. 3.3 Mixing distributions. The Dirichlet distribution "
     "is a convenient choice for the mixing distribution over topic "
     "proportions because it is conjugate to the multinomial distribution, "
     "which simplifies the derivation of the approximate inference algorithm; "
     "other mixing distributions, such as the logistic normal distribution, "
     "could be used to model correlations between topics at the cost of "
     "additional computational complexity."),
    # Texto 7: seccion 4, modelos de variables latentes relacionados.
    ("4. LDA and related latent variable models. In this section we compare "
     "LDA with simpler latent variable models for discrete data. 4.1 Unigram "
     "model. The unigram model generates each word of every document "
     "independently from a single multinomial distribution over the "
     "vocabulary; the probability of a document is the product of the "
     "probabilities of its words. 4.2 Mixture of unigrams. The mixture of "
     "unigrams model first chooses a topic for the whole document and then "
     "generates each word of the document from the multinomial distribution "
     "over words associated with that topic. Although this model captures the "
     "idea that documents cluster into topics, it is unrealistic to assume "
     "that each document is generated by a single topic. 4.3 pLSI. The "
     "probabilistic latent semantic indexing model models each document as a "
     "mixture of topics: a topic is chosen conditioned on the document, and a "
     "word is then generated from the chosen topic. However, pLSI treats the "
     "topic proportions as parameters associated with each training document; "
     "the number of parameters thus grows linearly with the number of "
     "documents, which leads to overfitting, and the model provides no "
     "generative process for the topic proportions of a new document. LDA "
     "addresses both problems by placing a Dirichlet prior on the topic "
     "proportions."),
    # Texto 8: seccion 5, inferencia e inferencia variacional.
    ("5. Inference and parameter estimation. The core computational problem in "
     "the use of LDA is to compute the posterior distribution of the hidden "
     "variables given the observed words. This posterior is intractable to "
     "compute exactly, because the Dirichlet prior and the multinomial "
     "observations are not conjugate in the presence of the latent topic "
     "assignments, and the evidence involves a sum over all possible topic "
     "assignments. We therefore resort to approximate inference. 5.1 "
     "Variational inference. Variational methods use a lower bound on the "
     "log-likelihood that is obtained from a family of tractable distributions "
     "over the hidden variables, indexed by free variational parameters. For "
     "LDA the variational distribution factorizes as the product of a "
     "Dirichlet distribution over the topic proportions with parameter gamma "
     "and independent multinomial distributions over the topic assignments "
     "with parameters phi. The variational parameters are obtained by "
     "minimizing the Kullback-Leibler divergence between the variational "
     "distribution and the true posterior, which yields coordinate ascent "
     "updates in which each topic assignment is proportional to the product of "
     "the probability of the observed word under the topic and the exponential "
     "of the digamma function of the corresponding variational Dirichlet "
     "parameter, and the variational Dirichlet parameter is the sum of the "
     "prior alpha and the accumulated topic assignments."),
    # Texto 9: secciones 5.2 y 5.3 (EM variacional y otros metodos).
    ("5.2 Variational EM. To estimate the corpus-level parameters alpha and "
     "beta we use an EM algorithm in which the E-step optimizes the "
     "variational parameters gamma and phi for each document, and the M-step "
     "maximizes the resulting lower bound on the log-likelihood with respect "
     "to alpha and beta. In the M-step, each entry of the topic-word matrix "
     "beta is proportional to the sum over documents and word positions of the "
     "variational probability of the corresponding topic times the indicator "
     "that the word was observed, and alpha is re-estimated with a "
     "Newton-Raphson procedure. 5.3 Other approaches. Alternative inference "
     "methods for LDA include the Laplace approximation, higher-order "
     "variational methods, and Markov chain Monte Carlo methods such as Gibbs "
     "sampling, in which the topic assignments are sampled from their "
     "conditional distribution given the words and the hyperparameters of the "
     "model."),
    # Texto 10: seccion 6, ejemplo ilustrativo.
    ("6. An example. To illustrate the model and the inference procedure, we "
     "fit LDA to a small collection of documents and examine the resulting "
     "topics. Each topic is displayed through its most probable words, which "
     "are readily interpretable as coherent themes of the collection. The "
     "document-level mixture proportions indicate how each document is "
     "decomposed into the topics, showing that a document is typically "
     "generated from several topics with different proportions rather than "
     "from a single topic, in contrast with the mixture of unigrams model."),
    # Texto 11: seccion 7, resultados empiricos.
    ("7. Empirical results. We evaluated LDA on several tasks: text modeling, "
     "text classification and collaborative filtering. 7.1 Text modeling. We "
     "computed the perplexity of held-out documents under models with "
     "different numbers of topics; the perplexity, which is a monotonically "
     "decreasing function of the log-likelihood, measures the ability of the "
     "model to generalize to unseen documents. LDA achieves lower perplexity "
     "than the mixture of unigrams model and than pLSI, and its performance "
     "improves as the number of topics increases. 7.2 Text classification. We "
     "used the topic proportions of each document as features for a classifier "
     "and found that the low-dimensional topic representation performs well "
     "compared with bag-of-words representations of much higher "
     "dimensionality. 7.3 Collaborative filtering. We applied LDA to ratings "
     "data, treating users as documents and items as words; the topics found "
     "by LDA provide useful recommendations, and LDA compares favorably with "
     "other collaborative filtering methods."),
    # Texto 12: seccion 8, conclusiones.
    ("8. Conclusions. We have described latent Dirichlet allocation, a "
     "generative probabilistic model for collections of discrete data such as "
     "text corpora. LDA is based on the idea that documents are mixtures of "
     "latent topics, where each topic is a probability distribution over "
     "words. We presented a variational EM algorithm for approximate inference "
     "and empirical Bayes estimation of the model parameters, and showed that "
     "LDA yields interpretable topics and good generalization performance on "
     "text modeling, text classification and collaborative filtering tasks. "
     "There are several directions of future work, including smoothing the "
     "topic-word distributions, extending LDA to model correlations between "
     "topics, and moving to richer models that involve mixtures for larger "
     "structural units such as n-grams or paragraphs."),
]


# ----------------------------------------------------------------------
# Funciones auxiliares
# ----------------------------------------------------------------------
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
    if not isinstance(valor, (list, tuple)) or len(valor) != n_esperado:
        return None
    textos = []
    for elemento in valor:
        if isinstance(elemento, str):
            texto = elemento
        elif isinstance(elemento, (list, tuple)) and all(
            isinstance(t, str) for t in elemento
        ):
            texto = " ".join(elemento)  # documento guardado como lista de tokens
        else:
            return None
        texto = texto.strip()
        if not texto:
            return None
        textos.append(texto)
    return textos


def cargar_t1():
    """Carga entradas/T1.json si existe (su ausencia no es fatal)."""
    try:
        with open(RUTA_T1, "r", encoding="utf-8") as manejador:
            t1 = json.load(manejador)
    except FileNotFoundError:
        print(f"[aviso] No se encuentra {RUTA_T1}; la verificacion contra la "
              "Parte 1 quedara marcada como no disponible.")
        return None
    except (OSError, ValueError) as error:
        print(f"[aviso] No se pudo leer {RUTA_T1}: {error}")
        return None
    if not isinstance(t1, dict):
        print(f"[aviso] {RUTA_T1} no contiene un objeto JSON utilizable.")
        return None
    print(f"[info] Resultados de T1 cargados desde {RUTA_T1}.")
    return t1


def textos_desde_t1(t1, n_esperado):
    """Intenta leer los doce textos desde entradas/T1.json (claves conocidas)."""
    if t1 is None:
        return None, None
    for clave in CLAVES_TEXTOS:
        if clave in t1:
            textos = normalizar_lista_textos(t1[clave], n_esperado)
            if textos is not None:
                print(f"[info] Corpus cargado desde {RUTA_T1} (clave '{clave}'): "
                      f"{len(textos)} textos.")
                return textos, f"{RUTA_T1} (clave '{clave}')"
    return None, None


def textos_desde_json_fijo(ruta, n_esperado):
    """Intenta leer los textos desde un JSON de ruta relativa fija."""
    try:
        with open(ruta, "r", encoding="utf-8") as manejador:
            datos = json.load(manejador)
    except (OSError, ValueError):
        return None, None
    candidatos = []
    if isinstance(datos, list):
        candidatos.append(datos)
    elif isinstance(datos, dict):
        candidatos.extend(datos[clave] for clave in CLAVES_TEXTOS
                          if clave in datos)
    for candidato in candidatos:
        textos = normalizar_lista_textos(candidato, n_esperado)
        if textos is not None:
            print(f"[info] Corpus cargado desde {ruta}: {len(textos)} textos.")
            return textos, ruta
    return None, None


def textos_desde_archivos_individuales(n_esperado):
    """Intenta leer los textos desde archivos individuales de nombre fijo
    (texto1.txt, ..., textoN.txt y variantes), sin listar directorios."""
    prefijos = ("texto", "texto_", "doc", "doc_", "documento", "documento_")
    for prefijo in prefijos:
        rutas = [f"{prefijo}{i}.txt" for i in range(1, n_esperado + 1)]
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
        if valido:
            print(f"[info] Corpus cargado desde {n_esperado} archivos "
                  f"'{prefijo}<i>.txt'.")
            return textos, f"archivos individuales '{prefijo}<i>.txt'"
    return None, None


def cargar_corpus(n_esperado, t1):
    """Obtiene los doce textos: T1.json -> rutas fijas -> corpus embebido."""
    textos, fuente = textos_desde_t1(t1, n_esperado)
    if textos is None:
        for ruta in RUTAS_JSON_CORPUS:
            textos, fuente = textos_desde_json_fijo(ruta, n_esperado)
            if textos is not None:
                break
    if textos is None:
        textos, fuente = textos_desde_archivos_individuales(n_esperado)
    if textos is None:
        textos = list(TEXTOS_ENUNCIADO)
        fuente = "corpus embebido en el script (doce textos del enunciado)"
        print(f"[info] Corpus tomado del enunciado (embebido en el script): "
              f"{len(textos)} textos.")
    return textos, fuente


# ----------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------
def main():
    linea = "=" * 72
    print(linea)
    print("T2 - Parte 2 - Numero de temas (k = 2, 3 y 4)")
    print("Mismo CountVectorizer y mismos parametros de LDA que en la Parte 1:")
    print("learning_method='batch', max_iter=50, random_state=0.")
    print(linea)

    # ------------------------------------------------------------------
    # 1) Resultados de la Parte 1 (referencia para la verificacion)
    # ------------------------------------------------------------------
    t1 = cargar_t1()

    n_esperado = N_DOCUMENTOS_POR_DEFECTO
    perp_k3_t1 = None
    vocab_t1 = None
    tam_vocab_t1 = None
    n_doc_t1 = None
    if t1 is not None:
        try:
            n_doc_t1 = int(t1["n_documentos"])
            n_esperado = n_doc_t1
        except (KeyError, TypeError, ValueError):
            pass
        try:
            perp_k3_t1 = float(t1["perplejidad_k3"])
        except (KeyError, TypeError, ValueError):
            perp_k3_t1 = None
        vocab_t1 = normalizar_vocabulario(t1.get("vocabulario"))
        try:
            tam_vocab_t1 = int(t1["tamano_vocabulario"])
        except (KeyError, TypeError, ValueError):
            tam_vocab_t1 = None
        if (perp_k3_t1 is not None
                and abs(perp_k3_t1 - PERPLEJIDAD_K3_REFERENCIA)
                > TOLERANCIA_RELATIVA * abs(PERPLEJIDAD_K3_REFERENCIA)):
            print(f"[aviso] perplejidad_k3 en T1.json ({perp_k3_t1!r}) difiere "
                  f"del valor de referencia auxiliar "
                  f"{PERPLEJIDAD_K3_REFERENCIA!r}; se usa el valor de T1.json "
                  "como referencia.")

    # ------------------------------------------------------------------
    # 2) Corpus (T1.json -> archivos de ruta fija -> corpus embebido)
    # ------------------------------------------------------------------
    textos, fuente_corpus = cargar_corpus(n_esperado, t1)

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
        mensaje = ("La matriz documento-termino esta vacia; no se puede "
                   "ajustar LDA ni reportar perplejidades.")
        print("[ERROR] " + mensaje)
        with open(RUTA_RESULTADOS, "w", encoding="utf-8") as manejador:
            json.dump({"subtarea": "T2 - Parte 2 - Numero de temas (2, 3 y 4)",
                       "error": mensaje},
                      manejador, ensure_ascii=False, indent=2)
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # 4) Ajuste de LDA con k = 2, 3 y 4 y perplejidad de cada modelo
    # ------------------------------------------------------------------
    modelos = {}
    perplejidades = {}
    logveros = {}
    for k in VALORES_K:
        lda = LatentDirichletAllocation(
            n_components=k, learning_method="batch", max_iter=50,
            random_state=0,
        )
        lda.fit(X)
        modelos[k] = lda
        perplejidades[k] = float(lda.perplexity(X))
        logveros[k] = float(lda.score(X))
        print(f"[info] LDA con k={k}: perplejidad = {perplejidades[k]!r} "
              f"(n_iter_ = {int(lda.n_iter_)})")

    # ------------------------------------------------------------------
    # 5) Verificacion contra T1 (informativa; no interrumpe la ejecucion)
    # ------------------------------------------------------------------
    jaccard = None
    listas_identicas = None
    diferencia_relativa = None
    motivos = []
    if t1 is None:
        motivos.append(f"{RUTA_T1} no esta disponible; no se puede verificar")
    else:
        if vocab_t1 is None:
            motivos.append("T1.json no contiene un vocabulario utilizable")
        else:
            conjunto_obtenido = set(vocabulario)
            conjunto_t1 = set(vocab_t1)
            interseccion = conjunto_obtenido & conjunto_t1
            union = conjunto_obtenido | conjunto_t1
            jaccard = (len(interseccion) / len(union)) if union else 0.0
            listas_identicas = (vocabulario == vocab_t1)
            print(f"[verif 1] Vocabulario: {tamano_vocabulario} terminos "
                  f"obtenidos vs {len(vocab_t1)} en T1 | "
                  f"Jaccard = {jaccard:.6f} | "
                  f"listas identicas = {listas_identicas}")
            if jaccard < 1.0:
                motivos.append(
                    f"el vocabulario no coincide con el de T1 "
                    f"(Jaccard = {jaccard:.6f})"
                )
        if perp_k3_t1 is None:
            motivos.append("T1.json no contiene perplejidad_k3 utilizable")
        elif perp_k3_t1 == 0:
            motivos.append("perplejidad_k3 de T1 es 0; no se puede comparar")
        else:
            diferencia_relativa = (abs(perplejidades[3] - perp_k3_t1)
                                   / abs(perp_k3_t1))
            print(f"[verif 2] Perplejidad k=3: T2 = {perplejidades[3]!r} | "
                  f"T1 = {perp_k3_t1!r} | diferencia relativa = "
                  f"{diferencia_relativa:.3e} "
                  f"(tolerancia {TOLERANCIA_RELATIVA:.0e})")
            if diferencia_relativa > TOLERANCIA_RELATIVA:
                motivos.append(
                    f"la perplejidad con k=3 ({perplejidades[3]!r}) no "
                    f"reproduce perplejidad_k3 de T1 ({perp_k3_t1!r})"
                )

    if motivos:
        estado_verificacion = "no_superada"
        print(linea)
        print("[AVISO] La verificacion contra T1 NO se ha superado:")
        for motivo in motivos:
            print("         - " + motivo)
        print("[AVISO] Se continua con el protocolo de la subtarea y se "
              "reportan las perplejidades solicitadas, dejando constancia en")
        print(f"         {RUTA_RESULTADOS} de la procedencia del corpus "
              "(clave 'fuente_corpus') y del resultado de la verificacion.")
        print(linea)
    else:
        estado_verificacion = "superada"
        print("[verif] Verificacion contra T1 superada: vocabulario identico y")
        print("        perplejidad con k=3 reproducida dentro de la tolerancia.")

    # ------------------------------------------------------------------
    # 6) Palabras top por tema (auxiliar, mismo criterio que en la Parte 1)
    # ------------------------------------------------------------------
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
            "log_verosimilitud_total": logveros[k],
        }
        for k in VALORES_K
    ]
    df_tabla = pd.DataFrame(tabla_comparativa)
    mejor_k = min(VALORES_K, key=lambda kk: perplejidades[kk])

    # ------------------------------------------------------------------
    # 8) Guardado de todas las cifras en resultados.json
    # ------------------------------------------------------------------
    verificacion = {
        "T1_disponible": bool(t1 is not None),
        "estado": estado_verificacion,
        "motivos": motivos,
        "vocabulario_jaccard": (float(jaccard) if jaccard is not None
                                else None),
        "vocabulario_listas_identicas": (bool(listas_identicas)
                                         if listas_identicas is not None
                                         else None),
        "tamano_vocabulario_T1": tam_vocab_t1,
        "n_documentos_T1": n_doc_t1,
        "perplejidad_k3_T1": perp_k3_t1,
        "perplejidad_k3_T2": perplejidades[3],
        "diferencia_relativa_k3": (float(diferencia_relativa)
                                   if diferencia_relativa is not None
                                   else None),
        "tolerancia_relativa": TOLERANCIA_RELATIVA,
        "valor_referencia_auxiliar": PERPLEJIDAD_K3_REFERENCIA,
    }

    resultados = {
        "subtarea": "T2 - Parte 2 - Numero de temas (2, 3 y 4)",
        "parametros_comunes": {
            "vectorizador": ("CountVectorizer (mismas palabras vacias que la "
                             "Parte 1, resto de parametros por defecto)"),
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
        "n_iteracion_final": {f"k{k}": int(modelos[k].n_iter_)
                              for k in VALORES_K},
        "log_verosimilitud_total": {f"k{k}": logveros[k] for k in VALORES_K},
        "tabla_comparativa": tabla_comparativa,
        "mejor_n_temas_por_perplejidad": int(mejor_k),
        "palabras_top_por_tema": {f"k{k}": palabras_top[k] for k in VALORES_K},
        "palabras_top_por_tema_con_pesos": {
            f"k{k}": palabras_top_con_pesos[k] for k in VALORES_K
        },
        "vocabulario": vocabulario,
        "verificacion_con_T1": verificacion,
        "nota_metodologica": (
            "Se ajusta LDA (learning_method='batch', max_iter=50, "
            "random_state=0) con n_components = 2, 3 y 4 sobre la matriz "
            "documento-termino de los doce textos, obtenida con el mismo "
            "CountVectorizer y la misma lista de palabras vacias que en la "
            "Parte 1. La perplejidad se evalua sobre el corpus, igual que en "
            "la Parte 1 (el enunciado no define conjunto de retencion), de "
            "modo que los valores de k = 2, 3 y 4 son comparables entre si y "
            "con perplejidad_k3 de T1. La fuente del corpus y el resultado de "
            "la verificacion contra entradas/T1.json (vocabulario y "
            "perplejidad con k=3) quedan registrados en 'fuente_corpus' y "
            "'verificacion_con_T1'."
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
    print("Mismo CountVectorizer y mismos parametros de LDA que en la Parte 1.")
    print(f"Fuente del corpus: {fuente_corpus}")
    print(linea)
    print(df_tabla.to_string(index=False,
                             float_format=lambda valor: f"{valor:.6f}"))
    print(linea)
    print("Cifras principales (precision completa):")
    print(f"  perplejidad_k2 = {perplejidades[2]!r}")
    print(f"  perplejidad_k3 = {perplejidades[3]!r}")
    print(f"  perplejidad_k4 = {perplejidades[4]!r}")
    print(f"  Menor perplejidad (mejor ajuste): k = {mejor_k} "
          f"({perplejidades[mejor_k]:.6f})")
    print(f"  Verificacion contra T1: {estado_verificacion}")
    print(linea)


if __name__ == "__main__":
    main()
