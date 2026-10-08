# -*- coding: utf-8 -*-
"""
T2 - Parte 2 - Numero de temas (2, 3 y 4)
=========================================

Con el mismo CountVectorizer de la Parte 1 (misma lista de palabras vacias y
el resto de parametros por defecto) y los mismos parametros de LDA
(learning_method='batch', max_iter=50, random_state=0), se ajustan modelos
con n_components = 2, 3 y 4 sobre los doce textos del corpus y se reporta la
perplejidad de cada modelo en una sola tabla comparativa.

Entradas:
  * entradas/T1.json : resultados de la Parte 1 (vocabulario, perplejidad_k3,
    parametros usados, ...). Se emplea para replicar la misma configuracion y
    comprobar la coherencia del modelo con k=3.
  * Corpus de la practica: los doce textos del enunciado (3 temas, 4 textos
    por tema), embebidos en este script para que sea autocontenido. Si T1.json
    incluyera los textos originales, se usarian preferentemente.

Salidas:
  * resultados.json : perplejidad_k2, perplejidad_k3, perplejidad_k4, la tabla
    comparativa completa y cifras auxiliares (todo sin redondear).
  * Impresion por pantalla de la tabla comparativa.

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


# ----------------------------------------------------------------------
# 1) Corpus de la practica: los doce textos (3 temas x 4 textos por tema)
# ----------------------------------------------------------------------
TEXTOS = [
    # --- Tema 1: deportes (futbol) ---
    ("El equipo local ganó el partido de liga con dos goles en el segundo tiempo. "
     "El delantero marcó el gol decisivo tras una gran jugada colectiva y la "
     "afición celebró la victoria desde la grada. Con los tres puntos, el "
     "conjunto se coloca en la parte alta de la tabla."),
    ("La final del torneo se disputó en el estadio municipal ante miles de "
     "aficionados. El portero detuvo un penalti en el último minuto y consiguió "
     "el título para su club. Los jugadores levantaron la copa entre cánticos y "
     "el entrenador elogió el esfuerzo de la plantilla."),
    ("El club anunció la contratación de un centrocampista internacional para "
     "reforzar el centro del campo. El jugador firmó un contrato de cuatro "
     "temporadas y se incorporará a la pretemporada. La directiva confía en que "
     "su experiencia permita luchar por el campeonato de liga."),
    ("La selección ganó su partido clasificatorio gracias a un gol de cabeza en "
     "el tiempo de descuento. El conjunto mostró un juego ordenado, presionó "
     "alto y creó numerosas ocasiones. Con este resultado, el equipo mantiene "
     "vivas sus opciones de acudir al mundial."),
    # --- Tema 2: tecnologia / inteligencia artificial ---
    ("La compañía presentó un modelo de inteligencia artificial capaz de "
     "generar imágenes a partir de texto. El sistema emplea redes neuronales "
     "profundas entrenadas con millones de ejemplos. Los investigadores "
     "aseguran que la tecnología acelerará la creación de contenido digital."),
    ("El algoritmo de aprendizaje automático analiza grandes volúmenes de "
     "datos para predecir el comportamiento de los usuarios. Las empresas "
     "aplican estos modelos para optimizar sus servicios en la nube y reducir "
     "costes. Los expertos subrayan la importancia de la calidad de los datos "
     "en el entrenamiento."),
    ("La startup desarrolló un chatbot basado en procesamiento del lenguaje "
     "natural que atiende consultas las veinticuatro horas. El asistente "
     "virtual aprende de cada conversación y mejora sus respuestas. La empresa "
     "planea integrar el sistema en su aplicación móvil."),
    ("Los investigadores publicaron un artículo sobre redes neuronales "
     "profundas aplicadas al reconocimiento de voz. El modelo entrenado con "
     "grabaciones reales supera la precisión de los métodos clásicos. La "
     "tecnología permitirá transcripciones automáticas y traducción simultánea."),
    # --- Tema 3: salud / medicina ---
    ("Los médicos recomiendan una dieta equilibrada y ejercicio regular para "
     "prevenir enfermedades cardiovasculares. El hospital lanzó una campaña de "
     "salud para fomentar hábitos de vida saludables. Los especialistas "
     "insisten en la importancia de las revisiones médicas anuales."),
    ("Un estudio del hospital asocia el consumo habitual de frutas y verduras "
     "con un menor riesgo de diabetes. Los pacientes que siguen una "
     "alimentación saludable presentan mejores niveles de azúcar en sangre. "
     "Los doctores aconsejan consultar con un nutricionista antes de modificar "
     "la dieta."),
    ("La vacuna contra la gripe estará disponible en los centros de salud a "
     "partir del próximo mes. Las autoridades sanitarias recomiendan la "
     "inmunización de las personas mayores y de los pacientes de riesgo. El "
     "ministerio recuerda que la vacunación reduce las hospitalizaciones."),
    ("El tratamiento del dolor crónico combina medicación, fisioterapia y "
     "apoyo psicológico. Los especialistas del hospital explican que el "
     "ejercicio suave y el descanso adecuado mejoran la calidad de vida de los "
     "pacientes. La unidad del dolor diseña un plan personalizado para cada "
     "enfermo."),
]

ETIQUETAS_REALES = ["deportes"] * 4 + ["tecnologia"] * 4 + ["salud"] * 4

# Lista de palabras vacias (la misma que aplica el CountVectorizer de la Parte 1)
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
# 2) Carga de los resultados de la Parte 1 (T1)
# ----------------------------------------------------------------------
t1 = {}
if os.path.exists(RUTA_T1):
    try:
        with open(RUTA_T1, "r", encoding="utf-8") as manejador:
            t1 = json.load(manejador)
        print(f"[info] Resultados de T1 cargados desde {RUTA_T1}.")
    except (OSError, ValueError) as error:
        print(f"[aviso] No se pudo leer {RUTA_T1}: {error}")
else:
    print(f"[aviso] No se encontro {RUTA_T1}; se continua sin el.")

# Si T1 hubiera guardado los textos originales, se usan; si no, el corpus embebido.
textos = None
for clave in ("textos", "corpus", "documentos"):
    valor = t1.get(clave)
    if isinstance(valor, list) and valor and all(isinstance(x, str) for x in valor):
        textos = valor
        print(f"[info] Corpus tomado de T1.json (clave '{clave}').")
        break
if textos is None:
    textos = TEXTOS

# ----------------------------------------------------------------------
# 3) Vectorizacion: mismo CountVectorizer que en la Parte 1
#    (mismas palabras vacias; el resto de parametros, por defecto)
# ----------------------------------------------------------------------
vectorizador = CountVectorizer(stop_words=PALABRAS_VACIAS)
X = vectorizador.fit_transform(textos)

try:
    vocabulario = [str(v) for v in vectorizador.get_feature_names_out()]
except AttributeError:  # compatibilidad con versiones antiguas de scikit-learn
    vocabulario = [str(v) for v in vectorizador.get_feature_names()]

n_documentos = int(X.shape[0])
tamano_vocabulario = int(X.shape[1])
total_palabras = int(X.sum())
print(f"[info] Matriz documento-termino: {n_documentos} documentos x "
      f"{tamano_vocabulario} terminos ({total_palabras} palabras en total).")

if t1.get("n_documentos") is not None:
    try:
        if int(t1["n_documentos"]) != n_documentos:
            print("[aviso] El numero de documentos no coincide con el indicado en T1.")
    except (TypeError, ValueError):
        pass

# Comparacion del vocabulario con el de T1 (diagnostico de coherencia)
coincidencia_vocab_t1 = None
vocab_t1 = t1.get("vocabulario")
if isinstance(vocab_t1, list) and vocab_t1:
    conjunto_propio = set(vocabulario)
    conjunto_t1 = {str(v) for v in vocab_t1}
    interseccion = len(conjunto_propio & conjunto_t1)
    union = len(conjunto_propio | conjunto_t1)
    coincidencia_vocab_t1 = float(interseccion / union) if union else 0.0
    print(f"[info] Vocabulario vs T1: {interseccion} terminos comunes "
          f"(Jaccard = {coincidencia_vocab_t1:.4f}).")

# ----------------------------------------------------------------------
# 4) Ajuste de LDA con k = 2, 3 y 4 (mismos parametros que en la Parte 1)
#
# Nota metodologica: la perplejidad se evalua sobre la matriz documento-
# termino del corpus, siguiendo exactamente el protocolo de la Parte 1
# (el enunciado define la perplejidad "sobre el corpus" y el ejercicio no
# establece conjunto de retencion); asi los valores de k = 2, 3 y 4 son
# comparables entre si y con perplejidad_k3 de T1.
# ----------------------------------------------------------------------
perplejidades = {}
cota_por_palabra = {}
cota_total = {}
n_iteracion_final = {}
modelos = {}
palabras_top_por_tema = {}
palabras_top_por_tema_con_pesos = {}

for k in VALORES_K:
    lda = LatentDirichletAllocation(
        n_components=k,
        learning_method="batch",
        max_iter=50,
        random_state=0,
    )
    lda.fit(X)

    perp = float(lda.perplexity(X))           # perplejidad sobre el corpus
    log_por_palabra = float(-np.log(perp))    # cota por palabra = -log(perplejidad)

    perplejidades[k] = perp
    cota_por_palabra[k] = log_por_palabra
    cota_total[k] = float(log_por_palabra * total_palabras)
    n_iteracion_final[k] = int(lda.n_iter_)
    modelos[k] = lda

    componentes = lda.components_
    probs = componentes / componentes.sum(axis=1, keepdims=True)
    top_indices = [np.argsort(componentes[t])[::-1][:N_PALABRAS_TOP]
                   for t in range(k)]
    palabras_top_por_tema[k] = [[vocabulario[i] for i in idxs] for idxs in top_indices]
    palabras_top_por_tema_con_pesos[k] = [
        [(vocabulario[i], float(probs[t, i])) for i in idxs]
        for t, idxs in enumerate(top_indices)
    ]

    print(f"[info] LDA con k={k}: perplejidad = {perp:.6f} "
          f"(n_iter_ = {n_iteracion_final[k]})")

# ----------------------------------------------------------------------
# 5) Tabla comparativa de perplejidades
# ----------------------------------------------------------------------
tabla_comparativa = [
    {
        "n_temas": int(k),
        "perplejidad": perplejidades[k],
        "cota_log_verosimilitud_por_palabra": cota_por_palabra[k],
        "n_iteracion_final": n_iteracion_final[k],
    }
    for k in VALORES_K
]
df_tabla = pd.DataFrame(tabla_comparativa)
mejor_k = min(VALORES_K, key=lambda kk: perplejidades[kk])

# Comprobacion de coherencia del modelo k=3 con la Parte 1
coherencia_con_t1 = {}
perp3_t1 = t1.get("perplejidad_k3")
if perp3_t1 is not None:
    try:
        perp3_t1 = float(perp3_t1)
        diferencia = abs(perplejidades[3] - perp3_t1)
        coherencia_con_t1 = {
            "perplejidad_k3_en_T1": perp3_t1,
            "perplejidad_k3_en_T2": perplejidades[3],
            "diferencia_absoluta": float(diferencia),
            "coincide": bool(diferencia <= 1e-6 * max(1.0, abs(perp3_t1))),
        }
    except (TypeError, ValueError):
        pass
for clave in ("learning_method", "max_iter", "random_state", "n_componentes",
              "tamano_vocabulario"):
    if clave in t1:
        coherencia_con_t1[f"{clave}_en_T1"] = t1[clave]

# ----------------------------------------------------------------------
# 6) Guardado de todas las cifras en resultados.json
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T2 - Parte 2 - Numero de temas (2, 3 y 4)",
    "parametros_comunes": {
        "vectorizador": "CountVectorizer (mismas palabras vacias que T1, resto por defecto)",
        "learning_method": "batch",
        "max_iter": 50,
        "random_state": 0,
        "valores_n_components": [2, 3, 4],
    },
    "n_documentos": n_documentos,
    "tamano_vocabulario": tamano_vocabulario,
    "total_palabras_corpus": total_palabras,
    "perplejidad_k2": perplejidades[2],
    "perplejidad_k3": perplejidades[3],
    "perplejidad_k4": perplejidades[4],
    "cota_log_verosimilitud_por_palabra": {f"k{k}": cota_por_palabra[k] for k in VALORES_K},
    "cota_log_verosimilitud_total": {f"k{k}": cota_total[k] for k in VALORES_K},
    "n_iteracion_final": {f"k{k}": n_iteracion_final[k] for k in VALORES_K},
    "tabla_comparativa": tabla_comparativa,
    "mejor_n_temas_por_perplejidad": int(mejor_k),
    "palabras_top_por_tema": {f"k{k}": palabras_top_por_tema[k] for k in VALORES_K},
    "palabras_top_por_tema_con_pesos": {
        f"k{k}": palabras_top_por_tema_con_pesos[k] for k in VALORES_K
    },
    "vocabulario": vocabulario,
    "etiquetas_reales_referencia": ETIQUETAS_REALES,
    "coincidencia_vocabulario_con_T1": coincidencia_vocab_t1,
    "coherencia_con_T1": coherencia_con_t1,
    "nota_metodologica": (
        "Perplejidad calculada con LatentDirichletAllocation.perplexity sobre la "
        "matriz documento-termino del corpus, con el mismo protocolo que en la "
        "Parte 1 (el enunciado define la perplejidad 'sobre el corpus' y el "
        "ejercicio no establece conjunto de retencion), de modo que los valores "
        "de k = 2, 3 y 4 son comparables entre si y con perplejidad_k3 de T1."
    ),
}

with open(RUTA_RESULTADOS, "w", encoding="utf-8") as manejador:
    json.dump(resultados, manejador, ensure_ascii=False, indent=2, default=a_nativo)
print(f"\n[ok] Cifras guardadas en {RUTA_RESULTADOS}")

# ----------------------------------------------------------------------
# 7) Impresion de la tabla comparativa y de las cifras principales
# ----------------------------------------------------------------------
linea = "=" * 72
print("\n" + linea)
print("T2 - Parte 2 - Perplejidad segun el numero de temas (k = 2, 3, 4)")
print("Mismo CountVectorizer y mismos parametros de LDA que en la Parte 1;")
print("perplejidad evaluada sobre el corpus (protocolo de la Parte 1).")
print(linea)
print(df_tabla.to_string(index=False, float_format=lambda valor: f"{valor:.6f}"))
print(linea)
print("Cifras principales (precision completa):")
print(f"  perplejidad_k2 = {perplejidades[2]!r}")
print(f"  perplejidad_k3 = {perplejidades[3]!r}")
print(f"  perplejidad_k4 = {perplejidades[4]!r}")
print(f"  Menor perplejidad (mejor ajuste): k = {mejor_k} "
      f"({perplejidades[mejor_k]:.6f})")
if "coincide" in coherencia_con_t1:
    veredicto = ("coincide con T1" if coherencia_con_t1["coincide"]
                 else "NO coincide con T1")
    print(f"  Comprobacion k=3: perplejidad_k3(T1) = "
          f"{coherencia_con_t1['perplejidad_k3_en_T1']!r} -> {veredicto}")
