# -*- coding: utf-8 -*-
"""
T2 - Parte 2: numero de temas (2, 3, 4) en LDA.

Reutiliza el mismo CountVectorizer de la Parte 1 (mismas palabras vacias,
resto de parametros por defecto) y los mismos parametros de LDA
(learning_method='batch', max_iter=50, random_state=0) para ajustar
LatentDirichletAllocation con n_components = 2, 3 y 4, y reportar la
perplejidad de cada modelo sobre el corpus en una sola tabla comparativa.

Notas:
- La perplejidad se calcula sobre el corpus, exactamente igual que pide el
  enunciado en la Parte 1 (este ejercicio no define un conjunto de prueba
  aparte, por lo que no hay ajuste de nada sobre datos de prueba).
- Esta subtarea NO requiere figura PNG.
- Todas las cifras se guardan en resultados.json y se imprimen por consola.
- Se lee entradas/T1.json solo con fines de verificacion/consistencia
  (no se usa para reajustar nada).
"""

import json
import os

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# ----------------------------------------------------------------------
# 1) Corpus: los doce textos de la Parte 1 (4 de deportes, 4 de economia
#    y politica, 4 de ciencia y tecnologia).
# ----------------------------------------------------------------------
textos = [
    # --- Deportes ---
    ("El equipo local ganó el partido de liga con dos goles en el segundo "
     "tiempo. El delantero marcó el gol de la victoria tras una gran jugada "
     "colectiva y los aficionados celebraron el triunfo en el estadio."),
    ("La selección se clasificó para el torneo internacional después de "
     "vencer a su rival por tres goles a uno. El entrenador destacó el "
     "esfuerzo de los jugadores durante todo el campeonato."),
    ("El club cerró el fichaje de un centrocampista para reforzar el centro "
     "del campo esta temporada. El jugador firmó su contrato y será "
     "presentado ante la afición en el estadio."),
    ("El ciclista ganó la etapa de montaña y se puso el maillot de líder de "
     "la vuelta. El corredor prepara la próxima carrera con su equipo."),
    # --- Economia y politica ---
    ("El banco central subió los tipos de interés para contener la "
     "inflación. Los mercados reaccionaron con caídas en la bolsa y la "
     "moneda se fortaleció frente a las demás divisas."),
    ("El gobierno aprobó los presupuestos generales con nuevas medidas "
     "fiscales. La oposición criticó la subida de impuestos y el aumento "
     "del gasto público."),
    ("La empresa presentó unos resultados trimestrales récord y anunció una "
     "gran inversión en innovación. Las acciones subieron en bolsa y los "
     "analistas mejoraron sus previsiones de beneficio."),
    ("La tasa de paro bajó en el último trimestre según los datos del "
     "ministerio de economía. Los expertos prevén un crecimiento moderado "
     "del producto interior bruto para el próximo año."),
    # --- Ciencia y tecnologia ---
    ("Los astrónomos descubrieron un nuevo exoplaneta que orbita una "
     "estrella cercana. El telescopio espacial observó la atmósfera del "
     "planeta y los datos del estudio se publicaron en una revista "
     "científica."),
    ("Los investigadores desarrollaron una vacuna eficaz contra el virus "
     "después de años de ensayos clínicos. El descubrimiento abre la puerta "
     "a nuevos tratamientos para la enfermedad."),
    ("Los modelos de inteligencia artificial generan textos e imágenes cada "
     "vez más realistas. El aprendizaje automático transforma la tecnología "
     "y la investigación científica."),
    ("El cohete despegó con éxito y colocó en órbita el nuevo satélite de "
     "comunicaciones. La agencia espacial confirmó que la misión cumplió "
     "todos sus objetivos."),
]

# ----------------------------------------------------------------------
# 2) Lista de palabras vacias (la misma lista de la Parte 1; el resto de
#    parametros del CountVectorizer por defecto).
# ----------------------------------------------------------------------
palabras_vacias = [
    "a", "al", "algo", "algun", "alguna", "algunas", "alguno", "algunos",
    "ante", "antes", "aqui", "aun", "aunque", "así", "asi",
    "bien", "cada", "como", "con", "contra", "cual", "cuales", "cuando",
    "de", "del", "desde", "donde", "durante",
    "e", "el", "ella", "ellas", "ello", "ellos", "en", "entre",
    "era", "eran", "es", "esa", "esas", "ese", "eso", "esos",
    "esta", "está", "estan", "están", "este", "esto", "estos",
    "fue", "fueron", "ha", "habia", "había", "han", "hasta", "hay",
    "la", "las", "le", "les", "lo", "los",
    "mas", "más", "me", "mi", "mis", "mucho", "muy",
    "nada", "ni", "no", "nos", "nosotros",
    "o", "otra", "otras", "otro", "otros",
    "para", "pero", "poco", "por", "porque", "que", "quien", "quienes",
    "se", "segun", "según", "ser", "sera", "será", "si", "sí", "sido",
    "sin", "sobre", "son", "su", "sus",
    "tal", "tan", "tanto", "te", "tambien", "también", "todas", "todo",
    "todos", "tras", "tu", "tus",
    "u", "un", "una", "unas", "uno", "unos", "usted", "ustedes",
    "va", "van", "y", "ya", "yo",
]

# ----------------------------------------------------------------------
# 3) Lectura de los resultados de la Parte 1 (entradas/T1.json), solo para
#    verificacion de consistencia (no se usa para reajustar nada).
# ----------------------------------------------------------------------
t1 = {}
ruta_t1 = os.path.join("entradas", "T1.json")
if os.path.exists(ruta_t1):
    try:
        with open(ruta_t1, "r", encoding="utf-8") as f:
            t1 = json.load(f)
    except Exception as exc:
        print("Aviso: no se pudo leer {} ({}); se continua sin el.".format(ruta_t1, exc))
else:
    print("Aviso: no existe {}; se continua sin el.".format(ruta_t1))

perplejidad_k3_t1 = None
if isinstance(t1, dict) and "perplejidad_k3" in t1:
    try:
        perplejidad_k3_t1 = float(t1["perplejidad_k3"])
    except (TypeError, ValueError):
        perplejidad_k3_t1 = None

vocabulario_t1 = None
if isinstance(t1, dict) and "vocabulario" in t1:
    try:
        vocabulario_t1 = list(t1["vocabulario"])
    except Exception:
        vocabulario_t1 = None

# ----------------------------------------------------------------------
# 4) Vectorizacion: mismo CountVectorizer que en la Parte 1
#    (stop_words = lista de la Parte 1, resto de parametros por defecto).
# ----------------------------------------------------------------------
vectorizer = CountVectorizer(stop_words=palabras_vacias)
X = vectorizer.fit_transform(textos)

try:
    vocabulario = list(vectorizer.get_feature_names_out())
except AttributeError:  # compatibilidad con versiones antiguas de sklearn
    vocabulario = list(vectorizer.get_feature_names())

n_documentos, tamano_vocabulario = X.shape
n_documentos = int(n_documentos)
tamano_vocabulario = int(tamano_vocabulario)

vocabulario_coincide_con_T1 = None
if vocabulario_t1 is not None:
    vocabulario_coincide_con_T1 = bool(set(vocabulario) == set(vocabulario_t1))

# ----------------------------------------------------------------------
# 5) LDA con k = 2, 3 y 4 (mismos parametros que en la Parte 1) y
#    perplejidad de cada modelo sobre el corpus.
# ----------------------------------------------------------------------
temas_evaluados = [2, 3, 4]
perplejidades = {}
for k in temas_evaluados:
    lda = LatentDirichletAllocation(
        n_components=k,
        learning_method="batch",
        max_iter=50,
        random_state=0,
    )
    lda.fit(X)
    # Perplejidad sobre el corpus, tal como pide el enunciado (igual que en
    # la Parte 1); este ejercicio no define un conjunto de prueba aparte.
    perplejidades[k] = float(lda.perplexity(X))

perplejidad_k2 = perplejidades[2]
perplejidad_k3 = perplejidades[3]
perplejidad_k4 = perplejidades[4]

# Tabla comparativa (una fila por numero de temas).
tabla_perplejidades = [
    {"n_componentes": int(k), "perplejidad": perplejidades[k]}
    for k in temas_evaluados
]

# Numero de temas con menor perplejidad sobre el corpus.
mejor_k = int(min(temas_evaluados, key=lambda k: perplejidades[k]))

diferencia_k3_vs_T1 = None
if perplejidad_k3_t1 is not None:
    diferencia_k3_vs_T1 = float(abs(perplejidad_k3 - perplejidad_k3_t1))

# ----------------------------------------------------------------------
# 6) Guardado de todas las cifras en resultados.json (valores con toda su
#    precision, tipos nativos de Python).
# ----------------------------------------------------------------------
resultados = {
    "perplejidad_k2": perplejidad_k2,
    "perplejidad_k3": perplejidad_k3,
    "perplejidad_k4": perplejidad_k4,
    "tabla_perplejidades": tabla_perplejidades,
    "mejor_k": mejor_k,
    "temas_evaluados": [int(k) for k in temas_evaluados],
    "n_documentos": n_documentos,
    "tamano_vocabulario": tamano_vocabulario,
    "parametros_lda": {
        "learning_method": "batch",
        "max_iter": 50,
        "random_state": 0,
    },
    "perplejidad_k3_en_T1": perplejidad_k3_t1,
    "diferencia_perplejidad_k3_vs_T1": diferencia_k3_vs_T1,
    "vocabulario_coincide_con_T1": vocabulario_coincide_con_T1,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 7) Impresion de las cifras principales.
# ----------------------------------------------------------------------
print("=" * 64)
print("T2 - Parte 2: perplejidad de LDA con 2, 3 y 4 temas")
print("=" * 64)
print("Documentos: {} | Tamano del vocabulario: {}".format(
    n_documentos, tamano_vocabulario))
print()
for k in temas_evaluados:
    print("Perplejidad (k={}): {:.10f}".format(k, perplejidades[k]))
print()
df_tabla = pd.DataFrame(tabla_perplejidades)
df_tabla.columns = ["n_componentes (temas)", "perplejidad"]
print("Tabla comparativa de perplejidades:")
print(df_tabla.to_string(index=False))
print()
print("Numero de temas con menor perplejidad sobre el corpus: k={}".format(mejor_k))
if perplejidad_k3_t1 is not None:
    print("Verificacion con T1: perplejidad_k3(T1)={:.10f} | "
          "diferencia absoluta={:.6e}".format(
              perplejidad_k3_t1, diferencia_k3_vs_T1))
if vocabulario_coincide_con_T1 is not None:
    print("Vocabulario identico al de T1: {}".format(vocabulario_coincide_con_T1))
print()
print("Resultados guardados en resultados.json")
