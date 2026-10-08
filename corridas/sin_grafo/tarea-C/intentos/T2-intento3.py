# -*- coding: utf-8 -*-
"""
T2 - Parte 2 - Numero de temas (2, 3, 4) para LDA.

Reutiliza los datos EXACTOS de la Parte 1 y los VERIFICA antes de reportar:

1) Los doce textos y la lista de palabras vacias se CARGAN de los archivos
   de la Parte 1: primero de entradas/T1.json y, si ahi no estan, de
   cualquier otro archivo JSON original de T1 presente en entradas/ o en la
   carpeta actual. Solo si ningun archivo los contiene se emplea, como
   ultimo respaldo, el corpus literal del enunciado (el mismo de la Parte 1).

2) Verificacion obligatoria antes de reportar, con los datos cargados:
   (a) vocabulario_coincide_con_T1 == True (igualdad de conjuntos entre el
       vocabulario del CountVectorizer reconstruido y el guardado por T1);
   (b) la perplejidad del modelo con k=3 (mismos hiperparametros que en la
       Parte 1) reproduce el valor de T1, 137.7877015669, dentro de una
       tolerancia numerica pequena (1e-6 relativa).
   Si alguna verificacion falla, el script se detiene con error y NO guarda
   resultados: solo con ambas comprobaciones superadas se recalcula y se
   guarda en resultados.json la tabla de perplejidades para k = 2, 3, 4.

3) Modelos: LatentDirichletAllocation(n_components=k, learning_method='batch',
   max_iter=50, random_state=0) con k = 2, 3, 4, sobre la matriz de conteos
   del mismo CountVectorizer de la Parte 1 (stop_words = lista de la Parte 1,
   resto de parametros por defecto).

La perplejidad se evalua sobre el corpus con la misma convencion que la
Parte 1 (el enunciado pide "la perplejidad del modelo sobre el corpus"; el
ejercicio no define un conjunto de prueba aparte y no se ajusta nada con
datos de prueba: se reproducen exactamente los modelos de la Parte 1).

Esta subtarea NO produce figura PNG.
"""

import json
import os

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# ======================================================================
# 1) Referencias de la Parte 1 (entradas/T1.json)
# ======================================================================
RUTA_T1 = os.path.join("entradas", "T1.json")
VALOR_K3_CORRECCION = 137.7877015669  # perplejidad k=3 de la Parte 1 (referencia)
TOL_RELATIVA = 1e-6                   # tolerancia numerica pequena


def cargar_json(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def _entero(valor):
    try:
        return int(valor)
    except (TypeError, ValueError):
        return None


def _flotante(valor):
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


t1 = {}
if os.path.isfile(RUTA_T1):
    try:
        t1 = cargar_json(RUTA_T1)
        if not isinstance(t1, dict):
            t1 = {}
        print("Datos de la Parte 1 cargados desde {}.".format(RUTA_T1))
    except Exception as exc:
        print("Aviso: no se pudo leer {} ({}).".format(RUTA_T1, exc))
        t1 = {}
else:
    print("Aviso: no existe {}.".format(RUTA_T1))

vocabulario_t1 = None
if isinstance(t1.get("vocabulario"), list) and len(t1["vocabulario"]) > 0:
    vocabulario_t1 = [str(p) for p in t1["vocabulario"]]
perplejidad_k3_t1 = _flotante(t1.get("perplejidad_k3"))
n_documentos_t1 = _entero(t1.get("n_documentos"))
if n_documentos_t1 is not None and n_documentos_t1 <= 0:
    n_documentos_t1 = None
tamano_vocabulario_t1 = _entero(t1.get("tamano_vocabulario"))
if tamano_vocabulario_t1 is not None and tamano_vocabulario_t1 <= 0:
    tamano_vocabulario_t1 = None

# Referencia que debe reproducir el modelo con k=3: el valor guardado por T1
# (que, segun la correccion, es 137.7877015669); si T1 no estuviera
# disponible, se usa el valor indicado en la correccion.
if perplejidad_k3_t1 is not None:
    referencia_k3 = perplejidad_k3_t1
    if abs(perplejidad_k3_t1 - VALOR_K3_CORRECCION) > TOL_RELATIVA * max(1.0, VALOR_K3_CORRECCION):
        print("Aviso: la perplejidad_k3 guardada por T1 ({!r}) difiere del valor de "
              "referencia indicado ({!r}).".format(perplejidad_k3_t1, VALOR_K3_CORRECCION))
else:
    referencia_k3 = VALOR_K3_CORRECCION

# ======================================================================
# 2) Respaldo: corpus literal del enunciado (los mismos doce textos de la
#    Parte 1) y su lista de palabras vacias. Se usa SOLO si ningun archivo
#    de la Parte 1 contiene los textos o las palabras vacias; en todos los
#    casos los datos empleados se VERIFICAN contra T1 (vocabulario
#    identico y perplejidad k=3 reproducida) antes de reportar: si este
#    respaldo no coincidiera con lo usado en la Parte 1, el script se
#    detiene con error en vez de reportar cifras no verificadas.
# ======================================================================
TEXTOS_PARTE_1 = [
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

PALABRAS_VACIAS_PARTE_1 = [
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

# ======================================================================
# 3) Carga de los textos y las palabras vacias EXACTOS de la Parte 1
#    desde los archivos disponibles (entradas/T1.json y cualquier otro
#    JSON original de T1 en entradas/ o en la carpeta actual).
# ======================================================================
CLAVES_POSIBLES_TEXTO = (
    "textos", "texto", "corpus", "documentos", "docs", "texts",
    "textos_parte1", "corpus_parte1", "documentos_parte1", "lista_textos",
)
CLAVES_POSIBLES_STOP = (
    "palabras_vacias", "stop_words", "stopwords", "stop_words_list",
    "palabras_vacias_parte1", "lista_palabras_vacias", "stop_words_parte1",
)


def _normalizar_lista_strings(valor):
    """Acepta list[str], dict[str,str] o dict[str,list[str]] y devuelve una
    lista plana de strings (o None si el valor no tiene ninguna de esas
    formas)."""
    if isinstance(valor, list) and valor and all(isinstance(s, str) for s in valor):
        return list(valor)
    if isinstance(valor, dict) and valor:
        valores = list(valor.values())
        if all(isinstance(s, str) for s in valores):
            return valores
        if all(isinstance(v, list) and v and all(isinstance(s, str) for s in v)
               for v in valores):
            plana = []
            for v in valores:
                plana.extend(v)
            return plana
    return None


def _buscar_lista_strings(obj, claves, profundidad=0):
    """Busca recursivamente la primera clave de 'claves' cuyo valor sea una
    lista de strings (o asimilable)."""
    if profundidad > 8:
        return None
    if isinstance(obj, dict):
        for clave in claves:
            if clave in obj:
                lista = _normalizar_lista_strings(obj[clave])
                if lista is not None:
                    return lista
        for valor in obj.values():
            resultado = _buscar_lista_strings(valor, claves, profundidad + 1)
            if resultado is not None:
                return resultado
    elif isinstance(obj, list):
        for valor in obj:
            resultado = _buscar_lista_strings(valor, claves, profundidad + 1)
            if resultado is not None:
                return resultado
    return None


rutas_candidatas = []
if os.path.isfile(RUTA_T1):
    rutas_candidatas.append(RUTA_T1)
if os.path.isdir("entradas"):
    for nombre in sorted(os.listdir("entradas")):
        ruta = os.path.join("entradas", nombre)
        if nombre.lower().endswith(".json") and os.path.isfile(ruta) and ruta != RUTA_T1:
            rutas_candidatas.append(ruta)
for nombre in sorted(os.listdir(".")):
    if (nombre.lower().endswith(".json") and os.path.isfile(nombre)
            and nombre != "resultados.json" and nombre not in rutas_candidatas):
        rutas_candidatas.append(nombre)

fuentes_textos = []  # (etiqueta_de_archivo, lista_de_textos)
fuentes_stop = []    # (etiqueta_de_archivo, lista_de_palabras_vacias)
for ruta in rutas_candidatas:
    try:
        datos = cargar_json(ruta)
    except Exception:
        continue
    etiqueta = ruta.replace(os.sep, "/")
    textos_arch = _buscar_lista_strings(datos, CLAVES_POSIBLES_TEXTO)
    if textos_arch is not None and len(textos_arch) >= 2:
        fuentes_textos.append((etiqueta, textos_arch))
    stop_arch = _buscar_lista_strings(datos, CLAVES_POSIBLES_STOP)
    if stop_arch is not None and len(stop_arch) >= 2:
        fuentes_stop.append((etiqueta, stop_arch))

# Los candidatos cuyo numero de textos coincide con n_documentos de T1 van primero.
if n_documentos_t1 is not None:
    fuentes_textos.sort(key=lambda par: 0 if len(par[1]) == n_documentos_t1 else 1)

textos_opciones = list(fuentes_textos) + [("respaldo_enunciado_parte1", TEXTOS_PARTE_1)]
stop_opciones = list(fuentes_stop) + [("respaldo_enunciado_parte1", PALABRAS_VACIAS_PARTE_1)]

# ======================================================================
# 4) Verificacion de candidatos: solo se acepta la combinacion
#    textos/palabras_vacias que supera las dos comprobaciones contra T1
#    (vocabulario identico y perplejidad k=3 reproducida).
# ======================================================================
def evaluar_candidato(textos_c, stop_c):
    """Devuelve (resultado, info). resultado es None si el candidato NO
    supera todas las verificaciones disponibles contra la Parte 1."""
    info = {"n_textos": int(len(textos_c)), "n_palabras_vacias": int(len(stop_c))}

    if n_documentos_t1 is not None and len(textos_c) != n_documentos_t1:
        info["motivo_descarte"] = "el numero de textos no coincide con n_documentos de T1"
        return None, info

    try:
        vec = CountVectorizer(stop_words=list(stop_c))
        Xc = vec.fit_transform(textos_c)
    except Exception as exc:
        info["motivo_descarte"] = "error al vectorizar: {}".format(exc)
        return None, info

    try:
        vocab = [str(p) for p in vec.get_feature_names_out()]
    except AttributeError:  # compatibilidad con versiones antiguas de sklearn
        vocab = [str(p) for p in vec.get_feature_names()]
    info["tamano_vocabulario"] = int(Xc.shape[1])

    # Verificacion (a): igualdad de conjuntos con el vocabulario de T1.
    coincide = None
    if vocabulario_t1 is not None:
        coincide = bool(set(vocab) == set(vocabulario_t1))
        info["vocabulario_coincide_con_T1"] = coincide
        if not coincide:
            solo_t1 = sorted(set(vocabulario_t1) - set(vocab))
            solo_cand = sorted(set(vocab) - set(vocabulario_t1))
            info["palabras_solo_en_T1"] = solo_t1[:10]
            info["palabras_solo_en_candidato"] = solo_cand[:10]
            info["motivo_descarte"] = "el vocabulario no coincide con el de T1"
            return None, info
    if tamano_vocabulario_t1 is not None and int(Xc.shape[1]) != tamano_vocabulario_t1:
        info["motivo_descarte"] = "el tamano del vocabulario no coincide con el de T1"
        return None, info

    # Verificacion (b): la perplejidad con k=3 debe reproducir el valor de T1.
    try:
        lda3 = LatentDirichletAllocation(
            n_components=3,
            learning_method="batch",
            max_iter=50,
            random_state=0,
        )
        lda3.fit(Xc)
        p3 = float(lda3.perplexity(Xc))
    except Exception as exc:
        info["motivo_descarte"] = "error al ajustar LDA k=3: {}".format(exc)
        return None, info

    info["perplejidad_k3"] = p3
    if referencia_k3 is not None:
        tolerancia = TOL_RELATIVA * max(1.0, abs(referencia_k3))
        diferencia = abs(p3 - referencia_k3)
        info["perplejidad_k3_referencia"] = float(referencia_k3)
        info["diferencia_absoluta_k3"] = float(diferencia)
        info["tolerancia_absoluta"] = float(tolerancia)
        if diferencia > tolerancia:
            info["motivo_descarte"] = "la perplejidad con k=3 no reproduce el valor de T1"
            return None, info

    info["verificacion"] = "superada"
    resultado = {
        "vectorizer": vec,
        "X": Xc,
        "vocabulario": vocab,
        "lda_k3": lda3,
        "perplejidad_k3": p3,
        "vocabulario_coincide_con_T1": coincide,
        "n_palabras_vacias": int(len(stop_c)),
    }
    return resultado, info


seleccion = None
info_seleccion = None
etiqueta_textos_usada = None
etiqueta_stop_usada = None
descartes = []

for etiqueta_t, textos_c in textos_opciones:
    for etiqueta_s, stop_c in stop_opciones:
        resultado, info = evaluar_candidato(textos_c, stop_c)
        if resultado is not None:
            seleccion = resultado
            info_seleccion = info
            etiqueta_textos_usada = etiqueta_t
            etiqueta_stop_usada = etiqueta_s
            break
        descartes.append({
            "fuente_textos": etiqueta_t,
            "fuente_palabras_vacias": etiqueta_s,
            "motivo": info.get("motivo_descarte"),
            "tamano_vocabulario": info.get("tamano_vocabulario"),
            "perplejidad_k3": info.get("perplejidad_k3"),
        })
    if seleccion is not None:
        break

if seleccion is None:
    print("\nNinguna combinacion textos/palabras vacias supero la verificacion contra la Parte 1:")
    for d in descartes:
        print("  - textos={} | palabras_vacias={} -> {}".format(
            d["fuente_textos"], d["fuente_palabras_vacias"], d["motivo"]))
    raise RuntimeError("Verificacion contra la Parte 1 fallida: "
                       "no se calculan ni se guardan resultados.")

if etiqueta_textos_usada == "respaldo_enunciado_parte1" or etiqueta_stop_usada == "respaldo_enunciado_parte1":
    print("Nota: los archivos de la Parte 1 no contienen los textos/palabras vacias crudos;")
    print("      se usa el corpus literal del enunciado (el de la Parte 1) y su identidad")
    print("      queda garantizada por la verificacion contra T1 (vocabulario y perplejidad k=3).")

# ======================================================================
# 5) Modelos LDA con k = 2, 3, 4 (mismos parametros que la Parte 1) y
#    perplejidad de cada uno sobre el corpus.
# ======================================================================
vec = seleccion["vectorizer"]
X = seleccion["X"]
vocabulario = seleccion["vocabulario"]
lda_k3 = seleccion["lda_k3"]

n_documentos = int(X.shape[0])
tamano_vocabulario = int(X.shape[1])

# k=3 ya esta ajustado y verificado; se ajustan k=2 y k=4 con los mismos
# hiperparametros (learning_method='batch', max_iter=50, random_state=0).
perplejidades = {3: float(seleccion["perplejidad_k3"])}
for k in (2, 4):
    lda_k = LatentDirichletAllocation(
        n_components=k,
        learning_method="batch",
        max_iter=50,
        random_state=0,
    )
    lda_k.fit(X)
    perplejidades[k] = float(lda_k.perplexity(X))

perplejidad_k2 = perplejidades[2]
perplejidad_k3 = perplejidades[3]
perplejidad_k4 = perplejidades[4]

tabla_perplejidades = [
    {"n_componentes": int(k), "perplejidad": perplejidades[k]}
    for k in (2, 3, 4)
]
perplejidades_por_k = {"2": perplejidad_k2, "3": perplejidad_k3, "4": perplejidad_k4}
mejor_k = int(min((2, 3, 4), key=lambda k: perplejidades[k]))

if vocabulario_t1 is not None:
    vocabulario_coincide_con_T1 = bool(set(vocabulario) == set(vocabulario_t1))
else:
    vocabulario_coincide_con_T1 = None  # no verificable (T1 sin vocabulario)
perplejidad_k3_reproduce_T1 = bool(referencia_k3 is not None)
diferencia_k3 = (abs(perplejidad_k3 - referencia_k3)
                 if referencia_k3 is not None else None)
tolerancia_abs = (TOL_RELATIVA * max(1.0, abs(referencia_k3))
                  if referencia_k3 is not None else None)

# ======================================================================
# 6) Guardado de todas las cifras en resultados.json (tipos nativos,
#    valores con toda su precision).
# ======================================================================
resultados = {
    "perplejidad_k2": perplejidad_k2,
    "perplejidad_k3": perplejidad_k3,
    "perplejidad_k4": perplejidad_k4,
    "tabla_perplejidades": tabla_perplejidades,
    "perplejidades_por_k": perplejidades_por_k,
    "mejor_k": mejor_k,
    "n_documentos": n_documentos,
    "tamano_vocabulario": tamano_vocabulario,
    "parametros_lda": {
        "n_componentes_evaluados": [2, 3, 4],
        "learning_method": "batch",
        "max_iter": 50,
        "random_state": 0,
    },
    "vocabulario_coincide_con_T1": vocabulario_coincide_con_T1,
    "perplejidad_k3_reproduce_T1": perplejidad_k3_reproduce_T1,
    "verificacion_parte1": {
        "vocabulario_coincide_con_T1": vocabulario_coincide_con_T1,
        "perplejidad_k3_reproduce_T1": perplejidad_k3_reproduce_T1,
        "perplejidad_k3_obtenida": perplejidad_k3,
        "perplejidad_k3_referencia": (float(referencia_k3)
                                      if referencia_k3 is not None else None),
        "perplejidad_k3_en_T1_json": perplejidad_k3_t1,
        "valor_k3_indicado_en_correccion": VALOR_K3_CORRECCION,
        "diferencia_absoluta_k3": (float(diferencia_k3)
                                   if diferencia_k3 is not None else None),
        "tolerancia_absoluta": (float(tolerancia_abs)
                                if tolerancia_abs is not None else None),
        "tolerancia_relativa": TOL_RELATIVA,
        "fuente_textos": etiqueta_textos_usada,
        "fuente_palabras_vacias": etiqueta_stop_usada,
        "n_palabras_vacias": int(seleccion["n_palabras_vacias"]),
        "n_candidatos_descartados": int(len(descartes)),
        "candidatos_descartados": descartes,
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ======================================================================
# 7) Impresion de las cifras principales.
# ======================================================================
print("=" * 72)
print("T2 - Parte 2: perplejidad de LDA con 2, 3 y 4 temas")
print("=" * 72)
print("Referencias de la Parte 1 disponibles:")
print("  vocabulario de T1          : {}".format(
    "{} palabras".format(len(vocabulario_t1)) if vocabulario_t1 is not None else "no disponible"))
print("  perplejidad_k3 de T1       : {}".format(
    repr(perplejidad_k3_t1) if perplejidad_k3_t1 is not None else "no disponible"))
print("  n_documentos de T1         : {}".format(
    n_documentos_t1 if n_documentos_t1 is not None else "no disponible"))
print("  tamano_vocabulario de T1   : {}".format(
    tamano_vocabulario_t1 if tamano_vocabulario_t1 is not None else "no disponible"))
print()
print("Textos usados          : {} ({})".format(etiqueta_textos_usada, n_documentos))
print("Palabras vacias usadas : {} ({} palabras)".format(
    etiqueta_stop_usada, seleccion["n_palabras_vacias"]))
print()
print("Verificacion contra la Parte 1 (exigida antes de reportar):")
print("  vocabulario_coincide_con_T1   : {}".format(vocabulario_coincide_con_T1))
print("  perplejidad k=3 obtenida      : {!r}".format(perplejidad_k3))
print("  perplejidad k=3 referencia T1 : {!r}".format(referencia_k3))
print("  diferencia absoluta           : {!r}".format(diferencia_k3))
print("  tolerancia absoluta           : {!r}".format(tolerancia_abs))
print("  perplejidad_k3_reproduce_T1   : {}".format(perplejidad_k3_reproduce_T1))
if descartes:
    print()
    print("Candidatos descartados por la verificacion: {}".format(len(descartes)))
    for d in descartes:
        print("  - textos={} | palabras_vacias={} -> {}".format(
            d["fuente_textos"], d["fuente_palabras_vacias"], d["motivo"]))
print()
print("Perplejidades sobre el corpus (misma convencion que la Parte 1):")
for k in (2, 3, 4):
    print("  perplejidad_k{} = {!r}".format(k, perplejidades[k]))
print()
df_tabla = pd.DataFrame(tabla_perplejidades)
df_tabla.columns = ["n_componentes (temas)", "perplejidad"]
print("Tabla comparativa de perplejidades:")
print(df_tabla.to_string(index=False, float_format="%.10f"))
print()
print("Numero de temas con menor perplejidad sobre el corpus: k={}".format(mejor_k))
print()
print("Cifras guardadas en resultados.json")
