"""Agente INDEXADOR — el GraphRAG del enunciado y del material del curso.

REGLA 2 — GraphRAG con base fija:

- **Base fija, sin LLM.** Cada sección del enunciado es un nodo. Cada referencia que una
  sección hace a otra («Parte 1», «Tarea 3», «P0», «la parte anterior») es una arista
  `depende_de` que se extrae con **expresiones regulares**: una referencia está escrita o no
  lo está, y una regla no la inventa ni la olvida (Parte 0.b).
- **Encima, las entidades del LLM**: datasets, métodos, métricas y conceptos de cada sección
  y de cada fragmento de los papers del corpus (el del Taller 02). Las entidades con el mismo
  nombre normalizado son el mismo nodo, y así una sección del enunciado queda unida a los
  fragmentos de papers que tratan lo mismo (`menciona`).
- La capa vectorial (bge-m3 + Qdrant, del Taller 02) ordena los fragmentos del corpus
  cuando varios comparten entidades, y es lo único que usa la versión «sin grafo».

El corpus no cambia entre tareas: sus entidades y embeddings se guardan en `.cache/` y solo
la primera corrida paga por ellos (unos 750 fragmentos). Se indexa en paralelo.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx

import pymupdf

from .config import CACHE, CORPUS
from .lector import Documento
from .llm import ClienteLLM, PresupuestoAgotado
from .rag import IndiceVectorial

ETIQUETA = re.compile(r"\b(Parte|Tarea|Pregunta|Secci[oó]n|Ejercicio|Respuesta)\s+(\d+)\b", re.I)
PREGUNTA_CORTA = re.compile(r"\b(P\d{1,2})\b")
ANAFORA = re.compile(r"\b(la|el)\s+(parte|pregunta|secci[oó]n|tarea|ejercicio)\s+anterior\b", re.I)
TIPOS_ENTIDAD = {"dataset", "metodo", "metrica", "concepto"}


@dataclass
class Seccion:
    id: str
    titulo: str
    texto: str
    pagina: int

    @property
    def literal(self) -> str:
        return f"{self.titulo}\n{self.texto}".strip()


# ──────────────────────────────────────────────────── secciones (sin LLM)
def dividir_en_secciones(doc: Documento) -> list[Seccion]:
    """Un encabezado es una línea toda en negrita, más grande que el cuerpo y corta. Las
    líneas de encabezado seguidas y del mismo tamaño son un solo título partido en dos."""
    cuerpo = doc.tamano_cuerpo

    def es_encabezado(l) -> bool:
        return (not l.es_tabla and l.negrita and len(l.texto) <= 110
                and re.search(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]{3}", l.texto) is not None
                and (l.tamano >= cuerpo + 0.5 or ETIQUETA.match(l.texto) is not None))

    tamano_titulo = max((l.tamano for l in doc.lineas if es_encabezado(l)), default=0)
    secciones = [Seccion("S0", "Preámbulo", "", 1)]
    anterior_fue_encabezado, tamano_anterior = False, 0.0
    for l in doc.lineas:
        if es_encabezado(l):
            if anterior_fue_encabezado and l.tamano == tamano_anterior:
                secciones[-1].titulo += " " + l.texto          # título partido en dos líneas
            elif l.tamano == tamano_titulo and len(secciones) == 1 and not secciones[0].texto:
                secciones[0].titulo = l.texto                  # el título del documento
            else:
                secciones.append(Seccion(f"S{len(secciones)}", l.texto, "", l.pagina))
            anterior_fue_encabezado, tamano_anterior = True, l.tamano
            continue
        anterior_fue_encabezado = False
        secciones[-1].texto += l.texto + "\n"
    for nombre, texto in doc.anexos.items():
        secciones.append(Seccion(f"S{len(secciones)}", f"Anexo: {nombre}", texto, 0))
    for s in secciones:
        s.texto = re.sub(r"([a-záéíóúñ])-\n([a-záéíóúñ])", r"\1\2", s.texto).strip()
    return secciones


def _clave(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", t).split())


def aristas_depende_de(secciones: list[Seccion]) -> list[dict]:
    """REGLA 2: las referencias entre secciones, con regex. Devuelve
    [{"origen", "destino", "evidencia"}], donde `origen` depende de `destino`."""
    anclas: dict[str, str] = {}
    for s in secciones:
        m = ETIQUETA.match(s.titulo)
        if m:
            anclas[_clave(f"{m.group(1)} {m.group(2)}")] = s.id
        for linea in s.texto.splitlines():             # «P0. Antes de ejecutar…» define P0
            m = re.match(r"\s*(P\d{1,2})\s*[.:)]", linea)
            if m:
                anclas.setdefault(_clave(m.group(1)), s.id)

    aristas: list[dict] = []
    vistas: set[tuple[str, str]] = set()
    for i, s in enumerate(secciones):
        menciones = [(m.group(0), m.start()) for m in ETIQUETA.finditer(s.texto)]
        menciones += [(m.group(1), m.start()) for m in PREGUNTA_CORTA.finditer(s.texto)]
        for texto, pos in menciones:
            destino = anclas.get(_clave(texto))
            if destino and destino != s.id and (s.id, destino) not in vistas:
                vistas.add((s.id, destino))
                aristas.append({"origen": s.id, "destino": destino,
                                "evidencia": " ".join(s.texto[max(0, pos - 50):pos + 40].split())})
        for m in ANAFORA.finditer(s.texto):
            if i > 1 and (s.id, secciones[i - 1].id) not in vistas:
                vistas.add((s.id, secciones[i - 1].id))
                aristas.append({"origen": s.id, "destino": secciones[i - 1].id,
                                "evidencia": m.group(0)})
    return aristas


# ──────────────────────────────────────────────────── corpus de papers
PALABRAS_FRAGMENTO = 300      # ≈ 450 tokens de bge-m3: como los fragmentos de 512 del Taller 02
SOLAPAMIENTO = 50
MIN_CARACTERES_PAGINA = 200   # Taller 02: una página casi sin texto no se indexa (¿escaneo?)


def fragmentos_del_corpus(carpeta: Path = CORPUS) -> list[dict]:
    """Los papers del Taller 02 en ventanas de palabras con solapamiento, por página, para
    que cada fragmento lleve su cita exacta (archivo y página)."""
    fragmentos: list[dict] = []
    for ruta in sorted(carpeta.glob("*.pdf")):
        doc = pymupdf.open(ruta)
        for pagina in doc:
            texto = unicodedata.normalize("NFKC", pagina.get_text())
            if sum(c.isalnum() for c in texto) < MIN_CARACTERES_PAGINA:
                continue
            palabras = texto.split()
            paso = PALABRAS_FRAGMENTO - SOLAPAMIENTO
            for inicio in range(0, max(len(palabras) - SOLAPAMIENTO, 1), paso):
                trozo = " ".join(palabras[inicio:inicio + PALABRAS_FRAGMENTO])
                fragmentos.append({"id": f"C:{ruta.stem}#p{pagina.number + 1}.{inicio // paso}",
                                   "archivo": ruta.name, "pagina": pagina.number + 1,
                                   "texto": trozo})
    return fragmentos


# ──────────────────────────────────────────────────── entidades (LLM)
PROMPT_ENTIDADES = """Extraes entidades de un texto de un curso de maestría en IA (un enunciado o
un fragmento de un paper, en español o en inglés).
Tipos permitidos: dataset, metodo, metrica, concepto.
- dataset: conjuntos de datos concretos (p. ej. «Breast Cancer Wisconsin», «Palmer Penguins»).
- metodo: algoritmos, modelos o técnicas (p. ej. «Naive Bayes gaussiano», «TF-IDF», «PCA», «SVD»).
- metrica: medidas de evaluación (p. ej. «accuracy», «F1 macro», «MRR», «entropía»).
- concepto: ideas teóricas (p. ej. «sesgo y varianza», «temperatura», «gradiente»).
Nombres cortos y canónicos, en español salvo nombres propios (sin artículos). Máximo 12 entidades. Responde solo JSON:
{"entidades": [{"nombre": "...", "tipo": "dataset|metodo|metrica|concepto"}]}"""


def _limpiar_entidades(datos: dict) -> list[dict]:
    salida, vistas = [], set()
    for e in datos.get("entidades", []) if isinstance(datos, dict) else []:
        if not isinstance(e, dict):
            continue
        nombre = str(e.get("nombre", "")).strip()
        tipo = _clave(str(e.get("tipo", "concepto"))).replace(" ", "")
        tipo = tipo if tipo in TIPOS_ENTIDAD else "concepto"
        clave = re.sub(r"^(el|la|los|las|un|una) ", "", _clave(nombre))
        if 2 <= len(clave) <= 60 and clave not in vistas:
            vistas.add(clave)
            salida.append({"nombre": nombre, "tipo": tipo, "clave": clave})
    return salida


class IndexadorGraphRAG:
    """`llm` es el de la corrida (entidades del enunciado, con su presupuesto). `llm_corpus`
    indexa el corpus la primera vez: es infraestructura compartida por todas las tareas, como
    la ingesta del Taller 02, y lleva su propia traza en .cache/indexado-corpus/."""

    def __init__(self, llm: ClienteLLM, codificador, traza, hilos: int = 8,
                 llm_corpus: ClienteLLM | None = None):
        self.llm = llm
        self.llm_corpus = llm_corpus or llm
        self.codificador = codificador
        self.traza = traza
        self.hilos = hilos

    def _entidades(self, texto: str, llm: ClienteLLM | None = None) -> list[dict]:
        datos = (llm or self.llm).pedir_json("indexador", [
            {"role": "system", "content": PROMPT_ENTIDADES},
            {"role": "user", "content": texto[:6000]}], max_tokens=6000, salida_estimada=800)
        return _limpiar_entidades(datos)

    def _entidades_en_paralelo(self, textos: list[str], llm: ClienteLLM | None = None) -> list[list[dict]]:
        def una(texto):
            try:
                return self._entidades(texto, llm)
            except PresupuestoAgotado:
                raise
            except Exception as err:  # noqa: BLE001 — sin entidades, la base fija sigue
                self.traza.registrar("aviso", agente="indexador", error=str(err)[:300])
                return []
        with ThreadPoolExecutor(self.hilos) as pool:
            return list(pool.map(una, textos))

    def _corpus_con_entidades(self) -> list[dict]:
        """El corpus con sus entidades; la caché depende del modelo y del texto."""
        fragmentos = fragmentos_del_corpus()
        ruta = CACHE / f"corpus-entidades-{_clave(self.llm_corpus.modelo).replace(' ', '_')}.json"
        cache = json.loads(ruta.read_text()) if ruta.exists() else {}
        hash_de = {f["id"]: hashlib.sha1(f["texto"].encode()).hexdigest() for f in fragmentos}
        faltan = [f for f in fragmentos if hash_de[f["id"]] not in cache]
        if faltan:
            self.traza.registrar("indexado_corpus", fragmentos=len(faltan), en_cache=len(fragmentos) - len(faltan))
            textos = [f["texto"] for f in faltan]
            for f, ents in zip(faltan, self._entidades_en_paralelo(textos, self.llm_corpus)):
                cache[hash_de[f["id"]]] = ents
            ruta.parent.mkdir(parents=True, exist_ok=True)
            ruta.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
        for f in fragmentos:
            f["entidades"] = cache.get(hash_de[f["id"]], [])
        return fragmentos

    def construir(self, doc: Documento, usar_entidades: bool = True) -> "GrafoTarea":
        secciones = dividir_en_secciones(doc)
        aristas = aristas_depende_de(secciones)
        g = nx.DiGraph()
        for s in secciones:
            g.add_node(s.id, tipo="seccion", titulo=s.titulo, texto=s.literal, pagina=s.pagina)
        for a in aristas:
            g.add_edge(a["origen"], a["destino"], tipo="depende_de", evidencia=a["evidencia"])

        corpus = self._corpus_con_entidades() if usar_entidades else fragmentos_del_corpus()
        for f in corpus:
            g.add_node(f["id"], tipo="corpus", titulo=f"{f['archivo']}, p. {f['pagina']}",
                       texto=f["texto"], archivo=f["archivo"])
        ents_secciones = self._entidades_en_paralelo([s.literal for s in secciones]) \
            if usar_entidades else [[] for _ in secciones]
        for nodo, ents in [(s.id, e) for s, e in zip(secciones, ents_secciones)] + \
                          [(f["id"], f.get("entidades", [])) for f in corpus]:
            for e in ents:
                eid = f"E:{e['clave']}"
                if eid not in g:
                    g.add_node(eid, tipo="entidad", titulo=e["nombre"], clase=e["tipo"])
                g.add_edge(nodo, eid, tipo="menciona")

        textos = {n: d["texto"] for n, d in g.nodes(data=True) if d["tipo"] in {"seccion", "corpus"}}
        indice = IndiceVectorial(self.codificador)
        indice.indexar(list(textos), list(textos.values()),
                       [{"tipo": g.nodes[n]["tipo"]} for n in textos])
        grafo = GrafoTarea(g, secciones, indice)
        self.traza.registrar("grafo", secciones=len(secciones), depende_de=aristas,
                             entidades=sum(1 for _, d in g.nodes(data=True) if d["tipo"] == "entidad"),
                             fragmentos_corpus=len(corpus), embeddings=self.codificador.nombre)
        return grafo


# ──────────────────────────────────────────────────── el grafo ya construido
class GrafoTarea:
    def __init__(self, g: nx.DiGraph, secciones: list[Seccion], indice: IndiceVectorial):
        self.g = g
        self.secciones = secciones
        self.orden = {s.id: i for i, s in enumerate(secciones)}
        self.indice = indice

    def cita(self, nodo: str) -> str:
        d = self.g.nodes[nodo]
        if d["tipo"] == "seccion":
            pag = f", p. {d['pagina']}" if d.get("pagina") else ""
            return f"[enunciado · {nodo} «{d['titulo']}»{pag}]"
        return f"[corpus · {d['titulo']}]"

    def texto(self, nodo: str) -> str:
        return self.g.nodes[nodo]["texto"]

    def dependencias(self, ids: list[str]) -> list[str]:
        """Cierre transitivo por `depende_de`, en el orden del enunciado."""
        vistos, pila = set(), list(ids)
        while pila:
            actual = pila.pop()
            for _, destino, d in self.g.out_edges(actual, data=True):
                if d["tipo"] == "depende_de" and destino not in vistos and destino not in ids:
                    vistos.add(destino)
                    pila.append(destino)
        return sorted(vistos, key=lambda n: self.orden.get(n, 999))

    def entidades_de(self, ids: list[str]) -> set[str]:
        return {d for n in ids if n in self.g
                for _, d, t in self.g.out_edges(n, data="tipo") if t == "menciona"}

    def corpus_relacionado(self, ids: list[str], consulta: str, k: int) -> list[tuple[str, int]]:
        """Fragmentos de papers que comparten entidades con las secciones; con muchos
        empates (750 fragmentos), desempata la similitud con la subtarea."""
        entidades = self.entidades_de(ids)
        cercanas = {r["nodo"]: r["score"] for r in self.indice.buscar(consulta, 60, "corpus")}
        puntaje: dict[str, int] = {}
        for e in entidades:
            for frag, _ in self.g.in_edges(e):
                if self.g.nodes[frag]["tipo"] == "corpus":
                    puntaje[frag] = puntaje.get(frag, 0) + 1
        orden = sorted(puntaje, key=lambda n: (-puntaje[n] - 2 * cercanas.get(n, 0)))
        if not orden:                       # sin entidades compartidas: la más parecida
            orden = list(cercanas)
        return [(n, puntaje.get(n, 0)) for n in orden[:k]]

    def buscar(self, consulta: str, k: int) -> list[dict]:
        """Versión «sin grafo»: solo similitud, sobre secciones y corpus a la vez (como 0.b)."""
        return self.indice.buscar(consulta, k)

    def guardar(self, ruta: Path) -> None:
        datos = nx.node_link_data(self.g, edges="aristas")
        for n in datos["nodes"]:
            if n.get("tipo") == "corpus":
                n["texto"] = n["texto"][:200] + " …"
        ruta.write_text(json.dumps({"secciones": [asdict(s) for s in self.secciones], **datos},
                                   ensure_ascii=False, indent=1), encoding="utf-8")

    def dibujar(self, ruta: Path, max_corpus: int = 10) -> None:
        """Tres columnas: secciones (con las aristas depende_de en rojo), entidades de las
        secciones y los fragmentos del corpus que comparten más entidades con ellas."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        secs = [s.id for s in self.secciones]
        ents = sorted(self.entidades_de(secs), key=lambda e: self.g.nodes[e]["titulo"].lower())
        puntaje: dict[str, int] = {}
        for e in ents:
            for n, _ in self.g.in_edges(e):
                if self.g.nodes[n]["tipo"] == "corpus":
                    puntaje[n] = puntaje.get(n, 0) + 1
        papers = sorted(puntaje, key=lambda n: -puntaje[n])[:max_corpus]

        alto = max(len(secs), len(ents) / 2.2, len(papers), 6)
        fig, ax = plt.subplots(figsize=(17, 0.55 * alto + 2))
        pos = {}
        for col, nodos, x in ((0, secs, 0.0), (1, ents, 1.0), (2, papers, 2.0)):
            for i, n in enumerate(nodos):
                pos[n] = (x, 1 - (i + 0.5) / max(len(nodos), 1))

        for o, d, t in self.g.edges(data="tipo"):
            if o in pos and d in pos and t == "menciona":
                ax.plot([pos[o][0], pos[d][0]], [pos[o][1], pos[d][1]], color="#bbbbbb", lw=0.5, zorder=1)
        for o, d, t in self.g.edges(data="tipo"):
            if t == "depende_de":
                ax.annotate("", xy=(pos[d][0] - 0.02, pos[d][1]), xytext=(pos[o][0] - 0.02, pos[o][1]),
                            arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=2,
                                            connectionstyle="arc3,rad=0.45"), zorder=3)
        estilos = {"seccion": ("#d6eaf8", 9), "entidad": ("#fdebd0", 7.5), "corpus": ("#e8f8f5", 7.5)}
        for n, (x, y) in pos.items():
            d = self.g.nodes[n]
            color, tam = estilos[d["tipo"]]
            etiqueta = f"{n} · {d['titulo']}" if d["tipo"] == "seccion" else d["titulo"]
            if d["tipo"] == "entidad":
                etiqueta = f"{d['titulo']} ({d.get('clase', '')[:4]})"
            ax.text(x, y, etiqueta[:58], ha="center", va="center", fontsize=tam, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="#555555", lw=0.6))
        ax.plot([], [], color="#c0392b", lw=2, label="depende_de (regla)")
        ax.plot([], [], color="#bbbbbb", lw=1, label="menciona (entidades del LLM)")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False)
        for x, t in ((0, "Secciones del enunciado"), (1, "Entidades"), (2, "Corpus (papers)")):
            ax.text(x, 1.04, t, ha="center", fontsize=11, weight="bold")
        ax.set_xlim(-0.45, 2.45)
        ax.set_ylim(-0.04, 1.08)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(ruta, dpi=130)
        plt.close(fig)
