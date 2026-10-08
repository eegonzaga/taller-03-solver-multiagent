"""Agente LECTOR — PDF → texto por página, con las tablas como tablas. Sin LLM.

Lo que revisa el código, porque son fallas que no fallan:

- **PDF escaneado**: sin capa de texto, cada página devuelve "" y el enunciado entero
  desaparecería sin error (Taller 02, Parte 0.a). Si casi no hay caracteres útiles, se marca
  `escaneado` y el solver se detiene diciendo por qué.
- **Tablas**: `page.find_tables()` las devuelve como Markdown y su texto se quita del flujo
  normal, para que una tabla no llegue como una lista de celdas sueltas.
- **Texto partido**: palabras cortadas con guion al final de línea se vuelven a unir; las
  ligaduras (ﬁ, ﬂ) se normalizan con NFKC.
- **Texto pegado**: una «palabra» de más de 40 letras sin espacios casi siempre son varias
  palabras pegadas por el extractor; se avisa.

Una tarea puede llegar como PDF o como carpeta (un paquete con su enunciado y sus datos). En
el segundo caso, los demás archivos son **adjuntos**: los .md/.txt se leen como parte del
enunciado y los datos se copian a la carpeta de cada script.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

MIN_CARACTERES_UTILES = 200
IGNORAR = {".venv", "__pycache__", "output", ".git", ".ipynb_checkpoints"}


@dataclass
class Linea:
    texto: str
    pagina: int
    tamano: float
    negrita: bool
    es_tabla: bool = False
    mono: bool = False          # código: no cuenta para el tamaño del cuerpo


@dataclass
class Documento:
    ruta: Path
    paginas: list[str]                       # texto por página, tablas en Markdown
    lineas: list[Linea]                      # con su formato, para encontrar encabezados
    tamano_cuerpo: float
    adjuntos: list[Path] = field(default_factory=list)        # datos para el sandbox
    anexos: dict[str, str] = field(default_factory=dict)       # .md/.txt del paquete
    avisos: list[str] = field(default_factory=list)
    escaneado: bool = False

    @property
    def texto(self) -> str:
        partes = [f"[página {i}]\n{t}" for i, t in enumerate(self.paginas, start=1)]
        partes += [f"[anexo {n}]\n{t}" for n, t in self.anexos.items()]
        return "\n\n".join(partes)


def normalizar(texto: str) -> str:
    return unicodedata.normalize("NFKC", texto).replace(" ", " ")


def _dentro(caja, rects) -> bool:
    x = (caja[0] + caja[2]) / 2
    y = (caja[1] + caja[3]) / 2
    return any(r.x0 <= x <= r.x1 and r.y0 <= y <= r.y1 for r in rects)


def leer_pdf(ruta: Path) -> Documento:
    doc = pymupdf.open(ruta)
    lineas: list[Linea] = []
    paginas: list[str] = []
    avisos: list[str] = []
    con_imagenes = 0
    for pagina in doc:
        n = pagina.number + 1
        tablas = pagina.find_tables().tables
        rects = [pymupdf.Rect(t.bbox) for t in tablas]
        con_imagenes += bool(pagina.get_images())
        elementos: list[tuple[float, Linea]] = []
        for bloque in pagina.get_text("dict", sort=True)["blocks"]:
            if bloque.get("type") != 0 or _dentro(bloque["bbox"], rects):
                continue
            for l in bloque["lines"]:
                spans = [s for s in l["spans"] if s["text"].strip()]
                if not spans:
                    continue
                texto = normalizar("".join(s["text"] for s in l["spans"])).strip()
                elementos.append((l["bbox"][1], Linea(
                    texto=texto, pagina=n, tamano=round(max(s["size"] for s in spans), 1),
                    negrita=all(s["flags"] & 16 or "Bold" in s["font"] for s in spans),
                    mono=all(s["flags"] & 8 or "Mono" in s["font"] or "Courier" in s["font"]
                             for s in spans))))
        for t in tablas:
            md = normalizar(t.to_markdown()).strip()
            elementos.append((t.bbox[1], Linea(texto=md, pagina=n, tamano=0, negrita=False,
                                               es_tabla=True)))
        elementos.sort(key=lambda e: e[0])
        lineas += [l for _, l in elementos]
        paginas.append(_unir_partidas("\n".join(l.texto for _, l in elementos)))

    utiles = sum(c.isalnum() for p in paginas for c in p)
    escaneado = utiles < MIN_CARACTERES_UTILES
    if escaneado:
        avisos.append(f"{ruta.name}: {utiles} caracteres útiles en {doc.page_count} páginas "
                      f"({con_imagenes} con imágenes). Parece escaneado: sin OCR no hay texto.")
    for p, texto in enumerate(paginas, start=1):
        pegadas = [w for w in re.findall(r"[^\W\d_]{41,}", texto)]
        if pegadas:
            avisos.append(f"página {p}: posible texto pegado: {pegadas[:3]}")

    # El tamaño del cuerpo es el del texto corrido: sin tablas ni código (en un enunciado
    # con mucho código, la letra del código ganaría y todo lo demás parecería encabezado).
    tamanos = Counter()
    for l in lineas:
        if not l.es_tabla and not l.mono:
            tamanos[l.tamano] += len(l.texto)
    cuerpo = tamanos.most_common(1)[0][0] if tamanos else 10.0
    return Documento(ruta=ruta, paginas=paginas, lineas=lineas, tamano_cuerpo=cuerpo,
                     avisos=avisos, escaneado=escaneado)


def _unir_partidas(texto: str) -> str:
    """«pala-\\nbra» → «palabra» (solo entre minúsculas, para no tocar «Breast-\\nCancer»)."""
    return re.sub(r"([a-záéíóúñ])-\n([a-záéíóúñ])", r"\1\2", texto)


def leer_entrada(ruta: str | Path) -> Documento:
    """Un PDF, o una carpeta-paquete: su PDF de enunciado más los adjuntos."""
    ruta = Path(ruta)
    if ruta.is_file():
        return leer_pdf(ruta)
    archivos = [p for p in sorted(ruta.rglob("*"))
                if p.is_file() and not (IGNORAR & set(p.relative_to(ruta).parts))
                and not p.name.startswith(".")]
    pdfs = [p for p in archivos if p.suffix.lower() == ".pdf"]
    if not pdfs:
        raise FileNotFoundError(f"{ruta}: la carpeta no trae ningún PDF de enunciado")
    principal = next((p for p in pdfs if "enunciado" in p.name.lower()), pdfs[0])
    doc = leer_pdf(principal)
    for p in archivos:
        if p == principal:
            continue
        if p.suffix.lower() in {".md", ".txt"}:
            doc.anexos[p.name] = normalizar(p.read_text(encoding="utf-8", errors="replace"))
        doc.adjuntos.append(p)
    doc.ruta = ruta
    return doc


def raiz_de_adjuntos(doc: Documento) -> Path:
    """Los adjuntos se copian conservando su ruta relativa al paquete (data/penguins.csv)."""
    return doc.ruta if doc.ruta.is_dir() else doc.ruta.parent
