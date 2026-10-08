"""El ORQUESTADOR — un grafo de estados de LangGraph.

    leer → indexar → planificar ⇄ validar_plan (REGLA 1)
         → elegir_subtarea → investigar → programar → ejecutar → revisar ─┐
                ↑                 │ (redacción)                 ↑          │ rechazado y quedan intentos
                └─────────────────┘                             └ programar┘
         → (sin subtareas o sin presupuesto) redactar ⇄ verificar (REGLA 5) → FIN

El estado de LangGraph solo lleva datos simples (plan, intentos, veredictos…); los objetos
pesados (el grafo de conocimiento, el cliente del LLM) viven en esta clase.

Los frenos de la Parte 3, en el código:
  FRENO 1 — `PresupuestoAgotado` en cualquier nodo → marca y pasa a redactar.
  FRENO 2 — `revisar`: al llegar a `max_intentos`, la subtarea queda fallida y se sigue;
            `ejecutar`: un script idéntico a uno rechazado no se vuelve a ejecutar.
  FRENO 3 — `sandbox.ejecutar`: tiempo máximo y `killpg`.
  FRENO 4 — `ejecutar`: un script que usa la red espera la aprobación de una persona.
"""
from __future__ import annotations

import functools
import json
import shutil
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import sandbox
from .config import CACHE, Config
from .grafo import IndexadorGraphRAG
from .investigador import como_texto, contexto_de
from .lector import leer_entrada, raiz_de_adjuntos
from .llm import ClienteLLM, Presupuesto, PresupuestoAgotado
from .planificador import Planificador, orden_topologico, validar_plan
from .programador import ARCHIVO_RESULTADOS, Programador, claves_de
from .rag import crear_codificador
from .redactor import Redactor, entregable_minimo, validar_entregable
from .revisor import Revisor, Veredicto
from .traza import Traza


class Estado(TypedDict, total=False):
    plan: dict
    problemas_plan: list[str]
    rondas_plan: int
    pendientes: list[str]
    actual: str | None
    contexto: str
    codigo: str
    ejecucion: dict
    veredicto: dict
    subtareas: dict            # id → {"id", "tipo", "status", "intentos"}
    rechazados: dict           # id → huellas de scripts rechazados (FRENO 2)
    presupuesto_agotado: bool
    faltantes: list[str]
    borrador: str
    problemas_entrega: list[str]
    rondas_redactor: int
    entregable: str
    status: str


def nodo(funcion):
    """FRENO 1 en cada nodo: si el presupuesto se acaba, se anota y el grafo pasa a redactar."""
    @functools.wraps(funcion)
    def envoltura(self, estado: Estado) -> dict:
        try:
            return funcion(self, estado)
        except PresupuestoAgotado as err:
            self.traza.registrar("freno_presupuesto", nodo=funcion.__name__, detalle=str(err))
            return {"presupuesto_agotado": True}
    return envoltura


class Orquestador:
    def __init__(self, entrada: str, salida: Path, llm: ClienteLLM, traza: Traza, config: Config):
        self.entrada = entrada
        self.salida = salida
        self.llm = llm
        self.traza = traza
        self.config = config
        self.planificador = Planificador(llm)
        self.programador = Programador(llm)
        self.revisor = Revisor(llm)
        self.redactor = Redactor(llm)
        self.confirmar_red = config.confirmar_red or sandbox.confirmar_por_consola
        self.doc = None
        self.grafo = None
        self.resultados: dict[str, dict] = {}     # id → {"json", "stdout", "figuras", "codigo"}
        self.app = self._construir()

    # ───────────────────────────────────────────────────────────── el grafo de estados
    def _construir(self):
        g = StateGraph(Estado)
        for nombre in ("leer", "indexar", "planificar", "validar_plan", "elegir_subtarea",
                       "investigar", "programar", "ejecutar", "revisar", "redactar", "verificar"):
            g.add_node(nombre, getattr(self, nombre))
        g.add_edge(START, "leer")
        g.add_conditional_edges("leer", lambda e: END if e.get("status") == "fallido" else "indexar",
                                ["indexar", END])
        g.add_conditional_edges("indexar", lambda e: END if e.get("status") == "fallido"
                                or e.get("presupuesto_agotado") else "planificar",
                                ["planificar", END])
        g.add_conditional_edges("planificar", lambda e: END if e.get("presupuesto_agotado") else "validar_plan",
                                ["validar_plan", END])
        g.add_conditional_edges("validar_plan", self._tras_validar_plan,
                                ["planificar", "elegir_subtarea", END])
        g.add_conditional_edges("elegir_subtarea", lambda e: "investigar" if e.get("actual") else "redactar",
                                ["investigar", "redactar"])
        g.add_conditional_edges("investigar", self._tras_investigar,
                                ["programar", "elegir_subtarea", "redactar"])
        g.add_conditional_edges("programar", lambda e: "redactar" if e.get("presupuesto_agotado") else "ejecutar",
                                ["ejecutar", "redactar"])
        g.add_edge("ejecutar", "revisar")
        g.add_conditional_edges("revisar", self._tras_revisar,
                                ["programar", "elegir_subtarea", "redactar"])
        g.add_edge("redactar", "verificar")
        g.add_conditional_edges("verificar", self._tras_verificar, ["redactar", END])
        return g.compile()

    def correr(self) -> Estado:
        return self.app.invoke({}, {"recursion_limit": 400})

    # ───────────────────────────────────────────────────────────── nodos
    @nodo
    def leer(self, estado: Estado) -> dict:
        self.doc = leer_entrada(self.entrada)
        self.traza.registrar("lector", entrada=str(self.entrada), paginas=len(self.doc.paginas),
                             caracteres=len(self.doc.texto), escaneado=self.doc.escaneado,
                             avisos=self.doc.avisos, adjuntos=[p.name for p in self.doc.adjuntos])
        if self.doc.escaneado:
            return {"status": "fallido"}
        (self.salida / "enunciado.txt").write_text(self.doc.texto, encoding="utf-8")
        return {}

    @nodo
    def indexar(self, estado: Estado) -> dict:
        codificador = crear_codificador(self.config.embeddings, CACHE)
        # El corpus se indexa una vez para todas las tareas: su costo no es de esta corrida.
        carpeta_corpus = CACHE / "indexado-corpus"
        carpeta_corpus.mkdir(parents=True, exist_ok=True)
        traza_corpus = Traza(carpeta_corpus, reiniciar=False)
        llm_corpus = ClienteLLM(self.llm.servidor, traza_corpus, Presupuesto(10**12, 0))
        indexador = IndexadorGraphRAG(self.llm, codificador, self.traza, self.config.hilos_indexado,
                                      llm_corpus)
        # La versión «sin grafo» no paga por las entidades: solo usa la similitud.
        self.grafo = indexador.construir(self.doc, usar_entidades=self.config.usar_grafo)
        self.grafo.guardar(self.salida / "grafo.json")
        self.grafo.dibujar(self.salida / "grafo.png")
        return {}

    @nodo
    def planificar(self, estado: Estado) -> dict:
        plan = self.planificador.planificar(self.grafo, self.doc.texto,
                                            estado.get("problemas_plan"), estado.get("plan"))
        return {"plan": plan, "rondas_plan": estado.get("rondas_plan", 0) + 1}

    @nodo
    def validar_plan(self, estado: Estado) -> dict:
        problemas = validar_plan(estado["plan"], self.grafo)                    # REGLA 1
        self.traza.registrar("validacion_plan", ronda=estado["rondas_plan"],
                             valido=not problemas, problemas=problemas)
        if problemas:
            return {"problemas_plan": problemas}
        plan = estado["plan"]
        (self.salida / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
        subtareas = {s["id"]: {"id": s["id"], "tipo": s["tipo"], "status": "pendiente", "intentos": 0}
                     for s in plan["subtareas"]}
        return {"problemas_plan": [], "pendientes": orden_topologico(plan),
                "subtareas": subtareas, "rechazados": {}}

    def _tras_validar_plan(self, estado: Estado) -> str:
        if estado.get("presupuesto_agotado"):
            return END
        if not estado.get("problemas_plan"):
            return "elegir_subtarea"
        if estado["rondas_plan"] < self.config.max_rondas_plan:
            return "planificar"
        self.traza.registrar("plan_invalido", problemas=estado["problemas_plan"])
        return END

    @nodo
    def elegir_subtarea(self, estado: Estado) -> dict:
        pendientes = list(estado.get("pendientes", []))
        if estado.get("presupuesto_agotado") or not pendientes:
            return {"actual": None, "pendientes": pendientes}
        return {"actual": pendientes.pop(0), "pendientes": pendientes, "codigo": "",
                "veredicto": {}, "ejecucion": {}}

    @nodo
    def investigar(self, estado: Estado) -> dict:
        subtarea = self._subtarea(estado)
        fragmentos = contexto_de(subtarea, estado["plan"], self.grafo, self.config.usar_grafo,
                                 self.config.k_fragmentos, self.config.k_corpus)
        self.traza.registrar("investigador", subtarea=subtarea["id"], usar_grafo=self.config.usar_grafo,
                             fragmentos=[{"cita": f.cita, "motivo": f.motivo} for f in fragmentos])
        cambios = {"contexto": como_texto(fragmentos)}
        if subtarea["tipo"] != "calculo":
            cambios["subtareas"] = self._con_estado(estado, subtarea["id"], status="para_redactor")
        return cambios

    def _tras_investigar(self, estado: Estado) -> str:
        if estado.get("presupuesto_agotado"):
            return "redactar"
        return "programar" if self._subtarea(estado)["tipo"] == "calculo" else "elegir_subtarea"

    @nodo
    def programar(self, estado: Estado) -> dict:
        subtarea = self._subtarea(estado)
        carpeta = self._preparar_carpeta(subtarea)
        entradas = {d: claves_de(self.resultados[d]["json"]) for d in subtarea.get("depende_de", [])
                    if d in self.resultados}
        datos = [str(p.relative_to(raiz_de_adjuntos(self.doc))) for p in self.doc.adjuntos
                 if p.suffix.lower() not in {".md", ".txt", ".pdf"}]
        veredicto = estado.get("veredicto") or {}
        anterior = None
        if veredicto and not veredicto.get("aprobado") and estado.get("codigo"):
            anterior = {"codigo": estado["codigo"], "correccion": veredicto.get("correccion", "")}
        codigo = self.programador.programar(subtarea, estado["contexto"], datos, entradas, anterior)
        intentos = estado["subtareas"][subtarea["id"]]["intentos"] + 1
        (carpeta / "script.py").write_text(codigo, encoding="utf-8")
        historial = self.salida / "intentos"
        historial.mkdir(exist_ok=True)
        (historial / f"{subtarea['id']}-intento{intentos}.py").write_text(codigo, encoding="utf-8")
        return {"codigo": codigo, "subtareas": self._con_estado(estado, subtarea["id"], intentos=intentos)}

    @nodo
    def ejecutar(self, estado: Estado) -> dict:
        subtarea = self._subtarea(estado)
        sid, codigo = subtarea["id"], estado["codigo"]
        intento = estado["subtareas"][sid]["intentos"]

        if sandbox.huella(codigo) in estado.get("rechazados", {}).get(sid, []):     # FRENO 2
            self.traza.registrar("freno_repeticion", subtarea=sid, intento=intento,
                                 huella=sandbox.huella(codigo),
                                 detalle="script idéntico a uno ya rechazado: no se ejecuta")
            return {"ejecucion": {"bloqueado": "repeticion", "problemas": [
                "el script es idéntico a uno que ya fue rechazado; cambia el enfoque"]}}

        revision = sandbox.revisar_codigo(codigo)                                    # REGLA 3
        self.traza.registrar("sandbox_revision", subtarea=sid, intento=intento,
                             permitido=revision.permitido, problemas=revision.problemas,
                             usa_red=revision.usa_red)
        if not revision.permitido:
            return {"ejecucion": {"bloqueado": "sandbox", "problemas": revision.problemas}}
        if revision.usa_red:                                                         # FRENO 4
            aprobado = bool(self.confirmar_red(sid, revision.usa_red, codigo))
            self.traza.registrar("freno_red", subtarea=sid, intento=intento, motivos=revision.usa_red,
                                 decision="aprobado" if aprobado else "denegado",
                                 quien=getattr(self.confirmar_red, "__name__", "persona"))
            if not aprobado:
                return {"ejecucion": {"bloqueado": "red", "problemas": [
                    f"el script quiere usar la red ({'; '.join(revision.usa_red)}) y una persona "
                    "no lo aprobó. Usa solo datos locales (scikit-learn incluye sus datasets "
                    "de ejemplo, como load_breast_cancer) o los del enunciado."]}}

        carpeta = self.salida / "trabajo" / sid
        ej = sandbox.ejecutar(carpeta / "script.py", self.config.timeout_s, self.salida / ".mpl")
        self.traza.registrar("ejecucion", subtarea=sid, intento=intento,                # REGLA 6
                             codigo_salida=ej.codigo_salida, duracion_s=ej.duracion_s,
                             archivos_creados=ej.archivos_creados, timeout=ej.timeout,
                             stderr=ej.stderr[-600:])
        if ej.timeout:
            self.traza.registrar("freno_timeout", subtarea=sid, intento=intento,
                                 limite_s=self.config.timeout_s)
        return {"ejecucion": vars(ej)}

    @nodo
    def revisar(self, estado: Estado) -> dict:
        subtarea = self._subtarea(estado)
        sid = subtarea["id"]
        ej = estado["ejecucion"]
        carpeta = self.salida / "trabajo" / sid
        if ej.get("bloqueado"):
            veredicto = Veredicto(False, ej["bloqueado"], ej["problemas"],
                                  "No se ejecutó:\n- " + "\n- ".join(ej["problemas"]))
        elif not self.config.usar_revisor:          # versión recortada: ninguna revisión
            veredicto = Veredicto(ej["codigo_salida"] == 0, "sin_revisor")
        else:
            from .sandbox import Ejecucion
            veredicto = self.revisor.revisar(subtarea, estado["contexto"], estado["codigo"],
                                             Ejecucion(**ej), carpeta)
        intentos = estado["subtareas"][sid]["intentos"]
        self.traza.registrar("revision", subtarea=sid, intento=intentos, aprobado=veredicto.aprobado,
                             por=veredicto.por, problemas=veredicto.problemas,
                             correccion=veredicto.correccion[:1500])

        rechazados = {k: list(v) for k, v in estado.get("rechazados", {}).items()}
        if veredicto.aprobado:
            self._guardar_resultados(sid, carpeta, estado["codigo"], ej)
            status = "completado"
        else:
            rechazados.setdefault(sid, []).append(sandbox.huella(estado["codigo"]))
            status = "en_curso"
            if intentos >= self.config.max_intentos:                                 # FRENO 2
                status = "fallida"
                self.traza.registrar("freno_intentos", subtarea=sid, intentos=intentos,
                                     detalle="máximo de intentos alcanzado: subtarea fallida, se sigue")
        return {"veredicto": vars(veredicto), "rechazados": rechazados,
                "subtareas": self._con_estado(estado, sid, status=status)}

    def _tras_revisar(self, estado: Estado) -> str:
        if estado.get("presupuesto_agotado"):
            return "redactar"
        status = estado["subtareas"][estado["actual"]]["status"]
        return "programar" if status == "en_curso" else "elegir_subtarea"

    @nodo
    def redactar(self, estado: Estado) -> dict:
        plan = estado["plan"]
        faltantes = self._faltantes(estado)
        subtareas = [{**s, "status": estado["subtareas"][s["id"]]["status"]} for s in plan["subtareas"]]
        try:
            borrador = self.redactor.redactar(plan["entrega"], subtareas, self.resultados,
                                              self._contexto_redactor(plan), faltantes,
                                              estado.get("problemas_entrega"), estado.get("borrador"))
        except PresupuestoAgotado as err:          # FRENO 1: ni la reserva alcanza
            self.traza.registrar("freno_presupuesto", nodo="redactar", detalle=str(err),
                                 accion="entregable mínimo escrito por código")
            borrador = entregable_minimo(plan["entrega"], plan["subtareas"], self.resultados, faltantes)
            return {"borrador": borrador, "faltantes": faltantes,
                    "rondas_redactor": self.config.max_rondas_redactor, "presupuesto_agotado": True}
        return {"borrador": borrador, "faltantes": faltantes,
                "rondas_redactor": estado.get("rondas_redactor", 0) + 1}

    @nodo
    def verificar(self, estado: Estado) -> dict:
        plan = estado["plan"]
        figuras = {sid: r["figuras"] for sid, r in self.resultados.items() if r["figuras"]}
        codigos = {sid: r["codigo"] for sid, r in self.resultados.items()}
        textos = [t for r in self.resultados.values() for t in (r["json_texto"], r["stdout"])]
        e = validar_entregable(estado["borrador"], plan["entrega"], self.salida, figuras, codigos,
                               textos, self.doc.texto, self.config.timeout_s)
        self.traza.registrar("verificacion_entrega", ronda=estado["rondas_redactor"],
                             archivo=str(e.ruta.name) if e.ruta else None, problemas=e.problemas,
                             palabras=e.palabras, paginas=e.paginas, cifras_sin_origen=e.cifras_huerfanas)
        return {"problemas_entrega": e.problemas, "entregable": str(e.ruta) if e.ruta else ""}

    def _tras_verificar(self, estado: Estado) -> str:
        if estado.get("problemas_entrega") and estado["rondas_redactor"] < self.config.max_rondas_redactor \
                and not estado.get("presupuesto_agotado"):
            return "redactar"
        return END

    # ───────────────────────────────────────────────────────────── ayudas
    def _subtarea(self, estado: Estado) -> dict:
        return next(s for s in estado["plan"]["subtareas"] if s["id"] == estado["actual"])

    @staticmethod
    def _con_estado(estado: Estado, sid: str, **cambios) -> dict:
        subtareas = {k: dict(v) for k, v in estado["subtareas"].items()}
        subtareas[sid].update(cambios)
        return subtareas

    def _preparar_carpeta(self, subtarea: dict) -> Path:
        """Una carpeta limpia por intento, con los datos del paquete y los resultados de las
        subtareas de las que depende (entradas/<id>.json)."""
        carpeta = self.salida / "trabajo" / subtarea["id"]
        if carpeta.exists():
            shutil.rmtree(carpeta)            # lo borra el orquestador, nunca el script
        carpeta.mkdir(parents=True)
        raiz = raiz_de_adjuntos(self.doc)
        for p in self.doc.adjuntos:
            if p.suffix.lower() not in {".md", ".txt", ".pdf"}:
                destino = carpeta / p.relative_to(raiz)
                destino.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, destino)
        for d in subtarea.get("depende_de", []):
            if d in self.resultados:
                (carpeta / "entradas").mkdir(exist_ok=True)
                (carpeta / "entradas" / f"{d}.json").write_text(self.resultados[d]["json_texto"],
                                                                encoding="utf-8")
        return carpeta

    def _guardar_resultados(self, sid: str, carpeta: Path, codigo: str, ej: dict) -> None:
        ruta = carpeta / ARCHIVO_RESULTADOS
        texto = ruta.read_text(encoding="utf-8") if ruta.exists() else "{}"
        try:
            datos = json.loads(texto)
        except json.JSONDecodeError:
            datos = {}
        figuras = sorted(str(p.relative_to(self.salida)) for p in carpeta.rglob("*.png")
                         if ".mpl" not in p.parts)
        self.resultados[sid] = {"json": datos, "json_texto": texto, "stdout": ej.get("stdout", ""),
                                "figuras": figuras, "codigo": codigo}

    def _faltantes(self, estado: Estado) -> list[str]:
        faltan = []
        for s in estado["plan"]["subtareas"]:
            st = estado["subtareas"][s["id"]]
            if st["status"] == "fallida":
                faltan.append(f"{s['id']} ({s.get('titulo', '')}): fallida tras {st['intentos']} intentos")
            elif st["status"] in {"pendiente", "en_curso"} and s["tipo"] == "calculo":
                motivo = "presupuesto agotado" if estado.get("presupuesto_agotado") else "no se completó"
                faltan.append(f"{s['id']} ({s.get('titulo', '')}): {motivo}")
        return faltan

    def _contexto_redactor(self, plan: dict) -> str:
        """El enunciado completo (es corto) y los papers del corpus de las partes de redacción."""
        partes = [f"{self.grafo.cita(s.id)}\n{s.literal}" for s in self.grafo.secciones]
        for s in plan["subtareas"]:
            if s["tipo"] == "redaccion":
                fr = contexto_de(s, plan, self.grafo, self.config.usar_grafo,
                                 self.config.k_fragmentos, self.config.k_corpus)
                partes += [f"{f.cita}\n{f.texto[:1500]}" for f in fr if not f.nodo.startswith("S")]
        return "\n\n---\n\n".join(dict.fromkeys(partes))[:30000]


def dibujar_orquestador(destino_png: Path, destino_mmd: Path) -> None:
    """El diagrama del orquestador, a partir del grafo compilado (sin red)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class _Falso:
        modelo = "diagrama"
    tmp = CACHE / "diagrama"
    tmp.mkdir(parents=True, exist_ok=True)
    o = Orquestador("", tmp, ClienteLLM(_Falso(), Traza(tmp), Presupuesto(1, 0)), Traza(tmp), Config())
    grafo = o.app.get_graph()
    destino_mmd.write_text(grafo.draw_mermaid(), encoding="utf-8")

    pos = {"__start__": (0, 10), "leer": (0, 9), "indexar": (0, 8), "planificar": (0, 7),
           "validar_plan": (0, 6), "elegir_subtarea": (0, 5), "investigar": (0, 4),
           "programar": (2.2, 4), "ejecutar": (2.2, 3), "revisar": (2.2, 2),
           "redactar": (0, 2), "verificar": (0, 1), "__end__": (0, 0)}
    fig, ax = plt.subplots(figsize=(8.5, 10))
    for e in grafo.edges:
        (x0, y0), (x1, y1) = pos[e.source], pos[e.target]
        rad = 0.35 if y1 >= y0 or (e.source, e.target) in {("revisar", "elegir_subtarea"),
                                                           ("verificar", "__end__")} else 0.0
        if e.source in {"leer", "indexar", "planificar", "validar_plan"} and e.target == "__end__":
            rad = 0.25
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#888" if e.conditional else "#222",
                                    linestyle="--" if e.conditional else "-",
                                    connectionstyle=f"arc3,rad={rad}", shrinkA=14, shrinkB=14))
    for n, (x, y) in pos.items():
        color = "#fdebd0" if n in {"programar", "ejecutar", "revisar"} else "#d6eaf8"
        ax.text(x, y, n.strip("_"), ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", fc=color, ec="#444"))
    ax.text(2.2, 1.2, "bucle de código:\nhasta aprobar o\nllegar a max_intentos", ha="center", fontsize=8)
    ax.text(-0.9, 1.5, "REGLA 5\n(redactar ⇄ verificar)", ha="center", fontsize=8)
    ax.text(-0.9, 6.5, "REGLA 1\n(planificar ⇄ validar)", ha="center", fontsize=8)
    ax.set_xlim(-1.6, 3.2)
    ax.set_ylim(-0.6, 10.6)
    ax.axis("off")
    ax.set_title("Orquestador (LangGraph): líneas punteadas = aristas condicionales")
    fig.tight_layout()
    fig.savefig(destino_png, dpi=130)
    plt.close(fig)
