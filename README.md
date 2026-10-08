# Taller 03 v2 — Solver multiagente con GraphRAG

MMIA 6013 IA Generativa y Agentes · USFQ · Estefanía Gonzaga

Un sistema de agentes que recibe el enunciado de una tarea en PDF y la resuelve: lo lee, arma
un grafo del enunciado y de los papers del curso, lo divide en subtareas, escribe y ejecuta
el código en un sandbox, revisa los resultados, corrige lo que falla y redacta el entregable
en el formato pedido (Markdown, PDF o notebook ejecutado).

Base de conocimiento: los 16 papers del corpus del **Taller 02** y las presentaciones de la
Semana 2 (`corpus/`), con la capa vectorial del RAG de ese taller (bge-m3 en la H200 + Qdrant).
Tareas: tres de práctica escritas a partir de esos papers (`enunciados/`) y dos tareas reales,
actividades de clase de la Semana 2 convertidas a PDF (`enunciados/reales/`).

## Cómo correrlo

```bash
uv venv --python 3.14 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
# VPN GlobalProtect conectada (H200: LLM en :12555, bge-m3 y gemma3 en :11434)

python enunciados/generar_pdfs.py        # enunciados/*.md → *.pdf (ya vienen hechos)
python enunciados/reales/convertir_actividades.py   # actividades de clase → PDF, sin soluciones
python verdades.py                        # las respuestas correctas, calculadas
# El índice del corpus (entidades + embeddings) viene en .cache/: no hace falta recalcularlo.
# python indexar_corpus.py               # solo si cambia el corpus (~24 min en la H200)

# Parte 0
python parte0/a_llm_sin_ejecutar.py       # H200
python parte0/b_rag_plano.py              # sin red
python parte0/c_exit_cero.py              # sin red

# Parte 1: una tarea
python -c "from solver import Solver; print(Solver().solve('enunciados/tarea-a-ng-jordan-digitos.pdf', 'corridas/prueba')['status'])"

# Parte 2.b: solver completo y versión recortada, mismo golden set y mismo modelo
python evaluar_solver.py --solver solver:Solver --ruta . --corridas corridas/completo \
    --salida resultados/resultados_solver_completo.csv
python evaluar_solver.py --solver solver:SolverSinGrafo --ruta . --corridas corridas/sin_grafo \
    --salida resultados/resultados_solver_sin_grafo.csv
# Si se corta (VPN), el mismo comando con --reanudar no repite las tareas ya terminadas.
python tablas.py                          # resultados/tablas.md, solo desde los CSV
python peores_casos.py                    # Parte 2.c: evidencia de las fallas en la traza

# Parte 3: los cuatro frenos, con un modelo de guion (sin gastar tokens)
python frenos.py

# Parte 4.E: juez de otra familia (gemma3) contra las comprobaciones
python juez_gemma.py

python diagrama_orquestador.py            # informe/diagrama_orquestador.png
python informe/construir_informe.py       # informe/informe.pdf (incluye las tablas generadas)
```

## Arquitectura

| Agente | Módulo | LLM | Qué revisa el código |
|---|---|---|---|
| Lector | `solver/lector.py` | no | PDF escaneado, tablas (`find_tables`), texto partido o pegado |
| Indexador (GraphRAG) | `solver/grafo.py`, `solver/rag.py` | entidades | secciones y aristas `depende_de` por regla |
| Planificador | `solver/planificador.py` | sí | `validar_plan`: ids, dependencias, ciclos, cobertura |
| Investigador | `solver/investigador.py` | no | sección literal + dependencias, cada fragmento con cita |
| Programador | `solver/programador.py` | sí | el filtro del sandbox |
| Ejecutor | `solver/sandbox.py` | no | entorno vacío, carpeta propia, tiempo máximo con `killpg` |
| Revisor | `solver/revisor.py` | sí, después del código | salida, resultados.json, NaN, figuras, fugas, plausibilidad |
| Redactor | `solver/redactor.py`, `solver/formatos.py` | sí | secciones, extensión, páginas, notebook ejecutado, origen de cada cifra |
| Orquestador | `solver/orquestador.py` | — | grafo de estados de LangGraph |

Las seis reglas están marcadas en el código con `REGLA n`, y los frenos con `FRENO n`:

1. **REGLA 1**, plan validado con código: `planificador.validar_plan`.
2. **REGLA 2**, GraphRAG con base fija: `grafo.aristas_depende_de` (regex) + entidades del LLM.
3. **REGLA 3**, sandbox: `sandbox.revisar_codigo`, `sandbox.entorno_limpio`, `sandbox.ejecutar`.
4. **REGLA 4**, `resultados.json` fijo y revisión con código primero: `revisor.revisiones_con_codigo`.
5. **REGLA 5**, origen de cada cifra: `procedencia.cifras_sin_origen`, en `redactor.validar_entregable`.
6. **REGLA 6**, traza siempre: `traza.Traza`, una línea por evento, escrita al momento.

Contrato: `Solver().solve(pdf, salida)` y `Solver().run(ruta)`, en `solver/solver.py`.

## Estructura

```
├── solver/                 el paquete (un módulo por agente)
├── corpus/                 16 papers del Taller 02 + 3 presentaciones de la Semana 2
├── enunciados/             tareas A (md), B (notebook), C (PDF de 2 páginas), en .md y .pdf;
│                           reales/: R1 y R2, actividades de clase de la Semana 2 (notebook)
├── golden_tareas.json      43 comprobaciones en 5 tareas; cada verdad es código (verdad_py → verdades.py)
├── verdades.py             las respuestas correctas, calculadas
├── evaluar_solver.py       evaluador (propio; no importa nada del solver)
├── parte0/                 los tres errores que no dan error, y sus salidas
├── frenos.py               Parte 3, con modelo de guion
├── juez_gemma.py           Parte 4.E
├── tablas.py, peores_casos.py
├── corridas/               completo/, sin_grafo/ (Parte 2.b oficial); fix/, fix2/ (re-mediciones de
│                           las correcciones); v1/ (primera medición, 3 tareas); control_sin_ejecutar/
│                           (Parte 4.E); frenos/ (Parte 3). Cada tarea: traza.jsonl, traza_llm/,
│                           plan.json, grafo.png, trabajo/ y el entregable
├── resultados/             CSV del evaluador y tablas derivadas
└── informe/                informe.md → informe.pdf (python informe/construir_informe.py) y diagramas
```

## Qué está en el repositorio y qué no

- **Sí:** el código, los enunciados (`.md` y `.pdf`), el corpus, el golden set, las corridas
  (`traza.jsonl`, `traza_llm/`, `plan.json`, el grafo, los scripts y el entregable), los CSV,
  el informe y el índice del corpus ya calculado (`.cache/corpus-entidades-*.json` y
  `.cache/embeddings-*.json`). Las entidades las extrae un LLM: reconstruirlas daría un grafo
  distinto, así que se versionan para que las corridas se puedan repetir con el mismo grafo.
- **No:** `.env`, `.venv/`, `__pycache__/` y las cachés de matplotlib y Jupyter (ver `.gitignore`).

## Credenciales

Ninguna. La H200 no pide clave (`Bearer local` no es una credencial) y el sandbox ejecuta los
scripts con un entorno construido desde cero: aunque existiera una clave en el `.env` o en el
entorno, no llegaría al código generado. `.env` está en `.gitignore`.

## Tareas reales (Parte 2.a)

R1 y R2 son las actividades de clase S2·MAR (ANN) y S2·MIÉ (RAG mínimo) de MMIA 6013.
`enunciados/reales/convertir_actividades.py` las convierte de notebook a PDF conservando el
texto y el código que da el profesor y retirando las soluciones de los ejercicios. Para añadir
otra tarea real: su PDF en `enunciados/reales/` y su entrada en `golden_tareas.json`, con al
menos cinco comprobaciones (de forma, dos de cifra con `verdad_py` y una de procedencia).
