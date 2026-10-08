# Taller 03 v2 — Un solver multiagente con GraphRAG

**MMIA 6013 IA Generativa y Agentes · USFQ** · Integrante: Estefanía Gonzaga (trabajo individual)

**Modelo (Partes 0–3):** `zai-org/GLM-5.3-Flash`, leído de `http://172.28.230.10:12555/v1/models` (vLLM de la H200, réplica en el puerto 12559). **Embeddings:** `bge-m3:latest` en el Ollama de la H200 (`:11434`). **Juez (Parte 4.E):** `gemma3:27b` en el mismo Ollama. **Fechas de las corridas:** 2026-10-06 y 2026-10-07. Costo: 0 USD.

**Integridad académica.** Las tareas de práctica (A, B y C) se escribieron para este taller a partir de los papers del corpus del Taller 02. Las tareas reales (R1 y R2) son actividades de clase de la Semana 2 de esta materia; se convirtieron de notebook a PDF retirando mis soluciones. Si el solver se usa en una tarea de otra materia, se declara y se adjunta `corridas/<tarea>/traza.jsonl`.

## Material

- **Base de conocimiento** (`corpus/`): los 16 papers del corpus del Taller 02 y las tres presentaciones de la Semana 2 (embeddings, bases vectoriales, fragmentación y recuperación). Son 1079 fragmentos de 300 palabras con 50 de solapamiento, cada uno con su cita (archivo y página). Entidades y embeddings se calculan una vez (`indexar_corpus.py`) y quedan en `.cache/`.
- **Tareas** (`enunciados/`):

| Tarea | Origen | Pide | Entregable | Qué pone a prueba |
|---|---|---|---|---|
| A | Ng y Jordan (2001) | NB contra regresión logística en `load_digits`, curva de aprendizaje | `reporte.md`, máx. 1000 palabras | una parte que depende de otra; una conclusión del paper que los datos no confirman |
| B | Vaswani et al. (2017) | atención escalada, efecto de √d_k, codificación posicional | notebook ejecutado | cifras exactas con un generador consumido en orden |
| C | Blei et al. (2003) | LDA sobre 12 oraciones, perplejidad, pureza | `reporte.pdf`, máx. 2 páginas | datos dentro de tablas, límite de páginas, no descargar un corpus |
| R1 | **Real:** actividad S2·MAR | kNN exacto, mini-IVF (recall contra `nprobe`), Qdrant | notebook ejecutado | código dado que usa la red (Qdrant en `localhost:6333`) |
| R2 | **Real:** actividad S2·MIÉ | RAG mínimo: chunking, recuperación, abstención | notebook ejecutado | generación por APIs de red; modo inspección del propio notebook |

## Parte 0 — Tres errores que no dan error

Los tres scripts son propios y se aplican a la Tarea A. Usan las piezas reales del solver: el lector y las reglas del grafo (0.b), el sandbox y el revisor (0.c).

### 0.a — El solver que no ejecuta

<!-- incluir: parte0/salidas/0a_llm_sin_ejecutar.txt -->

Una corrida anterior del mismo script (`parte0/salidas/0a_reporte_sin_ejecutar_corrida2.md`) concluyó bien que no hay cruce, pero con cifras igual de inventadas (NB con m = 20: 0.311 contra 0.3667 medido) y con una nota que admite que «las cifras son los valores esperados».

1. **¿Por qué no basta con que las cifras estén cerca?** Una cifra cercana no es una medida: no se puede reproducir, no tiene error conocido y no dice qué código la produjo. El control de la Parte 4 lo muestra: este reporte aprueba la comprobación A07 (regresión logística 0.969, dentro de la tolerancia de 0.9711) sin haber ejecutado nada.
2. **¿Qué cifra inventada cambiaría una conclusión?** La exactitud con m = 20. El reporte da NB 0.489 contra LR 0.589. Si el modelo hubiera escrito NB > LR, como dice la teoría del paper, el reporte «confirmaría» el cruce de Ng y Jordan; lo medido es 0.3667 contra 0.7044, sin cruce.
3. **¿Qué debe comprobar un verificador sin conocer las respuestas?** La procedencia: que cada cifra aparezca en algo que escribió una ejecución (`resultados.json`, stdout, salidas del notebook). Aquí 0 de 30 aparecen en algo así. Es la REGLA 5 del solver.

### 0.b — El RAG que pierde la referencia

<!-- incluir: parte0/salidas/0b_rag_plano.txt -->

1. **¿Por qué es un fallo de la búsqueda y no del modelo?** El modelo nunca ve la Parte 1: la similitud elige la Parte 3 y la Parte 4 porque comparten las palabras de la pregunta, y los datos están en un texto que no se parece a ella. Con el contexto correcto, cualquier modelo copia la partición.
2. **¿Qué haría un LLM con el primer contexto?** Inventaría la partición (lo típico: 80/20 con `random_state=42`) o la preguntaría. Si la inventa, la curva usa otra división y sus cifras no coinciden con las de la Parte 2.
3. **¿Por qué la conexión se saca con una regla?** Porque la referencia «Parte 1» está escrita literalmente y una expresión regular la encuentra siempre, con costo cero y sin poder inventar una referencia que no existe. Pedírsela al LLM añade una fuente de error justo en la base del grafo.

### 0.c — Terminar sin error no es acertar

<!-- incluir: parte0/salidas/0c_exit_cero.txt -->

1. **¿Qué fuga no detectaría la revisión del código?** Un escalador ajustado con todos los datos antes de dividir (`StandardScaler().fit_transform(X)` y luego `train_test_split`): ningún `.predict` usa la variable de un `.fit` supervisado, y la exactitud resultante es plausible, solo un poco optimista.
2. **¿Por qué se puede dejar la decisión final a un LLM revisor si las revisiones con código van antes?** Porque el código descarta lo verificable y barato (código de salida, NaN, fuga evidente, cifra implausible) sin dejarse convencer; al LLM solo le llega lo que pasó esas barreras, y su papel es juzgar lo que el código no puede: si el script hace lo que pide el enunciado. En la primera medición de la Tarea C, el LLM rechazó una partición 75/25 que el enunciado no pedía y que el código no podía ver.

## Parte 1 — El solver base

### Arquitectura

![Diagrama del orquestador](informe/diagrama_orquestador.png)

El orquestador es un `StateGraph` de LangGraph 1.2.12 (`solver/orquestador.py`). El diagrama sale del grafo compilado (`python diagrama_orquestador.py`); las líneas punteadas son aristas condicionales, que deciden con código a qué nodo se pasa.

| Rol | Módulo | Qué revisa su código |
|---|---|---|
| Lector | `lector.py` | PDF escaneado (< 200 caracteres útiles), tablas con `find_tables`, texto partido o pegado; el tamaño del cuerpo sin contar el código |
| Indexador (GraphRAG) | `grafo.py`, `rag.py` | secciones por tipografía; aristas `depende_de` por regex |
| Planificador | `planificador.py` | `validar_plan` |
| Investigador | `investigador.py` | sección literal + dependencias + preámbulo, cada fragmento con cita |
| Programador | `programador.py` | el filtro del sandbox |
| Ejecutor | `sandbox.py` (sin LLM) | entorno vacío, carpeta propia, tiempo máximo con `killpg` |
| Revisor | `revisor.py` | código de salida, `resultados.json`, NaN, figuras, fuga y plausibilidad |
| Redactor | `redactor.py`, `formatos.py` | secciones en orden, palabras, páginas, notebook ejecutado, origen de cada cifra |

**Las seis reglas, en el código** (buscar `REGLA n`):

1. **Plan validado con código** — `planificador.validar_plan`: ids únicos, dependencias que existen, sin ciclos (`networkx.find_cycle`), cada sección cubierta, formato de entrega válido. Si falla, `validar_plan → planificar` con la lista de problemas (máx. 3 rondas).
2. **GraphRAG con base fija** — `grafo.aristas_depende_de`: «Parte N», «Tarea N», «P0», «la parte anterior» → arista `depende_de` por regex. Encima, entidades del LLM (dataset, método, métrica, concepto) del enunciado y del corpus, unidas por nombre normalizado. El investigador entrega la sección literal, sus dependencias (cierre transitivo), el preámbulo y los fragmentos del corpus que comparten entidades, cada uno con su cita.
3. **Sandbox** — `sandbox.revisar_codigo` (AST: sin red, sin procesos, sin borrar, sin `eval`, sin rutas fuera de la carpeta), `sandbox.entorno_limpio` (el entorno se construye desde cero: ninguna variable se hereda) y `sandbox.ejecutar` (carpeta propia, `start_new_session` y `os.killpg` al vencer el tiempo). El notebook se ejecuta con el mismo entorno.
4. **Resultados en archivo fijo y revisión con código primero** — cada script escribe `resultados.json`; `revisor.revisiones_con_codigo` corre antes que el LLM y, si encuentra algo, el LLM no se consulta.
5. **Origen de cada cifra** — `procedencia.cifras_sin_origen`, dentro de `redactor.validar_entregable`: cada cifra del entregable debe aparecer en una ejecución o en el enunciado (salvo los enteros de 0 a 10, numeración). Si no, `verificar → redactar` con la lista.
6. **Traza siempre** — `traza.Traza`: una línea JSON por evento, escrita al momento (también si el proceso muere). Por llamada: agente, modelo, tokens de entrada y salida, latencia, error; por ejecución: código de salida, duración, archivos creados. Los mensajes completos van a `traza_llm/`.

**Contrato.** `Solver().solve(pdf, salida)` devuelve `status`, `entregables`, `subtareas`, `usage` (con el detalle por agente), `model` y `trace`; `Solver().run(ruta)` devuelve `answer`, `trace`, `status`, `model` y `usage` (Taller 4).

### El grafo de la Tarea A

![Grafo de la Tarea A](corridas/completo/tarea-A/grafo.png)

Las cuatro aristas `depende_de` (en rojo) salen de la regla: S3 → S1 y S3 → S2 («Con la partición de la Parte 1, entrena los dos modelos de la Parte 2»), S5 → S2 y S5 → S3 («la tabla de la Parte 2 y la figura de la Parte 3»). Las entidades unen el enunciado con las páginas del paper de Ng y Jordan.

### El plan de la Tarea A (`corridas/completo/tarea-A/plan.json`)

| Subtarea | Tipo | Secciones | Depende de | Figura |
|---|---|---|---|---|
| T1 Datos y partición única | cálculo | S1 | — | no |
| T2 GaussianNB contra regresión logística | cálculo | S2 | T1 | no |
| T3 Curva de aprendizaje con 20, 50, 100, 200, 400 y todos | cálculo | S3 | T1, T2 | sí |
| T4 Redacción del reporte y discusión | redacción | S4, S5 | T2, T3 | no |

Entrega: `md`, `reporte.md`, secciones Introducción, Metodología, Resultados, Discusión y Conclusiones, 1000 palabras. El plan pasó la validación en la primera ronda.

### Una corrida en la que se rechazaron scripts: tarea real R1 (`corridas/completo/tarea-R1/traza.jsonl`)

| Evento | Subtarea · intento | Qué pasó |
|---|---|---|
| 21 | T2 · 1 | **Rechazado por el sandbox**: «error de sintaxis en la línea 1» (el modelo devolvió texto sin bloque de código) |
| 24–26 | T2 · 2 | Ejecución con código 0; **aprobado por el LLM** |
| 30–31 | T3 · 1 | El script copia el código dado: `QdrantClient(url="http://localhost:6333")`. **FRENO 4**: nadie aprueba la red → **rechazado** |
| 34–36 | T3 · 2 | Qdrant en modo embebido (`:memory:`); **aprobado por el LLM** |

## Parte 2 — Evaluación

### 2.a — El golden set (`golden_tareas.json`)

43 comprobaciones: 9 en cada tarea de práctica y 8 en cada tarea real. Cada tarea tiene comprobaciones de forma (archivo, secciones en orden, palabras o páginas, notebook ejecutado), al menos dos de tipo cifra con su `verdad_py` y una de procedencia. **Ninguna verdad se escribe a mano**: `verdad_py` llama a `verdades.py`, que reproduce cada enunciado con sus parámetros.

- **Notebook:** B, R1, R2. **PDF:** C. **Trampas:** A (partición única, escalador solo con el entrenamiento y un cruce del paper que los datos no muestran); B (un solo generador consumido en orden fijo); C (no descargar un corpus); R1 (Qdrant por URL en el código dado); R2 (generación por APIs de red). Las tres últimas usan el tipo nuevo `codigo_sin` sobre el código que se ejecutó.
- **Tareas reales.** `enunciados/reales/convertir_actividades.py` convierte los notebooks de clase en PDF: conserva el Markdown y el código que da el profesor y, de las celdas de ejercicio y de reflexión, solo las líneas de comentario que dicen qué hacer. La verdad de R1 se validó contra las salidas guardadas del notebook (celdas de 1562, puntajes de Qdrant −0.102 y −0.112).
- **Lo que enseñó medir el golden set.**
  - *Una verdad no reproducible.* El recall@10 de R1 da 0.916 o 0.926 con `nprobe` = 1 en corridas idénticas: KMeans multihilo cambia el orden de las sumas. Solo lleva tolerancia 0.015; las comprobaciones exactas van sobre cifras deterministas.
  - *Un enunciado ambiguo* (A, primera versión): «NB y regresión logística con los atributos estandarizados» se leyó como estandarizar ambos; NB dio 0.8089 en lugar de 0.8556. Se corrigió el enunciado.
  - *Una verdad sobre texto que el solver no ve* (R204): la pregunta corta del ejercicio 2.1 solo estaba en mi solución, que la conversión retira. Se cambió por la pregunta del código dado.
  - *Una comprobación de trampa frágil* (`codigo_sin`): contaba intentos rechazados que nunca corrieron, comentarios y docstrings que solo mencionan una URL. Se afinó tres veces; lo robusto sería reutilizar el AST, como el sandbox.
- **Evaluador:** `evaluar_solver.py` es propio y no importa nada del solver. Los artefactos del solver (grafo, traza, plan, intentos) no cuentan como entrega ni como evidencia; `--reanudar` permite seguir una evaluación cortada (se usó tras una caída de la VPN).

### 2.b — Solver completo contra la versión sin grafo

Mismo golden set y mismo modelo. En la versión **sin grafo**, el investigador solo recibe los k = 4 fragmentos más parecidos a la subtarea (secciones y corpus juntos, con bge-m3), como en la Parte 0.b, y el indexador no extrae entidades. Las tablas salen de `resultados/*.csv` con `python tablas.py` (Checks = comprobaciones aprobadas; Proced. = fracción de cifras respaldadas por una ejecución; Fallidas = subtareas fallidas; s = duración en segundos).

<!-- incluir: resultados/tablas.md -->

**Lectura.** El grafo ganó 5 comprobaciones (41 contra 36), 3 subtareas fallidas menos y 6 intentos de código menos, con un 3 % menos de tokens (942 444 contra 974 763); tardó más (4932 s contra 4345 s), sobre todo por R1 y R2. Las diferencias están donde el contexto importa: en C, sin el preámbulo con la tabla de los 12 documentos, T2 no tuvo datos; en R1, la puntuación de Qdrant (R105) no llegó al entregable porque T3 falló; en R2, la versión sin grafo no reportó el número de fragmentos ni los puntajes (R203–R205). Ambas versiones fallaron la procedencia de R1 por un defecto del ensamblado del notebook (2.c).

### 2.c — Los tres peores casos

Evidencia en `resultados/peores_casos_completo.md` y `peores_casos_sin_grafo.md` (sacada de la traza con `peores_casos.py`). Cada corrección se midió después con el mismo golden set:

<!-- incluir: resultados/remediciones.md -->

1. **R206 — la abstención no llega al entregable (completo, R2). Agentes: programador y freno de presupuesto.** *Recibió* para T3 (generación) el código dado, que llama por red a tres APIs. *Produjo* un primer intento sin `resultados.json` (rechazado por el código) y después razonó hasta agotar `max_tokens` seis veces. *El código decidió*: el reintento de 65 536 tokens se autorizó con la estimación típica (unos 5000), así que una sola llamada pasó el límite (443 268 usados contra 390 000) y se comió la reserva del redactor; el redactor tampoco pudo llamar al LLM y el solver publicó el entregable mínimo escrito por código, sin la frase de abstención. *Corrección:* un reintento se autoriza con el nuevo `max_tokens`. Medido: el freno salta antes del reintento («usados 383 265 + estimados 71 897 > límite 390 000») y el redactor escribe el entregable.
2. **R108 — procedencia 0.81 (completo y sin grafo, R1). Agente: redactor, en el ensamblado del notebook.** R1 mide latencias, que cambian en cada ejecución. El redactor copió las del script aprobado, pero el notebook volvía a correr cada script en su carpeta y **sobrescribía** el `resultados.json` de donde salían. La REGLA 5 pasó porque comparaba contra los valores en memoria; el evaluador, contra los archivos. *Corrección:* el notebook corre sobre una copia (`trabajo_nb/`). Medido: procedencia 1.00 en las dos versiones.
3. **R105 — el puntaje de Qdrant no llega al entregable (sin grafo, R1). Agentes: revisor y sandbox, en conflicto.** T3 copió el código dado (`localhost:6333`) y el freno de red lo bloqueó (bien). El programador quitó la conexión y **el revisor LLM lo rechazó** porque «el enunciado exige intentar conexión a localhost:6333»: nadie le decía que la red está prohibida. Tercer intento con la URL otra vez, bloqueado; T3 fallida. En la re-medición apareció un segundo falso positivo: la revisión de plausibilidad rechazaba un recall@10 = 1.0, legítimo en un índice ANN. *Corrección:* el prompt del revisor dice que el sandbox prohíbe la red y que la alternativa local del enunciado es la correcta; la plausibilidad solo mira clasificadores. Medido: R1 8/8 en las dos versiones.

De la primera medición (v1, tres tareas) queda otra corrección medida: la revisión de fugas rechazaba `lda.fit(X)` + `lda.score(X)` en la Tarea C, que en un modelo no supervisado es lo que se pide; ahora solo cuenta ajustes supervisados (`corridas/v1/`).

## Parte 3 — Los cuatro frenos

Los cuatro se hicieron saltar con un modelo de guion (`frenos.py`, sin gastar tokens; trazas en `corridas/frenos/<escenario>/traza.jsonl`; salida en `resultados/frenos_salida.txt`), y tres saltaron solos en las corridas oficiales.

| Freno | Demostración (`frenos.py`) | Evento en la traza | En las corridas oficiales |
|---|---|---|---|
| Presupuesto, revisado antes de cada llamada, con reserva para el redactor | `presupuesto`: total 20 000, reserva 7000. T1 se completa; antes de programar T2, «usados 9080 + estimados 7534 > límite 13000». Se entrega `parcial` | `freno_presupuesto` | 3 veces (R2 en las dos versiones) |
| Máximo de intentos y detector de repetición | `intentos`: T2 entrega un árbol con fuga; el intento 1 se rechaza; los intentos 2 y 3 son idénticos (misma huella AST) y **no se ejecutan**; T2 queda fallida y T3 se completa | `freno_repeticion` ×2, `freno_intentos` | 3 veces (C, R1 y R2 sin grafo) |
| Tiempo máximo que mata todos los procesos | `timeout`: `while True` muerto a los 3 s; T1 se reescribe. Prueba directa: un script que lanza un hijo; tras el tiempo, el hijo ya no existe (`hijo_sigue_vivo: false`) | `freno_timeout` | no hizo falta |
| Confirmación humana antes de usar la red | `red`: T1 llama a `fetch_openml`; la persona responde «no»; el programador reescribe con `load_digits` | `freno_red` (decisión, quién) | 3 veces (Qdrant por URL en R1) |

Además, el reintento por `finish_reason=length` con contenido vacío saltó 14 veces en las corridas oficiales (`reintento_llm`).

## Parte 4 — Mejora E: un juez de otra familia

`juez_gemma.py`: `gemma3:27b` (otra familia que GLM) recibe el enunciado, el entregable y la lista de archivos, y decide criterio por criterio (la `descripcion` de cada comprobación del golden set) si se cumple, sin ver la verdad calculada. Se compara con las comprobaciones de código; el grupo `control_sin_ejecutar` es el reporte de la Parte 0.a entregado como si fuera la Tarea A.

<!-- incluir: resultados/juez_resumen.csv -->

<!-- incluir: resultados/juez_lectura.md -->

## Parte 5 — Preguntas

**1. ¿Quién decide que la tarea está resuelta?** Ninguno de los agentes con LLM. El planificador propone subtareas y el revisor aprueba scripts, pero el `status` final lo calcula `Solver._status` (`solver/solver.py`): «completado» solo si todas las subtareas de cálculo quedaron aprobadas, el entregable pasó `validar_entregable` (secciones, extensión, notebook ejecutado y REGLA 5) y no se agotó el presupuesto. Si algo de eso falla, «parcial»; sin entregable, «fallido». Y fuera del solver, la calificación la decide el evaluador, que es código y no comparte nada con el solver. Un LLM puede equivocarse al aprobar un script, pero no puede declarar resuelta una tarea cuyas cifras no salieron de una ejecución.

**2. ¿Qué agente gasta más tokens y qué ganó el grafo?** El programador: 88 190 de entrada y 490 320 de salida en la versión completa, el 61 % de los tokens (`tablas.md`). Gasta más porque el modelo razona antes de escribir cada script y esos tokens se cobran como salida; cuando la tarea lo enreda (la generación por red de R2) razona hasta agotar `max_tokens` varias veces. El revisor es el segundo, porque recibe código, salida y contexto completos. El grafo ganó 5 comprobaciones (41 contra 36), 3 subtareas fallidas menos y 6 intentos de código menos, con 32 319 tokens menos (−3 %); a cambio tardó 588 s más, por las entidades del índice y un contexto más largo que el revisor lee entero.

**3. Uso honesto y riesgos.** Usarlo de forma honesta exige declararlo, adjuntar la traza (qué recibió cada agente, qué código se ejecutó, qué rechazó el revisor) y los scripts, separar lo que escribió el sistema de lo que revisó o cambió la persona, y no presentar como propias las decisiones del planificador ni las cifras sin verificar su procedencia. Si el sandbox fallara, el código generado correría con los permisos del usuario: podría borrar archivos, leer credenciales del entorno o del disco y enviarlas por red, descargar y ejecutar código de terceros, o dejar procesos vivos consumiendo recursos. Por eso el entorno va vacío, la carpeta es desechable, la red pide una persona y todo proceso muere con su grupo; un aislamiento completo pediría además un contenedor sin red.

## Reproducibilidad

`README.md` tiene todos los comandos. Versiones exactas en `requirements.txt` (Python 3.14.7). Corridas oficiales en `corridas/completo/` y `corridas/sin_grafo/`; re-mediciones en `corridas/fix/` y `corridas/fix2/`; primera medición (tres tareas) en `corridas/v1/`; control del juez en `corridas/control_sin_ejecutar/`; frenos en `corridas/frenos/`. Cada tarea tiene `traza.jsonl`, `traza_llm/`, `plan.json`, `grafo.png`, `trabajo/` y su entregable. CSV en `resultados/` (y `resultados/fix*`, `resultados/v1`). Ninguna clave aparece en el código, las trazas ni los scripts generados: la H200 no la pide y el sandbox no hereda el entorno.
