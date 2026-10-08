# Comprobaciones fallidas — versión sin_grafo

7 fallidas. Las tres primeras son las peores (contenido antes que forma).

## C05 (cifra_presente) — Reporta correctamente la perplejidad del modelo de cuatro temas

Detalle del evaluador: `verdad=156.4198 ±0.06; no hallada`

### Traza de la tarea C (corridas/sin_grafo/tarea-C/traza.jsonl)

- **plan** ronda 1: válido
- **investigador** T1: [enunciado · S1 «Parte 1 — LDA con tres temas», p. 1] (similitud 0.8944); [enunciado · S2 «Parte 2 — Número de temas», p. 1] (similitud 0.6995); [enunciado · S0 «Tarea C — Temas latentes con LDA en un corpus mínimo (Blei, Ng y Jordan, 2003)», p. 1] (similitud 0.6826); [corpus · blei-2003-lda.pdf, p. 4] (similitud 0.6537)
- **ejecutor** T1 intento 1: código 0, 1.03 s, creó ['resultados.json']
- **revisor** T1 intento 1: APROBADO por llm
- **investigador** T2: [enunciado · S1 «Parte 1 — LDA con tres temas», p. 1] (similitud 0.8816); [enunciado · S2 «Parte 2 — Número de temas», p. 1] (similitud 0.8087); [corpus · blei-2003-lda.pdf, p. 1] (similitud 0.673); [corpus · blei-2003-lda.pdf, p. 4] (similitud 0.6537)
- **revisor** T2 intento 1: RECHAZADO por sandbox — error de sintaxis en la línea 1: invalid syntax
- **ejecutor** T2 intento 2: código 0, 1.05 s, creó ['resultados.json']
- **revisor** T2 intento 2: RECHAZADO por llm — El requisito central era reutilizar el mismo CountVectorizer de la Parte 1 (mismas palabras vacías), pero la propia verificación del script lo desmiente: 'vocabulario_coincide_con_T1': false. El vocabulario aquí (173 términos) no coincide con el registrado en T1.; La perplejidad con k=3 (303.0018) no reproduce la de la Parte 1 (137.7877, diferencia absoluta 165.21) aunque se usan los mismos paráme
- **revisor** T2 intento 3: RECHAZADO por sandbox — línea 243: llama .listdir() (borrar, procesos, entorno o salir de la carpeta); línea 239: llama .listdir() (borrar, procesos, entorno o salir de la carpeta)
- **freno_intentos** {"subtarea": "T2", "intentos": 3, "detalle": "máximo de intentos alcanzado: subtarea fallida, se sigue"}
- **investigador** T3: [enunciado · S3 «Parte 3 — ¿Recupera LDA los temas reales?», p. 1] (similitud 0.9769); [enunciado · S2 «Parte 2 — Número de temas», p. 1] (similitud 0.6564); [enunciado · S1 «Parte 1 — LDA con tres temas», p. 1] (similitud 0.6467); [corpus · blei-2003-lda.pdf, p. 10] (similitud 0.6379)
- **ejecutor** T3 intento 1: código 1, 0.1 s, creó [], stderr: 'abajo/T3/script.py", line 71, in <module>\n    doc_topic = np.array([bruta[c] for c in _claves_ordenadas(bruta)], dtype=float)\nTypeError: float() argument must be a string or a real number, not \'dict\'\n'
- **revisor** T3 intento 1: RECHAZADO por codigo — terminó con código de salida 1:
Traceback (most recent call last):
  File "/Users/estefania.gonzaga/Documents/Personal/Master-USFQ-Classes/AIGenerativa&Agentes/Semana3/taller03/taller-03-solver-multiagent/corridas/sin_grafo/tarea-C/trabajo/T3/script.py", line 71, in <module>
    doc_topic = np.array([bruta[c] for c in _claves_ordenadas(bruta)], dtype=float)
TypeError: float() argument must be a st
- **ejecutor** T3 intento 2: código 0, 0.1 s, creó ['resultados.json']
- **revisor** T3 intento 2: APROBADO por llm
- **investigador** T4: [enunciado · S5 «El reporte», p. 1] (similitud 0.8055); [enunciado · S3 «Parte 3 — ¿Recupera LDA los temas reales?», p. 1] (similitud 0.6763); [enunciado · S4 «Parte 4 — Pregunta conceptual», p. 1] (similitud 0.6652); [corpus · blei-2003-lda.pdf, p. 20] (similitud 0.6346)
- **verificación del entregable** ronda 1: 1012 palabras, 2 páginas; problemas: ["REGLA 5 — estas cifras no salieron de ninguna ejecución ni están en el enunciado; corrígelas copiando el valor medido o elimínalas: ['1999']"]
- **verificación del entregable** ronda 2: 1018 palabras, 2 páginas; problemas: ninguno

## R105 (cifra_presente) — Reporta el puntaje del mejor resultado de Qdrant filtrado por tema «pagos» (el signo menos no lo distingue el evaluador)

Detalle del evaluador: `verdad=0.1018 ±0.0006; no hallada`

### Traza de la tarea R1 (corridas/sin_grafo/tarea-R1/traza.jsonl)

- **plan** ronda 1: válido
- **investigador** T1: [enunciado · S1 «Parte 1 — kNN exacto: latencia vs. N», p. 1] (similitud 0.8028); [corpus · slides-s2-vector_databases.pdf, p. 4] (similitud 0.6925); [enunciado · S4 «Cierre», p. 3] (similitud 0.6249); [enunciado · S2 «Parte 2 — Mini-IVF: clustering + búsqueda local», p. 1] (similitud 0.6038)
- **ejecutor** T1 intento 1: código 0, 1.71 s, creó ['resultados.json', 't1_latencia_vs_N.png']
- **revisor** T1 intento 1: APROBADO por llm
- **investigador** T2: [enunciado · S2 «Parte 2 — Mini-IVF: clustering + búsqueda local», p. 1] (similitud 0.8743); [corpus · slides-s2-vector_databases.pdf, p. 8] (similitud 0.7193); [enunciado · S0 «S2·MAR — ANN: mide el trade-off recall / latencia», p. 1] (similitud 0.7056); [enunciado · S1 «Parte 1 — kNN exacto: latencia vs. N», p. 1] (similitud 0.6864)
- **ejecutor** T2 intento 1: código 0, 5.76 s, creó ['T2_parte2_recall_y_speedup_por_nprobe.png', 'T2_parte2_triangulo_recall_latencia.png', 'resultados.json']
- **revisor** T2 intento 1: APROBADO por llm
- **investigador** T3: [enunciado · S3 «Parte 3 — Qdrant embebido: una vector DB real sin Docker», p. 2] (similitud 0.8101); [corpus · slides-s2-vector_databases.pdf, p. 21] (similitud 0.6298); [corpus · slides-s2-vector_databases.pdf, p. 21] (similitud 0.6285); [corpus · slides-s2-vector_databases.pdf, p. 23] (similitud 0.6025)
- **freno_red** {"subtarea": "T3", "intento": 1, "motivos": ["línea 120: URL http://localhost:6333"], "decision": "denegado", "quien": "confirmar_por_consola"}
- **revisor** T3 intento 1: RECHAZADO por red — el script quiere usar la red (línea 120: URL http://localhost:6333) y una persona no lo aprobó. Usa solo datos locales (scikit-learn incluye sus datasets de ejemplo, como load_breast_cancer) o los del enunciado.
- **freno_red** {"subtarea": "T3", "intento": 2, "motivos": ["línea 214: URL http://localhost:6333 (timeout=3, omitido)"], "decision": "denegado", "quien": "confirmar_por_consola"}
- **revisor** T3 intento 2: RECHAZADO por red — el script quiere usar la red (línea 214: URL http://localhost:6333 (timeout=3, omitido)) y una persona no lo aprobó. Usa solo datos locales (scikit-learn incluye sus datasets de ejemplo, como load_breast_cancer) o los del enunciado.
- **ejecutor** T3 intento 3: código 0, 0.53 s, creó ['resultados.json']
- **revisor** T3 intento 3: RECHAZADO por llm — El enunciado exige como primer paso 'intentar conexión a http://localhost:6333 con timeout 3 y caer a modo embebido (:memory:) si no responde'. El código omitió ese intento por completo: crea directamente QdrantClient(':memory:'). Es verificable en el código (no existe ningún QdrantClient(url=...) ni get_collections() de sondeo) y en el stdout (no aparece 'Conectado al servidor Qdrant en http://lo
- **freno_intentos** {"subtarea": "T3", "intentos": 3, "detalle": "máximo de intentos alcanzado: subtarea fallida, se sigue"}
- **investigador** T4: [enunciado · S4 «Cierre», p. 3] (similitud 0.7255); [corpus · slides-s2-vector_databases.pdf, p. 8] (similitud 0.653); [enunciado · S0 «S2·MAR — ANN: mide el trade-off recall / latencia», p. 1] (similitud 0.6513); [enunciado · S1 «Parte 1 — kNN exacto: latencia vs. N», p. 1] (similitud 0.6405)
- **verificación del entregable** ronda 1: 1594 palabras, 0 páginas; problemas: ['el marcador T3 no corresponde a ninguna subtarea con código aprobado']
- **verificación del entregable** ronda 2: 1605 palabras, 0 páginas; problemas: ninguno

## R203 (cifra) — Reporta el número correcto de fragmentos (300 caracteres, solapamiento 80)

Detalle del evaluador: `verdad=6.0000 ±0; no hallada`

### Traza de la tarea R2 (corridas/sin_grafo/tarea-R2/traza.jsonl)

- **plan** ronda 1: válido
- **investigador** T1: [enunciado · S1 «Parte 1 — Chunking», p. 1] (similitud 0.9016); [corpus · slides-s2-fragmentacion_recuperacion.pdf, p. 13] (similitud 0.6298); [corpus · edge-2024-graphrag.pdf, p. 4] (similitud 0.6281); [corpus · slides-s2-fragmentacion_recuperacion.pdf, p. 14] (similitud 0.6277)
- **revisor** T1 intento 1: RECHAZADO por sandbox — línea 220: ruta fuera de la carpeta «..»
- **ejecutor** T1 intento 2: código 0, 0.3 s, creó ['resultados.json']
- **revisor** T1 intento 2: APROBADO por llm
- **investigador** T2: [enunciado · S2 «Parte 2 — Índice (embeddings + kNN)», p. 2] (similitud 0.8886); [corpus · slides-s2-fragmentacion_recuperacion.pdf, p. 11] (similitud 0.6651); [corpus · slides-s2-fragmentacion_recuperacion.pdf, p. 9] (similitud 0.6593); [corpus · slides-s2-fragmentacion_recuperacion.pdf, p. 12] (similitud 0.6549)
- **revisor** T2 intento 1: RECHAZADO por sandbox — error de sintaxis en la línea 1: invalid syntax
- **ejecutor** T2 intento 2: código 0, 1.05 s, creó ['resultados.json']
- **revisor** T2 intento 2: RECHAZADO por llm — El índice se construyó sobre 2 'chunks' extraídos incorrectamente de entradas/T1.json, cuando T1 reporta 12 chunks para overlap80 (el propio resultados.json lo registra: num_chunks_indexados=2 vs num_chunks_t1_reportado=12, coincide_con_num_chunks_t1=false). El enunciado exige usar los chunks de T1, así que V (2, 17), el vocabulario TF-IDF y la demo de recuperar están calculados sobre datos inváli
- **ejecutor** T2 intento 3: código 1, 0.19 s, creó [], stderr: 'n_frase_en_doc: 294, fin_frase_en_doc: 319, idea_aparece_completa_en_algun_chunk: True, idea_cortada: False, chunks_con_idea_completa_doc: list[1] de 1, chunks_con_idea_completa_global: list[1] de 1}\n'
- **revisor** T2 intento 3: RECHAZADO por codigo — terminó con código de salida 1:
Traceback (most recent call last):
  File "/Users/estefania.gonzaga/Documents/Personal/Master-USFQ-Classes/AIGenerativa&Agentes/Semana3/taller03/taller-03-solver-multiagent/corridas/sin_grafo/tarea-R2/trabajo/T2/script.py", line 224, in <module>
    raise ValueError(
    ...<2 lines>...
    )
ValueError: No se pudieron extraer chunks de entradas/T1.json. Estructura 
- **freno_intentos** {"subtarea": "T2", "intentos": 3, "detalle": "máximo de intentos alcanzado: subtarea fallida, se sigue"}
- **investigador** T3: [enunciado · S3 «Parte 3 — Generación con contexto», p. 3] (similitud 0.8011); [corpus · slides-s2-fragmentacion_recuperacion.pdf, p. 27] (similitud 0.6597); [corpus · slides-s2-fragmentacion_recuperacion.pdf, p. 16] (similitud 0.6062); [corpus · brown-2020-gpt3.pdf, p. 29] (similitud 0.5931)
- **freno_presupuesto** {"nodo": "programar", "detalle": "programador: usados 399472 + estimados 9337 > límite 390000 (total 450000, reserva del redactor 60000)"}
- **verificación del entregable** ronda 1: 1556 palabras, 0 páginas; problemas: ['el marcador T2 no corresponde a ninguna subtarea con código aprobado', 'el marcador T3 no corresponde a ninguna subtarea con código aprobado', "las secciones no están en el orden pedido: ['RAG mínimo en ~60 líneas (MMIA 6013 · Semana 2)', 'Parte 1 — Chunking', 'Parte 2 — Índice (embeddings + kNN)', 'Parte 3 — Generación con contexto', 'Diagnóstico final y cierre']", "REGLA 5 — estas cifras no salieron de ninguna ejecución ni están en el enunciado; corrígelas copiando el valor medido o elimínalas: ['36.3636 %', '128', '500', '512']"]

## R204 (cifra_presente) — Reporta el puntaje top-1 correcto para «¿Cuántos días de vacaciones puedo transferir al año siguiente?»

Detalle del evaluador: `verdad=0.4986 ±0.0006; no hallada`

## R205 (cifra_presente) — Reporta el puntaje top-1 correcto para «¿puedo trabajar desde el exterior?»

Detalle del evaluador: `verdad=0.3985 ±0.0006; no hallada`

## R108 (procedencia) — Las cifras del notebook salen de una ejecución

Detalle del evaluador: `66/80 cifras respaldadas (0.82)`

## R208 (procedencia) — Las cifras del notebook salen de una ejecución

Detalle del evaluador: `5/6 cifras respaldadas (0.83)`

