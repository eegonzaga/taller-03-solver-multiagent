# Comprobaciones fallidas — versión completo

2 fallidas. Las tres primeras son las peores (contenido antes que forma).

## R108 (procedencia) — Las cifras del notebook salen de una ejecución

Detalle del evaluador: `56/69 cifras respaldadas (0.81)`

### Traza de la tarea R1 (corridas/completo/tarea-R1/traza.jsonl)

- **plan** ronda 1: válido
- **investigador** T1: [enunciado · S0 «S2·MAR — ANN: mide el trade-off recall / latencia», p. 1] (preámbulo); [enunciado · S1 «Parte 1 — kNN exacto: latencia vs. N», p. 1] (sección propia); [enunciado · S3 «Parte 3 — Qdrant embebido: una vector DB real sin Docker», p. 2] (depende_de); [corpus · slides-s2-embeddings.pdf, p. 15] (papers del corpus, 3 entidades compartidas); [corpus · slides-s2-embeddings.pdf, p. 14] (papers del corpus, 3 entidades compartidas); [corpus · slides-s2-embeddings.pdf, p. 14] (papers del corpus, 3 entidades compartidas)
- **ejecutor** T1 intento 1: código 0, 1.43 s, creó ['parte1_latencia_vs_N.png', 'resultados.json']
- **revisor** T1 intento 1: APROBADO por llm
- **investigador** T2: [enunciado · S0 «S2·MAR — ANN: mide el trade-off recall / latencia», p. 1] (preámbulo); [enunciado · S2 «Parte 2 — Mini-IVF: clustering + búsqueda local», p. 1] (sección propia); [enunciado · S1 «Parte 1 — kNN exacto: latencia vs. N», p. 1] (sección de una subtarea de la que depende); [enunciado · S3 «Parte 3 — Qdrant embebido: una vector DB real sin Docker», p. 2] (depende_de); [corpus · slides-s2-vector_databases.pdf, p. 8] (papers del corpus, 5 entidades compartidas); [corpus · slides-s2-vector_databases.pdf, p. 12] (papers del corpus, 5 entidades compartidas); [corpus · slides-s2-vector_da
- **revisor** T2 intento 1: RECHAZADO por sandbox — error de sintaxis en la línea 1: invalid syntax
- **ejecutor** T2 intento 2: código 0, 7.1 s, creó ['T2_parte2_recall_vs_speedup.png', 'resultados.json']
- **revisor** T2 intento 2: APROBADO por llm
- **investigador** T3: [enunciado · S0 «S2·MAR — ANN: mide el trade-off recall / latencia», p. 1] (preámbulo); [enunciado · S3 «Parte 3 — Qdrant embebido: una vector DB real sin Docker», p. 2] (sección propia); [corpus · slides-s2-vector_databases.pdf, p. 21] (papers del corpus, 5 entidades compartidas); [corpus · slides-s2-vector_databases.pdf, p. 19] (papers del corpus, 3 entidades compartidas); [corpus · slides-s2-vector_databases.pdf, p. 8] (papers del corpus, 4 entidades compartidas)
- **freno_red** {"subtarea": "T3", "intento": 1, "motivos": ["línea 89: URL http://localhost:6333"], "decision": "denegado", "quien": "confirmar_por_consola"}
- **revisor** T3 intento 1: RECHAZADO por red — el script quiere usar la red (línea 89: URL http://localhost:6333) y una persona no lo aprobó. Usa solo datos locales (scikit-learn incluye sus datasets de ejemplo, como load_breast_cancer) o los del enunciado.
- **ejecutor** T3 intento 2: código 0, 0.55 s, creó ['resultados.json']
- **revisor** T3 intento 2: APROBADO por llm
- **investigador** T4: [enunciado · S0 «S2·MAR — ANN: mide el trade-off recall / latencia», p. 1] (preámbulo); [enunciado · S4 «Cierre», p. 3] (sección propia); [enunciado · S1 «Parte 1 — kNN exacto: latencia vs. N», p. 1] (sección de una subtarea de la que depende); [enunciado · S2 «Parte 2 — Mini-IVF: clustering + búsqueda local», p. 1] (sección de una subtarea de la que depende); [enunciado · S3 «Parte 3 — Qdrant embebido: una vector DB real sin Docker», p. 2] (sección de una subtarea de la que depende); [corpus · slides-s2-vector_databases.pdf, p. 2] (papers del corpus, 6 entidades compartidas); [corpus · slides
- **verificación del entregable** ronda 1: 1474 palabras, 0 páginas; problemas: ["REGLA 5 — estas cifras no salieron de ninguna ejecución ni están en el enunciado; corrígelas copiando el valor medido o elimínalas: ['133.2']"]
- **verificación del entregable** ronda 2: 1482 palabras, 0 páginas; problemas: ninguno

## R206 (contiene) — Para la pregunta sin respuesta (mascotas), el RAG se abstiene con la frase del curso

Detalle del evaluador: `faltan ['El corpus no contiene información suficiente']`

### Traza de la tarea R2 (corridas/completo/tarea-R2/traza.jsonl)

- **plan** ronda 1: válido
- **investigador** T1: [enunciado · S0 «S2·MIÉ — RAG mínimo en ~60 líneas», p. 1] (preámbulo); [enunciado · S1 «Parte 1 — Chunking», p. 1] (sección propia); [corpus · slides-s2-fragmentacion_recuperacion.pdf, p. 13] (papers del corpus, 4 entidades compartidas); [corpus · slides-s2-fragmentacion_recuperacion.pdf, p. 14] (papers del corpus, 3 entidades compartidas); [corpus · gao-2023-rag-survey.pdf, p. 8] (papers del corpus, 3 entidades compartidas)
- **ejecutor** T1 intento 1: código 0, 0.11 s, creó ['resultados.json']
- **revisor** T1 intento 1: APROBADO por llm
- **investigador** T2: [enunciado · S0 «S2·MIÉ — RAG mínimo en ~60 líneas», p. 1] (preámbulo); [enunciado · S2 «Parte 2 — Índice (embeddings + kNN)», p. 2] (sección propia); [enunciado · S1 «Parte 1 — Chunking», p. 1] (sección de una subtarea de la que depende); [enunciado · S3 «Parte 3 — Generación con contexto», p. 3] (depende_de); [corpus · slides-s2-fragmentacion_recuperacion.pdf, p. 11] (papers del corpus, 4 entidades compartidas); [corpus · slides-s2-embeddings.pdf, p. 19] (papers del corpus, 4 entidades compartidas); [corpus · slides-s2-fragmentacion_recuperacion.pdf, p. 12] (papers del corpus, 3 entidades co
- **ejecutor** T2 intento 1: código 0, 1.44 s, creó ['resultados.json']
- **revisor** T2 intento 1: APROBADO por llm
- **investigador** T3: [enunciado · S0 «S2·MIÉ — RAG mínimo en ~60 líneas», p. 1] (preámbulo); [enunciado · S3 «Parte 3 — Generación con contexto», p. 3] (sección propia); [enunciado · S2 «Parte 2 — Índice (embeddings + kNN)», p. 2] (sección de una subtarea de la que depende); [enunciado · S1 «Parte 1 — Chunking», p. 1] (depende_de); [corpus · gao-2023-rag-survey.pdf, p. 3] (papers del corpus, 4 entidades compartidas); [corpus · slides-s2-fragmentacion_recuperacion.pdf, p. 16] (papers del corpus, 3 entidades compartidas); [corpus · es-2023-ragas.pdf, p. 3] (papers del corpus, 3 entidades compartidas)
- **ejecutor** T3 intento 1: código 0, 0.11 s, creó []
- **revisor** T3 intento 1: RECHAZADO por codigo — no escribió resultados.json
- **freno_presupuesto** {"nodo": "programar", "detalle": "programador: usados 443268 + estimados 11397 > límite 390000 (total 450000, reserva del redactor 60000)"}
- **freno_presupuesto** {"nodo": "redactar", "detalle": "redactor: usados 443268 + estimados 14834 > límite 450000 (total 450000, reserva del redactor 60000)", "accion": "entregable mínimo escrito por código"}
- **verificación del entregable** ronda 3: 68 palabras, 0 páginas; problemas: ["faltan los marcadores [[CODIGO …]] de ['T1', 'T2']"]

