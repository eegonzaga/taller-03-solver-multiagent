# Comprobaciones fallidas — versión sin_grafo

2 fallidas. Las tres primeras son las peores (contenido antes que forma).

## C05 (cifra_presente) — Reporta correctamente la perplejidad del modelo de cuatro temas

Detalle del evaluador: `verdad=156.4198 ±0.06; no hallada`

### Traza de la tarea C (corridas/sin_grafo/tarea-C/traza.jsonl)

- **plan** ronda 1: válido
- **investigador** T1: [enunciado · S1 «Parte 1 — LDA con tres temas», p. 1] (similitud 0.9131); [enunciado · S2 «Parte 2 — Número de temas», p. 1] (similitud 0.7207); [enunciado · S0 «Tarea C — Temas latentes con LDA en un corpus mínimo (Blei, Ng y Jordan, 2003)», p. 1] (similitud 0.6673); [corpus · blei-2003-lda.pdf, p. 3] (similitud 0.6587)
- **ejecutor** T1 intento 1: código 0, 1.48 s, creó ['resultados.json']
- **revisor** T1 intento 1: APROBADO por llm
- **investigador** T2: [enunciado · S2 «Parte 2 — Número de temas», p. 1] (similitud 0.8387); [enunciado · S1 «Parte 1 — LDA con tres temas», p. 1] (similitud 0.8353); [corpus · blei-2003-lda.pdf, p. 3] (similitud 0.6363); [enunciado · S3 «Parte 3 — ¿Recupera LDA los temas reales?», p. 1] (similitud 0.6253)
- **ejecutor** T2 intento 1: código 0, 1.56 s, creó ['resultados.json']
- **revisor** T2 intento 1: RECHAZADO por llm — El corpus usado NO es el de la Parte 1: el propio script lo demuestra con sus comprobaciones de coherencia. La salida muestra 'Vocabulario vs T1: 16 terminos comunes (Jaccard = 0.0513)', es decir, el vocabulario coincide apenas en un 5% con el de T1, y la perplejidad con k=3 (409.9873) NO coincide con la de T1 (137.7877) pese a usar supuestamente los mismos datos, el mismo vectorizador y random_st
- **revisor** T2 intento 2: RECHAZADO por sandbox — línea 204: llama .listdir() (borrar, procesos, entorno o salir de la carpeta); línea 239: llama .listdir() (borrar, procesos, entorno o salir de la carpeta)
- **ejecutor** T2 intento 3: código 0, 1.71 s, creó ['resultados.json']
- **revisor** T2 intento 3: RECHAZADO por codigo — fuga de datos: se predice/evalúa sobre ['X'], la misma variable con que se ajustó
- **freno_intentos** {"subtarea": "T2", "intentos": 3, "detalle": "máximo de intentos alcanzado: subtarea fallida, se sigue"}
- **investigador** T3: [enunciado · S3 «Parte 3 — ¿Recupera LDA los temas reales?», p. 1] (similitud 0.9487); [enunciado · S1 «Parte 1 — LDA con tres temas», p. 1] (similitud 0.6648); [enunciado · S2 «Parte 2 — Número de temas», p. 1] (similitud 0.6565); [corpus · blei-2003-lda.pdf, p. 10] (similitud 0.6492)
- **freno_presupuesto** {"nodo": "programar", "detalle": "programador: usados 398806 + estimados 6440 > límite 390000 (total 450000, reserva del redactor 60000)"}
- **verificación del entregable** ronda 1: 765 palabras, 2 páginas; problemas: ["REGLA 5 — estas cifras no salieron de ninguna ejecución ni están en el enunciado; corrígelas copiando el valor medido o elimínalas: ['137.7878', '1999']"]

## C06 (cifra) — Reporta correctamente la pureza de la asignación de temas

Detalle del evaluador: `verdad=0.5000 ±0.006; no hallada`

