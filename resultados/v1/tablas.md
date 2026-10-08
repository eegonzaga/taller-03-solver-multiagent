# Tablas de la Parte 2.b (generadas por tablas.py desde los CSV)

## Por tarea y versión

| Tarea | Versión | Status | Checks | Proced. | Fallidas | Intentos | Tok. entrada | Tok. salida | s |
|---|---|---|---|---|---|---|---|---|---|
| A | completo | completado | 9/9 | 1.00 | 0/4 | 3 | 28295 | 31048 | 200.0 |
| A | sin_grafo | completado | 9/9 | 1.00 | 0/4 | 5 | 34670 | 55998 | 372.1 |
| B | completo | completado | 9/9 | 1.00 | 0/4 | 3 | 57198 | 78558 | 439.5 |
| B | sin_grafo | completado | 9/9 | 1.00 | 0/4 | 3 | 40300 | 43206 | 252.4 |
| C | completo | completado | 9/9 | 1.00 | 0/4 | 5 | 64766 | 104869 | 615.6 |
| C | sin_grafo | parcial | 7/9 | 0.88 | 1/4 | 4 | 63692 | 345284 | 2158.3 |

## Totales por versión

| Versión | Aprobadas | Total | Subtareas fallidas | Intentos | Tokens entrada | Tokens salida | Duración (s) |
|---|---|---|---|---|---|---|---|
| completo | 27 | 27 | 0 | 11 | 150259 | 214475 | 1255.1 |
| sin_grafo | 25 | 27 | 1 | 12 | 138662 | 444488 | 2782.8 |

## Tokens por agente (suma de las tareas)

| Agente | llamadas (completo) | entrada (completo) | salida (completo) | llamadas (sin_grafo) | entrada (sin_grafo) | salida (sin_grafo) |
|---|---|---|---|---|---|---|
| programador | 11 | 37971 | 98345 | 18 | 64312 | 379720 |
| revisor | 11 | 65967 | 59452 | 9 | 43481 | 19871 |
| redactor | 4 | 32869 | 38297 | 4 | 25329 | 37923 |
| planificador | 3 | 5540 | 6607 | 3 | 5540 | 6974 |
| indexador | 18 | 7912 | 11774 | 0 | 0 | 0 |

## Comprobaciones: completo frente a sin_grafo

| Tarea | Check | Tipo | Completo | Sin grafo | Detalle (completo) |
|---|---|---|---|---|---|
| A | A01 | archivo | ✓ | ✓ | 1 archivo(s) con reporte.md |
| A | A02 | archivo | ✓ | ✓ | 1 archivo(s) con **/*.png |
| A | A03 | secciones | ✓ | ✓ | todas, en orden |
| A | A04 | palabras_max | ✓ | ✓ | 793 palabras |
| A | A05 | cifra_presente | ✓ | ✓ | verdad=1797.0000 ±0; hallada |
| A | A06 | cifra | ✓ | ✓ | verdad=0.8556 ±0.006; hallada |
| A | A07 | cifra | ✓ | ✓ | verdad=0.9711 ±0.006; hallada |
| A | A08 | cifra_presente | ✓ | ✓ | verdad=0.3667 ±0.0006; hallada |
| A | A09 | procedencia | ✓ | ✓ | 24/24 cifras respaldadas (1.00) |
| B | B01 | archivo | ✓ | ✓ | 1 archivo(s) con *.ipynb |
| B | B02 | ipynb_ejecutado | ✓ | ✓ | 4 celdas, 0 sin ejecutar, 0 con error |
| B | B03 | cifra_presente | ✓ | ✓ | verdad=0.4981 ±0.00051; hallada |
| B | B04 | cifra_presente | ✓ | ✓ | verdad=0.8639 ±0.00051; hallada |
| B | B05 | cifra_presente | ✓ | ✓ | verdad=0.3537 ±0.00051; hallada |
| B | B06 | cifra_presente | ✓ | ✓ | verdad=3.0363 ±0.00051; hallada |
| B | B07 | cifra_presente | ✓ | ✓ | verdad=0.7108 ±0.00051; hallada |
| B | B08 | archivo | ✓ | ✓ | 1 archivo(s) con **/*.png |
| B | B09 | procedencia | ✓ | ✓ | 61/61 cifras respaldadas (1.00) |
| C | C01 | archivo | ✓ | ✓ | 1 archivo(s) con reporte.pdf |
| C | C02 | paginas_max | ✓ | ✓ | 2 páginas |
| C | C03 | secciones | ✓ | ✓ | todas, en orden |
| C | C04 | cifra | ✓ | ✓ | verdad=137.7877 ±0.06; hallada |
| C | C05 | cifra_presente | ✓ | ✗ | verdad=156.4198 ±0.06; hallada |
| C | C06 | cifra | ✓ | ✗ | verdad=0.5000 ±0.006; hallada |
| C | C07 | cifra_presente | ✓ | ✓ | verdad=71.0000 ±0; hallada |
| C | C08 | codigo_sin | ✓ | ✓ | ninguno |
| C | C09 | procedencia | ✓ | ✓ | 37/37 cifras respaldadas (1.00) |
