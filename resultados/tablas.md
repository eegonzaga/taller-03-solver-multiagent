# Tablas de la Parte 2.b (generadas por tablas.py desde los CSV)

## Por tarea y versión

| Tarea | Versión | Status | Checks | Proced. | Fallidas | Intentos | Tok. entrada | Tok. salida | s |
|---|---|---|---|---|---|---|---|---|---|
| A | completo | completado | 9/9 | 1.00 | 0/4 | 3 | 28517 | 41918 | 227.22 |
| A | sin_grafo | completado | 9/9 | 1.00 | 0/4 | 4 | 25777 | 67200 | 393.28 |
| B | completo | completado | 9/9 | 1.00 | 0/5 | 3 | 51065 | 58362 | 320.09 |
| B | sin_grafo | completado | 9/9 | 1.00 | 0/4 | 3 | 37793 | 52918 | 304.11 |
| C | completo | completado | 9/9 | 1.00 | 0/4 | 3 | 53479 | 89684 | 555.39 |
| C | sin_grafo | parcial | 8/9 | 1.00 | 1/4 | 6 | 59387 | 155599 | 946.35 |
| R1 | completo | completado | 7/8 | 0.81 | 0/4 | 5 | 76215 | 99936 | 1629.61 |
| R1 | sin_grafo | parcial | 6/8 | 0.82 | 1/4 | 5 | 60433 | 97396 | 629.95 |
| R2 | completo | parcial | 7/8 | 1.00 | 0/4 | 3 | 65266 | 378002 | 2200.04 |
| R2 | sin_grafo | parcial | 4/8 | 0.83 | 1/4 | 5 | 70567 | 347693 | 2070.87 |

## Totales por versión

| Versión | Aprobadas | Total | Subtareas fallidas | Intentos | Tokens entrada | Tokens salida | Duración (s) |
|---|---|---|---|---|---|---|---|
| completo | 41 | 43 | 0 | 17 | 274542 | 667902 | 4932.3 |
| sin_grafo | 36 | 43 | 3 | 23 | 253957 | 720806 | 4344.6 |

## Tokens por agente (suma de las tareas)

| Agente | llamadas (completo) | entrada (completo) | salida (completo) | llamadas (sin_grafo) | entrada (sin_grafo) | salida (sin_grafo) |
|---|---|---|---|---|---|---|
| programador | 24 | 88190 | 490320 | 30 | 100405 | 575352 |
| revisor | 15 | 103333 | 76072 | 14 | 80679 | 42723 |
| redactor | 6 | 55730 | 60468 | 8 | 60977 | 87057 |
| planificador | 5 | 11896 | 15211 | 5 | 11896 | 15674 |
| indexador | 28 | 15393 | 25831 | 0 | 0 | 0 |

## Tokens por tarea y agente (entrada / salida)

| Tarea | Agente | completo | sin_grafo |
|---|---|---|---|
| A | programador | 7000 / 23597 | 7375 / 49847 |
| A | revisor | 13381 / 5193 | 12473 / 4144 |
| A | redactor | 4662 / 6064 | 4583 / 11450 |
| A | planificador | 1346 / 2036 | 1346 / 1759 |
| A | indexador | 2128 / 5028 | 0 / 0 |
| B | programador | 8229 / 15182 | 5642 / 13149 |
| B | revisor | 17542 / 13119 | 13543 / 11571 |
| B | redactor | 20822 / 23637 | 16769 / 25112 |
| B | planificador | 1839 / 2052 | 1839 / 3086 |
| B | indexador | 2633 / 4372 | 0 / 0 |
| C | programador | 10631 / 34942 | 23135 / 130790 |
| C | revisor | 29290 / 38401 | 18215 / 11565 |
| C | redactor | 8052 / 10644 | 15682 / 11340 |
| C | planificador | 2355 / 1795 | 2355 / 1904 |
| C | indexador | 3151 / 3902 | 0 / 0 |
| R1 | programador | 23291 / 64972 | 18697 / 58889 |
| R1 | revisor | 24902 / 7180 | 21632 / 8470 |
| R1 | redactor | 22194 / 20123 | 17473 / 26837 |
| R1 | planificador | 2631 / 2286 | 2631 / 3200 |
| R1 | indexador | 3197 / 5375 | 0 / 0 |
| R2 | programador | 39039 / 351627 | 45556 / 322677 |
| R2 | revisor | 18218 / 12179 | 14816 / 6973 |
| R2 | redactor | 0 / 0 | 6470 / 12318 |
| R2 | planificador | 3725 / 7042 | 3725 / 5725 |
| R2 | indexador | 4284 / 7154 | 0 / 0 |

## Comprobaciones: completo frente a sin_grafo

| Tarea | Check | Tipo | Completo | Sin grafo | Detalle (completo) |
|---|---|---|---|---|---|
| A | A01 | archivo | ✓ | ✓ | 1 archivo(s) con reporte.md |
| A | A02 | archivo | ✓ | ✓ | 1 archivo(s) con **/*.png |
| A | A03 | secciones | ✓ | ✓ | todas, en orden |
| A | A04 | palabras_max | ✓ | ✓ | 792 palabras |
| A | A05 | cifra_presente | ✓ | ✓ | verdad=1797.0000 ±0; hallada |
| A | A06 | cifra | ✓ | ✓ | verdad=0.8556 ±0.006; hallada |
| A | A07 | cifra | ✓ | ✓ | verdad=0.9711 ±0.006; hallada |
| A | A08 | cifra_presente | ✓ | ✓ | verdad=0.3667 ±0.0006; hallada |
| A | A09 | procedencia | ✓ | ✓ | 37/37 cifras respaldadas (1.00) |
| B | B01 | archivo | ✓ | ✓ | 1 archivo(s) con *.ipynb |
| B | B02 | ipynb_ejecutado | ✓ | ✓ | 4 celdas, 0 sin ejecutar, 0 con error |
| B | B03 | cifra_presente | ✓ | ✓ | verdad=0.4981 ±0.00051; hallada |
| B | B04 | cifra_presente | ✓ | ✓ | verdad=0.8639 ±0.00051; hallada |
| B | B05 | cifra_presente | ✓ | ✓ | verdad=0.3537 ±0.00051; hallada |
| B | B06 | cifra_presente | ✓ | ✓ | verdad=3.0363 ±0.00051; hallada |
| B | B07 | cifra_presente | ✓ | ✓ | verdad=0.7108 ±0.00051; hallada |
| B | B08 | archivo | ✓ | ✓ | 2 archivo(s) con **/*.png |
| B | B09 | procedencia | ✓ | ✓ | 60/60 cifras respaldadas (1.00) |
| C | C01 | archivo | ✓ | ✓ | 1 archivo(s) con reporte.pdf |
| C | C02 | paginas_max | ✓ | ✓ | 2 páginas |
| C | C03 | secciones | ✓ | ✓ | todas, en orden |
| C | C04 | cifra | ✓ | ✓ | verdad=137.7877 ±0.06; hallada |
| C | C05 | cifra_presente | ✓ | ✗ | verdad=156.4198 ±0.06; hallada |
| C | C06 | cifra | ✓ | ✓ | verdad=0.5000 ±0.006; hallada |
| C | C07 | cifra_presente | ✓ | ✓ | verdad=71.0000 ±0; hallada |
| C | C08 | codigo_sin | ✓ | ✓ | ninguno |
| C | C09 | procedencia | ✓ | ✓ | 22/22 cifras respaldadas (1.00) |
| R1 | R101 | archivo | ✓ | ✓ | 1 archivo(s) con *.ipynb |
| R1 | R102 | ipynb_ejecutado | ✓ | ✓ | 4 celdas, 0 sin ejecutar, 0 con error |
| R1 | R103 | cifra_presente | ✓ | ✓ | verdad=1562.5000 ±0.6; hallada |
| R1 | R104 | cifra | ✓ | ✓ | verdad=0.9920 ±0.015; hallada |
| R1 | R105 | cifra_presente | ✓ | ✗ | verdad=0.1018 ±0.0006; hallada |
| R1 | R106 | archivo | ✓ | ✓ | 2 archivo(s) con **/*.png |
| R1 | R107 | codigo_sin | ✓ | ✓ | ninguno |
| R1 | R108 | procedencia | ✗ | ✗ | 56/69 cifras respaldadas (0.81) |
| R2 | R201 | archivo | ✓ | ✓ | 1 archivo(s) con *.ipynb |
| R2 | R202 | ipynb_ejecutado | ✓ | ✓ | 3 celdas, 0 sin ejecutar, 0 con error |
| R2 | R203 | cifra | ✓ | ✗ | verdad=6.0000 ±0; hallada |
| R2 | R204 | cifra_presente | ✓ | ✗ | verdad=0.4986 ±0.0006; hallada |
| R2 | R205 | cifra_presente | ✓ | ✗ | verdad=0.3985 ±0.0006; hallada |
| R2 | R206 | contiene | ✗ | ✓ | faltan ['El corpus no contiene información suficiente'] |
| R2 | R207 | codigo_sin | ✓ | ✓ | ninguno |
| R2 | R208 | procedencia | ✓ | ✗ | sin cifras con dos o más decimales |
