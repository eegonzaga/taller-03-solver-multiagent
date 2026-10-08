# Tarea C — Temas latentes con LDA en un corpus mínimo (Blei, Ng y Jordan, 2003)

**MMIA 6013 · Tarea de práctica para el solver multiagente (Taller 03 v2)**
**Entrega:** un reporte en **PDF** (`reporte.pdf`) de dos páginas como máximo, con el código.

*Latent Dirichlet Allocation* (corpus del curso) modela cada documento como una mezcla de temas
y cada tema como una distribución sobre palabras. El corpus son doce oraciones de tres temas;
la columna «tema» es la etiqueta real, que LDA **no** ve. Los datos están completos en este
enunciado: no descargues ningún otro corpus (por ejemplo 20 Newsgroups).

**Corpus**

| id | texto | tema |
|---|---|---|
| d01 | El delantero marcó dos goles en el partido de fútbol y el equipo ganó la liga. | deporte |
| d02 | El entrenador del equipo preparó la defensa para el partido final de la liga. | deporte |
| d03 | La tenista ganó el torneo después de un partido largo contra la campeona. | deporte |
| d04 | El equipo de baloncesto perdió el partido por un punto en el último segundo. | deporte |
| d05 | La receta lleva harina, huevos y azúcar; la masa se hornea durante treinta minutos. | cocina |
| d06 | Para la salsa se sofríe cebolla y ajo en aceite y se añade tomate. | cocina |
| d07 | El pan se hornea con harina, agua, sal y levadura después de amasar la masa. | cocina |
| d08 | La sopa de verduras lleva cebolla, zanahoria, ajo y sal, y se cocina a fuego lento. | cocina |
| d09 | El telescopio observó una galaxia lejana y varias estrellas de la nebulosa. | astronomía |
| d10 | Los planetas giran alrededor de la estrella; la órbita de la Tierra dura un año. | astronomía |
| d11 | La nebulosa es una nube de gas donde nacen estrellas nuevas en la galaxia. | astronomía |
| d12 | El telescopio espacial fotografió planetas y la órbita de una luna de Júpiter. | astronomía |

**Palabras vacías** (pásalas a `CountVectorizer(stop_words=...)` tal cual): el, la, los, las,
de, del, en, y, a, un, una, por, para, con, se, su, al, es, después, durante, contra, donde,
alrededor.

## Parte 1 — LDA con tres temas

Vectoriza los doce textos con `CountVectorizer` y la lista de palabras vacías de arriba (el
resto de parámetros por defecto). Ajusta `LatentDirichletAllocation(n_components=3,
learning_method="batch", max_iter=50, random_state=0)`. Reporta el tamaño del vocabulario, las
cinco palabras más probables de cada tema y la perplejidad del modelo sobre el corpus.

## Parte 2 — Número de temas

Con el mismo vectorizador de la Parte 1 y los mismos parámetros, ajusta LDA con 2, 3 y 4 temas
y reporta la perplejidad de cada uno en una sola tabla.

## Parte 3 — ¿Recupera LDA los temas reales?

Con el modelo de tres temas de la Parte 1, asigna a cada documento su tema dominante y
calcula la **pureza**: para cada tema de LDA, cuenta los documentos de la etiqueta real más
frecuente entre los asignados a él; suma esas cuentas y divide por 12. Presenta la asignación
por documento en una tabla y explica qué documentos quedan mal agrupados y por qué.

## Parte 4 — Pregunta conceptual

¿Qué añade LDA frente a pLSI según Blei et al. (el prior de Dirichlet y el modelo generativo
para documentos nuevos)? ¿Por qué un corpus de doce oraciones es un mal escenario para LDA?

## El reporte

En PDF, dos páginas como máximo, con las secciones **Objetivo**, **Método**, **Resultados**
(las tablas de las Partes 2 y 3) y **Discusión**. Toda cifra sale de la ejecución.
