# Tarea A — ¿Gana el generativo con pocos datos? Ng y Jordan sobre dígitos

**MMIA 6013 · Tarea de práctica para el solver multiagente (Taller 03 v2)**
**Entrega:** un reporte en Markdown (`reporte.md`) con el código que lo produce.

Ng y Jordan (2001), *On Discriminative vs. Generative Classifiers* (corpus del curso), sostienen
que un clasificador generativo como Naive Bayes se acerca a su error asintótico con menos
ejemplos que su par discriminativo, la regresión logística, aunque ese error asintótico sea
mayor. Esta tarea pone a prueba esa afirmación sobre un conjunto de datos de imágenes.

## Parte 1 — Datos y partición

Carga el conjunto de dígitos manuscritos que trae scikit-learn (`sklearn.datasets.load_digits`).
Reporta cuántos ejemplos, cuántos atributos y cuántas clases tiene. Divide los datos en
entrenamiento y prueba, 75 % y 25 %, estratificando por la clase, con `random_state=7`. Esa
partición es la única de toda la tarea: el conjunto de prueba no se usa para ajustar ni para
elegir nada.

## Parte 2 — Los dos clasificadores

Entrena un **Naive Bayes gaussiano** (`GaussianNB` con sus valores por defecto) sobre los
atributos originales, sin escalar, y una **regresión logística**
(`LogisticRegression(max_iter=3000)`) sobre los atributos estandarizados, con un
`StandardScaler` que se ajusta solo con el conjunto de entrenamiento. Reporta la
exactitud (*accuracy*) y el F1 macro de cada modelo sobre el conjunto de prueba, en una tabla.

## Parte 3 — La curva de aprendizaje

Con la partición de la Parte 1, entrena los dos modelos de la Parte 2 con 20, 50, 100, 200 y
400 ejemplos de entrenamiento y con el conjunto de entrenamiento completo. Cada submuestra se
obtiene con `train_test_split(X_train, y_train, train_size=m, stratify=y_train,
random_state=7)`. Evalúa siempre sobre el mismo conjunto de prueba. Dibuja la exactitud
contra el número de ejemplos, con una curva por modelo, y guarda la figura como PNG.

## Parte 4 — Discusión

Con tus cifras, di si se observa el cruce que predicen Ng y Jordan y desde qué tamaño conviene
cada modelo. Si no se observa, explica qué supuesto del modelo generativo puede fallar en
estos datos (piensa en la independencia condicional entre píxeles vecinos).

## El reporte

Un `reporte.md` con estas secciones, en este orden: **Introducción**, **Metodología**,
**Resultados** (la tabla de la Parte 2 y la figura de la Parte 3), **Discusión** y
**Conclusiones**. Máximo **1 000 palabras**. Toda cifra del reporte sale de la ejecución del
código entregado.
