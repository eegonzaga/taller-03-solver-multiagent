"""Las respuestas correctas del golden set, calculadas (nunca escritas a mano).

Cada función reproduce exactamente lo que pide su enunciado (`enunciados/*.md`), con los
mismos parámetros. Si un enunciado cambia, esta función cambia con él: `corpus_coincide()`
comprueba que el corpus de la Tarea C sea el mismo que trae su enunciado.

    python verdades.py          # imprime todas las verdades
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent

# ───────────────────────────────────────────── Tarea A — Ng y Jordan (2001) con load_digits
TAMANOS_A = [20, 50, 100, 200, 400]


@lru_cache(maxsize=None)
def tarea_a() -> dict:
    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=7)

    def modelos():
        return {"nb": GaussianNB(),
                "lr": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))}

    salida = {"n": int(X.shape[0]), "atributos": int(X.shape[1]), "clases": int(len(set(y))),
              "n_entrenamiento": int(len(ytr)), "n_prueba": int(len(yte))}
    for nombre, m in modelos().items():
        p = m.fit(Xtr, ytr).predict(Xte)
        salida[f"{nombre}_accuracy"] = accuracy_score(yte, p)
        salida[f"{nombre}_f1_macro"] = f1_score(yte, p, average="macro")
    curva = {}
    for t in TAMANOS_A + [len(ytr)]:
        if t < len(ytr):
            Xs, _, ys, _ = train_test_split(Xtr, ytr, train_size=t, stratify=ytr, random_state=7)
        else:
            Xs, ys = Xtr, ytr
        curva[t] = {n: accuracy_score(yte, m.fit(Xs, ys).predict(Xte)) for n, m in modelos().items()}
    salida["curva"] = curva
    return salida


# ───────────────────────────────────────────── Tarea B — Vaswani et al. (2017)
Q_B = np.array([[1.0, 0.0, 1.0, 0.5], [0.0, 2.0, 0.0, 1.0]])
K_B = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.5], [1.0, 0.0, 1.0, 1.0]])
V_B = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def atencion(Q, K, V):
    pesos = _softmax(Q @ K.T / np.sqrt(K.shape[1]))
    return pesos, pesos @ V


def entropia_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def codificacion_posicional(n_pos: int = 50, d_model: int = 16) -> np.ndarray:
    pos = np.arange(n_pos)[:, None]
    i = np.arange(0, d_model, 2)[None, :]
    angulos = pos / np.power(10000.0, i / d_model)
    pe = np.zeros((n_pos, d_model))
    pe[:, 0::2] = np.sin(angulos)
    pe[:, 1::2] = np.cos(angulos)
    return pe


@lru_cache(maxsize=None)
def tarea_b() -> dict:
    pesos, salida = atencion(Q_B, K_B, V_B)
    escala = {}
    rng = np.random.default_rng(0)
    for dk in (4, 64, 512):
        q = rng.standard_normal(dk)
        K = rng.standard_normal((10, dk))
        s = K @ q
        sin, con = _softmax(s), _softmax(s / np.sqrt(dk))
        escala[dk] = {"max_sin": float(sin.max()), "max_con": float(con.max()),
                      "H_sin": entropia_bits(sin), "H_con": entropia_bits(con)}
    pe = codificacion_posicional()
    pesos_pos, _ = atencion(pe[10:11], pe, pe)
    return {"pesos": pesos, "salida": salida, "escala": escala, "pe": pe,
            "pos_max": int(pesos_pos[0].argmax()), "peso_max": float(pesos_pos[0].max())}


# ───────────────────────────────────────────── Tarea C — Blei, Ng y Jordan (2003), LDA
CORPUS_C = {
    "d01": ("El delantero marcó dos goles en el partido de fútbol y el equipo ganó la liga.", "deporte"),
    "d02": ("El entrenador del equipo preparó la defensa para el partido final de la liga.", "deporte"),
    "d03": ("La tenista ganó el torneo después de un partido largo contra la campeona.", "deporte"),
    "d04": ("El equipo de baloncesto perdió el partido por un punto en el último segundo.", "deporte"),
    "d05": ("La receta lleva harina, huevos y azúcar; la masa se hornea durante treinta minutos.", "cocina"),
    "d06": ("Para la salsa se sofríe cebolla y ajo en aceite y se añade tomate.", "cocina"),
    "d07": ("El pan se hornea con harina, agua, sal y levadura después de amasar la masa.", "cocina"),
    "d08": ("La sopa de verduras lleva cebolla, zanahoria, ajo y sal, y se cocina a fuego lento.", "cocina"),
    "d09": ("El telescopio observó una galaxia lejana y varias estrellas de la nebulosa.", "astronomía"),
    "d10": ("Los planetas giran alrededor de la estrella; la órbita de la Tierra dura un año.", "astronomía"),
    "d11": ("La nebulosa es una nube de gas donde nacen estrellas nuevas en la galaxia.", "astronomía"),
    "d12": ("El telescopio espacial fotografió planetas y la órbita de una luna de Júpiter.", "astronomía"),
}
STOP_C = ["el", "la", "los", "las", "de", "del", "en", "y", "a", "un", "una", "por", "para",
          "con", "se", "su", "al", "es", "después", "durante", "contra", "donde", "alrededor"]


def _lda(k: int):
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    textos = [t for t, _ in CORPUS_C.values()]
    vec = CountVectorizer(stop_words=STOP_C)
    X = vec.fit_transform(textos)
    lda = LatentDirichletAllocation(n_components=k, learning_method="batch", max_iter=50,
                                    random_state=0).fit(X)
    return vec, X, lda


@lru_cache(maxsize=None)
def tarea_c() -> dict:
    perplejidad = {k: float(_lda(k)[2].perplexity(_lda(k)[1])) for k in (2, 3, 4)}
    vec, X, lda = _lda(3)
    dominante = lda.transform(X).argmax(axis=1)
    etiquetas = [e for _, e in CORPUS_C.values()]
    aciertos = 0
    for t in set(dominante):
        miembros = [etiquetas[i] for i in range(len(etiquetas)) if dominante[i] == t]
        aciertos += max(miembros.count(e) for e in set(miembros))
    vocab = vec.get_feature_names_out()
    top = [[vocab[j] for j in fila.argsort()[::-1][:5]] for fila in lda.components_]
    return {"perplejidad": perplejidad, "pureza": aciertos / len(etiquetas),
            "dominante": dominante.tolist(), "top5": top, "vocabulario": len(vocab)}


# ───────────────────────────────────────────── Tareas reales (actividades de la Semana 2)
# R1 — s2-mar (ANN). La verdad sigue el notebook EN ORDEN: el generador global
# `rng = default_rng(7)` se consume primero en la Parte 1 (20 consultas y las bases de 10k,
# 50k, 100k y 200k), después en la base de 100k de la Parte 2, la consulta de prueba del 2.1
# y las 50 consultas del 2.2. Quien regenere los datos por su cuenta obtiene otros vectores.
@lru_cache(maxsize=None)
def real_r1() -> dict:
    import hashlib
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(7)
    D = 128
    centros = np.random.default_rng(2026).normal(size=(200, D))

    def datos(n, d=D):
        asignacion = rng.integers(0, len(centros), size=n)
        return (centros[asignacion, :d] + 1.5 * rng.normal(size=(n, d))).astype(np.float32)

    def normalizar(X):
        return X / np.linalg.norm(X, axis=-1, keepdims=True)

    def knn(q, base, k=10):
        s = base @ q
        top = np.argpartition(-s, k)[:k]
        return top[np.argsort(-s[top])]

    normalizar(datos(20))                                   # Parte 1: consultas
    for n in [10_000, 50_000, 100_000, 200_000]:            # Parte 1: bases (solo tiempos)
        normalizar(datos(n))
    base = normalizar(datos(100_000))                       # Parte 2
    # Un solo hilo: con varios, el orden de las sumas de KMeans cambia y el recall@10 salía
    # 0.916 o 0.926 (nprobe = 1) en corridas idénticas. Con uno, siempre da lo mismo, y
    # coincide con las salidas guardadas del notebook del profesor.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        km = KMeans(n_clusters=64, n_init=3, random_state=7).fit(base)
    centroides = normalizar(km.cluster_centers_.astype(np.float32))
    celdas = [np.where(km.labels_ == c)[0] for c in range(64)]

    def knn_ivf(q, nprobe, k=10):
        top_c = np.argsort(-(centroides @ q))[:nprobe]
        cand = np.concatenate([celdas[c] for c in top_c])
        s = base[cand] @ q
        kk = min(k, len(cand))
        t = np.argpartition(-s, kk - 1)[:kk]
        return cand[t[np.argsort(-s[t])]]

    normalizar(datos(1))                                    # 2.1: consulta de prueba
    consultas = normalizar(datos(50))                       # 2.2
    verdad = [knn(q, base) for q in consultas]
    recall = {npb: float(np.mean([len(set(knn_ivf(q, npb)) & set(v)) / 10
                                  for q, v in zip(consultas, verdad)]))
              for npb in (1, 2, 4, 8, 16)}

    def embed_demo(texto):                                  # Parte 3, con la semilla del notebook
        semilla = int.from_bytes(hashlib.sha256(texto.encode()).digest()[:4], "little")
        v = np.random.default_rng(semilla).normal(size=64).astype(np.float32)
        return v / np.linalg.norm(v)
    pagos = ["Los reembolsos se procesan en 5 a 7 días hábiles.",
             "La factura electrónica se emite al confirmar el pago."]
    q = embed_demo("¿cuándo me devuelven el dinero?")
    scores = sorted((float(embed_demo(t) @ q) for t in pagos), reverse=True)
    return {"tamano_medio": float(np.mean([len(c) for c in celdas])), "recall": recall,
            "qdrant_scores": scores}


# R2 — s2-mie (RAG mínimo). chunk_fijo(300, 80) como lo describe el ejercicio 1.1, y el
# índice del notebook sin sentence-transformers (no está en el entorno del sandbox): el
# respaldo TF-IDF que el propio notebook trae.
DOCS_R2 = {
    "politica_vacaciones.md": """Política de vacaciones de NimbusSoft.
Los empleados a tiempo completo acumulan 1.5 días de vacaciones por mes trabajado,
hasta un máximo de 18 días por año. Las vacaciones deben solicitarse con al menos
15 días de anticipación a través del portal interno. Los días no utilizados pueden
transferirse al año siguiente hasta un máximo de 5 días. Durante el primer año,
los días solo pueden tomarse después de superar el período de prueba de 3 meses.""",
    "politica_remoto.md": """Política de trabajo remoto de NimbusSoft.
El trabajo remoto está permitido hasta 3 días por semana para todos los roles
excepto soporte de infraestructura on-site. Los días remotos se coordinan con el
líder de equipo. Para trabajar desde el exterior del país se requiere aprobación
de Recursos Humanos con 30 días de anticipación y un máximo de 60 días por año.""",
    "gastos.md": """Política de reembolso de gastos de NimbusSoft.
Los gastos de viaje se reembolsan presentando factura dentro de los 30 días
posteriores al gasto. El límite diario de alimentación en viajes es de 45 USD.
Los pasajes aéreos deben comprarse en clase económica salvo vuelos de más de
8 horas, donde se permite económica premium con aprobación del gerente de área.""",
}


@lru_cache(maxsize=None)
def real_r2() -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer

    def chunk_fijo(texto, tamano=300, overlap=80):
        paso, trozos = tamano - overlap, []
        for inicio in range(0, len(texto), paso):
            trozos.append(texto[inicio:inicio + tamano])
            if inicio + tamano >= len(texto):
                break
        return trozos

    chunks = [c for t in DOCS_R2.values() for c in chunk_fijo(t)]
    tf = TfidfVectorizer().fit(chunks)
    V = tf.transform(chunks).toarray()
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)

    def top1(pregunta):
        v = tf.transform([pregunta]).toarray()[0]
        s = V @ (v / (np.linalg.norm(v) + 1e-9))
        return float(s.max()), int(s.argmax())

    return {"n_chunks": len(chunks),
            # La pregunta del código dado (celda de generación); la versión corta del
            # ejercicio 2.1 solo estaba en la solución de la estudiante y no llega al PDF.
            "transferir": top1("¿Cuántos días de vacaciones puedo transferir al año siguiente?"),
            "exterior": top1("¿puedo trabajar desde el exterior?")}


def corpus_coincide() -> bool:
    texto = (AQUI / "enunciados" / "tarea-c-lda-blei.md").read_text(encoding="utf-8")
    return all(re.search(re.escape(t), texto) for t, _ in CORPUS_C.values())


if __name__ == "__main__":
    import json
    a = tarea_a()
    print("A", json.dumps(a, indent=1, default=float))
    b = tarea_b()
    print("B pesos", b["pesos"].round(4).tolist(), "salida", b["salida"].round(4).tolist())
    print("B escala", json.dumps(b["escala"], indent=1))
    print("B PE[10,0..1]", b["pe"][10, :2].round(4), "PE[25,6]", round(b["pe"][25, 6], 4),
          "pos_max", b["pos_max"], round(b["peso_max"], 4))
    print("C", tarea_c())
    print("R1", real_r1())
    print("R2", real_r2())
    if (AQUI / "enunciados" / "tarea-c-lda-blei.md").exists():
        print("corpus de C coincide con el enunciado:", corpus_coincide())
