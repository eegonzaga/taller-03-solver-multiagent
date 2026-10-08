"""La capa vectorial del GraphRAG, tomada del RAG del Taller 02 (`rag_pipeline.py`).

Del Taller 02 se conservan tres decisiones:

1. **Embeddings `bge-m3` en el Ollama de la H200** (`/api/embed`, puerto 11434), con el id
   servido buscado en `/api/tags` y no escrito a mano, y con `truncate: false`: si un texto
   no cabe, el servidor da error en vez de recortarlo en silencio.
2. **Vectores normalizados**: el producto punto es el coseno.
3. **Qdrant con la API del notebook** (`create_collection` + `upsert` + `query_points`), en
   modo `:memory:`, que corre sin Docker.

Lo nuevo: una caché en disco de embeddings (las notas del curso se vectorizan una sola vez)
y un codificador TF-IDF sin red, para las pruebas con el modelo de guion.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

H200_EMBED_URL = os.getenv("H200_EMBED_URL", "http://172.28.230.10:11434")


def _normalizar(vectores) -> np.ndarray:
    v = np.asarray(vectores, dtype=np.float32)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


class CodificadorH200:
    """bge-m3 en el Ollama de la H200, con caché en disco por hash del texto."""

    LOTE = 32
    TOPE_TOKENS = 8192

    def __init__(self, cache: Path | None = None, modelo: str = "bge-m3"):
        servidos = [m["name"] for m in self._pedir("/api/tags", timeout=10).get("models", [])]
        candidatos = sorted(n for n in servidos if n.lower().startswith(modelo))
        if not candidatos:
            raise RuntimeError(f"la H200 no sirve {modelo} (sirve: {', '.join(servidos)})")
        self.nombre = candidatos[0]
        self._ruta_cache = cache / f"embeddings-{self.nombre.replace(':', '_')}.json" if cache else None
        self._cache: dict[str, list[float]] = {}
        if self._ruta_cache and self._ruta_cache.exists():
            self._cache = json.loads(self._ruta_cache.read_text())
        self._lock = threading.Lock()

    def _pedir(self, ruta: str, cuerpo: dict | None = None, timeout: int = 300) -> dict:
        pet = urllib.request.Request(f"{H200_EMBED_URL}{ruta}",
                                     data=None if cuerpo is None else json.dumps(cuerpo).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(pet, timeout=timeout) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            raise ConnectionError(f"la H200 no responde en {H200_EMBED_URL} ({err}). "
                                  "¿Está conectada la VPN GlobalProtect?") from err

    def ajustar(self, textos: list[str]) -> None:
        """bge-m3 no se ajusta; existe para que los dos codificadores se usen igual."""

    def encode(self, textos: list[str]) -> np.ndarray:
        claves = [hashlib.sha1(t.encode()).hexdigest() for t in textos]
        faltan = [(c, t) for c, t in zip(claves, textos) if c not in self._cache]
        for i in range(0, len(faltan), self.LOTE):
            lote = faltan[i:i + self.LOTE]
            r = self._pedir("/api/embed", {
                "model": self.nombre, "input": [t for _, t in lote],
                "truncate": False, "options": {"num_ctx": self.TOPE_TOKENS}})
            with self._lock:
                for (c, _), v in zip(lote, r["embeddings"]):
                    self._cache[c] = v
        if faltan and self._ruta_cache:
            self._ruta_cache.parent.mkdir(parents=True, exist_ok=True)
            self._ruta_cache.write_text(json.dumps(self._cache))
        return _normalizar([self._cache[c] for c in claves])


class CodificadorTfidf:
    """Sin red: TF-IDF ajustado sobre los textos del índice. Peor que bge-m3, pero suficiente
    para probar el flujo y los frenos con el modelo de guion."""

    nombre = "tfidf"

    def __init__(self, cache: Path | None = None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer()
        self._ajustado = False

    def ajustar(self, textos: list[str]) -> None:
        self._vec.fit(textos)
        self._ajustado = True

    def encode(self, textos: list[str]) -> np.ndarray:
        if not self._ajustado:
            self.ajustar(textos)
        return _normalizar(self._vec.transform(textos).toarray() + 1e-9)


def crear_codificador(tipo: str, cache: Path | None):
    return {"h200": CodificadorH200, "tfidf": CodificadorTfidf}[tipo](cache=cache)


class IndiceVectorial:
    """Qdrant en memoria, con la misma API que `RagPipeline.index/retrieve` del Taller 02."""

    def __init__(self, codificador, coleccion: str = "graphrag"):
        self.codificador = codificador
        self.coleccion = coleccion
        self.cliente = QdrantClient(":memory:")
        self.ids: list[str] = []

    def indexar(self, ids: list[str], textos: list[str], cargas: list[dict]) -> None:
        self.codificador.ajustar(textos)
        vectores = self.codificador.encode(textos)
        if self.cliente.collection_exists(self.coleccion):
            self.cliente.delete_collection(self.coleccion)
        self.cliente.create_collection(
            collection_name=self.coleccion,
            vectors_config=models.VectorParams(size=len(vectores[0]),
                                               distance=models.Distance.COSINE))
        self.cliente.upsert(collection_name=self.coleccion, points=[
            models.PointStruct(id=i, vector=v.tolist(), payload={"nodo": nid, **carga})
            for i, (nid, v, carga) in enumerate(zip(ids, vectores, cargas))])
        self.ids = list(ids)

    def buscar(self, consulta: str, k: int, filtro_tipo: str | None = None) -> list[dict]:
        vector = self.codificador.encode([consulta])[0].tolist()
        filtro = None
        if filtro_tipo:
            filtro = models.Filter(must=[models.FieldCondition(
                key="tipo", match=models.MatchValue(value=filtro_tipo))])
        r = self.cliente.query_points(collection_name=self.coleccion, query=vector,
                                      limit=k, with_payload=True, query_filter=filtro)
        return [{"nodo": p.payload["nodo"], "score": round(p.score, 4)} for p in r.points]
