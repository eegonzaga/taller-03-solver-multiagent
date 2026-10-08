"""El acceso al LLM: un servidor (la H200 o un modelo de guion) y un cliente que cobra,
reintenta y deja traza.

- `ServidorH200`: el vLLM de la H200 (puerto 12555, réplica 12559). **El id del modelo no
  se escribe: se lee de `/v1/models`.** Solo biblioteca estándar, como `h200.py` del kit.
- `GuionLLM`: un «modelo» que responde con un guion fijo. Sirve para hacer saltar los frenos
  de la Parte 3 sin gastar tokens.
- `ClienteLLM`: lo que usan los agentes. Antes de cada llamada consulta el presupuesto
  (FRENO 1); si el modelo agota `max_tokens` razonando y devuelve vacío
  (`finish_reason=length`), reintenta con más cupo; y registra cada llamada (REGLA 6).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Callable

from .traza import Traza


# ─────────────────────────────────────────────────────────────── servidores
class ServidorH200:
    HOST = os.getenv("H200_HOST", "172.28.230.10")
    PUERTOS = [int(p) for p in os.getenv("H200_PUERTOS", "12555,12559").split(",")]

    def __init__(self, timeout: float = 900):
        self.timeout = timeout
        self.puerto = self.PUERTOS[0]
        self.modelo = self._pedir("v1/models", timeout=10)["data"][0]["id"]

    def _pedir(self, ruta: str, cuerpo: dict | None = None, timeout: float | None = None) -> dict:
        pet = urllib.request.Request(
            f"http://{self.HOST}:{self.puerto}/{ruta}",
            data=None if cuerpo is None else json.dumps(cuerpo).encode(),
            # vLLM no valida la clave; «local» no es una credencial
            headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
        try:
            with urllib.request.urlopen(pet, timeout=timeout or self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as err:
            raise RuntimeError(f"La H200 respondió {err.code}: {err.read()[:300]!r}") from err
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            raise ConnectionError(f"Sin respuesta de {self.HOST}:{self.puerto} ({err}). "
                                  "¿Está GlobalProtect conectada?") from err

    def cambiar_de_replica(self) -> None:
        i = self.PUERTOS.index(self.puerto)
        self.puerto = self.PUERTOS[(i + 1) % len(self.PUERTOS)]

    def chat(self, mensajes: list[dict], json_mode: bool = False, max_tokens: int = 16384,
             agente: str = "") -> dict:
        cuerpo = {"model": self.modelo, "messages": mensajes, "temperature": 0.0,
                  "max_tokens": max_tokens,
                  # El modelo razona siempre; esto solo separa el razonamiento del contenido.
                  "chat_template_kwargs": {"enable_thinking": True}}
        if json_mode:
            cuerpo["response_format"] = {"type": "json_object"}
        r = self._pedir("v1/chat/completions", cuerpo)
        eleccion = r["choices"][0]
        uso = r.get("usage", {})
        return {"contenido": eleccion["message"].get("content") or "",
                "fin": eleccion.get("finish_reason"),
                "tokens_entrada": uso.get("prompt_tokens", 0),
                "tokens_salida": uso.get("completion_tokens", 0)}


class GuionLLM:
    """Un modelo de guion: para cada agente, una lista de respuestas (se consumen en orden y
    la última se repite) o una función `f(mensajes) -> str`. Los tokens se estiman."""

    modelo = "guion"

    def __init__(self, guion: dict[str, list[str] | Callable[[list[dict]], str]]):
        self.guion = guion
        self._usadas: dict[str, int] = {}

    def chat(self, mensajes: list[dict], json_mode: bool = False, max_tokens: int = 16384,
             agente: str = "") -> dict:
        regla = self.guion.get(agente)
        if regla is None:
            raise KeyError(f"el guion no tiene respuestas para el agente «{agente}»")
        if callable(regla):
            texto = regla(mensajes)
        else:
            i = self._usadas.get(agente, 0)
            texto = regla[min(i, len(regla) - 1)]
            self._usadas[agente] = i + 1
        return {"contenido": texto, "fin": "stop",
                "tokens_entrada": estimar_tokens(" ".join(m["content"] for m in mensajes)),
                "tokens_salida": estimar_tokens(texto)}


# ─────────────────────────────────────────────────────────────── FRENO 1: presupuesto
class PresupuestoAgotado(Exception):
    """El orquestador la captura y pasa a redactar con lo que haya."""


class Presupuesto:
    """Tokens por corrida (entrada + salida). Se revisa ANTES de cada llamada con una
    estimación; la reserva solo la puede gastar el redactor, para que siempre haya entrega."""

    def __init__(self, total: int, reserva_redactor: int):
        self.total = total
        self.reserva = reserva_redactor
        self.usado = 0
        self._lock = threading.Lock()

    def autorizar(self, agente: str, estimacion: int) -> None:
        limite = self.total if agente == "redactor" else self.total - self.reserva
        if self.usado + estimacion > limite:
            raise PresupuestoAgotado(
                f"{agente}: usados {self.usado} + estimados {estimacion} > límite {limite} "
                f"(total {self.total}, reserva del redactor {self.reserva})")

    def cobrar(self, tokens: int) -> None:
        with self._lock:
            self.usado += tokens


def estimar_tokens(texto: str) -> int:
    return len(texto) // 3 + 1      # cota prudente para español y código


# ─────────────────────────────────────────────────────────────── cliente
class ClienteLLM:
    MAX_TOKENS_TOPE = 65536

    def __init__(self, servidor, traza: Traza, presupuesto: Presupuesto):
        self.servidor = servidor
        self.modelo = servidor.modelo
        self.traza = traza
        self.presupuesto = presupuesto
        self.uso_por_agente: dict[str, dict] = {}
        self._lock = threading.Lock()

    def pedir(self, agente: str, mensajes: list[dict], json_mode: bool = False,
              max_tokens: int = 16384, salida_estimada: int = 3000) -> str:
        estimacion = sum(estimar_tokens(m["content"]) for m in mensajes) + salida_estimada
        self.presupuesto.autorizar(agente, estimacion)          # FRENO 1: antes de llamar

        reintentos_red = 0
        while True:
            t0 = time.perf_counter()
            respuesta, error = None, None
            try:
                respuesta = self.servidor.chat(mensajes, json_mode=json_mode,
                                               max_tokens=max_tokens, agente=agente)
            except ConnectionError as err:
                error = str(err)
            except Exception as err:  # noqa: BLE001 — se registra y se re-lanza abajo
                error = f"{type(err).__name__}: {err}"
            latencia = round(time.perf_counter() - t0, 2)
            self._registrar(agente, mensajes, respuesta, error, latencia, max_tokens)

            if error:
                # Un corte de red se reintenta una vez, en la otra réplica si la hay.
                if reintentos_red == 0 and hasattr(self.servidor, "cambiar_de_replica"):
                    reintentos_red += 1
                    self.servidor.cambiar_de_replica()
                    time.sleep(3)
                    continue
                raise RuntimeError(f"LLM ({agente}): {error}")

            # El razonamiento sale del mismo cupo: si se acabó, el contenido llega vacío.
            if respuesta["fin"] == "length" and not respuesta["contenido"].strip() \
                    and max_tokens < self.MAX_TOKENS_TOPE:
                max_tokens = min(max_tokens * 2, self.MAX_TOKENS_TOPE)
                self.traza.registrar("reintento_llm", agente=agente,
                                     motivo="finish_reason=length con contenido vacío",
                                     nuevo_max_tokens=max_tokens)
                # FRENO 1 en el reintento: el modelo ya demostró que puede gastar todo el cupo,
                # así que se estima con el nuevo max_tokens, no con la salida típica. Con la
                # estimación típica, un reintento de 65 536 tokens se comió la reserva del
                # redactor (tarea R2, 2026-10-07).
                entrada = sum(estimar_tokens(m["content"]) for m in mensajes)
                self.presupuesto.autorizar(agente, entrada + max_tokens)
                continue
            return respuesta["contenido"]

    def pedir_json(self, agente: str, mensajes: list[dict], **kw) -> dict:
        """Una respuesta JSON. Si no se puede leer, se pide una vez más diciendo por qué."""
        texto = self.pedir(agente, mensajes, json_mode=True, **kw)
        try:
            return extraer_json(texto)
        except ValueError as err:
            self.traza.registrar("json_invalido", agente=agente, error=str(err))
            mensajes = mensajes + [{"role": "assistant", "content": texto[:4000]},
                                   {"role": "user", "content": "Tu respuesta no era un JSON "
                                    "válido. Devuelve solo el objeto JSON, sin texto alrededor."}]
            return extraer_json(self.pedir(agente, mensajes, json_mode=True, **kw))

    def _registrar(self, agente, mensajes, respuesta, error, latencia, max_tokens) -> None:
        ent = (respuesta or {}).get("tokens_entrada", 0)
        sal = (respuesta or {}).get("tokens_salida", 0)
        self.presupuesto.cobrar(ent + sal)
        with self._lock:
            uso = self.uso_por_agente.setdefault(
                agente, {"llamadas": 0, "tokens_entrada": 0, "tokens_salida": 0, "latencia_s": 0.0})
            uso["llamadas"] += 1
            uso["tokens_entrada"] += ent
            uso["tokens_salida"] += sal
            uso["latencia_s"] = round(uso["latencia_s"] + latencia, 2)
        evento = self.traza.registrar(
            "llm", agente=agente, modelo=self.modelo, tokens_entrada=ent, tokens_salida=sal,
            latencia_s=latencia, max_tokens=max_tokens,
            fin=(respuesta or {}).get("fin"), error=error,
            presupuesto_usado=self.presupuesto.usado,
            entrada=_recorte(mensajes[-1]["content"], 300),
            salida=_recorte((respuesta or {}).get("contenido", ""), 500))
        self.traza.guardar_intercambio(evento["n"], agente, mensajes, respuesta or {"error": error})

    @property
    def usage(self) -> dict:
        return {"tokens_entrada": sum(u["tokens_entrada"] for u in self.uso_por_agente.values()),
                "tokens_salida": sum(u["tokens_salida"] for u in self.uso_por_agente.values()),
                "por_agente": self.uso_por_agente}


# ─────────────────────────────────────────────────────────────── utilidades de respuesta
def extraer_json(texto: str) -> dict:
    """El primer objeto JSON del texto (el modelo a veces lo envuelve en ```json)."""
    texto = texto.strip()
    bloque = re.search(r"```(?:json)?\s*(\{.*\})\s*```", texto, re.S)
    if bloque:
        texto = bloque.group(1)
    inicio, fin = texto.find("{"), texto.rfind("}")
    if inicio < 0 or fin < 0:
        raise ValueError(f"no hay JSON en la respuesta: {texto[:200]!r}")
    try:
        return json.loads(texto[inicio:fin + 1])
    except json.JSONDecodeError as err:
        raise ValueError(f"JSON inválido ({err}): {texto[:200]!r}") from err


def extraer_codigo(texto: str) -> str:
    """El bloque ```python más largo; si no hay bloques, el texto entero."""
    bloques = re.findall(r"```(?:python|py)?\s*\n(.*?)```", texto, re.S)
    return (max(bloques, key=len) if bloques else texto).strip() + "\n"


def _recorte(texto: str, n: int) -> str:
    texto = " ".join(str(texto).split())
    return texto if len(texto) <= n else texto[:n] + " …"
