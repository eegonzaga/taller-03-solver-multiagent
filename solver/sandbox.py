"""El SANDBOX y el agente EJECUTOR (código, sin LLM).

REGLA 3 — El código corre en un sandbox:
  1. Se revisa ANTES de ejecutar (`revisar_codigo`, con el AST): sin red, sin procesos, sin
     borrar, sin eval/exec y sin salir de su carpeta.
  2. Se ejecuta SIN variables de entorno heredadas (`entorno_limpio`): ninguna clave llega
     al script, ni siquiera por accidente.
  3. En su propia carpeta y con un tiempo máximo que mata TODOS sus procesos.

FRENO 3 — Tiempo máximo: el script arranca en su propia sesión (`start_new_session`) y, al
vencer el tiempo, `os.killpg` mata el grupo entero, no solo el proceso padre.

FRENO 4 — Confirmación humana antes de usar la red: un script que quiere descargar datos
(`fetch_openml`, `load_dataset`, `download=True`, una URL…) no se rechaza sin más ni corre
sin más: lo decide una persona (`confirmar_red`).

Límite honesto: la revisión es estática. Un script ofuscado podría esquivarla; por eso,
además, el entorno va vacío y la carpeta es desechable. Un aislamiento de verdad pediría un
contenedor sin red.
"""
from __future__ import annotations

import ast
import hashlib
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

MODULOS_PROHIBIDOS = {
    "socket", "subprocess", "multiprocessing", "requests", "urllib", "urllib3", "http",
    "httpx", "aiohttp", "ftplib", "smtplib", "telnetlib", "paramiko", "ctypes", "shutil",
    "signal", "pty", "webbrowser", "concurrent", "importlib", "pickle", "dotenv",
}
LLAMADAS_PROHIBIDAS = {"eval", "exec", "compile", "__import__", "breakpoint", "input"}
# Prohibidos con cualquier receptor (Path(...).unlink(), os.system(...)): nombres que no
# chocan con métodos comunes de listas o cadenas.
ATRIBUTOS_PROHIBIDOS = {
    "system", "popen", "unlink", "rmdir", "removedirs", "rmtree", "kill", "killpg",
    "fork", "forkpty", "execv", "execve", "execvp", "spawnl", "spawnv", "startfile",
    "chdir", "getenv", "putenv", "chmod",
}
# Prohibidos solo sobre el módulo os (lista.remove() o texto.replace() son legítimos).
ATRIBUTOS_DE_OS = {"remove", "rename", "replace", "walk", "scandir", "listdir"}
# FRENO 4: lo que delata una descarga.
RED = re.compile(r"^(fetch_\w+|load_dataset|urlretrieve|urlopen|download\w*|hf_hub_download)$")


@dataclass
class Revision:
    problemas: list[str] = field(default_factory=list)      # bloquean la ejecución
    usa_red: list[str] = field(default_factory=list)        # piden confirmación humana

    @property
    def permitido(self) -> bool:
        return not self.problemas


def _nombre_llamada(nodo: ast.Call) -> str:
    f = nodo.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def revisar_codigo(codigo: str) -> Revision:
    """REGLA 3, antes de ejecutar. Solo AST: no se ejecuta nada para revisar."""
    rev = Revision()
    try:
        arbol = ast.parse(codigo)
    except SyntaxError as err:
        rev.problemas.append(f"error de sintaxis en la línea {err.lineno}: {err.msg}")
        return rev

    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.Import, ast.ImportFrom)):
            nombres = [a.name for a in nodo.names] if isinstance(nodo, ast.Import) else [nodo.module or ""]
            for n in nombres:
                if n.split(".")[0] in MODULOS_PROHIBIDOS:
                    rev.problemas.append(f"línea {nodo.lineno}: importa «{n}» (red, procesos o archivos del sistema)")
        elif isinstance(nodo, ast.Call):
            nombre = _nombre_llamada(nodo)
            receptor = nodo.func.value.id if isinstance(nodo.func, ast.Attribute) \
                and isinstance(nodo.func.value, ast.Name) else ""
            if isinstance(nodo.func, ast.Name) and nombre in LLAMADAS_PROHIBIDAS:
                rev.problemas.append(f"línea {nodo.lineno}: usa {nombre}()")
            elif isinstance(nodo.func, ast.Attribute) and (
                    nombre in ATRIBUTOS_PROHIBIDOS or (receptor == "os" and nombre in ATRIBUTOS_DE_OS)):
                rev.problemas.append(f"línea {nodo.lineno}: llama .{nombre}() (borrar, procesos, entorno o salir de la carpeta)")
            if RED.match(nombre):
                rev.usa_red.append(f"línea {nodo.lineno}: {nombre}()")
            for kw in nodo.keywords:
                if kw.arg == "download" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    rev.usa_red.append(f"línea {nodo.lineno}: download=True")
        elif isinstance(nodo, ast.Attribute) and nodo.attr == "environ":
            rev.problemas.append(f"línea {nodo.lineno}: lee os.environ")
        elif isinstance(nodo, ast.Constant) and isinstance(nodo.value, str):
            v = nodo.value.strip()
            if re.match(r"^https?://", v):
                rev.usa_red.append(f"línea {nodo.lineno}: URL {v[:60]}")
            elif re.match(r"^(/[A-Za-z]|~|[A-Za-z]:\\)", v) or re.search(r"(^|[/\\])\.\.([/\\]|$)", v):
                rev.problemas.append(f"línea {nodo.lineno}: ruta fuera de la carpeta «{v[:60]}»")
    rev.usa_red = sorted(set(rev.usa_red))
    return rev


def huella(codigo: str) -> str:
    """FRENO 2 (detector de repetición): el mismo programa aunque cambien comentarios o
    espacios produce la misma huella, porque se compara el AST re-impreso."""
    try:
        normal = ast.unparse(ast.parse(codigo))
    except SyntaxError:
        normal = "\n".join(l.strip() for l in codigo.splitlines() if l.strip())
    return hashlib.sha256(normal.encode()).hexdigest()[:16]


def entorno_limpio(carpeta: Path, cache_mpl: Path) -> dict[str, str]:
    """REGLA 3: el entorno se construye desde cero. Nada se hereda de `os.environ`, así
    que ninguna clave (API, tokens, la del .env) puede llegar al script ni a sus logs."""
    return {"PATH": "/usr/bin:/bin", "HOME": str(carpeta), "LANG": "C.UTF-8",
            "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0",
            "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(cache_mpl),
            "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}


@dataclass
class Ejecucion:
    codigo_salida: int | None
    duracion_s: float
    archivos_creados: list[str]
    stdout: str
    stderr: str
    timeout: bool = False


def _archivos(carpeta: Path) -> dict[str, float]:
    return {str(p.relative_to(carpeta)): p.stat().st_mtime for p in carpeta.rglob("*")
            if p.is_file() and ".mpl" not in p.parts}


def ejecutar(script: Path, timeout_s: int, cache_mpl: Path) -> Ejecucion:
    """EJECUTOR: corre `script` en su carpeta, sin entorno y con tiempo máximo."""
    carpeta = script.parent
    cache_mpl.mkdir(parents=True, exist_ok=True)
    antes = _archivos(carpeta)
    t0 = time.perf_counter()
    proc = subprocess.Popen([sys.executable, "-I", script.name], cwd=carpeta,
                            env=entorno_limpio(carpeta, cache_mpl),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)          # su propio grupo de procesos
    vencido = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        vencido = True                                       # FRENO 3: se mata el grupo entero
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        stderr += f"\n[sandbox] tiempo máximo de {timeout_s} s alcanzado: grupo de procesos terminado"
    duracion = round(time.perf_counter() - t0, 2)
    (carpeta / "salida.log").write_text(stdout, encoding="utf-8")     # evidencia de la ejecución
    if stderr.strip():
        (carpeta / "errores.log").write_text(stderr, encoding="utf-8")
    despues = _archivos(carpeta)
    creados = sorted(p for p, t in despues.items()
                     if (p not in antes or antes[p] != t) and p not in {"salida.log", "errores.log"})
    return Ejecucion(codigo_salida=None if vencido else proc.returncode, duracion_s=duracion,
                     archivos_creados=creados, stdout=stdout, stderr=stderr, timeout=vencido)


def confirmar_por_consola(subtarea: str, motivos: list[str], codigo: str) -> bool:
    """FRENO 4 por defecto: pregunta a la persona que corre el solver. Sin terminal (p. ej.
    dentro del evaluador en segundo plano) nadie puede aprobar, y se deniega."""
    if not sys.stdin or not sys.stdin.isatty():
        return False
    print(f"\n[FRENO 4] La subtarea {subtarea} quiere usar la red:")
    for m in motivos:
        print(f"   - {m}")
    return input("¿Permitir la ejecución? [s/N] ").strip().lower() in {"s", "si", "sí", "y", "yes"}
