# tools/python_executor.py
from __future__ import annotations

import ast
import contextlib
import io
import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict

import importlib


class UnsafeCodeError(ValueError):
    pass


# ✅ Single source of truth:
# Add/remove allowed top-level modules here.
# NOTE: Allowing third‑party libs (e.g. sympy) means you are trusting that library's code.
ALLOWED_IMPORTS = {
    "math",
    "cmath",
    "statistics",
    "fractions",
    "decimal",
    "sympy",
    "numpy"
}


_BANNED_NAMES = {
    # builtins / reflection / io / code execution
    "open", "exec", "eval", "compile", "input",
    "__import__", "globals", "locals", "vars", "dir", "help",
    "getattr", "setattr", "delattr", "hasattr",
    # "escape hatch" primitives
    "type", "object", "super",
}

_BANNED_NODES = (
    # NOTE: We allow *limited* imports via ALLOWED_IMPORTS (see visitor + safe_import below).
    ast.With, ast.AsyncWith,
    ast.Try, ast.Raise,
    ast.ClassDef,
    ast.FunctionDef, ast.AsyncFunctionDef,
    ast.Lambda,
    ast.Global, ast.Nonlocal,
)


class _SafetyVisitor(ast.NodeVisitor):
    def generic_visit(self, node: ast.AST) -> Any:
        if isinstance(node, _BANNED_NODES):
            raise UnsafeCodeError(f"Disallowed syntax: {type(node).__name__}")
        return super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id.startswith("__") or node.id in _BANNED_NAMES:
            raise UnsafeCodeError(f"Disallowed name: {node.id}")
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        # blocks __class__, __mro__, __subclasses__, etc.
        if node.attr.startswith("__"):
            raise UnsafeCodeError(f"Disallowed attribute: {node.attr}")
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        # blocks calling banned names, even if referenced
        if isinstance(node.func, ast.Name) and node.func.id in _BANNED_NAMES:
            raise UnsafeCodeError(f"Disallowed call: {node.func.id}()")
        if isinstance(node.func, ast.Attribute) and node.func.attr.startswith("__"):
            raise UnsafeCodeError(f"Disallowed call: .{node.func.attr}()")
        return self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> Any:
        # Allow only imports from ALLOWED_IMPORTS (top-level allowlist).
        for alias in node.names:
            base = alias.name.split(".")[0]
            if base not in ALLOWED_IMPORTS:
                raise UnsafeCodeError(f"Import not allowed: {alias.name}")
        return self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        # Disallow relative imports and allow only known safe/allowed modules.
        if node.level and node.level != 0:
            raise UnsafeCodeError("Relative imports not allowed")
        module = node.module or ""
        base = module.split(".")[0] if module else ""
        if base not in ALLOWED_IMPORTS:
            raise UnsafeCodeError(f"Import not allowed: {module}")
        return self.generic_visit(node)


def _auto_capture_last_expr(tree: ast.Module) -> ast.Module:
    """If last stmt is expression, rewrite to: result = <expr>"""
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        tree.body[-1] = ast.Assign(
            targets=[ast.Name(id="result", ctx=ast.Store())],
            value=tree.body[-1].value,
            type_comment=None,
        )
        ast.fix_missing_locations(tree)
    return tree


def _jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, complex):
        return {"__complex__": True, "re": x.real, "im": x.imag}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    return repr(x)


def _worker(code: str, out_q: mp.Queue) -> None:
    start = time.time()
    stdout_io = io.StringIO()
    stderr_io = io.StringIO()

    # Safe importer used by Python's `import` statement inside the sandbox.
    # - Only allows top-level modules in ALLOWED_IMPORTS
    # - Imports the *requested* module path when needed (important for "from sympy.solvers import ...")
    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level and level != 0:
            raise ImportError("Relative imports not allowed")

        name = str(name)
        base = name.split(".")[0]
        if base not in ALLOWED_IMPORTS:
            raise ImportError(f"Import not allowed: {name}")

        # If it's `from x.y import z` then fromlist is non-empty and Python expects x.y module.
        target = name if fromlist else base
        return importlib.import_module(target)

    safe_builtins = {
        # harmless builtins
        "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "len": len,
        "range": range, "enumerate": enumerate, "sorted": sorted, "pow": pow, "print": print,

        # basic types
        "int": int, "float": float, "complex": complex, "str": str, "bool": bool,
        "list": list, "tuple": tuple, "dict": dict, "set": set,

        "__import__": safe_import,
    }

    # Pre-populate common modules if available (optional convenience).
    # This is NOT required for imports to work, because safe_import handles them.
    preloaded: dict[str, Any] = {}
    for mod in sorted(ALLOWED_IMPORTS):
        try:
            preloaded[mod] = importlib.import_module(mod)
        except Exception:
            # If it's allowed but not installed, imports will raise ImportError later.
            pass

    env: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        **preloaded,
    }

    try:
        tree = ast.parse(code, mode="exec")
        _SafetyVisitor().visit(tree)
        tree = _auto_capture_last_expr(tree)

        compiled = compile(tree, "<python_calculator>", "exec")

        with contextlib.redirect_stdout(stdout_io), contextlib.redirect_stderr(stderr_io):
            exec(compiled, env, env)

        payload = {
            "ok": True,
            "result": _jsonable(env.get("result", None)),
            "stdout": stdout_io.getvalue(),
            "stderr": stderr_io.getvalue(),
            "runtime_ms": int((time.time() - start) * 1000),
        }
        out_q.put(payload)

    except Exception as e:
        out_q.put({
            "ok": False,
            "error": str(e),
            "stdout": stdout_io.getvalue(),
            "stderr": stderr_io.getvalue(),
            "traceback": traceback.format_exc(),
            "runtime_ms": int((time.time() - start) * 1000),
        })


@dataclass
class SafePythonExecutor:
    default_timeout_s: float = 8.0

    def execute(self, code: str, timeout_s: float | None = None) -> dict[str, Any]:
        timeout_s = float(timeout_s or self.default_timeout_s)

        out_q: mp.Queue = mp.Queue(maxsize=1)
        proc = mp.Process(target=_worker, args=(code, out_q), daemon=True)
        proc.start()
        proc.join(timeout=timeout_s)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=0.2)
            return {
                "ok": False,
                "error": f"Timed out after {timeout_s:.2f}s",
                "result": None,
                "stdout": "",
                "stderr": "",
                "runtime_ms": int(timeout_s * 1000),
            }

        try:
            return out_q.get_nowait()
        except queue.Empty:
            return {
                "ok": False,
                "error": "No output received from sandbox process",
                "result": None,
                "stdout": "",
                "stderr": "",
                "runtime_ms": 0,
            }
