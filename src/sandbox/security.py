import ast
from typing import Optional


FORBIDDEN_IMPORTS = {
    "os", "subprocess", "sys", "socket", "requests", "urllib",
    "httpx", "aiohttp", "ftplib", "smtplib", "telnetlib",
    "paramiko", "fabric", "shutil", "pathlib", "glob",
    "importlib", "ctypes", "cffi", "mmap", "signal",
    "threading", "multiprocessing", "concurrent",
    "pickle", "shelve", "marshal", "builtins",
}

ALLOWED_IMPORTS = {
    "polars", "numpy", "scipy", "statsmodels", "plotly",
    "json", "math", "statistics", "datetime", "re",
    "collections", "itertools", "functools", "typing",
    "warnings", "logging", "pprint", "copy", "time",
    "pandas", "scikit_posthocs", "sklearn",
}

FORBIDDEN_FUNCTIONS = {
    "eval", "exec", "compile", "input",
    "__import__", "breakpoint", "memoryview",
    "vars", "dir", "globals", "locals", "getattr",
    "setattr", "delattr",
}


class SecurityViolation(Exception):
    pass


class ASTSecurityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            root_module = alias.name.split(".")[0]
            if root_module in FORBIDDEN_IMPORTS:
                self.violations.append(
                    f"Forbidden import detected: '{alias.name}' at line {node.lineno}"
                )
            elif root_module not in ALLOWED_IMPORTS:
                self.violations.append(
                    f"Unrecognized import: '{alias.name}' at line {node.lineno} — not in allowlist"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module is None:
            self.generic_visit(node)
            return
        root_module = node.module.split(".")[0]
        if root_module in FORBIDDEN_IMPORTS:
            self.violations.append(
                f"Forbidden import detected: 'from {node.module}' at line {node.lineno}"
            )
        elif root_module not in ALLOWED_IMPORTS:
            self.violations.append(
                f"Unrecognized import: 'from {node.module}' at line {node.lineno} — not in allowlist"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_FUNCTIONS:
                self.violations.append(
                    f"Forbidden function call: '{node.func.id}()' at line {node.lineno}"
                )
        if isinstance(node.func, ast.Attribute):
            dangerous_attrs = {
                "system", "popen", "run", "call", "Popen",
                "check_output", "spawn", "execv", "execve",
            }
            if node.func.attr in dangerous_attrs:
                full_call = f"{getattr(node.func.value, 'id', '?')}.{node.func.attr}"
                self.violations.append(
                    f"Potentially dangerous method call: '{full_call}()' at line {node.lineno}"
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        dangerous_dunder = {
            "__class__", "__bases__", "__subclasses__",
            "__globals__", "__builtins__", "__code__",
        }
        if node.attr in dangerous_dunder:
            self.violations.append(
                f"Dangerous attribute access: '.{node.attr}' at line {node.lineno}"
            )
        self.generic_visit(node)


def check_code_safety(code: str) -> tuple[bool, list[str]]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"Syntax error in generated code: {e}"]

    visitor = ASTSecurityVisitor()
    visitor.visit(tree)

    if visitor.violations:
        return False, visitor.violations

    return True, []


def enforce_safety(code: str) -> str:
    is_safe, violations = check_code_safety(code)
    if not is_safe:
        violation_str = "\n".join(f"  - {v}" for v in violations)
        raise SecurityViolation(
            f"Code failed security check with {len(violations)} violation(s):\n{violation_str}"
        )
    return code