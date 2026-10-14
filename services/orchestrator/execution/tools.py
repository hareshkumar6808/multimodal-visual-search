from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any

import sqlparse
import sympy


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    success: bool
    result: Any = None
    normalized_input: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class UnsafeExpressionError(ValueError):
    pass


_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS = {
    "sqrt": sympy.sqrt,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "log": sympy.log,
    "exp": sympy.exp,
    "Abs": sympy.Abs,
}


def _sympy_from_ast(node: ast.AST, symbols: dict[str, sympy.Symbol]) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return sympy.sympify(node.value)
    if isinstance(node, ast.Name):
        if node.id.startswith("_"):
            raise UnsafeExpressionError("private or dunder names are forbidden")
        return symbols.setdefault(node.id, sympy.Symbol(node.id))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](
            _sympy_from_ast(node.left, symbols), _sympy_from_ast(node.right, symbols)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_sympy_from_ast(node.operand, symbols))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*[_sympy_from_ast(arg, symbols) for arg in node.args])
    raise UnsafeExpressionError(f"forbidden expression node: {type(node).__name__}")


def _parse_expression(text: str, symbols: dict[str, sympy.Symbol]) -> Any:
    normalized = text.strip().replace("^", "**")
    normalized = re.sub(r"(?<=\d)(?=[A-Za-z_(])", "*", normalized)
    tree = ast.parse(normalized, mode="eval")
    return _sympy_from_ast(tree.body, symbols)


def math_tool(operation: str, expression: str, variable: str = "x") -> ToolResult:
    started = perf_counter()
    normalized = {"operation": operation, "expression": expression, "variable": variable}
    try:
        symbols: dict[str, sympy.Symbol] = {}
        if operation == "solve":
            left, separator, right = expression.partition("=")
            if not separator:
                right = "0"
            expr = sympy.Eq(_parse_expression(left, symbols), _parse_expression(right, symbols))
            result = sympy.solve(expr, symbols.setdefault(variable, sympy.Symbol(variable)))
        else:
            expr = _parse_expression(expression, symbols)
            operations = {
                "arithmetic": lambda: sympy.simplify(expr),
                "simplify": lambda: sympy.simplify(expr),
                "factor": lambda: sympy.factor(expr),
                "expand": lambda: sympy.expand(expr),
                "differentiate": lambda: sympy.diff(
                    expr, symbols.setdefault(variable, sympy.Symbol(variable))
                ),
                "integrate": lambda: sympy.integrate(
                    expr, symbols.setdefault(variable, sympy.Symbol(variable))
                ),
            }
            if operation not in operations:
                raise ValueError(f"unsupported math operation: {operation}")
            result = operations[operation]()
        return ToolResult(
            "math", True, str(result), normalized, duration_ms=(perf_counter() - started) * 1000
        )
    except (SyntaxError, TypeError, ValueError, UnsafeExpressionError) as exc:
        return ToolResult(
            "math",
            False,
            normalized_input=normalized,
            error=str(exc),
            duration_ms=(perf_counter() - started) * 1000,
        )


def table_tool(operation: str, values: list[Any]) -> ToolResult:
    started = perf_counter()
    try:
        if operation == "count":
            result: int | float = len(values)
        else:
            numeric = [float(value) for value in values]
            if not numeric:
                raise ValueError("no numeric values")
            functions: dict[str, Callable[[list[float]], float]] = {
                "sum": sum,
                "mean": lambda items: sum(items) / len(items),
                "average": lambda items: sum(items) / len(items),
                "min": min,
                "max": max,
            }
            if operation not in functions:
                raise ValueError(f"unsupported table operation: {operation}")
            result = functions[operation](numeric)
        return ToolResult(
            "table",
            True,
            result,
            {"operation": operation, "values": values},
            duration_ms=(perf_counter() - started) * 1000,
        )
    except (TypeError, ValueError) as exc:
        return ToolResult(
            "table",
            False,
            normalized_input={"operation": operation, "values": values},
            error=str(exc),
            duration_ms=(perf_counter() - started) * 1000,
        )


def python_static_tool(source: str) -> ToolResult:
    started = perf_counter()
    try:
        tree = ast.parse(source)
        result = {"valid": True, "top_level_nodes": [type(node).__name__ for node in tree.body]}
        return ToolResult(
            "python_static",
            True,
            result,
            source,
            metadata={"executed": False},
            duration_ms=(perf_counter() - started) * 1000,
        )
    except SyntaxError as exc:
        result = {"valid": False, "line": exc.lineno, "offset": exc.offset, "message": exc.msg}
        return ToolResult(
            "python_static",
            True,
            result,
            source,
            metadata={"executed": False},
            duration_ms=(perf_counter() - started) * 1000,
        )


def sql_static_tool(source: str) -> ToolResult:
    started = perf_counter()
    statements = sqlparse.parse(source)
    result = [
        {
            "type": statement.get_type(),  # type: ignore[no-untyped-call]
            "tokens": [str(token) for token in statement.tokens if not token.is_whitespace],
            "formatted": sqlparse.format(str(statement), reindent=True, keyword_case="upper"),
        }
        for statement in statements
    ]
    return ToolResult(
        "sql_static",
        bool(statements),
        result,
        source,
        None if statements else "no SQL statement",
        {
            "executed": False,
            "validation_scope": "structural; not database-specific syntax validation",
        },
        (perf_counter() - started) * 1000,
    )
