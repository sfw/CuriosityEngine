"""Calculator tool — AST-based safe arithmetic + financial formulas.

Ported from loom/src/loom/tools/calculator.py, adapted to the sync Tool ABC.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from engine.tools.base import Tool, ToolError

_MAX_EXPONENT = 10_000
_MAX_EXPRESSION_LEN = 2_000


def _safe_pow(base, exp):
    if isinstance(exp, (int, float)) and abs(exp) > _MAX_EXPONENT:
        raise ValueError(f"exponent too large: {exp} (max {_MAX_EXPONENT})")
    return operator.pow(base, exp)


def _npv(rate: float, cashflows: list[float]) -> float:
    if rate <= -1:
        raise ValueError("rate must be greater than -1")
    return sum(cf / (1 + rate) ** i for i, cf in enumerate(cashflows))


def _cagr(beginning: float, ending: float, years: float) -> float:
    if beginning <= 0 or years <= 0:
        raise ValueError("beginning value and years must be positive")
    if ending < 0:
        raise ValueError("ending value cannot be negative")
    return (ending / beginning) ** (1 / years) - 1


def _wacc(equity: float, debt: float, cost_equity: float, cost_debt: float, tax_rate: float) -> float:
    total = equity + debt
    if total <= 0:
        raise ValueError("total capital must be positive")
    return (equity / total) * cost_equity + (debt / total) * cost_debt * (1 - tax_rate)


def _pmt(rate: float, nper: int, pv: float) -> float:
    if nper <= 0:
        raise ValueError("nper must be positive")
    if rate == 0:
        return -pv / nper
    return -pv * rate * (1 + rate) ** nper / ((1 + rate) ** nper - 1)


_SAFE_OPS: dict[str, Any] = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "len": len,
    "int": int, "float": float,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "log2": math.log2,
    "exp": math.exp, "pow": pow,
    "ceil": math.ceil, "floor": math.floor,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "pi": math.pi, "e": math.e, "tau": math.tau,
    "npv": _npv, "cagr": _cagr, "wacc": _wacc, "pmt": _pmt,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"unsupported constant: {type(node.value).__name__}")

    if isinstance(node, ast.Name):
        if node.id in _SAFE_OPS:
            return _SAFE_OPS[node.id]
        raise ValueError(f"unknown variable: {node.id}")

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}
        func = ops.get(type(node.op))
        if func is None:
            raise ValueError(f"unsupported unary op: {type(node.op).__name__}")
        return func(operand)

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        ops = {
            ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
            ast.Pow: _safe_pow,
        }
        func = ops.get(type(node.op))
        if func is None:
            raise ValueError(f"unsupported binary op: {type(node.op).__name__}")
        return func(left, right)

    if isinstance(node, ast.Call):
        func = _eval_node(node.func)
        if not callable(func):
            raise ValueError(f"not callable: {func}")
        args = [_eval_node(a) for a in node.args]
        return func(*args)

    if isinstance(node, ast.List):
        return [_eval_node(el) for el in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(el) for el in node.elts)

    raise ValueError(f"unsupported expression type: {type(node).__name__}")


def _safe_eval(expr: str):
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"invalid expression: {e}") from e
    return _eval_node(tree.body)


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Evaluate mathematical and financial expressions safely. Supports arithmetic "
        "(+, -, *, /, **, //, %), math functions (sqrt, log, log10, log2, exp, ceil, "
        "floor, abs, round, min, max, sum, sin, cos, tan, asin, acos, atan, atan2), "
        "constants (pi, e, tau), and financial functions: npv(rate, [cashflows]), "
        "cagr(start, end, years), wacc(equity, debt, cost_equity, cost_debt, tax_rate), "
        "pmt(rate, nper, pv). Numeric constants only (no strings)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Python-syntax math expression to evaluate.",
            },
        },
        "required": ["expression"],
    }
    timeout_seconds = 5.0

    def execute(self, args: dict) -> str:
        expression = (args.get("expression") or "").strip()
        if not expression:
            raise ToolError("no expression provided")
        if len(expression) > _MAX_EXPRESSION_LEN:
            raise ToolError(f"expression too long (max {_MAX_EXPRESSION_LEN} chars)")
        try:
            result = _safe_eval(expression)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
            raise ToolError(f"evaluation error: {e}") from e

        if isinstance(result, float):
            if result == int(result) and abs(result) < 1e15:
                formatted = str(int(result))
            else:
                formatted = f"{result:.10g}"
        else:
            formatted = str(result)
        return f"{expression} = {formatted}"
