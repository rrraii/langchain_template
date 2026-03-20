import ast
import operator
from datetime import datetime

from langchain.tools import tool

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_expression(expression: str) -> float | int:
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            return _BINARY_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression")

    parsed = ast.parse(expression, mode="eval")
    return _eval(parsed)


@tool
def get_current_time() -> str:
    """Get the current local time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression such as 12 * (4 + 3)."""
    try:
        return str(_safe_eval_expression(expression))
    except Exception as exc:
        raise ValueError(f"Invalid arithmetic expression: {expression}") from exc


COMMON_TOOLS = [get_current_time, calculator]
