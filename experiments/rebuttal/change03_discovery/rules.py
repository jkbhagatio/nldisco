"""Checkable rules on the behavior variables: a small, safe boolean expression language.

A rule is a Python-style expression over the variable names in `data.VARIABLES`,
numbers, comparisons, `and` / `or` / `not`, arithmetic, `abs(...)` and
`hypot(a, b)`, e.g. ``wheel_speed > 0.05 and patch_distance <= 100``.
"""
from __future__ import annotations

import ast

import numpy as np
from beartype import beartype

from experiments.rebuttal.change03_discovery.data import CAMERA_VARIABLES, VARIABLES

_COMPARE = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater,
            ast.GtE: np.greater_equal, ast.Eq: np.equal, ast.NotEq: np.not_equal}
_BINARY = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
           ast.Div: np.divide, ast.Pow: np.power, ast.Mod: np.mod}
_FUNCTIONS = {"abs": (np.abs, 1), "hypot": (np.hypot, 2)}


class RuleError(ValueError):
    """The rule is not a valid expression of the rule language."""


@beartype
def names_used(rule: str) -> set:
    """Variable names a rule refers to; raises RuleError if the rule is invalid."""
    used: set = set()
    _evaluate(_parse(rule), {name: np.zeros(1) for name in VARIABLES}, used)
    return used


def _parse(rule: str) -> ast.AST:
    try:
        return ast.parse(rule.strip(), mode="eval").body
    except SyntaxError as error:
        raise RuleError(f"Cannot parse rule: {error.msg}") from error


def _evaluate(node: ast.AST, variables: dict, used: set):
    if isinstance(node, ast.BoolOp):
        parts = [_boolean(_evaluate(value, variables, used)) for value in node.values]
        combine = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        return combine.reduce(parts)
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate(node.operand, variables, used)
        if isinstance(node.op, ast.Not):
            return ~_boolean(operand)
        if isinstance(node.op, (ast.USub, ast.UAdd)):
            return -_number(operand) if isinstance(node.op, ast.USub) else _number(operand)
    if isinstance(node, ast.Compare):
        left = _number(_evaluate(node.left, variables, used))
        result = None
        for operator, comparator in zip(node.ops, node.comparators):
            if type(operator) not in _COMPARE:
                raise RuleError("Unsupported comparison")
            right = _number(_evaluate(comparator, variables, used))
            with np.errstate(invalid="ignore"):
                step = _COMPARE[type(operator)](left, right)
            result = step if result is None else result & step
            left = right
        return result
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        with np.errstate(all="ignore"):
            return _BINARY[type(node.op)](_number(_evaluate(node.left, variables, used)),
                                          _number(_evaluate(node.right, variables, used)))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _FUNCTIONS and not node.keywords:
        function, arity = _FUNCTIONS[node.func.id]
        if len(node.args) != arity:
            raise RuleError(f"{node.func.id} takes {arity} argument(s)")
        return function(*[_number(_evaluate(arg, variables, used)) for arg in node.args])
    if isinstance(node, ast.Name):
        if node.id not in VARIABLES:
            raise RuleError(f"Unknown variable '{node.id}'. Allowed: {', '.join(VARIABLES)}")
        used.add(node.id)
        return variables[node.id]
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return float(node.value)
    raise RuleError(f"Unsupported expression element: {type(node).__name__}")


def _boolean(value):
    if not (isinstance(value, np.ndarray) and value.dtype == bool):
        raise RuleError("'and', 'or' and 'not' need comparisons, e.g. speed > 5")
    return value


def _number(value):
    if isinstance(value, np.ndarray) and value.dtype == bool:
        raise RuleError("A comparison cannot be used as a number")
    return value


@beartype
def evaluate(rule: str, variables: dict, support: dict) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a rule over windows.

    Args:
        rule: expression in the rule language.
        variables: endpoint variables (name -> float array over windows).
        support: 'camera' and 'wheel' 20-bin support masks over windows.

    Returns:
        (mask, valid): where the rule holds, and the windows it is compared within
        (complete tracking support if a camera variable is used, complete wheel
        support if wheel speed is used). Undefined values make a comparison false.
    """
    used: set = set()
    mask = _evaluate(_parse(rule), variables, used)
    if not (isinstance(mask, np.ndarray) and mask.dtype == bool):
        raise RuleError("The rule must be a comparison or a combination of comparisons")
    if not used:
        raise RuleError("The rule must use at least one behavior variable")
    valid = np.ones(len(mask), dtype=bool)
    if used & set(CAMERA_VARIABLES):
        valid &= support["camera"]
    if "wheel_speed" in used:
        valid &= support["wheel"]
    return mask & valid, valid
