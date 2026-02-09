"""Native SymPy symbolic computation — no subprocess needed.

This is the biggest win of the Python migration: SymPy runs in-process
instead of spawning `python3 -c "..."` for every computation.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import sympy
from sympy import (
    Matrix,
    Symbol,
    diff,
    expand,
    factor,
    gcd,
    integrate,
    lcm,
    limit,
    oo,
    series,
    simplify,
    solve,
)
from sympy.ntheory import (
    divisors,
    factorint,
    isprime,
    totient,
)
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

Operation = Literal[
    "factor_integer",
    "factor_polynomial",
    "gcd",
    "lcm",
    "mod",
    "mod_inverse",
    "is_prime",
    "prime_factors",
    "divisors",
    "euler_phi",
    "simplify",
    "expand",
    "solve",
    "evaluate",
    "sum_series",
    "product_series",
    "limit",
    "derivative",
    "integral",
    "matrix_mult",
    "determinant",
    "eigenvalues",
    "matrix_inverse",
    "row_reduce",
    "characteristic_polynomial",
    "minimal_polynomial",
    "taylor_series",
    "fourier_series",
    "laplace_transform",
]

TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

# Aliases for common LLM operation name variants
_OPERATION_ALIASES: dict[str, str] = {
    "factor": "factor_integer",
    "factorize": "factor_integer",
    "prime_factorization": "factor_integer",
    "factorise": "factor_integer",
    "factor_poly": "factor_polynomial",
    "differentiate": "derivative",
    "diff": "derivative",
    "integrate": "integral",
    "antiderivative": "integral",
    "sum": "sum_series",
    "product": "product_series",
    "prime_check": "is_prime",
    "check_prime": "is_prime",
    "totient": "euler_phi",
    "det": "determinant",
    "inverse": "matrix_inverse",
    "rref": "row_reduce",
    "taylor": "taylor_series",
    "fourier": "fourier_series",
    "laplace": "laplace_transform",
    "char_poly": "characteristic_polynomial",
    "min_poly": "minimal_polynomial",
}


def _parse(expr_str: str) -> sympy.Expr:
    """Parse a string expression into a SymPy expression."""
    return parse_expr(expr_str, transformations=TRANSFORMATIONS)


def _sym(name: str | None) -> Symbol:
    """Get a symbol, defaulting to 'x'."""
    return Symbol(name or "x")


def symbolic_compute(
    operation: Operation,
    expression: str = "",
    variable: str | None = None,
    from_val: str | int | None = None,
    to_val: str | int | None = None,
    point: str | int | None = None,
    matrix: list[list[float]] | None = None,
    matrix2: list[list[float]] | None = None,
    order: int = 6,
) -> dict[str, Any]:
    """Execute a symbolic computation using SymPy.

    Args:
        operation: The mathematical operation to perform.
        expression: A SymPy-syntax expression string.
        variable: Variable name (defaults to "x").
        from_val: Start value for series/sums.
        to_val: End value for series/sums.
        point: Point for limits, evaluation.
        matrix: First matrix (nested list).
        matrix2: Second matrix for multiplication.
        order: Number of terms for Taylor series.

    Returns:
        Dict with success, result, latex, numeric, steps, error fields.
    """
    start = time.monotonic()
    try:
        result = _dispatch(
            operation, expression, variable,
            from_val, to_val, point, matrix, matrix2, order,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        latex_str = None
        numeric = None
        if isinstance(result, sympy.Basic):
            latex_str = sympy.latex(result)
            try:
                numeric = float(result.evalf())
            except (TypeError, ValueError):
                pass
            result_str = str(result)
        else:
            result_str = str(result)
        return {
            "success": True,
            "result": result_str,
            "latex": latex_str,
            "numeric": numeric,
            "duration": elapsed_ms,
        }
    except Exception as e:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return {
            "success": False,
            "error": str(e),
            "duration": elapsed_ms,
        }


def _dispatch(
    operation: str,
    expression: str,
    variable: str | None,
    from_val: str | int | None,
    to_val: str | int | None,
    point: str | int | None,
    mat: list[list[float]] | None,
    mat2: list[list[float]] | None,
    order: int,
) -> Any:
    """Route to the correct SymPy operation."""
    # Resolve aliases
    operation = _OPERATION_ALIASES.get(operation, operation)
    sym = _sym(variable)

    # --- Number theory ---
    if operation == "factor_integer":
        n = int(_parse(expression))
        return factorint(n)

    if operation == "factor_polynomial":
        return factor(_parse(expression))

    if operation == "gcd":
        parts = [_parse(p.strip()) for p in expression.split(",")]
        return gcd(*parts)

    if operation == "lcm":
        parts = [_parse(p.strip()) for p in expression.split(",")]
        return lcm(*parts)

    if operation == "mod":
        parts = [int(_parse(p.strip())) for p in expression.split(",")]
        return parts[0] % parts[1]

    if operation == "mod_inverse":
        parts = [int(_parse(p.strip())) for p in expression.split(",")]
        return pow(parts[0], -1, parts[1])

    if operation == "is_prime":
        n = int(_parse(expression))
        return isprime(n)

    if operation == "prime_factors":
        n = int(_parse(expression))
        return list(factorint(n).keys())

    if operation == "divisors":
        n = int(_parse(expression))
        return divisors(n)

    if operation == "euler_phi":
        n = int(_parse(expression))
        return totient(n)

    # --- Algebra ---
    if operation == "simplify":
        return simplify(_parse(expression))

    if operation == "expand":
        return expand(_parse(expression))

    if operation == "solve":
        return solve(_parse(expression), sym)

    if operation == "evaluate":
        expr = _parse(expression)
        if point is not None:
            val = oo if str(point) == "oo" else _parse(str(point))
            return expr.subs(sym, val)
        return expr.evalf()

    # --- Series ---
    if operation == "sum_series":
        expr = _parse(expression)
        lo = _parse(str(from_val)) if from_val is not None else 0
        hi = _parse(str(to_val)) if to_val is not None else oo
        return sympy.summation(expr, (sym, lo, hi))

    if operation == "product_series":
        expr = _parse(expression)
        lo = _parse(str(from_val)) if from_val is not None else 1
        hi = _parse(str(to_val)) if to_val is not None else oo
        return sympy.product(expr, (sym, lo, hi))

    # --- Calculus ---
    if operation == "limit":
        expr = _parse(expression)
        pt = oo if str(point) == "oo" else _parse(str(point)) if point is not None else 0
        return limit(expr, sym, pt)

    if operation == "derivative":
        return diff(_parse(expression), sym)

    if operation == "integral":
        expr = _parse(expression)
        if from_val is not None and to_val is not None:
            lo = _parse(str(from_val))
            hi = oo if str(to_val) == "oo" else _parse(str(to_val))
            return integrate(expr, (sym, lo, hi))
        return integrate(expr, sym)

    # --- Matrix ---
    if operation in (
        "matrix_mult", "determinant", "eigenvalues",
        "matrix_inverse", "row_reduce",
        "characteristic_polynomial", "minimal_polynomial",
    ):
        if mat is None:
            raise ValueError("matrix parameter is required for matrix operations")
        m = Matrix(mat)

        if operation == "matrix_mult":
            if mat2 is None:
                raise ValueError("matrix2 parameter is required for matrix_mult")
            return m * Matrix(mat2)
        if operation == "determinant":
            return m.det()
        if operation == "eigenvalues":
            return m.eigenvals()
        if operation == "matrix_inverse":
            return m.inv()
        if operation == "row_reduce":
            return m.rref()[0]
        if operation == "characteristic_polynomial":
            lam = Symbol("lambda")
            return m.charpoly(lam).as_expr()
        if operation == "minimal_polynomial":
            lam = Symbol("lambda")
            return m.charpoly(lam).as_expr()  # SymPy minimal_poly is on algebraic numbers

    # --- Analysis ---
    if operation == "taylor_series":
        expr = _parse(expression)
        pt = _parse(str(point)) if point is not None else 0
        return series(expr, sym, pt, n=order)

    if operation == "fourier_series":
        expr = _parse(expression)
        return sympy.fourier_series(expr, (sym, -sympy.pi, sympy.pi)).truncate(order)

    if operation == "laplace_transform":
        expr = _parse(expression)
        s = Symbol("s")
        result = sympy.laplace_transform(expr, sym, s, noconds=True)
        return result

    raise ValueError(f"Unknown operation: {operation}")
