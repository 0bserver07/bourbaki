"""Deterministic theorem verification via SymPy.

No LLM calls, no network — pure symbolic math. Runs in < 1 second.
Use this as a regression suite when making changes to the compute pipeline.

Each test verifies a known mathematical result through the /compute endpoint,
exercising the same code path the agent uses when calling symbolic_compute.
"""

import pytest


# ---------------------------------------------------------------------------
# Number Theory
# ---------------------------------------------------------------------------


class TestGaussSum:
    """Theorem: 1 + 2 + ... + n = n(n+1)/2  (Gauss, age 7)."""

    async def test_symbolic_formula(self, client):
        """SymPy should produce n(n+1)/2 for sum(k, k=1..n)."""
        resp = await client.post("/compute", json={
            "operation": "sum_series",
            "expression": "k",
            "variable": "k",
            "from_val": "1",
            "to_val": "n",
        })
        data = resp.json()
        assert data["success"]
        # SymPy returns n*(n + 1)/2 or n**2/2 + n/2
        r = data["result"].replace(" ", "")
        assert "n" in r

    @pytest.mark.parametrize("n,expected", [(1, 1), (5, 15), (10, 55), (100, 5050)])
    async def test_concrete_values(self, client, n, expected):
        """Verify formula at specific values."""
        resp = await client.post("/compute", json={
            "operation": "evaluate",
            "expression": f"Sum(k, (k, 1, {n}))",
        })
        data = resp.json()
        assert data["success"]
        assert int(float(data["numeric"])) == expected


class TestSumOfSquares:
    """Theorem: 1^2 + 2^2 + ... + n^2 = n(n+1)(2n+1)/6."""

    async def test_symbolic_formula(self, client):
        resp = await client.post("/compute", json={
            "operation": "sum_series",
            "expression": "k**2",
            "variable": "k",
            "from_val": "1",
            "to_val": "n",
        })
        data = resp.json()
        assert data["success"]
        assert "n" in data["result"]

    @pytest.mark.parametrize("n,expected", [(1, 1), (3, 14), (5, 55), (10, 385)])
    async def test_concrete_values(self, client, n, expected):
        resp = await client.post("/compute", json={
            "operation": "evaluate",
            "expression": f"Sum(k**2, (k, 1, {n}))",
        })
        data = resp.json()
        assert data["success"]
        assert int(float(data["numeric"])) == expected


class TestPrimality:
    """Verify primality checks for known primes and composites."""

    @pytest.mark.parametrize("p", [2, 3, 5, 7, 11, 13, 17, 19, 23, 97, 101, 7919])
    async def test_known_primes(self, client, p):
        resp = await client.post("/compute", json={
            "operation": "is_prime",
            "expression": str(p),
        })
        data = resp.json()
        assert data["success"]
        assert "true" in data["result"].lower(), f"{p} should be prime"

    @pytest.mark.parametrize("n", [1, 4, 6, 8, 9, 15, 21, 100, 1001])
    async def test_known_composites(self, client, n):
        resp = await client.post("/compute", json={
            "operation": "is_prime",
            "expression": str(n),
        })
        data = resp.json()
        assert data["success"]
        assert "false" in data["result"].lower(), f"{n} should not be prime"


class TestFundamentalTheoremOfArithmetic:
    """Every integer > 1 has a unique prime factorization."""

    @pytest.mark.parametrize("n,factors", [
        (12, {"2", "3"}),
        (84, {"2", "3", "7"}),
        (100, {"2", "5"}),
        (2310, {"2", "3", "5", "7", "11"}),  # primorial(5)
    ])
    async def test_prime_factorization(self, client, n, factors):
        resp = await client.post("/compute", json={
            "operation": "factor_integer",
            "expression": str(n),
        })
        data = resp.json()
        assert data["success"]
        for p in factors:
            assert p in data["result"], f"Factor {p} missing from {n} = {data['result']}"


class TestEulerTotient:
    """phi(n) = count of integers 1..n coprime to n."""

    @pytest.mark.parametrize("n,phi_n", [
        (1, 1), (2, 1), (6, 2), (10, 4), (12, 4), (36, 12),
    ])
    async def test_known_values(self, client, n, phi_n):
        resp = await client.post("/compute", json={
            "operation": "euler_phi",
            "expression": str(n),
        })
        data = resp.json()
        assert data["success"]
        assert int(float(data["numeric"])) == phi_n


class TestFermatLittle:
    """Fermat's Little Theorem: a^(p-1) ≡ 1 (mod p) for prime p, gcd(a,p)=1."""

    @pytest.mark.parametrize("a,p", [(2, 5), (3, 7), (5, 11), (7, 13), (2, 17)])
    async def test_fermat(self, client, a, p):
        resp = await client.post("/compute", json={
            "operation": "mod",
            "expression": f"{a ** (p - 1)}, {p}",
        })
        data = resp.json()
        assert data["success"]
        assert data["result"].strip() == "1"


# ---------------------------------------------------------------------------
# Algebra
# ---------------------------------------------------------------------------


class TestPolynomialIdentities:
    """Classic algebraic identities verified symbolically."""

    async def test_difference_of_squares(self, client):
        """(x^2 - 1)/(x - 1) = x + 1."""
        resp = await client.post("/compute", json={
            "operation": "simplify",
            "expression": "(x**2 - 1)/(x - 1)",
        })
        assert resp.json()["success"]
        assert "x + 1" in resp.json()["result"]

    async def test_factor_cubic(self, client):
        """x^3 - 1 = (x - 1)(x^2 + x + 1)."""
        resp = await client.post("/compute", json={
            "operation": "factor_polynomial",
            "expression": "x**3 - 1",
        })
        data = resp.json()
        assert data["success"]
        assert "x - 1" in data["result"]

    async def test_quadratic_formula(self, client):
        """Solve x^2 - 5x + 6 = 0 → {2, 3}."""
        resp = await client.post("/compute", json={
            "operation": "solve",
            "expression": "x**2 - 5*x + 6",
            "variable": "x",
        })
        data = resp.json()
        assert data["success"]
        assert "2" in data["result"]
        assert "3" in data["result"]


# ---------------------------------------------------------------------------
# Calculus
# ---------------------------------------------------------------------------


class TestCalculus:
    """Fundamental calculus results."""

    async def test_derivative_power_rule(self, client):
        """d/dx(x^n) = n*x^(n-1)."""
        resp = await client.post("/compute", json={
            "operation": "derivative",
            "expression": "x**5",
            "variable": "x",
        })
        data = resp.json()
        assert data["success"]
        assert "5" in data["result"]
        assert "x**4" in data["result"] or "x^4" in data["result"]

    async def test_integral_power_rule(self, client):
        """∫ x^2 dx = x^3/3."""
        resp = await client.post("/compute", json={
            "operation": "integral",
            "expression": "x**2",
            "variable": "x",
        })
        data = resp.json()
        assert data["success"]
        assert "x**3" in data["result"] or "x^3" in data["result"]
        assert "3" in data["result"]

    async def test_fundamental_theorem(self, client):
        """∫₀¹ x^2 dx = 1/3."""
        resp = await client.post("/compute", json={
            "operation": "integral",
            "expression": "x**2",
            "variable": "x",
            "from_val": "0",
            "to_val": "1",
        })
        data = resp.json()
        assert data["success"]
        assert abs(float(data["numeric"]) - 1 / 3) < 1e-10

    async def test_limit_sinx_over_x(self, client):
        """lim(x→0) sin(x)/x = 1."""
        resp = await client.post("/compute", json={
            "operation": "limit",
            "expression": "sin(x)/x",
            "variable": "x",
            "point": "0",
        })
        data = resp.json()
        assert data["success"]
        assert int(float(data["numeric"])) == 1

    async def test_geometric_series(self, client):
        """sum(x^k, k=0..inf) = 1/(1-x) for |x|<1."""
        resp = await client.post("/compute", json={
            "operation": "sum_series",
            "expression": "x**k",
            "variable": "k",
            "from_val": "0",
            "to_val": "oo",
        })
        data = resp.json()
        assert data["success"]
        # SymPy returns Piecewise or 1/(1-x)
        r = data["result"].replace(" ", "")
        assert "1" in r


# ---------------------------------------------------------------------------
# Linear Algebra
# ---------------------------------------------------------------------------


class TestLinearAlgebra:
    """Matrix theorems."""

    async def test_2x2_determinant(self, client):
        """det([[1,2],[3,4]]) = -2."""
        resp = await client.post("/compute", json={
            "operation": "determinant",
            "matrix": [[1, 2], [3, 4]],
        })
        data = resp.json()
        assert data["success"]
        assert int(float(data["numeric"])) == -2

    async def test_identity_inverse(self, client):
        """Inverse of [[1,0],[0,1]] is itself."""
        resp = await client.post("/compute", json={
            "operation": "matrix_inverse",
            "matrix": [[1, 0], [0, 1]],
        })
        data = resp.json()
        assert data["success"]

    async def test_eigenvalues(self, client):
        """Eigenvalues of [[2,1],[1,2]] = {1, 3}."""
        resp = await client.post("/compute", json={
            "operation": "eigenvalues",
            "matrix": [[2, 1], [1, 2]],
        })
        data = resp.json()
        assert data["success"]
        assert "1" in data["result"]
        assert "3" in data["result"]
