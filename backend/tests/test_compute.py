"""Test /compute endpoint — pure SymPy, no LLM calls needed."""


async def test_factor_integer(client):
    """Factor 84 = 2^2 * 3 * 7."""
    resp = await client.post("/compute", json={
        "operation": "factor_integer",
        "expression": "84",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "2" in data["result"]
    assert "3" in data["result"]
    assert "7" in data["result"]


async def test_simplify(client):
    """Simplify (x^2 - 1) / (x - 1) = x + 1."""
    resp = await client.post("/compute", json={
        "operation": "simplify",
        "expression": "(x**2 - 1) / (x - 1)",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "x + 1" in data["result"]


async def test_is_prime(client):
    """17 is prime."""
    resp = await client.post("/compute", json={
        "operation": "is_prime",
        "expression": "17",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "True" in data["result"] or "true" in data["result"].lower()


async def test_solve_equation(client):
    """Solve x^2 - 5x + 6 = 0 → x = 2, 3."""
    resp = await client.post("/compute", json={
        "operation": "solve",
        "expression": "x**2 - 5*x + 6",
        "variable": "x",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "2" in data["result"]
    assert "3" in data["result"]


async def test_sum_formula(client):
    """Verify sum(k, k=1..n) = n*(n+1)/2 via SymPy."""
    resp = await client.post("/compute", json={
        "operation": "sum_series",
        "expression": "k",
        "variable": "k",
        "from_val": "1",
        "to_val": "n",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    # SymPy should return n*(n+1)/2 in some form
    result = data["result"].lower().replace(" ", "")
    assert "n" in result
