"""Mathematical problem database — ported from src/problems/database.ts.

13 classic problems for practice and benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MathProblem:
    id: str
    title: str
    statement: str
    domain: str
    techniques: list[str]
    difficulty: int  # 1-5
    tags: list[str]
    status: str = "solved"  # solved, open, partially-solved
    famous: bool = False
    source: str | None = None
    hints: list[str] = field(default_factory=list)
    oeis: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "statement": self.statement,
            "domain": self.domain,
            "techniques": self.techniques,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "status": self.status,
            "famous": self.famous,
            "source": self.source,
            "hints": self.hints,
        }


# ── Number Theory ────────────────────────────────────────────────────────────

_NUMBER_THEORY = [
    MathProblem(
        "sum-of-integers", "Sum of First n Integers",
        "Prove that 1 + 2 + ... + n = n(n+1)/2 for all n >= 1.",
        "number-theory", ["induction"], 1, ["induction", "natural-numbers", "sum"],
    ),
    MathProblem(
        "sum-of-squares", "Sum of First n Squares",
        "Prove that 1² + 2² + ... + n² = n(n+1)(2n+1)/6 for all n >= 1.",
        "number-theory", ["induction"], 2, ["induction", "natural-numbers", "sum"],
    ),
    MathProblem(
        "euclid-primes", "Infinitude of Primes",
        "Prove that there are infinitely many prime numbers.",
        "number-theory", ["contradiction"], 2, ["primes", "infinity"], famous=True, source="Euclid",
    ),
    MathProblem(
        "sqrt2-irrational", "Irrationality of √2",
        "Prove that √2 is irrational.",
        "number-theory", ["contradiction"], 2, ["irrationality", "real-numbers"], famous=True,
    ),
    MathProblem(
        "bezout-identity", "Bézout's Identity",
        "Prove that for any integers a, b, there exist integers x, y such that ax + by = gcd(a,b).",
        "number-theory", ["strong-induction"], 3, ["gcd", "linear-combination"],
    ),
    MathProblem(
        "fermat-little", "Fermat's Little Theorem",
        "Prove that if p is prime and gcd(a,p) = 1, then a^(p-1) ≡ 1 (mod p).",
        "number-theory", ["induction"], 3, ["modular-arithmetic", "primes"], famous=True,
    ),
]

# ── Combinatorics ────────────────────────────────────────────────────────────

_COMBINATORICS = [
    MathProblem(
        "pigeonhole-basic", "Socks in a Drawer",
        "A drawer has red and blue socks. What is the minimum number you must draw to guarantee a pair?",
        "combinatorics", ["pigeonhole"], 1, ["pigeonhole", "existence"],
    ),
    MathProblem(
        "handshaking-lemma", "Handshaking Lemma",
        "Prove that in any graph, the sum of vertex degrees equals twice the number of edges.",
        "combinatorics", ["counting"], 2, ["graph-theory", "counting"],
    ),
    MathProblem(
        "ramsey-r33", "R(3,3) = 6",
        "Prove that R(3,3) = 6: any 2-coloring of K_6 contains a monochromatic triangle.",
        "combinatorics", ["pigeonhole", "cases"], 3, ["ramsey", "graph-theory"], famous=True,
    ),
    MathProblem(
        "erdos-ko-rado", "Erdős–Ko–Rado Theorem",
        "If F is a family of k-element subsets of {1,...,n} with n >= 2k, all pairwise intersecting, then |F| <= C(n-1,k-1).",
        "combinatorics", ["counting", "contradiction"], 4, ["extremal", "set-theory"],
        famous=True, source="Erdős, Ko, Rado 1961",
    ),
]

# ── Open Problems ────────────────────────────────────────────────────────────

_OPEN = [
    MathProblem(
        "goldbach", "Goldbach's Conjecture",
        "Every even integer greater than 2 can be expressed as the sum of two primes.",
        "number-theory", [], 5, ["conjecture", "primes", "additive"], status="open", famous=True,
    ),
    MathProblem(
        "twin-primes", "Twin Prime Conjecture",
        "There are infinitely many pairs of primes (p, p+2).",
        "number-theory", [], 5, ["conjecture", "primes", "gaps"], status="open", famous=True,
    ),
    MathProblem(
        "collatz", "Collatz Conjecture",
        "For any positive integer n, the sequence defined by n → n/2 (if even) or n → 3n+1 (if odd) eventually reaches 1.",
        "number-theory", [], 5, ["conjecture", "dynamics"], status="open", famous=True,
    ),
]

ALL_PROBLEMS: list[MathProblem] = _NUMBER_THEORY + _COMBINATORICS + _OPEN
_PROBLEM_MAP: dict[str, MathProblem] = {p.id: p for p in ALL_PROBLEMS}


def get_problem(problem_id: str) -> MathProblem | None:
    return _PROBLEM_MAP.get(problem_id)


def get_problems_by_domain(domain: str) -> list[MathProblem]:
    return [p for p in ALL_PROBLEMS if p.domain == domain]


def get_problems_by_technique(technique: str) -> list[MathProblem]:
    return [p for p in ALL_PROBLEMS if technique in p.techniques]


def get_problems_by_difficulty(min_d: int, max_d: int) -> list[MathProblem]:
    return [p for p in ALL_PROBLEMS if min_d <= p.difficulty <= max_d]


def get_famous_problems() -> list[MathProblem]:
    return [p for p in ALL_PROBLEMS if p.famous]


def get_random_problem(
    domain: str | None = None,
    difficulty: int | None = None,
    technique: str | None = None,
) -> MathProblem | None:
    import random
    candidates = ALL_PROBLEMS
    if domain:
        candidates = [p for p in candidates if p.domain == domain]
    if difficulty:
        candidates = [p for p in candidates if p.difficulty == difficulty]
    if technique:
        candidates = [p for p in candidates if technique in p.techniques]
    return random.choice(candidates) if candidates else None


def format_problem(problem: MathProblem, include_hints: bool = False) -> str:
    """Format a problem as markdown."""
    lines = [
        f"## {problem.title}",
        "",
        f"**Domain:** {problem.domain}",
        f"**Difficulty:** {'★' * problem.difficulty}{'☆' * (5 - problem.difficulty)}",
        f"**Techniques:** {', '.join(problem.techniques) or 'unknown'}",
        f"**Status:** {problem.status}",
        "",
        problem.statement,
    ]
    if include_hints and problem.hints:
        lines.extend(["", "**Hints:**"])
        for hint in problem.hints:
            lines.append(f"- {hint}")
    return "\n".join(lines)
