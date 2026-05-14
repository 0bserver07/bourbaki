"""z.ai compatibility shim for pydantic_ai 1.56's
:meth:`ToolCallPart.args_as_dict`.

z.ai's Anthropic-compatible endpoint occasionally returns tool-use
responses whose ``input`` field is not a JSON string or a dict —
empirically we see lists, and probably other shapes too. pydantic_ai
stores that as ``ToolCallPart.args`` (typed ``str | dict | None``), and
on retry-loop message re-mapping calls ``args_as_dict`` which does::

    if not self.args: return {}
    if isinstance(self.args, dict): return self.args
    args = pydantic_core.from_json(self.args)   # crashes here

``from_json`` only accepts str/bytes/bytearray; anything else raises
``TypeError: Expected bytes, bytearray or str``. This module replaces
``args_as_dict`` with a version that handles any input shape gracefully.

Imported once at startup from :mod:`bourbaki.prover.__init__`. Delete
this file (and the import line) once upstream pydantic_ai ships a fix.
See GitHub issue #13.
"""

from __future__ import annotations

from typing import Any

import pydantic_core
from pydantic_ai.messages import ToolCallPart


def _safe_args_as_dict(self: ToolCallPart) -> dict[str, Any]:
    """Drop-in replacement for ``ToolCallPart.args_as_dict``.

    Coerces any input to a ``dict[str, Any]`` without raising:

    - ``None`` / empty → ``{}``
    - ``dict`` → returned as-is
    - ``str`` / ``bytes`` / ``bytearray`` → ``from_json`` (the original
      path); if it parses to a non-dict, wrap in ``{"_value": ...}``;
      if it fails to parse, wrap the raw payload
    - anything else (list, int, bool, …) → ``{"_raw": ...}``
    """
    a = self.args
    if not a:
        return {}
    if isinstance(a, dict):
        return a
    if isinstance(a, (str, bytes, bytearray)):
        try:
            parsed = pydantic_core.from_json(a)
        except Exception:
            return {"_raw": a}
        if isinstance(parsed, dict):
            return parsed
        return {"_value": parsed}
    return {"_raw": a}


# Apply once at module import. Idempotent — replacing the bound method
# with itself is a no-op.
ToolCallPart.args_as_dict = _safe_args_as_dict  # type: ignore[method-assign]
