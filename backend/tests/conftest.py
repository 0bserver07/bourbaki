"""Shared fixtures for Bourbaki backend tests."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from bourbaki.server.app import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
