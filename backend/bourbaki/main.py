"""Uvicorn entry point for Bourbaki backend."""

from bourbaki.server.app import create_app

app = create_app()
