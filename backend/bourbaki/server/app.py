"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bourbaki.server.routes.autonomous import router as autonomous_router
from bourbaki.server.routes.compute import router as compute_router
from bourbaki.server.routes.export import router as export_router
from bourbaki.server.routes.health import router as health_router
from bourbaki.server.routes.problems import router as problems_router
from bourbaki.server.routes.prove import router as prove_router
from bourbaki.server.routes.query import router as query_router
from bourbaki.server.routes.search import router as search_router
from bourbaki.server.routes.sessions import router as sessions_router
from bourbaki.server.routes.skills import router as skills_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Bourbaki",
        description="Mathematical reasoning agent backend",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(compute_router)
    app.include_router(prove_router)
    app.include_router(search_router)
    app.include_router(export_router)
    app.include_router(query_router)
    app.include_router(sessions_router)
    app.include_router(problems_router)
    app.include_router(autonomous_router)
    app.include_router(skills_router)

    return app
