"""
FastAPI application entry point for the IncidentOps server.

Creates and exposes the IncidentOpsEnvironment over HTTP/WebSocket
endpoints as required by the OpenEnv specification.

Running locally (without Docker):
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

Running via uv:
    uv run --project . server

Endpoints (provided by openenv-core):
    POST /reset           Reset environment, returns initial observation
    POST /step            Execute an action, returns observation + reward
    GET  /state           Returns current episode state
    GET  /health          Health check
    GET  /schema          JSON schemas for Action/Observation/State
    GET  /metadata        Environment description and capabilities
    WS   /ws              WebSocket for streaming interaction
    GET  /web             Browser-based debug UI (if ENABLE_WEB_INTERFACE=true)
"""

from __future__ import annotations

# Support both in-repo and standalone (pip-installed) imports
try:
    from openenv.core.env_server.http_server import create_app
    from .environment import IncidentOpsEnvironment
    from ..models import IncidentAction, IncidentObservation
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from server.environment import IncidentOpsEnvironment

    import sys
    import os
    # Make the root package importable when running from /app/env in Docker
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from models import IncidentAction, IncidentObservation


# Pass the class (not an instance) so the server creates per-session instances.
# This is required for WebSocket-based multi-step rollouts.
app = create_app(
    IncidentOpsEnvironment,
    IncidentAction,
    IncidentObservation,
    env_name="incident_ops_env",
)

@app.get("/")
def read_root():
    # Hugging Face Spaces proxy strips absolute redirects like /metadata, causing 404s.
    # By returning the environment metadata directly on the root endpoint, 
    # the browser will immediately display the rich JSON payload.
    return {
        "name": "incident_ops_env",
        "description": "IncidentOps SRE training environment.",
        "version": "1.0.0",
        "spec": "openenv",
        "tasks": [
            "service-restart",
            "config-drift",
            "cascading-failure"
        ]
    }

def main() -> None:
    """
    Entry point for direct execution.

    Enables:
        python -m server.app
        uv run --project . server
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
