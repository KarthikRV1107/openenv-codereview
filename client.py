"""
IncidentOps Environment Client.

Provides the client for connecting to a running IncidentOps environment server.
IncidentOpsEnv extends EnvClient with type-safe action/observation handling.

Example (sync):

    with IncidentOpsEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset()
        result = env.step(IncidentAction(command="alerts"))

Example (async):

    async with IncidentOpsEnv(base_url="https://your-space.hf.space") as env:
        result = await env.reset()
        result = await env.step(IncidentAction(command="status"))

Example (from Docker):

    env = IncidentOpsEnv.from_docker_image("incident-ops-env:latest")
    try:
        env.reset()
        env.step(IncidentAction(command="alerts"))
    finally:
        env.close()
"""

from openenv.core.env_client import EnvClient

from .models import IncidentAction, IncidentObservation


class IncidentOpsEnv(EnvClient):
    """
    Client for the IncidentOps SRE Incident Response Environment.

    Connects to a running IncidentOps server and provides type-safe wrappers
    around the standard OpenEnv reset() / step() / state() interface.

    The agent interacts using text commands that mirror a real ops terminal.
    See IncidentAction and IncidentObservation in models.py for the full
    action/observation schema.

    Attributes:
        base_url: URL of the running environment server.

    Usage:
        with IncidentOpsEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset(task_name="service-restart")
            while not result.observation.done:
                action = IncidentAction(command="alerts")
                result = env.step(action)
            print(f"Final score: {result.reward:.3f}")
    """

    action_class = IncidentAction
    observation_class = IncidentObservation
