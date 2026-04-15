"""
IncidentOps Environment.

An OpenEnv environment for training AI agents on real-world Site Reliability
Engineering (SRE) incident response tasks. The agent acts as an on-call
engineer who must triage, diagnose, and resolve production incidents through a
realistic text-based terminal interface.

Quick start (synchronous):

    from incident_ops_env import IncidentAction, IncidentObservation, IncidentOpsEnv

    with IncidentOpsEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset()
        obs = result.observation
        print(obs.output)

        result = env.step(IncidentAction(command="alerts"))
        obs = result.observation
        print(obs.output)

        result = env.step(IncidentAction(command="logs payment-processor"))
        result = env.step(IncidentAction(command="restart payment-processor"))
        result = env.step(IncidentAction(command="resolve"))
        print(f"Score: {result.reward:.3f}")

Quick start (async):

    import asyncio
    from incident_ops_env import IncidentAction, IncidentOpsEnv

    async def main():
        async with IncidentOpsEnv(base_url="http://localhost:8000") as env:
            result = await env.reset()
            result = await env.step(IncidentAction(command="alerts"))
            result = await env.step(IncidentAction(command="resolve"))

    asyncio.run(main())
"""

from .client import IncidentOpsEnv
from .models import IncidentAction, IncidentObservation

__all__ = ["IncidentAction", "IncidentObservation", "IncidentOpsEnv"]
__version__ = "1.0.0"
