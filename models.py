"""
IncidentOps Environment Models.

Defines the typed Pydantic models for actions, observations, and state used
in the IncidentOps OpenEnv environment.

Action space:
    IncidentAction - A text command issued by the agent (e.g. "logs api-gateway")

Observation space:
    IncidentObservation - Rich terminal-style output plus structured metadata

State:
    Uses the base openenv State model (episode_id, step_count)
"""

from typing import Any, Dict, List, Optional

from pydantic import model_validator
from openenv.core.env_server.types import Action, Observation


# ─────────────────────────────────────────────────────────
#  Strict open-interval clamp for ALL rewards/scores.
#  The OpenEnv validator rejects 0.0 and 1.0 exactly.
#  We use 0.01 / 0.99 to stay safely inside (0, 1).
# ─────────────────────────────────────────────────────────

def _strict_clamp(value: float) -> float:
    """Clamp a float to the open interval (0.01, 0.99)."""
    if value <= 0.0:
        return 0.01
    if value >= 1.0:
        return 0.99
    # Also guard against values very close to the boundaries
    if value < 0.01:
        return 0.01
    if value > 0.99:
        return 0.99
    return float(value)


class IncidentAction(Action):
    """
    An action issued by the agent to the incident response environment.

    The agent interacts via a text command interface that mirrors a real
    ops terminal. Each command targets a specific investigation or
    remediation operation.

    Attributes:
        command: The full command string, including arguments.
                 Examples:
                   "help"
                   "logs api-gateway"
                   "metrics payment-processor"
                   "restart payment-processor"
                   "config api-gateway pool_size 100"
                   "resolve"

    Available top-level commands:
        help                          -- List all available commands
        status                        -- System-wide status dashboard
        alerts                        -- View active alerts with severity
        logs <service>                -- Recent log entries for a service
        metrics <service>             -- CPU / mem / latency / error-rate
        trace <request_id>            -- Follow a request through services
        diagnose <service>            -- Deep diagnostic report for service
        restart <service>             -- Restart a service pod
        scale <service> <replicas>    -- Scale service replica count
        rollback <service>            -- Roll back last deployment
        failover <service>            -- Trigger standby failover
        config <service> <key> <val>  -- Update a live config value
        notify <channel> <message>    -- Post status update
        resolve                       -- Mark incident resolved (ends episode)
    """

    command: str


class IncidentObservation(Observation):
    """
    Observation returned after each action.

    Attributes:
        output:            Terminal-style text output from the command.
        timestamp:         Simulated wall-clock time (ISO-8601 string).
        alert_count:       Number of currently active alerts.
        severity:          Current overall incident severity.
                           One of: "critical", "high", "medium", "low", "none".
        affected_services: List of service names showing degradation.
        done:              True when the episode has ended.
        reward:            Scalar reward in open interval (0.01, 0.99).
                           Clamped automatically by model validator.
        metadata:          Pass-through dict for additional info.
    """

    output: str
    timestamp: str
    alert_count: int = 0
    severity: str = "none"
    affected_services: List[str] = []

    @model_validator(mode="before")
    @classmethod
    def _enforce_reward_open_interval(cls, data: Any) -> Any:
        """
        Defense-in-depth: ensure reward is ALWAYS strictly within (0, 1).

        This validator fires on every construction of an IncidentObservation,
        guaranteeing that no matter what the environment logic produces,
        the serialized reward (via model_dump / HTTP JSON) will never be
        exactly 0.0 or 1.0.

        WHY mode="before" (not "after"):
          Pydantic v2 mode="after" validators are called with a fully-built
          model instance and MUST return `self` — returning a model_copy is
          silently discarded when validated via __init__.  mode="before" gives
          us the raw input dict/object before construction, so we can normalize
          the reward in-place.  The clamped value is then stored through
          Pydantic's proper field machinery, making it visible to model_dump()
          and therefore to the HTTP serialiser.
        """
        if isinstance(data, dict):
            r = data.get("reward")
            if r is not None:
                data = dict(data)          # don't mutate the caller's dict
                data["reward"] = _strict_clamp(float(r))
        return data
