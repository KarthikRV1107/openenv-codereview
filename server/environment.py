"""
IncidentOps Core Environment.

Implements the OpenEnv Environment base class for the IncidentOps SRE
incident-response training environment.

Episode lifecycle:
  1. reset(task_name=...) — loads a scenario and returns the initial briefing
  2. step(action)         — executes a command, returns observation + reward
  3. state()              — returns current State (episode_id, step_count)

The environment terminates (done=True) when:
  - The agent calls 'resolve'
  - step_count reaches scenario.max_steps (episode timeout)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.interfaces import Environment  # type: ignore[no-redef]
    from openenv.core.env_server.types import Action, Observation, State

try:
    from models import IncidentAction, IncidentObservation
    from server.graders import grade
    from server.scenarios import ALL_TASK_NAMES, Scenario, get_scenario
    from server.simulation import SimulationEngine
except ImportError:
    from ..models import IncidentAction, IncidentObservation
    from .graders import grade
    from .scenarios import ALL_TASK_NAMES, Scenario, get_scenario
    from .simulation import SimulationEngine


# Default task if none specified at reset
DEFAULT_TASK = os.getenv("INCIDENT_TASK", "service-restart")


# ---------------------------------------------------------
#  Strict open-interval clamp for ALL rewards/scores.
#  Duplicated here as a second defense layer.
# ---------------------------------------------------------

def _safe_reward(value: float) -> float:
    """
    Clamp reward to the OPEN interval (0.01, 0.99).

    The OpenEnv Phase 2 validator rejects scores that are exactly 0.0 or 1.0.
    We never allow a reward of 0 or 1 to leave this module.
    """
    v = float(value)
    if v <= 0.0:
        return 0.01
    if v >= 1.0:
        return 0.99
    if v < 0.01:
        return 0.01
    if v > 0.99:
        return 0.99
    return v


class IncidentOpsEnvironment(Environment):
    """
    IncidentOps: AI Incident Response Training Environment.

    The agent acts as an on-call SRE engineer, interacting with a simulated
    production system through a text-based terminal interface.

    Three tasks of increasing difficulty are supported:
      - service-restart   (easy)
      - config-drift      (medium)
      - cascading-failure (hard)

    Each task is self-contained and can be selected at reset time by passing
    task_name in the reset request body:
        {"task_name": "config-drift"}

    Attributes:
        SUPPORTS_CONCURRENT_SESSIONS: False — environment is stateful per session.
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self) -> None:
        self._state: State = State(episode_id=str(uuid4()), step_count=0)
        self._scenario: Optional[Scenario] = None
        self._engine: Optional[SimulationEngine] = None
        self._done: bool = False
        self._final_score: float = 0.5
        self._task_name: str = DEFAULT_TASK
        self._cumulative_reward: float = 0.0

    # --- Core API -------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs: Any,
    ) -> IncidentObservation:
        """
        Reset the environment for a new episode.

        Args:
            seed:       Optional seed for reproducible scenario generation.
            episode_id: Optional custom episode identifier.
            task_name:  Which task to load. One of:
                        "service-restart" | "config-drift" | "cascading-failure"
                        Defaults to the INCIDENT_TASK env var or "service-restart".
            **kwargs:   Ignored extra keyword arguments.

        Returns:
            IncidentObservation with initial briefing text and alert counts.
        """
        self._task_name = task_name or DEFAULT_TASK

        if self._task_name not in ALL_TASK_NAMES:
            valid = ", ".join(ALL_TASK_NAMES)
            self._task_name = DEFAULT_TASK
            briefing = (
                f"  [WARNING] Unknown task_name. Valid options: {valid}\n"
                f"Loading default task: '{self._task_name}'.\n"
            )
        else:
            briefing = ""

        self._scenario = get_scenario(self._task_name, seed=seed)
        self._engine = SimulationEngine(self._scenario)
        self._done = False
        self._cumulative_reward = 0.0
        self._final_score = 0.5
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        briefing += self._build_briefing()

        # Reset reward must be returned as a valid open interval float.
        # The Phase 1 validator crashes if it receives null/None here.
        return IncidentObservation(
            output=briefing,
            timestamp=self._sim_time(),
            alert_count=len(self._engine.active_alerts),
            severity=self._engine.current_severity,
            affected_services=self._engine.affected_services,
            done=False,
            reward=_safe_reward(0.5),
            metadata={
                "task_name": self._task_name,
                "episode_id": self._state.episode_id,
                "difficulty": self._scenario.difficulty,
                "max_steps": self._scenario.max_steps,
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> IncidentObservation:
        """
        Execute one agent action.

        Args:
            action:    An IncidentAction with a text command string.
            timeout_s: Ignored (commands are synchronous and fast).
            **kwargs:  Extra keyword arguments are ignored.

        Returns:
            IncidentObservation with command output, updated metrics, and reward.
        """
        if self._done:
            return self._terminal_observation("Episode already finished. Call reset() to start a new one.")

        if self._scenario is None or self._engine is None:
            return self._terminal_observation("Environment not initialised. Call reset() first.")

        # Extract command string from action
        if isinstance(action, IncidentAction):
            command = action.command
        elif isinstance(action, dict):
            command = action.get("command", "")
        else:
            command = str(getattr(action, "command", action))

        self._state.step_count += 1
        step_num = self._state.step_count

        # Execute command in simulation
        output, incremental_reward, done_by_resolve = self._engine.execute(command)

        # Check episode termination
        timeout_reached = step_num >= self._scenario.max_steps
        self._done = done_by_resolve or timeout_reached

        # Compute reward — always pass through _safe_reward
        if self._done:
            self._final_score = grade(
                scenario=self._scenario,
                engine=self._engine,
                step_count=step_num,
            )
            # _safe_reward is defense-in-depth (grader already clamps)
            reward = _safe_reward(self._final_score)
        else:
            # Incremental reward centred at 0.5, clamped to (0.01, 0.99)
            reward = _safe_reward(0.5 + incremental_reward)

        self._cumulative_reward += incremental_reward

        # Append timeout notice if applicable
        if timeout_reached and not done_by_resolve:
            output += (
                f"\n\n  [TIMEOUT] Episode timeout reached ({self._scenario.max_steps} steps).\n"
                f"      Final score computed based on actions taken so far.\n"
            )

        return IncidentObservation(
            output=output,
            timestamp=self._sim_time(offset_minutes=step_num),
            alert_count=len(self._engine.active_alerts),
            severity=self._engine.current_severity,
            affected_services=self._engine.affected_services,
            done=self._done,
            reward=reward,
            metadata={
                "task_name": self._task_name,
                "episode_id": self._state.episode_id,
                "step": step_num,
                "command": command,
                "final_score": _safe_reward(self._final_score) if self._done else None,
            },
        )

    @property
    def state(self) -> State:
        """Return the current episode state."""
        return self._state

    def close(self) -> None:
        """Release any resources (nothing to release in this implementation)."""
        pass

    # --- Helpers ---------------------------------------------

    def _build_briefing(self) -> str:
        """Build the initial incident briefing shown to the agent."""
        s = self._scenario
        e = self._engine

        lines = [
            "",
            "+------------------------------------------------------------------╗",
            "|          IncidentOps — Incident Response Environment              |",
            "+------------------------------------------------------------------╝",
            "",
           f"  [Task]         {s.display_name}",
           f"  [Difficulty]   {s.difficulty.upper()}",
           f"  [Sim-Time]     {s.start_time}",
           f"  [Alerts]       {len(e.active_alerts)} active",
           f"  [Severity]     {e.current_severity.upper()}",
            "",
           f"  DESCRIPTION:",
        ]
        # Wrap description text
        import textwrap
        for line in textwrap.wrap(s.description, width=66):
            lines.append(f"  {line}")

        lines += [
            "",
            "  -------------------------------------------------------------",
            "  You are the on-call engineer. Triage the incident, find the",
            "  root cause, remediate it, and type 'resolve' when done.",
            "",
            "  Type 'help' to see all available commands.",
            "  Type 'alerts' to see active alerts.",
            "  Type 'status' for a system-wide dashboard.",
            "  -------------------------------------------------------------",
            "",
        ]
        return "\n".join(lines)

    def _sim_time(self, offset_minutes: int = 0) -> str:
        """Return a simulated ISO-8601 timestamp."""
        if self._scenario:
            base = self._scenario.start_time  # e.g. "2026-04-10T04:00:00Z"
            try:
                from datetime import timedelta
                dt = datetime.fromisoformat(base.replace("Z", "+00:00"))
                dt += timedelta(minutes=offset_minutes)
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                pass
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _terminal_observation(self, message: str) -> IncidentObservation:
        """Return a terminal observation for error/edge cases."""
        return IncidentObservation(
            output=f"\n  {message}\n",
            timestamp=self._sim_time(),
            alert_count=0,
            severity="none",
            affected_services=[],
            done=True,
            reward=_safe_reward(0.5),   # Always use _safe_reward
            metadata={},
        )
