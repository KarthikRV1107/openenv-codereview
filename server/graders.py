"""
IncidentOps Task Graders.

Each grader evaluates the agent's performance on one task and produces a
deterministic score in the open interval (0.01, 0.99).

Grading rubric (applies to all tasks):
  - investigation (0.0–0.30): Credit for checking relevant logs/metrics/diagnose
  - root_cause    (0.0–0.30): Whether the agent correctly identified the root cause
  - remediation   (0.0–0.30): Whether the correct fix was applied to the correct service
  - communication (0.0–0.05): Whether the agent sent at least one notify update
  - efficiency    (0.0–0.05): Fewer unnecessary steps → higher bonus

Total is clamped to (0.01, 0.99).
"""

from __future__ import annotations

from typing import List

from .scenarios import Scenario
from .simulation import SimulationEngine


# ---------------------------------------------------------
#  Base grader
# ---------------------------------------------------------

def _clamp(score: float) -> float:
    """Clamp to open interval (0.01, 0.99)."""
    return max(0.01, min(0.99, score))


def _normalise_cmd(cmd: str) -> str:
    return cmd.strip().lower()


def _any_cmd_matches(actions: List[str], *prefixes: str) -> bool:
    """Return True if any taken action starts with one of the given prefixes."""
    normalised = [_normalise_cmd(a) for a in actions]
    for norm in normalised:
        for prefix in prefixes:
            if norm.startswith(prefix.lower()):
                return True
    return False


def _count_cmd_matches(actions: List[str], *prefixes: str) -> int:
    """Count how many taken actions match any of the given prefixes."""
    normalised = [_normalise_cmd(a) for a in actions]
    count = 0
    for norm in normalised:
        for prefix in prefixes:
            if norm.startswith(prefix.lower()):
                count += 1
                break
    return count


# ---------------------------------------------------------
#  Task graders
# ---------------------------------------------------------

def grade_service_restart(
    scenario: Scenario,
    engine: SimulationEngine,
    step_count: int,
) -> float:
    """
    Grade the 'service-restart' (easy) task.

    Max score breakdown:
      investigation  0.30  (check alerts, logs/metrics of payment-processor)
      root_cause     0.30  (looked at payment-processor specifically)
      remediation    0.30  (restarted payment-processor)
      communication  0.05  (at least one notify)
      efficiency     0.05  (finished in ≤ 6 steps)
    """
    actions = engine.actions_taken
    score = 0.0

    # -- Investigation (0.30) ----------------------------
    investigation = 0.0

    if _any_cmd_matches(actions, "alerts"):
        investigation += 0.05              # looked at alerts
    if _any_cmd_matches(actions, "status"):
        investigation += 0.03              # got an overview
    if _any_cmd_matches(actions, "logs payment-processor", "metrics payment-processor", "diagnose payment-processor"):
        investigation += 0.12              # checked the right service
    if _any_cmd_matches(actions, "logs api-gateway", "logs order-service"):
        investigation += 0.05              # checked downstream services
    if _any_cmd_matches(actions, "trace"):
        investigation += 0.05              # used distributed trace

    score += min(investigation, 0.30)

    # -- Root-cause identification (0.30) -----------------
    # Proxy: did the agent check payment-processor before restarting it?
    checked_root = _any_cmd_matches(
        actions,
        "logs payment-processor",
        "metrics payment-processor",
        "diagnose payment-processor",
    )
    score += 0.30 if checked_root else 0.10

    # -- Remediation (0.30) -------------------------------
    if _any_cmd_matches(actions, "restart payment-processor"):
        score += 0.30
    elif _any_cmd_matches(actions, "restart"):
        score += 0.05      # restarted something wrong

    # -- Communication (0.05) ------------------------------
    if _any_cmd_matches(actions, "notify"):
        score += 0.05

    # -- Efficiency (0.05) --------------------------------
    optimal_steps = 4
    penalty_per_extra = 0.005
    extras = max(0, step_count - optimal_steps)
    score += max(0.0, 0.05 - extras * penalty_per_extra)

    return _clamp(score)


def grade_config_drift(
    scenario: Scenario,
    engine: SimulationEngine,
    step_count: int,
) -> float:
    """
    Grade the 'config-drift' (medium) task.

    Max score breakdown:
      investigation  0.30  (check multiple services; identify api-gateway config)
      root_cause     0.30  (correctly identified api-gateway pool_size regression)
      remediation    0.30  (rollback or config fix on api-gateway)
      communication  0.05  (at least one notify)
      efficiency     0.05  (finished in ≤ 10 steps)
    """
    actions = engine.actions_taken
    score = 0.0

    # -- Investigation (0.30) ----------------------------
    investigation = 0.0

    if _any_cmd_matches(actions, "alerts"):
        investigation += 0.04
    if _any_cmd_matches(actions, "status"):
        investigation += 0.03
    if _any_cmd_matches(actions, "logs api-gateway", "metrics api-gateway", "diagnose api-gateway"):
        investigation += 0.10              # checked actual root cause service
    if _any_cmd_matches(actions, "logs checkout", "metrics checkout", "diagnose checkout"):
        investigation += 0.05
    if _any_cmd_matches(actions, "logs payment-processor", "metrics payment-processor"):
        investigation += 0.04
    if _any_cmd_matches(actions, "trace"):
        investigation += 0.04

    score += min(investigation, 0.30)

    # -- Root-cause identification (0.30) -----------------
    # Agent must have inspected api-gateway metrics/config to find pool_size
    inspected_root = _any_cmd_matches(
        actions,
        "metrics api-gateway",
        "diagnose api-gateway",
    )
    score += 0.30 if inspected_root else 0.05

    # -- Remediation (0.30) -------------------------------
    if _any_cmd_matches(actions, "rollback api-gateway"):
        score += 0.30
    elif _any_cmd_matches(actions, "config api-gateway pool_size 100"):
        score += 0.30
    elif _any_cmd_matches(actions, "config api-gateway"):
        score += 0.10      # tried to fix config but wrong value
    elif _any_cmd_matches(actions, "rollback"):
        score += 0.05      # rolled back wrong service

    # -- Communication (0.05) ------------------------------
    if _any_cmd_matches(actions, "notify"):
        score += 0.05

    # -- Efficiency (0.05) --------------------------------
    optimal_steps = 8
    penalty_per_extra = 0.003
    extras = max(0, step_count - optimal_steps)
    score += max(0.0, 0.05 - extras * penalty_per_extra)

    return _clamp(score)


def grade_cascading_failure(
    scenario: Scenario,
    engine: SimulationEngine,
    step_count: int,
) -> float:
    """
    Grade the 'cascading-failure' (hard) task.

    Max score breakdown:
      investigation  0.30  (must explore the full causal chain)
      root_cause     0.25  (identified database-primary disk I/O as root cause)
      remediation    0.30  (failover DB primary AND restart cache-layer)
      communication  0.05  (at least one notify)
      efficiency     0.10  (finished in ≤ 18 steps; bonus for identifying red herrings)
    """
    actions = engine.actions_taken
    score = 0.0

    # -- Investigation (0.30) ----------------------------
    investigation = 0.0

    if _any_cmd_matches(actions, "alerts"):
        investigation += 0.04
    if _any_cmd_matches(actions, "status"):
        investigation += 0.02
    if _any_cmd_matches(actions, "logs database-primary", "metrics database-primary", "diagnose database-primary"):
        investigation += 0.10              # found root cause service
    if _any_cmd_matches(actions, "logs cache-layer", "metrics cache-layer", "diagnose cache-layer"):
        investigation += 0.05
    if _any_cmd_matches(actions, "logs read-service", "metrics read-service"):
        investigation += 0.04
    if _any_cmd_matches(actions, "logs database-replica", "metrics database-replica"):
        investigation += 0.03
    if _any_cmd_matches(actions, "trace"):
        investigation += 0.04
    # Checking red-herring services (not penalised, just no bonus)

    score += min(investigation, 0.30)

    # -- Root-cause identification (0.25) -----------------
    identified_db = _any_cmd_matches(
        actions,
        "diagnose database-primary",
        "metrics database-primary",
        "logs database-primary",
    )
    score += 0.25 if identified_db else 0.05

    # -- Remediation (0.30) -------------------------------
    did_failover = _any_cmd_matches(actions, "failover database-primary")
    did_cache_restart = _any_cmd_matches(actions, "restart cache-layer")

    if did_failover and did_cache_restart:
        score += 0.30                  # both steps correct
    elif did_failover:
        score += 0.18                  # partial — DB fixed but cache still saturated
    elif did_cache_restart:
        score += 0.08                  # fixed symptom but not root cause
    elif _any_cmd_matches(actions, "failover", "restart"):
        score += 0.03                  # tried something but wrong targets

    # -- Communication (0.05) ------------------------------
    if _any_cmd_matches(actions, "notify"):
        score += 0.05

    # -- Efficiency (0.10) --------------------------------
    optimal_steps = 15
    penalty_per_extra = 0.004
    extras = max(0, step_count - optimal_steps)
    efficiency = max(0.0, 0.10 - extras * penalty_per_extra)

    # Bonus: agent avoided restarting red-herring services
    bounced_herring = _any_cmd_matches(
        actions,
        "restart notification-service",
        "restart inventory-service",
        "restart user-service",
    )
    if not bounced_herring:
        efficiency = min(efficiency + 0.02, 0.10)

    score += efficiency

    return _clamp(score)


# ---------------------------------------------------------
#  Dispatch
# ---------------------------------------------------------

GRADERS = {
    "service-restart":    grade_service_restart,
    "config-drift":       grade_config_drift,
    "cascading-failure":  grade_cascading_failure,
}


def grade(
    scenario: Scenario,
    engine: SimulationEngine,
    step_count: int,
) -> float:
    """
    Grade the completed episode.

    Returns:
        float: Final score in open interval (0.01, 0.99).
    """
    grader = GRADERS.get(scenario.name)
    if grader is None:
        raise ValueError(f"No grader registered for scenario '{scenario.name}'")
    return grader(scenario, engine, step_count)
