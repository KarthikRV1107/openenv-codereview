"""
Unit tests for IncidentOps OpenEnv environment.

Tests verify:
  1. All three tasks initialise and return valid observations from reset()
  2. step() returns observations with reward values strictly in (0.0, 1.0)
  3. Episode lifecycle (reset → step → done)
  4. Graders produce deterministic, in-range scores on representative trajectories
  5. Simulator handles all valid commands without exceptions
"""

from __future__ import annotations

import sys
import os

# Allow running from repo root without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from models import IncidentAction, IncidentObservation
from server.environment import IncidentOpsEnvironment
from server.scenarios import ALL_TASK_NAMES, get_scenario
from server.graders import grade, GRADERS
from server.simulation import SimulationEngine


# -----------------------------------------------------------------------------
#  Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def env() -> IncidentOpsEnvironment:
    return IncidentOpsEnvironment()


def _make_action(command: str) -> IncidentAction:
    return IncidentAction(command=command)


# -----------------------------------------------------------------------------
#  1. Reset returns valid observations for all tasks
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("task_name", ALL_TASK_NAMES)
def test_reset_returns_observation(env, task_name):
    obs = env.reset(task_name=task_name)
    assert isinstance(obs, IncidentObservation)
    assert isinstance(obs.output, str) and len(obs.output) > 10
    assert isinstance(obs.timestamp, str) and "T" in obs.timestamp
    assert isinstance(obs.alert_count, int) and obs.alert_count > 0
    assert obs.severity in ("critical", "high", "medium", "low", "none")
    assert isinstance(obs.affected_services, list)
    assert obs.done is False


@pytest.mark.parametrize("task_name", ALL_TASK_NAMES)
def test_reset_reward_in_open_interval(env, task_name):
    obs = env.reset(task_name=task_name)
    assert obs.reward is not None
    assert 0.0 < obs.reward < 1.0, f"Reset reward {obs.reward} not in (0, 1)"


# -----------------------------------------------------------------------------
#  2. step() returns valid observations with reward in (0, 1)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("task_name", ALL_TASK_NAMES)
def test_step_reward_strictly_in_open_interval(env, task_name):
    env.reset(task_name=task_name)
    for cmd in ["alerts", "status", "help"]:
        obs = env.step(_make_action(cmd))
        assert isinstance(obs, IncidentObservation)
        assert 0.0 < obs.reward < 1.0, (
            f"[{task_name}] reward {obs.reward} for '{cmd}' is not in open interval (0, 1)"
        )
        assert isinstance(obs.output, str) and len(obs.output) > 0


@pytest.mark.parametrize("task_name", ALL_TASK_NAMES)
def test_step_output_is_non_empty(env, task_name):
    env.reset(task_name=task_name)
    obs = env.step(_make_action("status"))
    assert len(obs.output.strip()) > 0


# -----------------------------------------------------------------------------
#  3. Episode lifecycle — resolve ends the episode
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("task_name", ALL_TASK_NAMES)
def test_resolve_terminates_episode(env, task_name):
    env.reset(task_name=task_name)
    obs = env.step(_make_action("resolve"))
    assert obs.done is True
    assert 0.0 < obs.reward < 1.0


def test_step_after_done_returns_done(env):
    env.reset(task_name="service-restart")
    env.step(_make_action("resolve"))
    obs = env.step(_make_action("alerts"))
    assert obs.done is True


# -----------------------------------------------------------------------------
#  4. Graders produce deterministic in-range scores
# -----------------------------------------------------------------------------

def test_grade_service_restart_optimal():
    """Agent checks payment-processor then restarts it — should score ≥ 0.7."""
    scenario = get_scenario("service-restart", seed=42)
    engine = SimulationEngine(scenario)
    # Simulate an optimal trajectory
    for cmd in ["alerts", "logs payment-processor", "restart payment-processor", "resolve"]:
        engine.execute(cmd)
    score = grade(scenario, engine, step_count=4)
    assert 0.0 < score < 1.0, f"Score {score} not in (0, 1)"
    assert score >= 0.70, f"Optimal run score {score:.3f} too low (expected ≥ 0.70)"


def test_grade_service_restart_blind_resolve():
    """Agent immediately resolves without investigating — should score low."""
    scenario = get_scenario("service-restart", seed=42)
    engine = SimulationEngine(scenario)
    engine.execute("resolve")
    score = grade(scenario, engine, step_count=1)
    assert 0.0 < score < 1.0, f"Score {score} not in (0, 1)"
    assert score < 0.35, f"Blind-resolve score {score:.3f} too high (expected < 0.35)"


def test_grade_config_drift_optimal():
    """Agent inspects api-gateway and rolls it back — should score ≥ 0.65."""
    scenario = get_scenario("config-drift", seed=99)
    engine = SimulationEngine(scenario)
    for cmd in [
        "alerts", "status",
        "logs api-gateway", "metrics api-gateway",
        "diagnose api-gateway",
        "rollback api-gateway",
        "notify oncall resolved",
        "resolve",
    ]:
        engine.execute(cmd)
    score = grade(scenario, engine, step_count=8)
    assert 0.0 < score < 1.0, f"Score {score} not in (0, 1)"
    assert score >= 0.65, f"Config-drift optimal score {score:.3f} too low (expected ≥ 0.65)"


def test_grade_cascading_failure_full_remediation():
    """Agent does DB failover + cache restart — should score ≥ 0.60."""
    scenario = get_scenario("cascading-failure", seed=7)
    engine = SimulationEngine(scenario)
    for cmd in [
        "alerts", "status",
        "logs api-gateway",
        "logs cache-layer", "metrics cache-layer",
        "logs database-primary", "metrics database-primary",
        "diagnose database-primary",
        "failover database-primary",
        "restart cache-layer",
        "notify oncall failover-complete",
        "resolve",
    ]:
        engine.execute(cmd)
    score = grade(scenario, engine, step_count=12)
    assert 0.0 < score < 1.0, f"Score {score} not in (0, 1)"
    assert score >= 0.60, f"Cascading-failure score {score:.3f} too low (expected ≥ 0.60)"


def test_all_graders_clamp_scores():
    """No grader should produce 0.0 or 1.0 exactly."""
    for task_name in ALL_TASK_NAMES:
        scenario = get_scenario(task_name, seed=1)
        engine = SimulationEngine(scenario)
        score = grade(scenario, engine, step_count=1)
        assert 0.0 < score < 1.0, (
            f"[{task_name}] score {score} not strictly in (0, 1)"
        )


def test_graders_deterministic():
    """Running grade twice on same state should return the same score."""
    for task_name in ALL_TASK_NAMES:
        scenario1 = get_scenario(task_name, seed=42)
        engine1 = SimulationEngine(scenario1)
        for cmd in ["alerts", "status"]:
            engine1.execute(cmd)
        score_a = grade(scenario1, engine1, step_count=2)

        scenario2 = get_scenario(task_name, seed=42)
        engine2 = SimulationEngine(scenario2)
        for cmd in ["alerts", "status"]:
            engine2.execute(cmd)
        score_b = grade(scenario2, engine2, step_count=2)

        assert score_a == score_b, (
            f"[{task_name}] grader not deterministic: {score_a:.4f} != {score_b:.4f}"
        )


# -----------------------------------------------------------------------------
#  5. Simulator handles all commands without exceptions
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("task_name", ALL_TASK_NAMES)
def test_all_commands_execute_without_exception(env, task_name):
    env.reset(task_name=task_name)
    scenario = get_scenario(task_name)
    services = list(scenario.services.keys())
    req_ids = scenario.request_ids

    commands_to_test = [
        "help", "status", "alerts",
        f"logs {services[0]}",
        f"metrics {services[0]}",
        f"diagnose {services[0]}",
        f"trace {req_ids[0]}",
        f"scale {services[0]} 3",
        "notify oncall investigating",
    ]

    for cmd in commands_to_test:
        obs = env.step(_make_action(cmd))
        assert isinstance(obs, IncidentObservation), f"Non-observation for '{cmd}'"
        assert 0.0 < obs.reward < 1.0, f"Reward {obs.reward} out of range for '{cmd}'"


def test_unknown_command_does_not_crash(env):
    env.reset(task_name="service-restart")
    obs = env.step(_make_action("nonexistent_op"))
    assert isinstance(obs, IncidentObservation)
    assert "unknown" in obs.output.lower() or "not found" in obs.output.lower()


def test_step_without_service_arg(env):
    """Commands missing required args should return usage hint, not crash."""
    env.reset(task_name="service-restart")
    for cmd in ["logs", "metrics", "diagnose", "restart", "rollback", "failover"]:
        obs = env.step(_make_action(cmd))
        assert isinstance(obs, IncidentObservation)
        assert obs.reward is not None
        assert 0.0 < obs.reward < 1.0, f"Reward {obs.reward} out of range for '{cmd}'"


# -----------------------------------------------------------------------------
#  6. Scenario data integrity
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("task_name", ALL_TASK_NAMES)
def test_scenario_has_minimum_structure(task_name):
    scenario = get_scenario(task_name)
    assert len(scenario.services) >= 3, "Scenario must have ≥ 3 services"
    assert len(scenario.alerts) >= 1, "Scenario must have ≥ 1 alert"
    assert scenario.root_cause_service in scenario.services
    assert scenario.max_steps >= 10
    assert scenario.difficulty in ("easy", "medium", "hard")
    assert len(scenario.request_ids) >= 1


@pytest.mark.parametrize("task_name", ALL_TASK_NAMES)
def test_scenario_reproducible_with_same_seed(task_name):
    s1 = get_scenario(task_name, seed=123)
    s2 = get_scenario(task_name, seed=123)
    assert s1.name == s2.name
    assert s1.root_cause_service == s2.root_cause_service
    assert len(s1.alerts) == len(s2.alerts)


# -----------------------------------------------------------------------------
#  7. Model-level reward clamping
# -----------------------------------------------------------------------------

def test_model_validator_clamps_zero():
    """reward=0.0 should be clamped to 0.01."""
    obs = IncidentObservation(
        output="test", timestamp="2026-01-01T00:00:00Z",
        done=False, reward=0.0,
    )
    assert obs.reward == 0.01


def test_model_validator_clamps_one():
    """reward=1.0 should be clamped to 0.99."""
    obs = IncidentObservation(
        output="test", timestamp="2026-01-01T00:00:00Z",
        done=False, reward=1.0,
    )
    assert obs.reward == 0.99


def test_model_validator_clamps_negative():
    """Negative reward should be clamped to 0.01."""
    obs = IncidentObservation(
        output="test", timestamp="2026-01-01T00:00:00Z",
        done=False, reward=-5.0,
    )
    assert obs.reward == 0.01


def test_model_validator_preserves_valid():
    """Valid reward should pass through unchanged."""
    obs = IncidentObservation(
        output="test", timestamp="2026-01-01T00:00:00Z",
        done=False, reward=0.42,
    )
    assert obs.reward == 0.42


def test_model_validator_none_stays_none():
    """None reward should remain None."""
    obs = IncidentObservation(
        output="test", timestamp="2026-01-01T00:00:00Z",
        done=False, reward=None,
    )
    assert obs.reward is None


def test_model_validator_bool_true():
    """bool True (== 1) should be clamped to 0.99."""
    obs = IncidentObservation(
        output="test", timestamp="2026-01-01T00:00:00Z",
        done=False, reward=True,
    )
    assert obs.reward == 0.99


def test_model_validator_bool_false():
    """bool False (== 0) should be clamped to 0.01."""
    obs = IncidentObservation(
        output="test", timestamp="2026-01-01T00:00:00Z",
        done=False, reward=False,
    )
    assert obs.reward == 0.01
