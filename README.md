---
title: IncidentOps SRE Environment
emoji: 🚨
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8000
---

# IncidentOps — AI Incident Response Training Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Team: CodeBlockers](https://img.shields.io/badge/Team-CodeBlockers-purple)](https://huggingface.co/chandan123467896uyjh)

**IncidentOps** is a production-grade OpenEnv environment that trains AI agents to respond to real-world production incidents, just like an on-call Site Reliability Engineer (SRE).

The agent interacts with a simulated microservice infrastructure through a realistic text-based terminal, executing commands to triage alerts, investigate logs and metrics, identify root causes, and apply remediations.

> **Why this matters**: Incident response is one of the highest-stakes, time-sensitive tasks in software engineering. Training AI agents to perform it well has immediate real-world value for DevOps/SRE teams.

---

## Environment Description

The agent acts as an on-call SRE engineer who receives a pager alert. They must:

1. **Triage** — Check active alerts and system-wide dashboard to understand impact
2. **Investigate** — Dig into logs, metrics, distributed traces, and diagnostics
3. **Identify Root Cause** — Determine which service failed and why
4. **Remediate** — Apply the correct fix (restart, rollback, failover, config change)
5. **Resolve** — Confirm recovery and close the incident

The environment simulates realistic microservice architectures with:
- Service dependency graphs (5–8 services per scenario)
- Timestamped log streams showing the failure progression
- Performance metrics (CPU, memory, latency percentiles, error rates)
- Distributed request traces across the service mesh
- Multiple alerts with varying severity — including red-herring alerts

---

## Action Space

The agent issues text commands through a terminal-style interface:

| Command | Description | Example |
|---------|-------------|---------|
| `help` | List all available commands | `help` |
| `status` | System-wide service dashboard | `status` |
| `alerts` | Active alerts with severity and details | `alerts` |
| `logs <service>` | Recent log entries for a service | `logs api-gateway` |
| `metrics <service>` | CPU / memory / latency / error-rate | `metrics database-primary` |
| `trace <request_id>` | Follow a request across services | `trace req-48219` |
| `diagnose <service>` | Deep diagnostic with recommendations | `diagnose payment-processor` |
| `restart <service>` | Restart a service | `restart payment-processor` |
| `scale <service> <n>` | Scale replica count | `scale read-service 6` |
| `rollback <service>` | Roll back last deployment | `rollback api-gateway` |
| `failover <service>` | Promote standby to primary | `failover database-primary` |
| `config <service> <key> <val>` | Update live configuration | `config api-gateway pool_size 100` |
| `notify <channel> <msg>` | Post status update to team | `notify oncall investigating` |
| `resolve` | Declare the incident resolved (ends episode) | `resolve` |

### Action Model

```python
class IncidentAction(Action):
    command: str    # Full command string including arguments
```

---

## Observation Space

After each action, the environment returns:

```python
class IncidentObservation(Observation):
    output: str              # Terminal-style output from the command
    timestamp: str           # Current simulation time (ISO-8601)
    alert_count: int         # Number of active alerts remaining
    severity: str            # Incident severity: "critical" | "high" | "medium" | "low" | "none"
    affected_services: list  # Services currently degraded or down
    done: bool               # True when episode has ended
    reward: float            # Score in open interval (0.01, 0.99)
    metadata: dict           # task_name, episode_id, step, final_score
```

---

## Tasks

Three tasks of increasing difficulty. Select at reset time via `task_name` parameter.

### Task 1: `service-restart` — OOM Service Crash (Easy)

**Scenario**: A `payment-processor` microservice has been OOM-killed (Out of Memory). It has entered CrashLoopBackOff. Three alerts are firing including a `critical` alert.

**Objective**: Identify the crashing service and restart it.

**Expected episode length**: 3–6 steps

**Grading**:
- Checked alerts: +5%
- Checked payment-processor logs/metrics: +12%
- Found root cause (payment-processor): +30%
- Applied correct fix (restart payment-processor): +30%
- Sent a status notification: +5%
- Efficiency bonus (≤ 4 steps): +5%

**Baseline score**: ~0.65

---

### Task 2: `config-drift` — Connection Pool Exhaustion (Medium)

**Scenario**: A recent deployment to `api-gateway` (v3.1.5) introduced a config regression — the connection pool size was set to 5 (was 100). This causes connection pool exhaustion, cascading timeouts across 3 downstream services, and a `critical` + `high` alert combo.

**Objective**: Identify the config regression and fix it (rollback or `config api-gateway pool_size 100`).

**Expected episode length**: 6–12 steps

**Grading**:
- Investigated 2+ services: +10%
- Diagnosed api-gateway: +30%
- Applied correct fix (rollback or config): +30%
- Sent notification: +5%
- Efficiency bonus: +5%

**Baseline score**: ~0.58

---

### Task 3: `cascading-failure` — Disk I/O → Cache Stampede → API Overload (Hard)

**Scenario**: A complex multi-service cascading failure:
- `database-primary` disk I/O saturated (iowait 82%)
- Replication lag builds to 182 seconds
- `cache-layer` gets a stampede (cache hit rate drops from 94% to 16%, CPU hits 99.8%)
- `api-gateway` becomes overloaded (72% error rate)
- Two **red-herring** alerts are included (notification-service delay, inventory maintenance window)

**Objective**: Full incident lifecycle — identify disk I/O root cause, execute `failover database-primary` then `restart cache-layer`, verify recovery.

**Expected episode length**: 10–20 steps

**Grading**:
- Explored full causal chain (DB → cache → API): +30%
- Identified database-primary as root cause: +25%
- DB failover: +18%
- Cache restart: +12%
- Both steps executed: +30% total (instead of +18+12)
- Notification: +5%
- Efficiency + avoided red herrings: up to +10%

**Baseline score**: ~0.48

---

## Reward Function

Rewards are **dense** — they provide signal on every step, not just at episode end.

| Phase | Range | Signal |
|-------|-------|--------|
| Investigation steps | 0.01–0.05 incremental | +0.05 for root-cause service, +0.03 for affected, +0.01 for healthy |
| Root cause identification | 0.05–0.30 | Based on depth of investigation |
| Correct remediation | 0.30 | On correct fix applied to correct service |
| Wrong remediation | −0.05 | Restarting healthy services penalised |
| Communication | 0.05 | At least one `notify` issued |
| Efficiency | 0.00–0.10 | Inverse of excess steps taken |

All rewards are clamped to the **open interval (0.01, 0.99)** to satisfy the OpenEnv validator.

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- `pip install openenv-core`

### Local Development (without Docker)

```bash
# Install
pip install -e ".[dev]"

# Run the server locally
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# In a separate terminal — connect and interact
python - <<'EOF'
from incident_ops_env import IncidentAction, IncidentOpsEnv

with IncidentOpsEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_name="service-restart")
    print(result.observation.output)

    for cmd in ["alerts", "logs payment-processor", "restart payment-processor", "resolve"]:
        result = env.step(IncidentAction(command=cmd))
        print(result.observation.output)
        if result.observation.done:
            print(f"\nFinal score: {result.reward:.3f}")
            break
EOF
```

### Docker

```bash
# Build
docker build -f server/Dockerfile -t incident-ops-env:latest .

# Run
docker run -p 8000:8000 incident-ops-env:latest

# Validate
openenv validate
```

### Running the Baseline Inference Script

```bash
export HF_TOKEN="your-token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export INCIDENT_BASE_URL="http://localhost:8000"

python inference.py
```

### Running Tests

```bash
pip install ".[dev]"
pytest tests/test_env.py -v
```

---

## Baseline Scores

Scores achieved by `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference:

| Task | Difficulty | Baseline Score |
|------|-----------|---------------|
| `service-restart` | Easy | ~0.65 |
| `config-drift` | Medium | ~0.58 |
| `cascading-failure` | Hard | ~0.48 |

A perfect agent would score ~0.95 on all tasks. These baseline scores leave significant headroom for RL training to improve agent performance.

---

## Project Structure

```
.
├── openenv.yaml          # OpenEnv manifest
├── pyproject.toml        # Package and dependency configuration
├── README.md             # This file
├── inference.py          # Baseline inference script
├── __init__.py           # Package exports
├── models.py             # Pydantic Action/Observation models
├── client.py             # EnvClient subclass
├── tests/
│   └── test_env.py       # Unit tests
└── server/
    ├── __init__.py
    ├── app.py             # FastAPI application
    ├── Dockerfile         # Container image
    ├── environment.py     # Core IncidentOpsEnvironment class
    ├── scenarios.py       # Three incident scenario definitions
    ├── graders.py         # Task-specific scoring functions
    └── simulation.py      # System simulation engine
```

---

## HuggingFace Space

**URL**: https://huggingface.co/spaces/Karthikrv/openenv-codereview

**Team**: CodeBlockers 

---

## License 

MIT License. See [LICENSE](LICENSE) file.
