"""
IncidentOps Scenario Definitions.

Each scenario is a complete, self-contained incident that simulates a
realistic production event. Scenarios define:
  - The microservice topology (services and their dependencies)
  - Initial state of each service (healthy, degraded, down)
  - Active alert definitions
  - Pre-generated log streams (timestamped entries per service)
  - Metric snapshots (cpu, memory_mb, latency_ms, error_rate)
  - The true root cause (service + issue)
  - Valid resolution criteria (what actions fix the incident)
  - Configuration values (current and correct)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────

@dataclass
class ServiceConfig:
    """Current and desired configuration values for a service."""
    current: Dict[str, Any] = field(default_factory=dict)
    desired: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceMetrics:
    """Snapshot of service performance metrics."""
    cpu_pct: float          # 0–100
    memory_mb: float        # absolute MB used
    memory_limit_mb: float  # container memory limit
    latency_p50_ms: float   # median latency
    latency_p99_ms: float   # 99th-percentile latency
    error_rate_pct: float   # percentage of requests returning 5xx
    rps: float              # requests per second
    replica_count: int      # running replicas
    max_replicas: int       # maximum allowed replicas


@dataclass
class ServiceState:
    """Full state of a simulated microservice."""
    name: str
    display_name: str
    status: str                         # "running" | "degraded" | "down"
    version: str                        # e.g. "v2.4.1"
    metrics: ServiceMetrics
    config: ServiceConfig = field(default_factory=ServiceConfig)
    logs: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    restart_count: int = 0
    oom_killed: bool = False


@dataclass
class Alert:
    """An active monitoring alert."""
    id: str
    severity: str       # "critical" | "high" | "medium" | "low"
    service: str
    title: str
    description: str
    fired_at: str       # simulated ISO timestamp
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Scenario:
    """A complete incident scenario."""
    name: str
    display_name: str
    description: str
    difficulty: str                               # "easy" | "medium" | "hard"
    start_time: str                               # simulation epoch
    services: Dict[str, ServiceState]
    alerts: List[Alert]
    root_cause_service: str
    root_cause_issue: str
    resolution_actions: List[str]                 # valid command prefixes that resolve
    resolution_services: List[str]                # services that must be acted on
    max_steps: int
    hint: str                                     # internal hint for grader
    request_ids: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────
#  Scenario 1 — "service-restart" (EASY)
#  payment-processor OOM-killed, all other services healthy
# ─────────────────────────────────────────────────────────

def _build_service_restart_scenario(seed: int = 42) -> Scenario:
    rng = random.Random(seed)

    t_base = "2026-04-10T04:00"
    req_ids = [f"req-{rng.randint(10000, 99999)}" for _ in range(4)]

    services: Dict[str, ServiceState] = {

        "api-gateway": ServiceState(
            name="api-gateway",
            display_name="API Gateway",
            status="running",
            version="v3.1.2",
            metrics=ServiceMetrics(
                cpu_pct=34.2, memory_mb=412, memory_limit_mb=1024,
                latency_p50_ms=12, latency_p99_ms=45, error_rate_pct=0.1,
                rps=820, replica_count=2, max_replicas=8,
            ),
            config=ServiceConfig(
                current={"timeout_s": 30, "pool_size": 100},
                desired={"timeout_s": 30, "pool_size": 100},
            ),
            dependencies=[],
            logs=[
                f"{t_base}:00Z [INFO ] api-gateway: serving request {req_ids[0]}",
                f"{t_base}:01Z [WARN ] api-gateway: upstream payment-processor timeout (retrying)",
                f"{t_base}:02Z [ERROR] api-gateway: payment endpoint returning 503 (upstream down)",
                f"{t_base}:03Z [INFO ] api-gateway: all other routes healthy",
            ],
        ),

        "payment-processor": ServiceState(
            name="payment-processor",
            display_name="Payment Processor",
            status="down",
            version="v1.9.3",
            metrics=ServiceMetrics(
                cpu_pct=0.0, memory_mb=0, memory_limit_mb=512,
                latency_p50_ms=0, latency_p99_ms=0, error_rate_pct=100.0,
                rps=0, replica_count=0, max_replicas=4,
            ),
            config=ServiceConfig(
                current={"max_heap_mb": 480, "workers": 4},
                desired={"max_heap_mb": 480, "workers": 4},
            ),
            dependencies=["database-primary"],
            oom_killed=True,
            restart_count=3,
            logs=[
                f"{t_base}:00Z [INFO ] payment-processor: processing batch job — heap usage 430MB",
                f"{t_base}:00Z [WARN ] payment-processor: heap usage approaching limit 480MB",
                f"{t_base}:01Z [ERROR] payment-processor: java.lang.OutOfMemoryError: GC overhead limit exceeded",
                f"{t_base}:01Z [ERROR] payment-processor: OOMKilled — container terminated by OS",
                f"{t_base}:01Z [INFO ] payment-processor: restart attempt 1/3 failed — backoff 10s",
                f"{t_base}:02Z [INFO ] payment-processor: restart attempt 2/3 failed — backoff 30s",
                f"{t_base}:03Z [INFO ] payment-processor: restart attempt 3/3 failed — CrashLoopBackOff",
            ],
        ),

        "database-primary": ServiceState(
            name="database-primary",
            display_name="Database Primary",
            status="running",
            version="postgresql-15.4",
            metrics=ServiceMetrics(
                cpu_pct=18.1, memory_mb=3200, memory_limit_mb=8192,
                latency_p50_ms=3, latency_p99_ms=18, error_rate_pct=0.0,
                rps=120, replica_count=1, max_replicas=1,
            ),
            config=ServiceConfig(
                current={"max_connections": 200, "shared_buffers_mb": 1024},
                desired={"max_connections": 200, "shared_buffers_mb": 1024},
            ),
            dependencies=[],
            logs=[
                f"{t_base}:00Z [INFO ] database-primary: checkpoint complete — 12 pages written",
                f"{t_base}:02Z [INFO ] database-primary: connection from payment-processor closed (client disconnect)",
                f"{t_base}:03Z [INFO ] database-primary: all systems nominal",
            ],
        ),

        "order-service": ServiceState(
            name="order-service",
            display_name="Order Service",
            status="running",
            version="v4.2.0",
            metrics=ServiceMetrics(
                cpu_pct=22.0, memory_mb=256, memory_limit_mb=512,
                latency_p50_ms=18, latency_p99_ms=82, error_rate_pct=0.05,
                rps=340, replica_count=2, max_replicas=6,
            ),
            config=ServiceConfig(
                current={"payment_fallback": "false", "timeout_s": 15},
                desired={"payment_fallback": "false", "timeout_s": 15},
            ),
            dependencies=["payment-processor", "database-primary"],
            logs=[
                f"{t_base}:00Z [INFO ] order-service: order #88214 submitted",
                f"{t_base}:01Z [WARN ] order-service: payment-processor unreachable — payment step failed",
                f"{t_base}:02Z [WARN ] order-service: order #88215 held pending payment retry",
            ],
        ),
    }

    alerts = [
        Alert(
            id="alert-001",
            severity="critical",
            service="payment-processor",
            title="Service Down — payment-processor",
            description="payment-processor has 0 healthy replicas. Error rate 100%. OOMKilled.",
            fired_at="2026-04-10T04:01:30Z",
            labels={"team": "payments", "env": "prod"},
        ),
        Alert(
            id="alert-002",
            severity="high",
            service="api-gateway",
            title="Upstream 503 — /api/v1/payments",
            description="api-gateway is returning 503 on /api/v1/payments endpoint for 2m.",
            fired_at="2026-04-10T04:02:00Z",
            labels={"team": "platform", "env": "prod"},
        ),
        Alert(
            id="alert-003",
            severity="medium",
            service="order-service",
            title="Payment step failing — order-service",
            description="order-service orders are stalling at payment step. New orders queued.",
            fired_at="2026-04-10T04:02:15Z",
            labels={"team": "commerce", "env": "prod"},
        ),
    ]

    return Scenario(
        name="service-restart",
        display_name="OOM Service Crash",
        description=(
            "A critical microservice has crashed due to an Out of Memory (OOM) condition. "
            "Multiple downstream alerts have fired. Identify the failing service and restart it."
        ),
        difficulty="easy",
        start_time="2026-04-10T04:00:00Z",
        services=services,
        alerts=alerts,
        root_cause_service="payment-processor",
        root_cause_issue="oom_killed",
        resolution_actions=["restart payment-processor"],
        resolution_services=["payment-processor"],
        max_steps=15,
        hint="payment-processor was OOMKilled — restart it",
        request_ids=req_ids,
    )


# ─────────────────────────────────────────────────────────
#  Scenario 2 — "config-drift" (MEDIUM)
#  api-gateway pool_size misconfigured → cascading timeouts
# ─────────────────────────────────────────────────────────

def _build_config_drift_scenario(seed: int = 99) -> Scenario:
    rng = random.Random(seed)
    t_base = "2026-04-10T05:30"
    req_ids = [f"req-{rng.randint(10000, 99999)}" for _ in range(6)]

    services: Dict[str, ServiceState] = {

        "api-gateway": ServiceState(
            name="api-gateway",
            display_name="API Gateway",
            status="degraded",
            version="v3.1.5",
            metrics=ServiceMetrics(
                cpu_pct=88.7, memory_mb=920, memory_limit_mb=1024,
                latency_p50_ms=4200, latency_p99_ms=29800, error_rate_pct=62.4,
                rps=760, replica_count=2, max_replicas=8,
            ),
            config=ServiceConfig(
                current={"timeout_s": 30, "pool_size": 5},   # WRONG — should be 100
                desired={"timeout_s": 30, "pool_size": 100},
            ),
            dependencies=[],
            logs=[
                f"{t_base}:02Z [INFO ] api-gateway: deployment v3.1.5 complete",
                f"{t_base}:05Z [WARN ] api-gateway: connection pool exhausted (5/5 in use)",
                f"{t_base}:06Z [ERROR] api-gateway: request {req_ids[0]} timeout — pool wait 4.2s",
                f"{t_base}:07Z [ERROR] api-gateway: request {req_ids[1]} timeout — pool wait 4.8s",
                f"{t_base}:08Z [ERROR] api-gateway: request {req_ids[2]} timeout — pool exhausted, rejecting",
                f"{t_base}:09Z [ERROR] api-gateway: thread pool utilisation 100% — queueing requests",
            ],
        ),

        "checkout-service": ServiceState(
            name="checkout-service",
            display_name="Checkout Service",
            status="degraded",
            version="v2.8.0",
            metrics=ServiceMetrics(
                cpu_pct=55.0, memory_mb=640, memory_limit_mb=1024,
                latency_p50_ms=5100, latency_p99_ms=30000, error_rate_pct=58.1,
                rps=320, replica_count=3, max_replicas=6,
            ),
            config=ServiceConfig(
                current={"upstream_timeout_s": 10},
                desired={"upstream_timeout_s": 10},
            ),
            dependencies=["api-gateway", "payment-processor"],
            logs=[
                f"{t_base}:06Z [WARN ] checkout-service: api-gateway response time 4.2s (threshold: 1s)",
                f"{t_base}:07Z [ERROR] checkout-service: checkout flow {req_ids[3]} failed — upstream timeout",
                f"{t_base}:08Z [ERROR] checkout-service: 58% of checkout attempts failing",
                f"{t_base}:09Z [WARN ] checkout-service: retry storm detected — circuit breaker opened",
            ],
        ),

        "payment-processor": ServiceState(
            name="payment-processor",
            display_name="Payment Processor",
            status="degraded",
            version="v1.9.3",
            metrics=ServiceMetrics(
                cpu_pct=71.3, memory_mb=390, memory_limit_mb=512,
                latency_p50_ms=3800, latency_p99_ms=18000, error_rate_pct=44.0,
                rps=210, replica_count=2, max_replicas=4,
            ),
            config=ServiceConfig(
                current={"max_heap_mb": 480, "workers": 4},
                desired={"max_heap_mb": 480, "workers": 4},
            ),
            dependencies=["database-primary", "api-gateway"],
            logs=[
                f"{t_base}:05Z [WARN ] payment-processor: upstream api-gateway latency elevated",
                f"{t_base}:07Z [ERROR] payment-processor: request {req_ids[4]} timed out from api-gateway",
                f"{t_base}:08Z [ERROR] payment-processor: 44% error rate — upstream dependency degraded",
            ],
        ),

        "database-primary": ServiceState(
            name="database-primary",
            display_name="Database Primary",
            status="running",
            version="postgresql-15.4",
            metrics=ServiceMetrics(
                cpu_pct=21.0, memory_mb=3100, memory_limit_mb=8192,
                latency_p50_ms=4, latency_p99_ms=22, error_rate_pct=0.0,
                rps=115, replica_count=1, max_replicas=1,
            ),
            config=ServiceConfig(
                current={"max_connections": 200, "shared_buffers_mb": 1024},
                desired={"max_connections": 200, "shared_buffers_mb": 1024},
            ),
            dependencies=[],
            logs=[
                f"{t_base}:00Z [INFO ] database-primary: all systems nominal",
                f"{t_base}:09Z [INFO ] database-primary: query performance normal",
            ],
        ),

        "user-service": ServiceState(
            name="user-service",
            display_name="User Service",
            status="running",
            version="v5.0.1",
            metrics=ServiceMetrics(
                cpu_pct=12.0, memory_mb=180, memory_limit_mb=512,
                latency_p50_ms=8, latency_p99_ms=35, error_rate_pct=0.0,
                rps=95, replica_count=2, max_replicas=4,
            ),
            config=ServiceConfig(
                current={"session_ttl_s": 3600},
                desired={"session_ttl_s": 3600},
            ),
            dependencies=[],
            logs=[
                f"{t_base}:00Z [INFO ] user-service: auth token refreshed for user u-8812",
                f"{t_base}:09Z [INFO ] user-service: all systems nominal",
            ],
        ),
    }

    alerts = [
        Alert(
            id="alert-101",
            severity="critical",
            service="checkout-service",
            title="High Error Rate — checkout-service",
            description="checkout-service error rate at 58% — checkout flow failing for majority of users.",
            fired_at="2026-04-10T05:30:07Z",
            labels={"team": "commerce", "env": "prod"},
        ),
        Alert(
            id="alert-102",
            severity="critical",
            service="api-gateway",
            title="API Gateway P99 Latency >30s",
            description="api-gateway P99 response time exceeded 30s threshold. Connection pool may be exhausted.",
            fired_at="2026-04-10T05:30:08Z",
            labels={"team": "platform", "env": "prod"},
        ),
        Alert(
            id="alert-103",
            severity="high",
            service="payment-processor",
            title="Elevated Error Rate — payment-processor",
            description="payment-processor error rate 44%. Upstream dependency appears degraded.",
            fired_at="2026-04-10T05:30:09Z",
            labels={"team": "payments", "env": "prod"},
        ),
        Alert(
            id="alert-104",
            severity="medium",
            service="api-gateway",
            title="Deployment Completed — api-gateway v3.1.5",
            description="api-gateway deployment to v3.1.5 completed 8 minutes ago. Coincides with latency spike.",
            fired_at="2026-04-10T05:30:05Z",
            labels={"team": "platform", "env": "prod"},
        ),
    ]

    return Scenario(
        name="config-drift",
        display_name="Config Drift — Connection Pool Exhaustion",
        description=(
            "A recent deployment to api-gateway introduced a configuration regression. "
            "Multiple services are now reporting elevated latency and errors. Find the "
            "root cause and apply the correct fix."
        ),
        difficulty="medium",
        start_time="2026-04-10T05:30:00Z",
        services=services,
        alerts=alerts,
        root_cause_service="api-gateway",
        root_cause_issue="pool_size_misconfigured",
        resolution_actions=[
            "rollback api-gateway",
            "config api-gateway pool_size 100",
        ],
        resolution_services=["api-gateway"],
        max_steps=20,
        hint="api-gateway pool_size was set to 5 during deployment (was 100) — rollback or fix config",
        request_ids=req_ids,
    )


# ─────────────────────────────────────────────────────────
#  Scenario 3 — "cascading-failure" (HARD)
#  Disk I/O on DB primary → replication lag → cache stampede
#  → API overload → 5xx. Multiple red-herring alerts.
# ─────────────────────────────────────────────────────────

def _build_cascading_failure_scenario(seed: int = 7) -> Scenario:
    rng = random.Random(seed)
    t_base = "2026-04-10T02:15"
    req_ids = [f"req-{rng.randint(10000, 99999)}" for _ in range(8)]

    services: Dict[str, ServiceState] = {

        "api-gateway": ServiceState(
            name="api-gateway",
            display_name="API Gateway",
            status="degraded",
            version="v3.1.2",
            metrics=ServiceMetrics(
                cpu_pct=96.1, memory_mb=988, memory_limit_mb=1024,
                latency_p50_ms=8400, latency_p99_ms=60000, error_rate_pct=72.0,
                rps=700, replica_count=2, max_replicas=8,
            ),
            config=ServiceConfig(
                current={"timeout_s": 30, "pool_size": 100},
                desired={"timeout_s": 30, "pool_size": 100},
            ),
            dependencies=[],
            logs=[
                f"{t_base}:12Z [WARN ] api-gateway: upstream response time increasing",
                f"{t_base}:15Z [ERROR] api-gateway: {req_ids[0]} — upstream read-service timeout 8.4s",
                f"{t_base}:16Z [ERROR] api-gateway: {req_ids[1]} — upstream cache-layer not responding",
                f"{t_base}:18Z [ERROR] api-gateway: 72% requests returning 503/504 — thread pool saturated",
                f"{t_base}:20Z [ERROR] api-gateway: circuit breaker open for cache-layer, read-service",
            ],
        ),

        "cache-layer": ServiceState(
            name="cache-layer",
            display_name="Cache Layer (Redis Cluster)",
            status="degraded",
            version="redis-7.2.4",
            metrics=ServiceMetrics(
                cpu_pct=99.8, memory_mb=7800, memory_limit_mb=8192,
                latency_p50_ms=12000, latency_p99_ms=30000, error_rate_pct=84.0,
                rps=12000, replica_count=3, max_replicas=3,
            ),
            config=ServiceConfig(
                current={"maxmemory_policy": "allkeys-lru", "maxmemory_mb": 7680},
                desired={"maxmemory_policy": "allkeys-lru", "maxmemory_mb": 7680},
            ),
            dependencies=["database-primary", "database-replica"],
            logs=[
                f"{t_base}:10Z [WARN ] cache-layer: cache hit rate dropping (was 94%, now 61%)",
                f"{t_base}:12Z [WARN ] cache-layer: stampede detected — 8,000 concurrent DB fallback queries",
                f"{t_base}:14Z [ERROR] cache-layer: CPU 99.8% — unable to serve requests",
                f"{t_base}:15Z [ERROR] cache-layer: connection queue full — dropping new connections",
                f"{t_base}:16Z [ERROR] cache-layer: GETs timing out — underlying data still stale from lagging replica",
            ],
        ),

        "read-service": ServiceState(
            name="read-service",
            display_name="Read Service",
            status="degraded",
            version="v6.3.0",
            metrics=ServiceMetrics(
                cpu_pct=81.0, memory_mb=620, memory_limit_mb=1024,
                latency_p50_ms=9200, latency_p99_ms=45000, error_rate_pct=68.0,
                rps=420, replica_count=3, max_replicas=10,
            ),
            config=ServiceConfig(
                current={"cache_ttl_s": 300, "db_timeout_s": 5},
                desired={"cache_ttl_s": 300, "db_timeout_s": 5},
            ),
            dependencies=["cache-layer", "database-replica"],
            logs=[
                f"{t_base}:10Z [WARN ] read-service: cache miss rate climbing (38% misses)",
                f"{t_base}:12Z [WARN ] read-service: falling back to DB — cache-layer latency 12s",
                f"{t_base}:14Z [ERROR] read-service: database-replica returning stale data (lag: 182s)",
                f"{t_base}:16Z [ERROR] read-service: 68% error rate — both cache and replica degraded",
            ],
        ),

        "database-primary": ServiceState(
            name="database-primary",
            display_name="Database Primary",
            status="degraded",
            version="postgresql-15.4",
            metrics=ServiceMetrics(
                cpu_pct=45.0, memory_mb=6100, memory_limit_mb=8192,
                latency_p50_ms=320, latency_p99_ms=4800, error_rate_pct=8.0,
                rps=80, replica_count=1, max_replicas=1,
            ),
            config=ServiceConfig(
                current={"max_connections": 200, "shared_buffers_mb": 1024},
                desired={"max_connections": 200, "shared_buffers_mb": 1024},
            ),
            dependencies=[],
            logs=[
                f"{t_base}:08Z [WARN ] database-primary: disk I/O wait elevated (iowait: 68%)",
                f"{t_base}:09Z [WARN ] database-primary: WAL write rate degraded — replication lag building",
                f"{t_base}:10Z [ERROR] database-primary: replication slot lag 120s and growing",
                f"{t_base}:12Z [ERROR] database-primary: checkpoint taking 45s (normal: 2s) — disk I/O saturation",
                f"{t_base}:14Z [ERROR] database-primary: iowait 82% — disk throughput 4MB/s (normal: 200MB/s)",
                f"{t_base}:15Z [INFO ] database-primary: standby 'database-standby' available and up-to-date",
            ],
        ),

        "database-replica": ServiceState(
            name="database-replica",
            display_name="Database Replica",
            status="degraded",
            version="postgresql-15.4",
            metrics=ServiceMetrics(
                cpu_pct=38.0, memory_mb=5800, memory_limit_mb=8192,
                latency_p50_ms=150, latency_p99_ms=900, error_rate_pct=0.0,
                rps=960, replica_count=1, max_replicas=1,
            ),
            config=ServiceConfig(
                current={"max_connections": 300, "hot_standby": "on"},
                desired={"max_connections": 300, "hot_standby": "on"},
            ),
            dependencies=["database-primary"],
            logs=[
                f"{t_base}:10Z [WARN ] database-replica: replication lag 95s from primary",
                f"{t_base}:12Z [WARN ] database-replica: replication lag 182s — serving stale reads",
                f"{t_base}:14Z [WARN ] database-replica: lag still growing — primary write throughput limited",
            ],
        ),

        "database-standby": ServiceState(
            name="database-standby",
            display_name="Database Standby",
            status="running",
            version="postgresql-15.4",
            metrics=ServiceMetrics(
                cpu_pct=12.0, memory_mb=4200, memory_limit_mb=8192,
                latency_p50_ms=5, latency_p99_ms=30, error_rate_pct=0.0,
                rps=0, replica_count=1, max_replicas=1,
            ),
            config=ServiceConfig(
                current={"max_connections": 200, "hot_standby": "on"},
                desired={"max_connections": 200, "hot_standby": "on"},
            ),
            dependencies=[],
            logs=[
                f"{t_base}:00Z [INFO ] database-standby: streaming replication healthy — lag 0.1s",
                f"{t_base}:08Z [INFO ] database-standby: diverging from primary (primary I/O issues)",
                f"{t_base}:15Z [INFO ] database-standby: healthy and ready for promotion",
            ],
        ),

        "notification-service": ServiceState(
            name="notification-service",
            display_name="Notification Service",
            status="running",
            version="v2.1.0",
            metrics=ServiceMetrics(
                cpu_pct=5.2, memory_mb=128, memory_limit_mb=512,
                latency_p50_ms=22, latency_p99_ms=110, error_rate_pct=0.0,
                rps=40, replica_count=2, max_replicas=4,
            ),
            config=ServiceConfig(
                current={"smtp_pool_size": 10, "retry_attempts": 3},
                desired={"smtp_pool_size": 10, "retry_attempts": 3},
            ),
            dependencies=[],
            logs=[
                f"{t_base}:00Z [INFO ] notification-service: email queue processing normally",
                f"{t_base}:20Z [INFO ] notification-service: all systems nominal",
            ],
        ),

        "inventory-service": ServiceState(
            name="inventory-service",
            display_name="Inventory Service",
            status="running",
            version="v3.0.4",
            metrics=ServiceMetrics(
                cpu_pct=18.0, memory_mb=310, memory_limit_mb=1024,
                latency_p50_ms=15, latency_p99_ms=60, error_rate_pct=0.0,
                rps=180, replica_count=2, max_replicas=6,
            ),
            config=ServiceConfig(
                current={"cache_backend": "local", "write_timeout_s": 5},
                desired={"cache_backend": "local", "write_timeout_s": 5},
            ),
            dependencies=["database-primary"],
            logs=[
                f"{t_base}:00Z [INFO ] inventory-service: stock sync complete",
                f"{t_base}:20Z [INFO ] inventory-service: write operations normal (uses primary directly)",
            ],
        ),
    }

    alerts = [
        Alert(
            id="alert-301",
            severity="critical",
            service="api-gateway",
            title="API Gateway 72% Error Rate",
            description="api-gateway serving 72% 5xx responses. P99 latency >60s. Multiple upstreams degraded.",
            fired_at="2026-04-10T02:15:18Z",
            labels={"team": "platform", "env": "prod", "impact": "user-facing"},
        ),
        Alert(
            id="alert-302",
            severity="critical",
            service="cache-layer",
            title="Cache Layer CPU Saturation",
            description="Redis cluster at 99.8% CPU. Cache hit rate dropped from 94% to 16%. Stampede detected.",
            fired_at="2026-04-10T02:15:14Z",
            labels={"team": "infrastructure", "env": "prod"},
        ),
        Alert(
            id="alert-303",
            severity="high",
            service="database-primary",
            title="Database Replication Lag >180s",
            description="Primary-to-replica replication lag is 182s and growing. Disk I/O iowait at 82%.",
            fired_at="2026-04-10T02:15:12Z",
            labels={"team": "dba", "env": "prod"},
        ),
        Alert(
            id="alert-304",
            severity="high",
            service="read-service",
            title="Read Service Serving Stale/Error Data",
            description="read-service returning 68% errors. Cache misses falling back to lagging replica.",
            fired_at="2026-04-10T02:15:16Z",
            labels={"team": "backend", "env": "prod"},
        ),
        Alert(
            id="alert-305",
            severity="medium",
            service="notification-service",
            title="Email Delivery Delay >5min",
            description="Notification emails delayed. Email queue depth 1,200 (normal: 50). SMTP pool busy.",  # red herring
            fired_at="2026-04-10T02:15:20Z",
            labels={"team": "comms", "env": "prod"},
        ),
        Alert(
            id="alert-306",
            severity="medium",
            service="inventory-service",
            title="Inventory Sync Scheduled Downtime",
            description="Inventory sync scheduled maintenance window 02:00–03:00 UTC. This is expected.",  # red herring
            fired_at="2026-04-10T02:00:00Z",
            labels={"team": "commerce", "env": "prod", "type": "maintenance"},
        ),
    ]

    return Scenario(
        name="cascading-failure",
        display_name="Cascading Failure — Disk I/O → Cache Stampede → API Overload",
        description=(
            "A complex multi-service cascading failure is causing widespread 5xx errors. "
            "Multiple services are reporting alerts including two red herrings. "
            "You must identify the root cause, execute a multi-step remediation, and verify recovery."
        ),
        difficulty="hard",
        start_time="2026-04-10T02:15:00Z",
        services=services,
        alerts=alerts,
        root_cause_service="database-primary",
        root_cause_issue="disk_io_saturation",
        resolution_actions=[
            "failover database-primary",
            "restart cache-layer",
        ],
        resolution_services=["database-primary", "cache-layer"],
        max_steps=25,
        hint=(
            "Root cause: database-primary disk I/O saturated → replication lag → "
            "cache stampede → API overload. Fix: failover DB primary then restart cache-layer."
        ),
        request_ids=req_ids,
    )


# ─────────────────────────────────────────────────────────
#  Public factory
# ─────────────────────────────────────────────────────────

SCENARIOS = {
    "service-restart": _build_service_restart_scenario,
    "config-drift": _build_config_drift_scenario,
    "cascading-failure": _build_cascading_failure_scenario,
}

ALL_TASK_NAMES = list(SCENARIOS.keys())


def get_scenario(name: str, seed: Optional[int] = None) -> Scenario:
    """
    Return a freshly-built Scenario instance.

    Args:
        name: One of "service-restart", "config-drift", "cascading-failure".
        seed: Optional random seed for reproducibility.

    Returns:
        A populated Scenario dataclass.

    Raises:
        ValueError: If the scenario name is not recognised.
    """
    if name not in SCENARIOS:
        valid = ", ".join(SCENARIOS)
        raise ValueError(f"Unknown scenario '{name}'. Valid options: {valid}")

    factory = SCENARIOS[name]
    if seed is not None:
        return factory(seed=seed)
    return factory()
