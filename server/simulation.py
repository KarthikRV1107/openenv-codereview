"""
IncidentOps Simulation Engine.

Tracks live service states and translates agent commands into state mutations
and rich terminal-style output. This module is the 'physics engine' of the
environment — it knows how services respond to restarts, config changes, etc.
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional, Tuple

from .scenarios import Scenario, ServiceState


# ---------------------------------------------------------
#  Helpers
# ---------------------------------------------------------

def _bar(value: float, max_val: float, width: int = 20) -> str:
    """Render a text progress bar, e.g. [########----] 73%"""
    filled = int(round(width * value / max(max_val, 0.001)))
    filled = min(filled, width)
    bar = "#" * filled + "-" * (width - filled)
    pct = value / max(max_val, 0.001) * 100
    return f"[{bar}] {pct:.1f}%"


def _severity_icon(severity: str) -> str:
    icons = {
        "critical": "[CRIT]",
        "high": "[HIGH]",
        "medium": "[MED]",
        "low": "[LOW]",
    }
    return icons.get(severity, "[NONE]")


def _status_icon(status: str) -> str:
    icons = {
        "running": "[OK]",
        "degraded": "[WARN] ",
        "down": "[ERROR]",
    }
    return icons.get(status, "[?]")


# ---------------------------------------------------------
#  SimulationEngine
# ---------------------------------------------------------

class SimulationEngine:
    """
    Manages the live state of all services in a scenario and processes
    agent commands, returning human-readable terminal output.
    """

    AVAILABLE_COMMANDS = [
        ("help",                              "List all available commands"),
        ("status",                            "System-wide dashboard of all services"),
        ("alerts",                            "Show active alerts with severity and details"),
        ("logs <service>",                    "View recent log entries for a service"),
        ("metrics <service>",                 "View CPU / memory / latency / error-rate"),
        ("trace <request_id>",               "Follow a request through the service mesh"),
        ("diagnose <service>",               "Run a full diagnostic report on a service"),
        ("restart <service>",                "Restart a service (brings healthy replicas up)"),
        ("scale <service> <replicas>",       "Scale service to N replicas"),
        ("rollback <service>",               "Roll back service to the previous deployment"),
        ("failover <service>",               "Promote standby and demote primary"),
        ("config <service> <key> <value>",   "Update a live configuration value"),
        ("notify <channel> <message>",       "Post a status update (e.g. notify oncall 'msg')"),
        ("resolve",                          "Declare the incident resolved and end the episode"),
    ]

    def __init__(self, scenario: Scenario) -> None:
        self._scenario = scenario
        # Deep-copy service states so mutations don't affect the scenario definition
        self._services: Dict[str, ServiceState] = {
            k: v for k, v in scenario.services.items()
        }
        self._resolved = False
        self._actions_taken: List[str] = []
        self._notified_channels: List[str] = []
        self._alert_list = list(scenario.alerts)  # mutable copy

    # -- Accessors --------------------------------------

    @property
    def resolved(self) -> bool:
        return self._resolved

    @property
    def actions_taken(self) -> List[str]:
        return list(self._actions_taken)

    @property
    def active_alerts(self) -> list:
        return list(self._alert_list)

    @property
    def affected_services(self) -> List[str]:
        return [
            name for name, svc in self._services.items()
            if svc.status in ("degraded", "down")
        ]

    @property
    def current_severity(self) -> str:
        sevs = [a.severity for a in self._alert_list]
        for level in ("critical", "high", "medium", "low"):
            if level in sevs:
                return level
        return "none"

    # -- Command dispatcher -----------------------------

    def execute(self, command: str) -> Tuple[str, float, bool]:
        """
        Execute a command string.

        Returns:
            (output_text, incremental_reward, done)
        """
        cmd = command.strip()
        if not cmd:
            return "[WARN]  Empty command. Type 'help' to see available commands.", 0.0, False

        self._actions_taken.append(cmd)
        parts = cmd.split(None, 3)
        verb = parts[0].lower()

        dispatch = {
            "help":     self._cmd_help,
            "status":   self._cmd_status,
            "alerts":   self._cmd_alerts,
            "logs":     self._cmd_logs,
            "metrics":  self._cmd_metrics,
            "trace":    self._cmd_trace,
            "diagnose": self._cmd_diagnose,
            "restart":  self._cmd_restart,
            "scale":    self._cmd_scale,
            "rollback": self._cmd_rollback,
            "failover": self._cmd_failover,
            "config":   self._cmd_config,
            "notify":   self._cmd_notify,
            "resolve":  self._cmd_resolve,
        }

        handler = dispatch.get(verb)
        if handler is None:
            return (
                f"[ERROR]  Unknown command: '{verb}'. Type 'help' to see available commands.",
                0.0,
                False,
            )

        try:
            return handler(parts)
        except Exception as exc:
            return f"[ERROR]  Command error: {exc}", 0.0, False

    # -- Individual command handlers --------------------

    def _cmd_help(self, parts: list) -> Tuple[str, float, bool]:
        lines = ["", "+------------------------------------------------------╗"]
        lines.append(  "|         IncidentOps Command Reference                |")
        lines.append(  "+------------------------------------------------------╝")
        lines.append("")
        for cmd, desc in self.AVAILABLE_COMMANDS:
            lines.append(f"  {cmd:<42} {desc}")
        lines.append("")
        lines.append("TIP: Start with 'status' and 'alerts' to orient yourself.")
        return "\n".join(lines), 0.0, False

    def _cmd_status(self, parts: list) -> Tuple[str, float, bool]:
        lines = [
            "",
            "+------------------------------------------------------╗",
            "|         System Status Dashboard                      |",
           f"|  Sim Time: {self._scenario.start_time:<43}|",
            "+------------------------------------------------------╝",
            "",
            f"  Active Alerts: {len(self._alert_list)}   "
            f"Severity: {self.current_severity.upper()}   "
            f"Affected Services: {len(self.affected_services)}",
            "",
            f"  {'SERVICE':<28} {'STATUS':<12} {'ERR%':<8} {'P99ms':<8} {'CPU%':<8}",
            "  " + "-" * 68,
        ]
        for name, svc in self._services.items():
            icon = _status_icon(svc.status)
            m = svc.metrics
            lines.append(
                f"  {icon} {svc.display_name:<26} {svc.status:<12} "
                f"{m.error_rate_pct:<8.1f} {m.latency_p99_ms:<8.0f} {m.cpu_pct:<8.1f}"
            )
        lines.append("")
        return "\n".join(lines), 0.02, False

    def _cmd_alerts(self, parts: list) -> Tuple[str, float, bool]:
        if not self._alert_list:
            return "\n  [OK]  No active alerts. All systems nominal.\n", 0.02, False

        lines = [
            "",
            "+------------------------------------------------------╗",
            "|         Active Alerts                                |",
            "+------------------------------------------------------╝",
            "",
        ]
        for alert in sorted(self._alert_list, key=lambda a: ["critical","high","medium","low"].index(a.severity)):
            icon = _severity_icon(alert.severity)
            lines.append(f"  {icon} [{alert.severity.upper()}] {alert.title}")
            lines.append(f"     Service:  {alert.service}")
            lines.append(f"     Fired at: {alert.fired_at}")
            lines.append(f"     Details:  {alert.description}")
            lines.append("")
        return "\n".join(lines), 0.03, False

    def _cmd_logs(self, parts: list) -> Tuple[str, float, bool]:
        if len(parts) < 2:
            return "Usage: logs <service>\nExample: logs api-gateway", 0.0, False

        service_name = parts[1].lower()
        svc = self._find_service(service_name)
        if svc is None:
            return self._unknown_service(service_name), 0.0, False

        # Determine investigative value
        is_root_cause = svc.name == self._scenario.root_cause_service
        is_affected = svc.status in ("degraded", "down")
        reward = 0.05 if is_root_cause else (0.03 if is_affected else 0.01)

        lines = [
            "",
            f"  Log stream — {svc.display_name} ({svc.name})",
            f"  Status: {svc.status}  |  Version: {svc.version}  |  Restarts: {svc.restart_count}",
            "  " + "-" * 60,
            "",
        ]
        if not svc.logs:
            lines.append("  (no log entries)")
        else:
            for entry in svc.logs:
                lines.append(f"  {entry}")
        lines.append("")
        return "\n".join(lines), reward, False

    def _cmd_metrics(self, parts: list) -> Tuple[str, float, bool]:
        if len(parts) < 2:
            return "Usage: metrics <service>\nExample: metrics payment-processor", 0.0, False

        service_name = parts[1].lower()
        svc = self._find_service(service_name)
        if svc is None:
            return self._unknown_service(service_name), 0.0, False

        is_root_cause = svc.name == self._scenario.root_cause_service
        is_affected = svc.status in ("degraded", "down")
        reward = 0.05 if is_root_cause else (0.03 if is_affected else 0.01)

        m = svc.metrics
        lines = [
            "",
            f"  Metrics — {svc.display_name} ({svc.name})",
            f"  Status: {_status_icon(svc.status)} {svc.status}  |  Version: {svc.version}",
            "  " + "-" * 60,
            "",
            f"  CPU Usage        {_bar(m.cpu_pct, 100)}  {m.cpu_pct:.1f}%",
            f"  Memory           {_bar(m.memory_mb, m.memory_limit_mb)}  {m.memory_mb:.0f}/{m.memory_limit_mb:.0f} MB",
            "",
            f"  Latency P50:     {m.latency_p50_ms:.0f} ms",
            f"  Latency P99:     {m.latency_p99_ms:.0f} ms",
            f"  Error Rate:      {m.error_rate_pct:.1f}%",
            f"  Throughput:      {m.rps:.0f} req/s",
            f"  Replicas:        {m.replica_count}/{m.max_replicas}",
        ]

        if svc.oom_killed:
            lines.append("")
            lines.append("  [WARN]   OOMKilled: YES — container was terminated due to memory limit exceeded")

        if svc.config.current:
            lines.append("")
            lines.append("  Configuration:")
            for k, v in svc.config.current.items():
                desired = svc.config.desired.get(k)
                flag = "  [WARN]  DRIFTED" if desired is not None and str(v) != str(desired) else ""
                lines.append(f"    {k}: {v}{flag}")

        if svc.dependencies:
            lines.append("")
            lines.append(f"  Dependencies: {', '.join(svc.dependencies)}")

        lines.append("")
        return "\n".join(lines), reward, False

    def _cmd_trace(self, parts: list) -> Tuple[str, float, bool]:
        if len(parts) < 2:
            return "Usage: trace <request_id>\nExample: trace req-48219", 0.0, False

        req_id = parts[1]
        valid_ids = self._scenario.request_ids
        if req_id not in valid_ids:
            return (
                f"[ERROR]  Request ID '{req_id}' not found in traces.\n"
                f"   Known request IDs: {', '.join(valid_ids[:3])} ...",
                0.0,
                False,
            )

        scenario_name = self._scenario.name
        lines = [
            "",
            f"  Distributed Trace — {req_id}",
            "  " + "-" * 60,
            "",
        ]

        if scenario_name == "service-restart":
            lines += [
                f"  api-gateway          → RECEIVED  {req_id}  +0ms",
                f"  api-gateway          → ROUTING   /api/v1/payments       +1ms",
                f"  payment-processor    → TIMEOUT   upstream not responding +5000ms",
                f"  api-gateway          → RESPONDED 503 Service Unavailable +5001ms",
                "",
                "  [CRIT]  Trace shows payment-processor as the failing upstream.",
            ]
        elif scenario_name == "config-drift":
            lines += [
                f"  api-gateway          → RECEIVED  {req_id}  +0ms",
                f"  api-gateway          → POOL WAIT awaiting connection slot +4200ms",
                f"  api-gateway          → TIMEOUT   connection pool exhausted +4201ms",
                f"  api-gateway          → RESPONDED 504 Gateway Timeout     +4202ms",
                "",
                "  [HIGH]  Trace shows api-gateway connection pool starvation.",
                "      Pool size config may have regressed (check 'metrics api-gateway').",
            ]
        else:  # cascading-failure
            lines += [
                f"  api-gateway          → RECEIVED  {req_id}  +0ms",
                f"  api-gateway          → ROUTING   /api/v1/products       +1ms",
                f"  read-service         → CACHE-HIT? NO — cache miss         +2ms",
                f"  cache-layer          → TIMEOUT   12,000ms wait           +12002ms",
                f"  read-service         → DB-FETCH   falling back to replica +12003ms",
                f"  database-replica     → STALE DATA lag=182s               +12050ms",
                f"  api-gateway          → RESPONDED 503 upstream error       +12100ms",
                "",
                "  [CRIT]  Trace shows: cache miss → cache timeout → stale DB replica.",
                "      The underlying cause is further upstream — check database-primary.",
            ]

        return "\n".join(lines), 0.05, False

    def _cmd_diagnose(self, parts: list) -> Tuple[str, float, bool]:
        if len(parts) < 2:
            return "Usage: diagnose <service>\nExample: diagnose database-primary", 0.0, False

        service_name = parts[1].lower()
        svc = self._find_service(service_name)
        if svc is None:
            return self._unknown_service(service_name), 0.0, False

        is_root_cause = svc.name == self._scenario.root_cause_service
        reward = 0.08 if is_root_cause else 0.02

        lines = [
            "",
            f"  -- Diagnostic Report: {svc.display_name} ------------------",
            f"  Status:   {_status_icon(svc.status)} {svc.status.upper()}",
            f"  Version:  {svc.version}",
            f"  Restarts: {svc.restart_count}",
            "",
        ]

        if svc.name == "payment-processor" and svc.oom_killed:
            lines += [
                "  ROOT CAUSE INDICATORS:",
                "  ✗ OOMKilled:    Container exceeded memory limit and was killed by the OS.",
                "  ✗ Replicas:     0/4 healthy (CrashLoopBackOff)",
                "  ✗ Heap usage:   Was at 430MB / 480MB limit before OOM event",
                "",
                "  RECOMMENDATION: Restart payment-processor to restore service.",
                "  (Consider increasing heap limit in a follow-up change.)",
            ]
        elif svc.name == "api-gateway" and self._scenario.name == "config-drift":
            lines += [
                "  ANOMALY DETECTED:",
                "  ✗ Connection pool exhausted: pool_size=5 (expected: 100)",
                "  ✗ Deployment v3.1.5 occurred 8 min ago — config may have regressed",
                "  ✓ CPU and memory are within normal bounds",
                "  ✓ No code changes — config-only regression suspected",
                "",
                "  RECOMMENDATION: Rollback api-gateway OR fix config:",
                "    config api-gateway pool_size 100",
            ]
        elif svc.name == "database-primary" and self._scenario.name == "cascading-failure":
            lines += [
                "  ROOT CAUSE INDICATORS:",
                "  ✗ Disk I/O:      iowait=82% — disk throughput severely degraded",
                "  ✗ WAL writes:    Checkpoint taking 45s (normal: 2s)",
                "  ✗ Replication:   Lag building (182s) — replica serving stale data",
                "  ✓ CPU/Memory:    Within normal limits (disk I/O is the bottleneck)",
                "  ✓ Standby:       database-standby is healthy and ready for promotion",
                "",
                "  RECOMMENDATION:",
                "  1. failover database-primary  (promote standby, reduce replication lag)",
                "  2. restart cache-layer        (flush stale cache, recover from stampede)",
            ]
        else:
            # Generic degraded service
            m = svc.metrics
            lines += [
                f"  Metrics Summary:",
                f"  {'cpu_pct':<20} {m.cpu_pct:.1f}%",
                f"  {'memory_mb':<20} {m.memory_mb:.0f}/{m.memory_limit_mb:.0f} MB",
                f"  {'error_rate_pct':<20} {m.error_rate_pct:.1f}%",
                f"  {'latency_p99_ms':<20} {m.latency_p99_ms:.0f} ms",
                "",
                "  Config values:",
            ]
            for k, v in svc.config.current.items():
                desired = svc.config.desired.get(k)
                flag = "  ← DRIFTED (expected: {})".format(desired) if desired is not None and str(v) != str(desired) else ""
                lines.append(f"    {k}: {v}{flag}")

            lines.append("")
            if svc.status == "running":
                lines.append("  [OK]  No anomalies detected. This service appears healthy.")
            else:
                lines.append("  [WARN]   Service is degraded but no direct root cause identified here.")
                lines.append("       Check its dependencies.")

        lines.append("")
        return "\n".join(lines), reward, False

    def _cmd_restart(self, parts: list) -> Tuple[str, float, bool]:
        if len(parts) < 2:
            return "Usage: restart <service>\nExample: restart payment-processor", 0.0, False

        service_name = parts[1].lower()
        svc = self._find_service(service_name)
        if svc is None:
            return self._unknown_service(service_name), 0.0, False

        scenario = self._scenario
        is_correct = service_name in [s.lower() for s in scenario.resolution_services]

        if svc.status == "running":
            # Restarting a healthy service is penalised
            return (
                f"  [WARN]   {svc.display_name} is already healthy (status: running).\n"
                f"       Restarting a healthy service may cause unnecessary downtime.\n"
                f"       Restart aborted. Use 'status' to review which services need attention.\n",
                -0.05,
                False,
            )

        # Perform restart
        svc.restart_count += 1
        was_oom = svc.oom_killed

        if is_correct and "restart" in " ".join(scenario.resolution_actions):
            # Resolving restart
            svc.status = "running"
            svc.oom_killed = False
            svc.metrics.error_rate_pct = 0.1
            svc.metrics.latency_p50_ms = 14
            svc.metrics.latency_p99_ms = 52
            svc.metrics.rps = svc.metrics.rps if svc.metrics.rps > 0 else 200
            svc.metrics.replica_count = max(svc.metrics.replica_count, 1)
            # Clear alerts for this service
            self._alert_list = [a for a in self._alert_list if a.service != service_name]
            suffix = "\n  [OK]  Service successfully restarted. Replicas healthy. Alerts cleared."
            reward = 0.35
        else:
            # Partial restart (wrong service or not the fix)
            svc.metrics.restart_count = svc.restart_count
            suffix = f"\n  [WARN]   {svc.display_name} restarted but root cause may still be present."
            reward = -0.03

        lines = [
            "",
            f"  [RESTART]  Restarting {svc.display_name} ...",
            f"      Sending SIGTERM to {svc.metrics.replica_count} pod(s) ...",
            "      Waiting for graceful shutdown (30s timeout) ...",
            "      Pulling image ... done",
            "      Starting new pod(s) ...",
            "      Health checks passing [OK]",
            suffix,
            "",
        ]
        return "\n".join(lines), reward, False

    def _cmd_scale(self, parts: list) -> Tuple[str, float, bool]:
        if len(parts) < 3:
            return "Usage: scale <service> <replicas>\nExample: scale web-server 5", 0.0, False

        service_name = parts[1].lower()
        svc = self._find_service(service_name)
        if svc is None:
            return self._unknown_service(service_name), 0.0, False

        try:
            target = int(parts[2])
        except ValueError:
            return f"[ERROR]  '{parts[2]}' is not a valid replica count.", 0.0, False

        if target > svc.metrics.max_replicas:
            return (
                f"  [ERROR]  Cannot scale beyond max replicas ({svc.metrics.max_replicas}).\n"
                f"       Requested: {target}. Use a value ≤ {svc.metrics.max_replicas}.",
                0.0,
                False,
            )

        old = svc.metrics.replica_count
        svc.metrics.replica_count = target
        return (
            f"\n  [OK]  {svc.display_name} scaled from {old} → {target} replicas.\n",
            0.02,
            False,
        )

    def _cmd_rollback(self, parts: list) -> Tuple[str, float, bool]:
        if len(parts) < 2:
            return "Usage: rollback <service>\nExample: rollback api-gateway", 0.0, False

        service_name = parts[1].lower()
        svc = self._find_service(service_name)
        if svc is None:
            return self._unknown_service(service_name), 0.0, False

        scenario = self._scenario
        is_correct = (
            service_name in [s.lower() for s in scenario.resolution_services]
            and any("rollback" in act for act in scenario.resolution_actions)
        )

        if svc.status == "running" and not is_correct:
            return (
                f"  [WARN]   {svc.display_name} appears healthy. Rollback unnecessary.\n"
                f"       Check 'metrics {service_name}' before rolling back.\n",
                -0.02,
                False,
            )

        if is_correct:
            # Apply the desired config values
            svc.config.current = dict(svc.config.desired)
            svc.status = "running"
            svc.metrics.error_rate_pct = 0.05
            svc.metrics.latency_p50_ms = 10
            svc.metrics.latency_p99_ms = 42
            self._alert_list = [a for a in self._alert_list if a.service != service_name]
            suffix = "  [OK]  Rollback successful. Config restored to previous known-good values."
            reward = 0.35
        else:
            suffix = f"  [WARN]   {svc.display_name} rolled back but underlying problem may persist."
            reward = -0.02

        lines = [
            "",
            f"  [ROLLBACK]   Rolling back {svc.display_name} ...",
            f"      Previous version restore initiated ...",
            "      Config values reverted to pre-deployment snapshot ...",
            "      Health checks passing [OK]",
            "",
            suffix,
            "",
        ]
        return "\n".join(lines), reward, False

    def _cmd_failover(self, parts: list) -> Tuple[str, float, bool]:
        if len(parts) < 2:
            return "Usage: failover <service>\nExample: failover database-primary", 0.0, False

        service_name = parts[1].lower()
        svc = self._find_service(service_name)
        if svc is None:
            return self._unknown_service(service_name), 0.0, False

        scenario = self._scenario
        is_correct = (
            service_name in [s.lower() for s in scenario.resolution_services]
            and any("failover" in act for act in scenario.resolution_actions)
        )

        if is_correct:
            svc.status = "degraded"          # old primary now demoted
            svc.metrics.iowait = 0            # type: ignore[attr-defined] — illustrative
            # Promote standby if it exists
            standby_key = service_name.replace("primary", "standby")
            if standby_key in self._services:
                self._services[standby_key].status = "running"
            # Update replica too
            replica_key = service_name.replace("primary", "replica")
            if replica_key in self._services:
                self._services[replica_key].metrics.latency_p99_ms = 25
                self._services[replica_key].metrics.error_rate_pct = 0.0

            self._alert_list = [
                a for a in self._alert_list
                if not (a.service == service_name and "replication" in a.title.lower())
            ]
            suffix = (
                "  [OK]  Failover complete. Standby promoted to primary.\n"
                "      Replication lag is now resolving. Cache should recover shortly."
            )
            reward = 0.30
        else:
            suffix = (
                f"  [WARN]   Failover for {svc.display_name} completed but may not be necessary.\n"
                f"       Verify with 'diagnose {service_name}'."
            )
            reward = -0.05

        lines = [
            "",
            f"  [FAILOVER]  Initiating failover for {svc.display_name} ...",
            "      Verifying standby readiness ...",
            "      Promoting standby to primary ...",
            "      Updating DNS / connection strings ...",
            "      Draining connections from old primary ...",
            "      Failover complete [OK]",
            "",
            suffix,
            "",
        ]
        return "\n".join(lines), reward, False

    def _cmd_config(self, parts: list) -> Tuple[str, float, bool]:
        if len(parts) < 4:
            return (
                "Usage: config <service> <key> <value>\n"
                "Example: config api-gateway pool_size 100",
                0.0,
                False,
            )

        service_name = parts[1].lower()
        key = parts[2]
        value = parts[3]

        svc = self._find_service(service_name)
        if svc is None:
            return self._unknown_service(service_name), 0.0, False

        scenario = self._scenario
        desired = svc.config.desired.get(key)
        is_correct_service = service_name in [s.lower() for s in scenario.resolution_services]
        is_correct_value = desired is not None and str(value) == str(desired)

        old_value = svc.config.current.get(key, "N/A")
        svc.config.current[key] = value

        if is_correct_service and is_correct_value:
            svc.status = "running"
            svc.metrics.error_rate_pct = 0.05
            svc.metrics.latency_p50_ms = 11
            svc.metrics.latency_p99_ms = 44
            self._alert_list = [a for a in self._alert_list if a.service != service_name]
            suffix = f"  [OK]  Config applied and verified. Service recovering."
            reward = 0.35
        else:
            suffix = f"  [WARN]   Config updated. Monitor service to confirm stability."
            reward = 0.01

        lines = [
            "",
            f"  [CONFIG]   Updating config on {svc.display_name} ...",
            f"       {key}: {old_value} → {value}",
            "      Hot-reload applied (no restart required) [OK]",
            "",
            suffix,
            "",
        ]
        return "\n".join(lines), reward, False

    def _cmd_notify(self, parts: list) -> Tuple[str, float, bool]:
        if len(parts) < 3:
            return (
                "Usage: notify <channel> <message>\n"
                "Example: notify oncall 'investigating high error rate on payment-processor'",
                0.0,
                False,
            )
        channel = parts[1]
        message = " ".join(parts[2:]).strip("'\"")
        self._notified_channels.append(channel)
        return (
            f"\n  [NOTIFY]  Notification sent to #{channel}:\n"
            f"       \"{message}\"\n",
            0.02,
            False,
        )

    def _cmd_resolve(self, parts: list) -> Tuple[str, float, bool]:
        self._resolved = True
        lines = [
            "",
            "+------------------------------------------------------╗",
            "|         Incident Resolved                            |",
            "+------------------------------------------------------╝",
            "",
            "  The incident has been marked as resolved.",
            "  Final score will be calculated based on your investigation",
            "  and remediation quality.",
            "",
        ]
        return "\n".join(lines), 0.0, True

    # -- Utilities --------------------------------------

    def _find_service(self, name: str) -> Optional[ServiceState]:
        # Exact match first
        if name in self._services:
            return self._services[name]
        # Prefix / partial match
        for key, svc in self._services.items():
            if key.startswith(name) or name in key:
                return svc
        return None

    def _unknown_service(self, name: str) -> str:
        known = ", ".join(sorted(self._services.keys()))
        return (
            f"[ERROR]  Service '{name}' not found.\n"
            f"   Known services: {known}"
        )
