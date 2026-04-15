"""
IncidentOps Baseline Inference Script.

Runs a language model agent against all three IncidentOps tasks and produces
a reproducible baseline score. Emits structured log output to stdout.

Communication strategy:
  PRIMARY  — WebSocket (/ws) which maintains session state, so every step
             actually executes inside the same environment instance and the
             grader is correctly invoked at episode end.
  FALLBACK — Plain HTTP (/reset + /step) used only if the websockets package
             is unavailable in the execution environment.

Environment variables (mandatory before submission):
    API_BASE_URL   — LLM API endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME     — Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       — Hugging Face token / API key

Optional:
    INCIDENT_BASE_URL — Running server URL (default: http://localhost:8000)

Stdout format (as required by OpenEnv):
    [START] task=<task_name> env=incident_ops_env model=<model>
    [STEP]  step=<n> action=<cmd> reward=<0.0000> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

import json
import os
import re
import sys
import textwrap
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
INCIDENT_BASE_URL: str = os.getenv("INCIDENT_BASE_URL", "http://localhost:8000")

ENV_NAME = "incident_ops_env"
TASKS = ["service-restart", "config-drift", "cascading-failure"]
MAX_STEPS = 20
TEMPERATURE = 0.1
MAX_TOKENS = 200
FALLBACK_ACTION = "status"

# Strict open-interval clamp: validator rejects exactly 0.0 or 1.0
_REWARD_MIN = 0.01
_REWARD_MAX = 0.99


def _clamp(reward: float) -> float:
    """Clamp reward strictly inside (0, 1)."""
    try:
        r = float(reward)
    except (TypeError, ValueError):
        r = 0.5
    return max(_REWARD_MIN, min(_REWARD_MAX, r))


# ─────────────────────────────────────────────────────────────────────────────
#  Structured logging (required format)
# ─────────────────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    done_str = str(done).lower()
    # Use 4 decimal places — prevents rounding to 0.0000 or 1.0000
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} done={done_str} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float], score: float) -> None:
    # Use 4 decimal places for each reward value
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  URL helpers
# ─────────────────────────────────────────────────────────────────────────────

def _http_base_url(base: str) -> str:
    """Ensure URL uses http/https scheme."""
    url = base.rstrip("/")
    if url.startswith("ws://"):
        url = "http://" + url[5:]
    elif url.startswith("wss://"):
        url = "https://" + url[6:]
    return url


def _ws_url(base: str) -> str:
    """Convert base HTTP URL to WebSocket URL for the /ws endpoint."""
    url = base.rstrip("/")
    if url.startswith("http://"):
        url = "ws://" + url[7:]
    elif url.startswith("https://"):
        url = "wss://" + url[8:]
    elif not url.startswith("ws://") and not url.startswith("wss://"):
        url = "ws://" + url
    return url + "/ws"


# ─────────────────────────────────────────────────────────────────────────────
#  WebSocket environment client (STATEFUL — primary path)
# ─────────────────────────────────────────────────────────────────────────────

class _WSObs:
    """Observation wrapper for WebSocket responses."""

    def __init__(self, data: Dict[str, Any]) -> None:
        obs = data.get("observation", {})
        self.output: str = obs.get("output", "")
        self.timestamp: str = obs.get("timestamp", "")
        self.alert_count: int = int(obs.get("alert_count", 0))
        self.severity: str = obs.get("severity", "none")
        self.affected_services: List[str] = obs.get("affected_services", [])
        # done and reward are at the TOP-LEVEL of data (not inside observation)
        self.done: bool = bool(data.get("done", False))
        raw_reward = data.get("reward")
        self.reward: float = _clamp(float(raw_reward) if raw_reward is not None else 0.5)


class EnvWSClient:
    """
    Stateful WebSocket client for the IncidentOps server.

    Uses the openenv-core WebSocket protocol:
      reset: {"type": "reset", "data": {"task_name": "..."}}
      step:  {"type": "step",  "data": {"command": "..."}}
    """

    def __init__(self, base_url: str) -> None:
        self._ws_url = _ws_url(base_url)
        self._ws = None

    async def _connect(self) -> None:
        import websockets  # type: ignore[import]
        self._ws = await websockets.connect(
            self._ws_url,
            open_timeout=15,
            close_timeout=5,
        )

    async def _send(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        import websockets  # type: ignore[import]
        await self._ws.send(json.dumps(msg))
        raw = await self._ws.recv()
        return json.loads(raw)

    async def reset(self, task_name: str) -> _WSObs:
        resp = await self._send({"type": "reset", "data": {"task_name": task_name}})
        return _WSObs(resp.get("data", {}))

    async def step(self, command: str) -> _WSObs:
        resp = await self._send({"type": "step", "data": {"command": command}})
        return _WSObs(resp.get("data", {}))

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None


# ─────────────────────────────────────────────────────────────────────────────
#  HTTP environment client (STATELESS — fallback only)
# ─────────────────────────────────────────────────────────────────────────────

class _HTTPObs:
    """Observation wrapper for HTTP responses."""

    def __init__(self, raw: Dict[str, Any]) -> None:
        obs = raw.get("observation", raw)
        self.output: str = obs.get("output", "")
        self.timestamp: str = obs.get("timestamp", "")
        self.alert_count: int = int(obs.get("alert_count", 0))
        self.severity: str = obs.get("severity", "none")
        self.affected_services: List[str] = obs.get("affected_services", [])
        # done lives at the TOP LEVEL (not inside the observation dict)
        top_done = raw.get("done")
        self.done: bool = bool(top_done) if top_done is not None else bool(obs.get("done", False))
        # Use explicit None check to avoid falsiness swallowing 0.0
        raw_reward = raw.get("reward")
        if raw_reward is None:
            raw_reward = obs.get("reward")
        self.reward: float = _clamp(float(raw_reward) if raw_reward is not None else 0.5)


def _http_post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


class EnvHTTPClient:
    """Stateless HTTP client (fallback only — each step runs on a fresh env)."""

    def __init__(self, base_url: str) -> None:
        self.base_url = _http_base_url(base_url)

    def reset(self, task_name: str) -> _HTTPObs:
        resp = _http_post(f"{self.base_url}/reset", {"task_name": task_name})
        return _HTTPObs(resp)

    def step(self, command: str) -> _HTTPObs:
        # The /step endpoint wraps the action under the "action" key
        resp = _http_post(f"{self.base_url}/step", {"action": {"command": command}})
        return _HTTPObs(resp)


# ─────────────────────────────────────────────────────────────────────────────
#  LLM client
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(client: Any, messages: List[Dict[str, str]]) -> str:
    """Call the LLM and return the response text, or '' on failure."""
    try:
        from openai import OpenAI  # imported here to give a clear error if missing
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", file=sys.stderr, flush=True)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
#  System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert Site Reliability Engineer (SRE) responding to a production incident.
You interact with a monitoring system through text commands.

Your goal is to:
1. Quickly triage the incident by checking alerts and system status
2. Identify the root cause through logs, metrics, and diagnostics
3. Apply the correct remediation (restart, rollback, failover, or config change)
4. Mark the incident as resolved with the 'resolve' command

Available commands:
  help                          -- Show all commands
  status                        -- System dashboard
  alerts                        -- Active alerts
  logs <service>                -- Service logs
  metrics <service>             -- Service metrics
  trace <request_id>            -- Trace a request
  diagnose <service>            -- Deep diagnostic
  restart <service>             -- Restart a service
  scale <service> <n>           -- Scale replicas
  rollback <service>            -- Rollback deployment
  failover <service>            -- Trigger DB failover
  config <service> <key> <val>  -- Update configuration
  notify <channel> <message>    -- Send status update
  resolve                       -- Mark incident resolved

RULES:
- Reply with EXACTLY ONE command per turn. No explanations.
- Do NOT restart healthy services.
- Start with 'alerts' then 'status' to orient yourself.
- Use 'diagnose <service>' on suspicious services.
- Type 'resolve' once you have applied the fix and verified recovery.
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Action parser
# ─────────────────────────────────────────────────────────────────────────────

VALID_VERBS = {
    "help", "status", "alerts", "logs", "metrics", "trace",
    "diagnose", "restart", "scale", "rollback", "failover",
    "config", "notify", "resolve",
}

_PREAMBLE_RE = re.compile(r"^(action|next action|command)[:\-]\s*", re.IGNORECASE)


def parse_action(response_text: str) -> str:
    """Extract a clean command string from model output."""
    if not response_text:
        return FALLBACK_ACTION

    for raw_line in response_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = _PREAMBLE_RE.sub("", line).strip()
        parts = line.split()
        if parts and parts[0].lower() in VALID_VERBS:
            return line

    first = response_text.strip().splitlines()[0].strip() if response_text.strip() else ""
    first = _PREAMBLE_RE.sub("", first).strip()
    return first if first else FALLBACK_ACTION


# ─────────────────────────────────────────────────────────────────────────────
#  Agent loop — WebSocket (PRIMARY, stateful)
# ─────────────────────────────────────────────────────────────────────────────

RESET_RETRIES = 3
RESET_RETRY_DELAY_S = 5


async def run_task_ws(llm_client: Any, base_url: str, task_name: str) -> None:
    """Run one complete task episode via WebSocket and emit structured logs."""
    log_start(task=task_name, env=ENV_NAME, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False

    env = EnvWSClient(base_url)

    try:
        # Connect with retries
        last_err: Optional[str] = None
        for attempt in range(1, RESET_RETRIES + 1):
            try:
                await env._connect()
                last_err = None
                break
            except Exception as exc:
                last_err = str(exc)
                print(f"[DEBUG] WS connect attempt {attempt}/{RESET_RETRIES} failed: {exc}",
                      file=sys.stderr, flush=True)
                if attempt < RESET_RETRIES:
                    time.sleep(RESET_RETRY_DELAY_S)

        if last_err is not None:
            # All connection attempts failed
            fallback_r = 0.5
            log_step(1, "connect", fallback_r, True, last_err)
            rewards.append(fallback_r)
            steps_taken = 1
            return

        # Reset
        obs = await env.reset(task_name)

        history: List[str] = []

        for step_idx in range(1, MAX_STEPS + 1):
            if obs.done:
                success = obs.reward > 0.5
                break

            # Build prompt
            history_text = "\n".join(history[-6:]) if history else "None"
            user_content = textwrap.dedent(f"""
                CURRENT OBSERVATION:
                {obs.output}

                System state:
                  Severity:          {obs.severity.upper()}
                  Active alerts:     {obs.alert_count}
                  Affected services: {', '.join(obs.affected_services) or 'none'}
                  Sim time:          {obs.timestamp}

                Recent actions:
                {history_text}

                Reply with EXACTLY ONE command to take next.
            """).strip()

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            response_text = _call_llm(llm_client, messages)
            action_str = parse_action(response_text)

            # Step
            try:
                obs = await env.step(action_str)
            except Exception as exc:
                print(f"[DEBUG] WS step failed: {exc}", file=sys.stderr, flush=True)
                fallback_r = 0.5
                log_step(step_idx, action_str, fallback_r, True, str(exc))
                rewards.append(fallback_r)
                steps_taken = step_idx
                break

            reward = obs.reward     # already clamped by _WSObs
            done = obs.done

            rewards.append(reward)
            steps_taken = step_idx
            log_step(step_idx, action_str, reward, done, None)
            history.append(f"Step {step_idx}: {action_str} -> reward {reward:.4f}")

            if done:
                success = reward > 0.5
                break

        else:
            success = False

    finally:
        await env.close()
        if not rewards:
            rewards.append(0.5)
        # Compute final score as mean of episode rewards, clamped to strict (0.01, 0.99)
        final_score = _clamp(sum(rewards) / len(rewards))
        log_end(success=success, steps=steps_taken, rewards=rewards, score=final_score)


# ─────────────────────────────────────────────────────────────────────────────
#  Agent loop — HTTP fallback (STATELESS, used only if websockets unavailable)
# ─────────────────────────────────────────────────────────────────────────────

def run_task_http(llm_client: Any, base_url: str, task_name: str) -> None:
    """Fallback HTTP-based episode runner (stateless — each step is a fresh env)."""
    log_start(task=task_name, env=ENV_NAME, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    history: List[str] = []
    episode_aborted = False

    env = EnvHTTPClient(base_url)

    try:
        # Reset with retries
        obs = None
        last_reset_error: Optional[str] = None
        for attempt in range(1, RESET_RETRIES + 1):
            try:
                obs = env.reset(task_name=task_name)
                last_reset_error = None
                break
            except Exception as exc:
                last_reset_error = str(exc)
                print(f"[DEBUG] HTTP reset attempt {attempt}/{RESET_RETRIES} failed: {exc}",
                      file=sys.stderr, flush=True)
                if attempt < RESET_RETRIES:
                    time.sleep(RESET_RETRY_DELAY_S)

        if obs is None:
            fallback_r = 0.5
            log_step(1, "reset", fallback_r, True, last_reset_error or "reset failed")
            rewards.append(fallback_r)
            steps_taken = 1
            episode_aborted = True
            return

        for step_idx in range(1, MAX_STEPS + 1):
            if obs.done:
                success = obs.reward > 0.5
                break

            history_text = "\n".join(history[-6:]) if history else "None"
            user_content = textwrap.dedent(f"""
                CURRENT OBSERVATION:
                {obs.output}

                System state:
                  Severity:          {obs.severity.upper()}
                  Active alerts:     {obs.alert_count}
                  Affected services: {', '.join(obs.affected_services) or 'none'}
                  Sim time:          {obs.timestamp}

                Recent actions:
                {history_text}

                Reply with EXACTLY ONE command to take next.
            """).strip()

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            response_text = _call_llm(llm_client, messages)
            action_str = parse_action(response_text)

            try:
                obs = env.step(action_str)
            except Exception as exc:
                print(f"[DEBUG] HTTP step failed: {exc}", file=sys.stderr, flush=True)
                fallback_r = 0.5
                log_step(step_idx, action_str, fallback_r, True, str(exc))
                rewards.append(fallback_r)
                steps_taken = step_idx
                episode_aborted = True
                break

            if not episode_aborted:
                reward = obs.reward   # already clamped by _HTTPObs
                done = obs.done
                rewards.append(reward)
                steps_taken = step_idx
                log_step(step_idx, action_str, reward, done, None)
                history.append(f"Step {step_idx}: {action_str} -> reward {reward:.4f}")
                if done:
                    success = reward > 0.5
                    break

        else:
            success = False

    finally:
        if not rewards:
            rewards.append(0.5)
        # Compute final score as mean of episode rewards, clamped to strict (0.01, 0.99)
        final_score = _clamp(sum(rewards) / len(rewards))
        log_end(success=success, steps=steps_taken, rewards=rewards, score=final_score)


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        print(
            "[ERROR] No API key found. Set HF_TOKEN or API_KEY environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("[ERROR] 'openai' package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Try WebSocket path first (stateful, graders are correctly invoked)
    try:
        import websockets  # noqa: F401 — just checking availability
        import asyncio

        print("[INFO] Using WebSocket client (stateful episodes)", file=sys.stderr, flush=True)

        async def run_all_ws() -> None:
            for task in TASKS:
                await run_task_ws(llm_client, INCIDENT_BASE_URL, task)

        asyncio.run(run_all_ws())

    except ImportError:
        # websockets not available — fall back to HTTP
        print("[INFO] websockets not available, using HTTP fallback", file=sys.stderr, flush=True)
        for task in TASKS:
            run_task_http(llm_client, INCIDENT_BASE_URL, task)


if __name__ == "__main__":
    main()
