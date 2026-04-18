"""CTOX as a Harbor installed agent.

The contract:

1. `install(environment)` uploads a pre-built CTOX bundle (a .tgz containing
   the full source tree plus a release `target/release/ctox` binary) into the
   task container and symlinks the binary onto PATH as `/usr/local/bin/ctox`.
   The bundle path on the host is supplied via the `CTOX_HOST_TARBALL`
   environment variable so the same adapter works across machines.

2. `run(instruction, environment, context)` starts `ctox service --foreground`
   inside the container, submits the task through `ctox chat --wait`, and
   points `--atif-out` at `/logs/agent/trajectory.json`. The provider API key
   and CTOX_ROOT are forwarded into the container via the `env=` kwarg. After
   the run, we copy the trajectory and forensic artifacts back to `self.logs_dir`
   on the host so Harbor can inspect the real service path.

3. `populate_context_post_run(context)` reads the downloaded trajectory and
   fills in token/cost accounting on the AgentContext. It is tolerant of a
   missing trajectory file — a failed run may not have produced one.

Design intent: the adapter is deliberately thin. All CTOX-specific behaviour
(queueing, continuity, plan execution, skill invocation) lives inside CTOX
itself. The adapter's job is just to bridge Harbor's installed-agent contract
to CTOX's canonical service + chat flow.
"""

from __future__ import annotations

import json
import os
import shlex
import time
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent


CTOX_HOST_TARBALL_ENV = "CTOX_HOST_TARBALL"
DEFAULT_CONTAINER_ROOT = "/opt/ctox"
DEFAULT_CODEX_HOME = "/root/.codex"
DEFAULT_ATIF_CONTAINER_PATH = "/logs/agent/trajectory.json"
DEFAULT_CONTEXT_LOG_CONTAINER_PATH = f"{DEFAULT_CONTAINER_ROOT}/runtime/context-log.jsonl"
DEFAULT_RUNTIME_DB_CONTAINER_PATH = f"{DEFAULT_CONTAINER_ROOT}/runtime/ctox.sqlite3"
DEFAULT_SERVICE_LOG_CONTAINER_PATH = "/logs/agent/service.log"
DEFAULT_RUN_TIMEOUT_SEC = 1800


class CtoxAgent(BaseInstalledAgent):
    """Harbor adapter that runs each task through the CTOX service path."""

    @staticmethod
    def name() -> str:  # type: ignore[override]
        return "ctox"

    async def install(self, environment: Any) -> None:  # type: ignore[override]
        tarball = os.environ.get(CTOX_HOST_TARBALL_ENV)
        if not tarball:
            raise RuntimeError(
                f"{CTOX_HOST_TARBALL_ENV} must point to a pre-built CTOX "
                "bundle tarball on the host"
            )
        tarball_path = Path(tarball).expanduser().resolve()
        if not tarball_path.is_file():
            raise RuntimeError(
                f"CTOX bundle tarball not found: {tarball_path}"
            )

        # Push the tarball and extract into the container.
        await environment.upload_file(str(tarball_path), "/tmp/ctox-bundle.tgz")

        turn_timeout = os.environ.get("CTOX_CHAT_TURN_TIMEOUT_SECS", "1200")
        extract_cmd = f"""
set -e
mkdir -p {DEFAULT_CONTAINER_ROOT}
tar -xzf /tmp/ctox-bundle.tgz -C {DEFAULT_CONTAINER_ROOT} --strip-components=1
chmod +x {DEFAULT_CONTAINER_ROOT}/target/release/ctox
optional_engine={DEFAULT_CONTAINER_ROOT}/tools/model-runtime/target/release/ctox-engine
[ -f "$optional_engine" ] && chmod +x "$optional_engine"

loader="{DEFAULT_CONTAINER_ROOT}/lib/ld-linux-x86-64.so.2"
libpath="{DEFAULT_CONTAINER_ROOT}/lib"

wrap_binary() {{
  target="$1"
  if [ ! -f "$target" ]; then
    return 0
  fi
  name="$(basename "$target")"
  if [ ! -f "$target.real" ]; then
    mv "$target" "$target.real"
  fi
  cat > "$target" <<EOF
#!/bin/sh
exec "$loader" --argv0 "$name" --library-path "$libpath" "$target.real" "\\$@"
EOF
  chmod +x "$target"
}}

wrap_binary "{DEFAULT_CONTAINER_ROOT}/target/release/ctox"
wrap_binary "{DEFAULT_CONTAINER_ROOT}/tools/model-runtime/target/release/ctox-engine"

mkdir -p {DEFAULT_CONTAINER_ROOT}/src
[ -f {DEFAULT_CONTAINER_ROOT}/src/main.rs ] || printf '%s\n' 'fn main() {{}}' > {DEFAULT_CONTAINER_ROOT}/src/main.rs

ln -sf {DEFAULT_CONTAINER_ROOT}/target/release/ctox /usr/local/bin/ctox
mkdir -p {DEFAULT_CONTAINER_ROOT}/runtime
rm -f {DEFAULT_CONTAINER_ROOT}/runtime/*.db \
  {DEFAULT_CONTAINER_ROOT}/runtime/*.db-shm \
  {DEFAULT_CONTAINER_ROOT}/runtime/*.db-wal \
  {DEFAULT_CONTAINER_ROOT}/runtime/*.log \
  {DEFAULT_CONTAINER_ROOT}/runtime/chat_plan.json \
  {DEFAULT_CONTAINER_ROOT}/runtime/runtime_state.json \
  {DEFAULT_CONTAINER_ROOT}/runtime/continuity.json \
  {DEFAULT_CONTAINER_ROOT}/runtime/cto-agent.lock
python3 - <<'PY'
import sqlite3
db_path = "{DEFAULT_RUNTIME_DB_CONTAINER_PATH}"
conn = sqlite3.connect(db_path)
conn.execute("CREATE TABLE IF NOT EXISTS runtime_env_kv (env_key TEXT PRIMARY KEY, env_value TEXT NOT NULL)")
conn.execute(
    "INSERT INTO runtime_env_kv (env_key, env_value) VALUES (?, ?) "
    "ON CONFLICT(env_key) DO UPDATE SET env_value = excluded.env_value",
    ("CTOX_CHAT_TURN_TIMEOUT_SECS", "{turn_timeout}")
)
conn.commit()
conn.close()
PY
rm -f /tmp/ctox-bundle.tgz
"""
        await self.exec_as_root(environment, command=extract_cmd)

        # Smoke-check — if the binary doesn't run, fail install loudly.
        # `ctox` with no args enters TUI mode; we probe the usage bail
        # path by invoking an unknown subcommand which is cheap and prints
        # to stderr without needing a TTY or model.
        await self.exec_as_agent(
            environment,
            command="/usr/local/bin/ctox help >/dev/null",
        )

    async def run(  # type: ignore[override]
        self,
        instruction: str,
        environment: Any,
        context: Any,
    ) -> None:
        # Any of these keys may be present on the host; forward whatever we
        # have so CTOX' provider routing picks the right upstream. At least
        # one must be set — we don't know yet which one the model needs
        # (that's resolved inside CTOX via the model registry).
        provider_keys = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
            "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY"),
            "MINIMAX_API_KEY": os.environ.get("MINIMAX_API_KEY"),
        }
        if not any(provider_keys.values()):
            raise RuntimeError(
                "No provider API key on the host — set at least one of "
                "OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY, "
                "MINIMAX_API_KEY"
            )

        # Ensure the log directory exists inside the container.
        await self.exec_as_agent(
            environment,
            command="mkdir -p /logs/agent",
        )

        thread_key = f"tbench-{int(time.time() * 1000)}"
        env = {
            "CTOX_ROOT": DEFAULT_CONTAINER_ROOT,
            "PATH": f"/usr/local/bin:{DEFAULT_CONTAINER_ROOT}/target/release:/usr/bin:/bin",
            "HOME": "/root",
            # Bench tasks can take minutes; the 180s default turn timeout is too
            # tight. Allow override via CTOX_CHAT_TURN_TIMEOUT_SECS in host env
            # and default to 1200s (20 min) for bench mode.
            "CTOX_CHAT_TURN_TIMEOUT_SECS": os.environ.get(
                "CTOX_CHAT_TURN_TIMEOUT_SECS", "1200"
            ),
        }
        for key in (
            "CTOX_CHAT_MODEL_REALIZED_CONTEXT",
            "CTOX_CHAT_MODEL_MAX_CONTEXT",
            "CTOX_CHAT_COMPACTION_THRESHOLD_PERCENT",
            "CTOX_CHAT_COMPACTION_MIN_TOKENS",
            "CTOX_REFRESH_OUTPUT_BUDGET_PCT",
            "CTOX_CONTINUITY_REFRESH_EVERY_N_TURNS",
            "CTOX_DISABLE_AUXILIARY_BACKENDS",
        ):
            value = os.environ.get(key)
            if value:
                env[key] = value
        for key, value in provider_keys.items():
            if value:
                env[key] = value

        # Terminal-Bench tasks stage their files in /app/ and their
        # verifiers check /app/ for the produced artefacts. CTOX's cwd and
        # workspace must point there, not at /opt/ctox (the installed
        # agent tree), otherwise the agent writes relative files into the
        # wrong directory and the verifier can't find them.
        workspace = "/app"
        bench_instruction = self._render_benchmark_instruction(
            instruction=instruction,
            workspace=workspace,
        )
        bench_model = os.environ.get("CTOX_BENCH_MODEL", "gpt-5.4")
        bench_preset = os.environ.get("CTOX_BENCH_PRESET", "quality")
        inline_exports = " ".join(
            f"{key}={shlex.quote(value)}"
            for key, value in (
                ("CTOX_ROOT", DEFAULT_CONTAINER_ROOT),
                ("CODEX_HOME", DEFAULT_CODEX_HOME),
                ("HOME", DEFAULT_CODEX_HOME),
                (
                    "PATH",
                    f"/usr/local/bin:{DEFAULT_CONTAINER_ROOT}/target/release:/usr/bin:/bin",
                ),
            )
        )
        for key in (
            "CTOX_CHAT_MODEL_REALIZED_CONTEXT",
            "CTOX_CHAT_MODEL_MAX_CONTEXT",
            "CTOX_CHAT_COMPACTION_THRESHOLD_PERCENT",
            "CTOX_CHAT_COMPACTION_MIN_TOKENS",
            "CTOX_REFRESH_OUTPUT_BUDGET_PCT",
            "CTOX_CONTINUITY_REFRESH_EVERY_N_TURNS",
            "CTOX_DISABLE_AUXILIARY_BACKENDS",
        ):
            value = env.get(key)
            if value:
                inline_exports += f" {key}={shlex.quote(value)}"
        cmd = (
            f"cd {shlex.quote(workspace)} && "
            f"export {inline_exports} && "
            f"/usr/local/bin/ctox runtime switch "
            f"{shlex.quote(bench_model)} {shlex.quote(bench_preset)} && "
            f"/usr/local/bin/ctox service --foreground --autonomy progressive "
            f"> {DEFAULT_SERVICE_LOG_CONTAINER_PATH} 2>&1 & "
            f"service_pid=$!; "
            f"cleanup() {{ "
            f"/usr/local/bin/ctox stop >/dev/null 2>&1 || true; "
            f"kill \"$service_pid\" >/dev/null 2>&1 || true; "
            f"wait \"$service_pid\" >/dev/null 2>&1 || true; "
            f"}}; "
            f"trap cleanup EXIT; "
            f"for _ in $(seq 1 100); do "
            f"if /usr/local/bin/ctox status >/tmp/ctox-status.json 2>/dev/null && "
            f"grep -q '\"running\": true' /tmp/ctox-status.json; then break; fi; "
            f"sleep 0.2; "
            f"done; "
            f"/usr/local/bin/ctox chat "
            f"{shlex.quote(bench_instruction)} "
            f"--wait "
            f"--workspace {shlex.quote(workspace)} "
            f"--thread-key {shlex.quote(thread_key)} "
            f"--atif-out {DEFAULT_ATIF_CONTAINER_PATH} "
            f"--timeout-secs {DEFAULT_RUN_TIMEOUT_SEC}"
        )

        run_exc: Exception | None = None
        try:
            await self.exec_as_agent(
                environment,
                command=cmd,
                env=env,
                timeout_sec=DEFAULT_RUN_TIMEOUT_SEC,
            )
        except Exception as exc:  # noqa: BLE001 — propagate after trajectory copy
            run_exc = exc

        host_logs_dir = Path(self.logs_dir)
        host_logs_dir.mkdir(parents=True, exist_ok=True)
        for container_path, host_name in (
            (DEFAULT_ATIF_CONTAINER_PATH, "trajectory.json"),
            (DEFAULT_CONTEXT_LOG_CONTAINER_PATH, "context-log.jsonl"),
            (DEFAULT_RUNTIME_DB_CONTAINER_PATH, "ctox.sqlite3"),
            (DEFAULT_SERVICE_LOG_CONTAINER_PATH, "service.log"),
        ):
            try:
                await environment.download_file(
                    container_path, str(host_logs_dir / host_name)
                )
            except Exception as copy_exc:  # noqa: BLE001
                print(
                    f"ctox-harbor: failed to copy {container_path} from container: "
                    f"{copy_exc}"
                )

        if run_exc is not None:
            raise run_exc

    @staticmethod
    def _render_benchmark_instruction(instruction: str, workspace: str) -> str:
        cleaned = instruction.strip()
        if cleaned.startswith("Work only inside this workspace:"):
            return cleaned
        return (
            f"Work only inside this workspace:\n{workspace}\n\n"
            "This is a non-interactive benchmark harness. No human will answer "
            "follow-up questions. Do not ask for clarification. Inspect the "
            "workspace, repository state, tests, and local artifacts, infer the "
            "most likely intended fix from the available evidence, apply the "
            "change directly, and verify it before replying.\n\n"
            f"{cleaned}"
        )

    def populate_context_post_run(self, context: Any) -> None:  # type: ignore[override]
        trajectory_path = Path(self.logs_dir) / "trajectory.json"
        if not trajectory_path.is_file():
            # Nothing to report — run probably crashed before emit.
            return
        try:
            with trajectory_path.open("r", encoding="utf-8") as fh:
                trajectory = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"ctox-harbor: failed to parse trajectory.json: {exc}")
            return

        final_metrics = trajectory.get("final_metrics") or {}
        context.n_input_tokens = int(final_metrics.get("total_prompt_tokens", 0))
        context.n_output_tokens = int(
            final_metrics.get("total_completion_tokens", 0)
        )
        context.n_cache_tokens = int(final_metrics.get("total_cached_tokens", 0))
        # Cost is not tracked in CTOX's lo-fi ATIF export yet — surface as 0
        # so Harbor's aggregation doesn't break on None.
        context.cost_usd = float(final_metrics.get("total_cost_usd") or 0.0)
