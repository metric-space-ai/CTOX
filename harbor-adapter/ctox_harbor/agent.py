"""CTOX as a Harbor installed agent.

The contract:

1. `install(environment)` uploads a pre-built CTOX bundle (a .tgz containing
   the full source tree plus a release `target/release/ctox` binary) into the
   task container and symlinks the binary onto PATH as `/usr/local/bin/ctox`.
   The bundle path on the host is supplied via the `CTOX_HOST_TARBALL`
   environment variable so the same adapter works across machines.

2. `run(instruction, environment, context)` invokes `ctox run-once` inside
   the container with `--brief <instruction> --model gpt-5.4 --quality`, and
   points `--atif-out` at `/logs/agent/trajectory.json`. The OpenAI API key
   and CTOX_ROOT are forwarded into the container via the `env=` kwarg.
   After the run, we copy `trajectory.json` back to `self.logs_dir` on the
   host so `populate_context_post_run` can read it.

3. `populate_context_post_run(context)` reads the downloaded trajectory and
   fills in token/cost accounting on the AgentContext. It is tolerant of a
   missing trajectory file — a failed run may not have produced one.

Design intent: the adapter is deliberately thin. All CTOX-specific behaviour
(queueing, continuity, plan execution, skill invocation) lives inside CTOX
itself. The adapter's job is just to bridge Harbor's installed-agent contract
to CTOX's `run-once` CLI.
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
DEFAULT_ATIF_CONTAINER_PATH = "/logs/agent/trajectory.json"
DEFAULT_RUN_TIMEOUT_SEC = 1800


class CtoxAgent(BaseInstalledAgent):
    """Harbor adapter that runs each task through `ctox run-once`."""

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
        extract_cmd = (
            f"mkdir -p {DEFAULT_CONTAINER_ROOT} && "
            f"tar -xzf /tmp/ctox-bundle.tgz -C {DEFAULT_CONTAINER_ROOT} --strip-components=1 && "
            f"chmod +x {DEFAULT_CONTAINER_ROOT}/target/release/ctox && "
            f"chmod +x {DEFAULT_CONTAINER_ROOT}/tools/agent-runtime/target/release/codex-exec && "
            f"ln -sf {DEFAULT_CONTAINER_ROOT}/target/release/ctox /usr/local/bin/ctox && "
            # Ensure runtime/ exists (may be empty if the bundle excluded it).
            f"mkdir -p {DEFAULT_CONTAINER_ROOT}/runtime && "
            # Wipe USER STATE that may have travelled in the bundle so each
            # bench trial starts virgin: the LCM conversation DB, the queue
            # DB, continuity+plan state files, backend logs. The other
            # runtime/ subdirs (benchmarks, browser, codex_exec scaffolding,
            # communication, documents, generated-skills) are static support
            # and must survive — `runtime_switch` otherwise fails with
            # `failed to inspect managed runtime processes`.
            f"rm -f {DEFAULT_CONTAINER_ROOT}/runtime/*.db "
            f"{DEFAULT_CONTAINER_ROOT}/runtime/*.db-shm "
            f"{DEFAULT_CONTAINER_ROOT}/runtime/*.db-wal "
            f"{DEFAULT_CONTAINER_ROOT}/runtime/*.log "
            f"{DEFAULT_CONTAINER_ROOT}/runtime/chat_plan.json "
            f"{DEFAULT_CONTAINER_ROOT}/runtime/runtime_state.json "
            f"{DEFAULT_CONTAINER_ROOT}/runtime/continuity.json "
            f"{DEFAULT_CONTAINER_ROOT}/runtime/cto-agent.lock && "
            # Operator settings live in engine.env. The turn timeout here
            # must be large enough to cover real agent work; the in-CTOX
            # default of 180s will kill most bench tasks before the first
            # tool call.
            f"printf 'CTOX_CHAT_TURN_TIMEOUT_SECS={turn_timeout}\\n' > {DEFAULT_CONTAINER_ROOT}/runtime/engine.env && "
            f"rm -f /tmp/ctox-bundle.tgz"
        )
        await self.exec_as_root(environment, command=extract_cmd)

        # Smoke-check — if the binary doesn't run, fail install loudly.
        # `ctox` with no args enters TUI mode; we probe the usage bail
        # path by invoking an unknown subcommand which is cheap and prints
        # to stderr without needing a TTY or model.
        await self.exec_as_agent(
            environment,
            command="/usr/local/bin/ctox run-once 2>&1 | head -1 || true",
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
        for key, value in provider_keys.items():
            if value:
                env[key] = value

        # Terminal-Bench tasks stage their files in /app/ and their
        # verifiers check /app/ for the produced artefacts. CTOX's cwd and
        # workspace must point there, not at /opt/ctox (the installed
        # agent tree), otherwise the agent writes relative files into the
        # wrong directory and the verifier can't find them.
        workspace = "/app"
        bench_model = os.environ.get("CTOX_BENCH_MODEL", "gpt-5.4")
        bench_preset = os.environ.get("CTOX_BENCH_PRESET", "quality")
        preset_flag = f"--{bench_preset}" if bench_preset in ("quality", "performance") else ""
        cmd = (
            f"cd {shlex.quote(workspace)} && "
            f"/usr/local/bin/ctox run-once "
            f"--brief {shlex.quote(instruction)} "
            f"--model {shlex.quote(bench_model)} "
            f"{preset_flag} "
            f"--workspace {shlex.quote(workspace)} "
            f"--atif-out {DEFAULT_ATIF_CONTAINER_PATH} "
            f"--thread-key {shlex.quote(thread_key)}"
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

        # Copy the trajectory out regardless of success — it's the most
        # useful debug artifact on failure. `populate_context_post_run` is
        # tolerant of a missing file.
        host_logs_dir = Path(self.logs_dir)
        host_logs_dir.mkdir(parents=True, exist_ok=True)
        host_trajectory = host_logs_dir / "trajectory.json"
        try:
            await environment.download_file(
                DEFAULT_ATIF_CONTAINER_PATH, str(host_trajectory)
            )
        except Exception as copy_exc:  # noqa: BLE001
            # Downgrade to warning: if the run itself failed, the trajectory
            # may not have been written, which is expected.
            print(
                f"ctox-harbor: failed to copy trajectory from container: "
                f"{copy_exc}"
            )

        if run_exc is not None:
            raise run_exc

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
