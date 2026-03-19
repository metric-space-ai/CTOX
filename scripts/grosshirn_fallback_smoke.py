#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-port", type=int, default=18443)
    parser.add_argument("--local-port", type=int, default=12431)
    parser.add_argument("--grosshirn-port", type=int, default=12432)
    parser.add_argument("--tick-ms", type=int, default=250)
    parser.add_argument("--wait-secs", type=int, default=6)
    return parser.parse_args()


def start_mock(
    root: Path,
    base_env: dict[str, str],
    *,
    port: int,
    model_id: str,
    mode: str,
    state_name: str,
    procs: list[subprocess.Popen],
) -> None:
    state_file = root / state_name
    state_file.write_text(json.dumps({"mode": mode}))
    env = base_env.copy()
    env.update(
        {
            "CTO_MOCK_PORT": str(port),
            "CTO_MOCK_MODEL_ID": model_id,
            "CTO_MOCK_STATE_FILE": str(state_file),
        }
    )
    proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "scripts/mock_kleinhirn.py")],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    line = proc.stdout.readline().strip()
    if not line:
        raise RuntimeError(f"mock {port} failed to start: {proc.stderr.read()}")
    print(f"mock {port} {line}", flush=True)
    procs.append(proc)


def wait_ready(agent_port: int, agent_log_path: Path) -> dict:
    ctx = ssl._create_unverified_context()
    ready_url = f"https://127.0.0.1:{agent_port}/readyz"
    ready_payload = None
    for _ in range(80):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(ready_url),
                context=ctx,
                timeout=2,
            ) as resp:
                ready_payload = json.loads(resp.read().decode())
                if ready_payload.get("ready") is True:
                    return ready_payload
        except Exception:
            pass
        time.sleep(0.5)
    log_tail = agent_log_path.read_text()[-4000:] if agent_log_path.exists() else ""
    raise RuntimeError(
        f"readyz never became ready: {json.dumps(ready_payload)}\nLOG:\n{log_tail}"
    )


def main() -> int:
    args = parse_args()
    root = Path(tempfile.mkdtemp(prefix="cto_grosshirn_smoke_"))
    base_env = os.environ.copy()
    base_env.update(
        {
            "CTO_AGENT_ROOT": str(root),
            "CTO_AGENT_PORT": str(args.agent_port),
            "CTO_AGENT_SUPERVISOR_TICK_MS": str(args.tick_ms),
            "CTO_AGENT_KLEINHIRN_BASE_URL": f"http://127.0.0.1:{args.local_port}/v1",
            "CTO_AGENT_KLEINHIRN_MODEL": "gpt-oss-20b",
            "CTO_AGENT_KLEINHIRN_RUNTIME_MODEL": "openai/gpt-oss-20b",
            "CTO_AGENT_KLEINHIRN_AGENTIC_ADAPTER": "openai_chatcompletions",
            "CTO_AGENT_GROSSHIRN_BASE_URL": f"http://127.0.0.1:{args.grosshirn_port}/v1",
            "CTO_AGENT_GROSSHIRN_API_KEY": "test-key",
            "CTO_AGENT_GROSSHIRN_MODEL": "gpt-5.4",
            "CTO_AGENT_GROSSHIRN_AGENTIC_ADAPTER": "openai_responses",
        }
    )
    procs: list[subprocess.Popen] = []
    try:
        start_mock(
            root,
            base_env,
            port=args.local_port,
            model_id="openai/gpt-oss-20b",
            mode="healthy",
            state_name="local_mock.json",
            procs=procs,
        )
        start_mock(
            root,
            base_env,
            port=args.grosshirn_port,
            model_id="gpt-5.4",
            mode="error",
            state_name="grosshirn_mock.json",
            procs=procs,
        )

        subprocess.run(
            [str(REPO_ROOT / "target/debug/cto-agent"), "--init-only"],
            cwd=REPO_ROOT,
            env=base_env,
            check=True,
        )
        print(f"init-only ok {root}", flush=True)

        db_path = root / "runtime/cto_agent.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            "update owner_trust set brain_access_mode = 'kleinhirn_plus_grosshirn' where singleton = 1"
        )
        conn.commit()
        conn.close()
        print("brain access switched", flush=True)

        agent_log_path = root / "agent.log"
        agent_log = open(agent_log_path, "w")
        agent = subprocess.Popen(
            [str(REPO_ROOT / "target/debug/cto-agent")],
            cwd=REPO_ROOT,
            env=base_env,
            stdout=agent_log,
            stderr=subprocess.STDOUT,
        )
        procs.append(agent)
        print(f"agent pid {agent.pid}", flush=True)

        ready_payload = wait_ready(args.agent_port, agent_log_path)
        print(f"readyz ok {json.dumps(ready_payload)}", flush=True)

        send = subprocess.run(
            [
                str(REPO_ROOT / "target/debug/cto-agent"),
                "send",
                "Michael Welsch: Bitte arbeite weiter und pruefe deine aktuellen Modellressourcen.",
            ],
            cwd=REPO_ROOT,
            env=base_env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f"send ok {send.stdout.strip()}", flush=True)

        time.sleep(args.wait_secs)

        conn = sqlite3.connect(db_path)
        resources = conn.execute(
            "select name, status, detail from resources "
            "where category = 'agentic_loop' "
            "and name in ('brain_source','brain_fallback','primary_brain_error','brain_fallback_activation') "
            "order by rowid desc limit 10"
        ).fetchall()
        focus = conn.execute(
            "select mode, active_task_id, active_task_title, queue_depth, note "
            "from focus_state where singleton = 1"
        ).fetchone()
        turns = conn.execute(
            "select id, task_id, status, summary from agent_turns order by id desc limit 5"
        ).fetchall()
        tasks = conn.execute(
            "select id, status, task_kind, title, run_count, last_checkpoint_summary "
            "from tasks order by id desc limit 8"
        ).fetchall()
        conn.close()

        print(
            json.dumps(
                {
                    "root": str(root),
                    "resources": resources,
                    "focus": focus,
                    "turns": turns,
                    "tasks": tasks,
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    finally:
        for proc in reversed(procs):
            try:
                proc.terminate()
            except Exception:
                pass
        for proc in reversed(procs):
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())
