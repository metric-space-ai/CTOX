#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import signal
import sqlite3
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def run(cmd, *, env=None, cwd=None, check=True, capture_output=True):
    result = subprocess.run(
        cmd,
        env=env,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=capture_output,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def http_json(url, timeout=2.0):
    context = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(url, timeout=timeout, context=context) as response:
            return response.getcode(), json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read().decode("utf-8"))
        return exc.code, payload


def wait_http_status(url, expected_status, timeout_secs=30):
    start = time.time()
    last = None
    while time.time() - start < timeout_secs:
        try:
            status, payload = http_json(url)
            last = (status, payload)
            if status == expected_status:
                return payload
        except Exception as exc:
            last = exc
        time.sleep(0.25)
    raise RuntimeError(f"timeout waiting for {url} to become {expected_status}; last={last}")


def wait_until(fn, timeout_secs=30, sleep_secs=0.25, label="condition"):
    start = time.time()
    last = None
    while time.time() - start < timeout_secs:
        last = fn()
        if last:
            return last
        time.sleep(sleep_secs)
    raise RuntimeError(f"timeout waiting for {label}; last={last}")


def sqlite_rows(db_path: Path, query: str, params=()):
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(query, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def write_mock_state(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2))


def terminate_process(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description="Kernel stress harness for CTO-Agent")
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--binary", default="target/debug/cto-agent")
    parser.add_argument("--mock-port", type=int, default=12391)
    parser.add_argument("--agent-port", type=int, default=9443)
    parser.add_argument("--keep-root", action="store_true")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    binary = workspace / args.binary
    if not binary.exists():
        raise RuntimeError(f"missing binary: {binary}")

    temp_root = Path(tempfile.mkdtemp(prefix="cto-kernel-stress."))
    state_file = temp_root / "mock_state.json"
    write_mock_state(state_file, {"mode": "healthy"})

    env = os.environ.copy()
    env.update(
        {
            "CTO_AGENT_ROOT": str(temp_root),
            "CTO_AGENT_PORT": str(args.agent_port),
            "CTO_AGENT_KLEINHIRN_BASE_URL": f"http://127.0.0.1:{args.mock_port}/v1",
            "CTO_AGENT_KLEINHIRN_API_KEY": "local-kleinhirn",
            "CTO_AGENT_SUPERVISOR_TICK_MS": "250",
            "CTO_AGENT_HEARTBEAT_STALE_SECS": "4",
            "CTO_AGENT_ACTIVE_TURN_STALE_SECS": "4",
        }
    )

    mock_env = env.copy()
    mock_env["CTO_MOCK_PORT"] = str(args.mock_port)
    mock_env["CTO_MOCK_STATE_FILE"] = str(state_file)

    print(f"[stress] temp root: {temp_root}")
    print(f"[stress] mock state: {state_file}")

    mock_proc = subprocess.Popen(
        [sys.executable, str(workspace / "scripts/mock_kleinhirn.py")],
        cwd=workspace,
        env=mock_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_until(
            lambda: state_file.exists(),
            timeout_secs=5,
            label="mock state file",
        )

        run([str(binary), "--init-only"], env=env, cwd=workspace)

        agent_proc = subprocess.Popen(
            [str(binary)],
            cwd=workspace,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            wait_http_status(
                f"https://127.0.0.1:{args.agent_port}/readyz",
                200,
                timeout_secs=20,
            )
            print("[stress] phase 1 ok: startup + readyz")

            homepage_policy_path = temp_root / "contracts/homepage/homepage-policy.json"
            homepage_artifact = temp_root / "runtime/agent-artifacts/homepage-bootstrap.txt"
            wait_until(
                lambda: homepage_artifact.exists()
                and json.loads(homepage_policy_path.read_text()).get("currentTitle")
                == "CTO-Agent BIOS Bridge",
                timeout_secs=20,
                label="starter homepage work",
            )
            print("[stress] phase 1b ok: starter task hat Homepage sichtbar gebaut")

            run(
                [str(binary), "send", "Michael Welsch: Bitte kalibriere dich zuerst sauber auf mich."],
                env=env,
                cwd=workspace,
            )
            db_path = temp_root / "runtime/cto_agent.db"
            wait_until(
                lambda: sqlite_rows(
                    db_path,
                    "select id, status, task_kind from tasks where speaker = 'Michael Welsch' order by id desc limit 1",
                ),
                timeout_secs=10,
                label="owner task creation",
            )
            print("[stress] phase 2 ok: owner interrupt aufgenommen")

            write_mock_state(state_file, {"mode": "question_compaction"})
            run(
                [str(binary), "send", "Michael Welsch: Mir kommt die letzte Verdichtung komisch vor, pruefe die alte Festlegung."],
                env=env,
                cwd=workspace,
            )
            wait_until(
                lambda: sqlite_rows(
                    db_path,
                    "select id, status, task_kind from tasks where task_kind = 'historical_research' order by id desc limit 1",
                ),
                timeout_secs=12,
                label="historical research task",
            )
            print("[stress] phase 3 ok: historical_research aus agentischer Kontextentscheidung erzeugt")

            write_mock_state(
                state_file,
                {
                    "mode": "healthy",
                    "temporaryChatModes": ["overflow", "healthy"],
                },
            )
            run(
                [str(binary), "send", "Michael Welsch: Bearbeite bitte noch einen bounded Schritt trotz schwerem Kontext."],
                env=env,
                cwd=workspace,
            )
            wait_until(
                lambda: any(
                    row["detail"] == "kernel_emergency_retry_minimal" or row["detail"] == "kernel_emergency_minimal"
                    for row in sqlite_rows(
                        db_path,
                        "select detail from resources where category = 'agentic_loop' and name = 'context_strategy'",
                    )
                ),
                timeout_secs=12,
                label="kernel emergency overflow fallback",
            )
            print("[stress] phase 4 ok: hard overflow ueberlebt, Kernel-Fallback sichtbar")

            write_mock_state(state_file, {"mode": "timeout"})
            wait_http_status(
                f"https://127.0.0.1:{args.agent_port}/readyz",
                503,
                timeout_secs=20,
            )
            incidents = sqlite_rows(
                db_path,
                "select incident_key, status from loop_incidents order by id desc limit 10",
            )
            if not any(row["incident_key"] == "kleinhirn_unavailable" for row in incidents):
                raise RuntimeError("kleinhirn_unavailable incident not observed")
            print("[stress] phase 5 ok: idle/loop-time kleinhirn-Ausfall erkannt")

            run(
                [str(binary), "hard-reset-report", "stress harness forced restart after kleinhirn timeout"],
                env=env,
                cwd=workspace,
            )
            terminate_process(agent_proc)
            write_mock_state(state_file, {"mode": "healthy"})
            agent_proc = subprocess.Popen(
                [str(binary)],
                cwd=workspace,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            wait_http_status(
                f"https://127.0.0.1:{args.agent_port}/readyz",
                200,
                timeout_secs=20,
            )
            recovery_rows = sqlite_rows(
                db_path,
                "select id, task_kind, status from tasks where task_kind = 'recovery' order by id desc limit 3",
            )
            if not recovery_rows:
                raise RuntimeError("recovery task not observed after hard reset")
            print("[stress] phase 6 ok: hard reset + recovery task + restart")

            write_mock_state(state_file, {"mode": "healthy"})
            run(
                [str(binary), "send", "Extern: Niedrig priorisierte Mailartige Nachfrage."],
                env=env,
                cwd=workspace,
            )
            run(
                [str(binary), "send", "Michael Welsch: Das hier ist jetzt wichtiger als der Rest."],
                env=env,
                cwd=workspace,
            )
            priority_rows = sqlite_rows(
                db_path,
                "select speaker, source_channel, priority_score, task_kind from tasks order by id desc limit 6",
            )
            owner_scores = [row["priority_score"] for row in priority_rows if row["speaker"] == "Michael Welsch"]
            external_scores = [row["priority_score"] for row in priority_rows if row["speaker"] == "Extern"]
            if not owner_scores or not external_scores or max(owner_scores) <= max(external_scores):
                raise RuntimeError(f"owner priority not above external priority: {priority_rows}")
            print("[stress] phase 7 ok: owner priority ueberdeckt low-trust work")

            print("[stress] kernel stress harness passed")
            return 0
        finally:
            terminate_process(agent_proc)
    finally:
        terminate_process(mock_proc)
        if not args.keep_root:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
