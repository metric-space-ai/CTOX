#!/usr/bin/env python3
import argparse
import atexit
import difflib
import json
import os
import signal
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SYMLINK_MIRROR_ENTRIES = {
    ".codex-paramiko-venv",
    ".venv",
    "engine",
    "old-legacy-for-transplation-only",
    "references",
    "skills",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_manifest(root: Path) -> Dict[str, Any]:
    return json.loads((root / "benchmarks/airbnb_clone/manifest.json").read_text())


def load_capability_reference(root: Path) -> Optional[Dict[str, Any]]:
    candidates = [
        repo_root().parent / "Capability_Extractor" / "airbnb_test_v2.json",
        root.parent / "Capability_Extractor" / "airbnb_test_v2.json",
        root.parents[0] / "Capability_Extractor" / "airbnb_test_v2.json" if len(root.parents) > 0 else None,
        root.parents[1] / "Capability_Extractor" / "airbnb_test_v2.json" if len(root.parents) > 1 else None,
        root.parents[2] / "Capability_Extractor" / "airbnb_test_v2.json" if len(root.parents) > 2 else None,
        root.parents[3] / "Capability_Extractor" / "airbnb_test_v2.json" if len(root.parents) > 3 else None,
    ]
    candidate = next((path for path in candidates if path and path.is_file()), None)
    if candidate is None:
        return None
    payload = json.loads(candidate.read_text())
    profile = payload.get("final_profile", {})
    audit = payload.get("final_audit", {})
    capability_groups = profile.get("capability_groups", [])
    capabilities = profile.get("capabilities", [])
    control_loops = profile.get("control_loops", [])
    integration_capabilities = profile.get("integration_capabilities", [])
    skill_candidates = profile.get("skill_candidates", [])
    evidence_count = int(payload.get("evidence_count", 0))
    return {
        "source_path": str(candidate),
        "company_entity": payload.get("company_entity", "Airbnb"),
        "analysis_boundary": payload.get("analysis_boundary", ""),
        "evidence_count": evidence_count,
        "audit_status": audit.get("status", ""),
        "audit_reason": audit.get("reason", ""),
        "capability_groups": capability_groups,
        "capabilities": capabilities,
        "control_loops": control_loops,
        "integration_capabilities": integration_capabilities,
        "skill_candidates": skill_candidates,
        "raw_payload": payload,
    }


def sync_repo_source(source_root: Path, bench_root: Path) -> None:
    if source_root == bench_root:
        return
    bench_root.mkdir(parents=True, exist_ok=True)
    preserved = {"runtime", "target"}
    ignored = {".git", "runtime", "target"}

    for entry in bench_root.iterdir():
        if entry.name in preserved:
            continue
        if entry.is_symlink() or entry.is_file():
            entry.unlink(missing_ok=True)
        else:
            shutil.rmtree(entry)

    for entry in source_root.iterdir():
        if entry.name in ignored:
            continue
        destination = bench_root / entry.name
        if destination.exists() or destination.is_symlink():
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink(missing_ok=True)
        if entry.name in SYMLINK_MIRROR_ENTRIES:
            destination.symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
        elif entry.is_symlink():
            if destination.exists() or destination.is_symlink():
                destination.unlink(missing_ok=True)
            destination.symlink_to(os.readlink(entry))
        elif entry.is_dir():
            shutil.copytree(entry, destination, symlinks=True, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, destination, follow_symlinks=False)


def load_env_file(path: Path) -> Dict[str, str]:
    env_map: Dict[str, str] = {}
    if not path.is_file():
        return env_map
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_map[key.strip()] = value.strip()
    return env_map


def save_env_file(path: Path, env_map: Dict[str, str]) -> None:
    lines = [f"{key}={env_map[key]}" for key in sorted(env_map)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_bench_runtime_env(root: Path, source_root: Path, model: str) -> Path:
    runtime_dir = root / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    source_env_path = source_root / "runtime/engine.env"
    target_env_path = runtime_dir / "engine.env"
    env_map = load_env_file(source_env_path)
    env_map.update(load_env_file(target_env_path))
    for derived_key in (
        "CTOX_UPSTREAM_BASE_URL",
        "CTOX_ENGINE_REALIZED_MAX_SEQ_LEN",
        "CTOX_CHAT_MODEL_REALIZED_CONTEXT",
        "CTOX_ENGINE_REALIZED_MODEL",
        "CTOX_EMBEDDING_MODEL",
        "CTOX_EMBEDDING_BASE_URL",
        "CTOX_STT_MODEL",
        "CTOX_STT_BASE_URL",
        "CTOX_TTS_MODEL",
        "CTOX_TTS_BASE_URL",
        "CTOX_AUXILIARY_CUDA_VISIBLE_DEVICES",
        "CTOX_EMBEDDING_CUDA_VISIBLE_DEVICES",
        "CTOX_STT_CUDA_VISIBLE_DEVICES",
        "CTOX_TTS_CUDA_VISIBLE_DEVICES",
    ):
        env_map.pop(derived_key, None)
    env_map["CTOX_CHAT_SOURCE"] = "local"
    env_map["CTOX_CHAT_MODEL"] = model
    env_map["CTOX_CHAT_MODEL_BASE"] = model
    env_map["CTOX_ACTIVE_MODEL"] = model
    env_map["CTOX_DISABLE_AUXILIARY_BACKENDS"] = "1"
    env_map["CODEX_HOME"] = str((runtime_dir / "codex_home").resolve())
    save_env_file(target_env_path, env_map)
    Path(env_map["CODEX_HOME"]).mkdir(parents=True, exist_ok=True)
    return target_env_path


def ensure_ctox_bin(root: Path, *, force_rebuild: bool = False) -> Path:
    release = root / "target/release/ctox"
    debug = root / "target/debug/ctox"
    if release.is_file() and not force_rebuild:
        return release
    if debug.is_file() and not force_rebuild:
        return debug
    subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", str(root / "Cargo.toml")],
        cwd=root,
        check=True,
    )
    return release


def bench_service_pid_path(root: Path) -> Path:
    return root / "runtime/airbnb_clone_bench/bench_service.pid"


def bench_service_log_path(root: Path) -> Path:
    return root / "runtime/airbnb_clone_bench/bench_service.log"


def bench_service_unit_name() -> str:
    return "ctox-bench-service"


def bench_service_unit_path(root: Path) -> Path:
    return root / "runtime/airbnb_clone_bench/bench_service.unit"


def bench_trace_log_path(root: Path) -> Path:
    return root / "runtime/airbnb_clone_bench/bench_trace.log"


def bench_error_log_path(root: Path) -> Path:
    return root / "runtime/airbnb_clone_bench/bench_error.log"


def note_phase(root: Path, label: str) -> None:
    path = bench_trace_log_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {label}\n")


def install_bench_signal_handlers(root: Path) -> None:
    def handle_signal(signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        note_phase(root, f"signal:{signal_name}")
        if signum in (signal.SIGTERM, signal.SIGHUP):
            with bench_error_log_path(root).open("a", encoding="utf-8") as handle:
                handle.write(f"benchmark received {signal_name} and ignored it\n")
            return
        bench_error_log_path(root).write_text(
            f"benchmark received {signal_name}\n",
            encoding="utf-8",
        )
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(signum, handle_signal)


def read_bench_service_pid(root: Path) -> Optional[int]:
    path = bench_service_pid_path(root)
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (TypeError, ValueError):
        return None


def read_pid_file(path: Path) -> Optional[int]:
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (TypeError, ValueError):
        return None


def has_systemd_user() -> bool:
    try:
        proc = subprocess.run(
            ["systemctl", "--user", "is-active", "default.target"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False
    return proc.returncode == 0


def use_systemd_bench_service() -> bool:
    return os.environ.get("CTOX_BENCH_USE_SYSTEMD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def read_bench_service_unit(root: Path) -> Optional[str]:
    path = bench_service_unit_path(root)
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 5
    while time.time() < deadline:
        if not process_is_alive(pid):
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def stop_pid_file(path: Path) -> None:
    pid = read_pid_file(path)
    if pid is not None:
        terminate_pid(pid)
    path.unlink(missing_ok=True)


def listener_pids_for_port(port: int) -> List[int]:
    try:
        proc = subprocess.run(
            ["lsof", f"-tiTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    pids: List[int] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return pids


def runtime_listener_ports(root: Path) -> List[int]:
    env_map = load_env_file(root / "runtime/engine.env")
    ports: List[int] = []
    defaults = {
        "CTOX_PROXY_PORT": 12434,
        "CTOX_ENGINE_PORT": 1234,
        "CTOX_EMBEDDING_PORT": 1237,
        "CTOX_STT_PORT": 1238,
        "CTOX_TTS_PORT": 1239,
    }
    for key, default in defaults.items():
        raw = env_map.get(key, str(default)).strip()
        try:
            ports.append(int(raw))
        except ValueError:
            continue
    return ports


def stop_runtime_sidecars(root: Path) -> None:
    runtime_dir = root / "runtime"
    for name in (
        "ctox_proxy.pid",
        "ctox_chat_backend.pid",
        "ctox_embedding_backend.pid",
        "ctox_stt_backend.pid",
        "ctox_tts_backend.pid",
    ):
        stop_pid_file(runtime_dir / name)
    seen: set[int] = set()
    for port in runtime_listener_ports(root):
        for pid in listener_pids_for_port(port):
            if pid in seen:
                continue
            seen.add(pid)
            terminate_pid(pid)


def stop_bench_managed_service(root: Path) -> None:
    if has_systemd_user() and use_systemd_bench_service():
        unit_name = read_bench_service_unit(root) or bench_service_unit_name()
        subprocess.run(
            ["systemctl", "--user", "stop", unit_name],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        bench_service_unit_path(root).unlink(missing_ok=True)
    else:
        bench_service_unit_path(root).unlink(missing_ok=True)
    pid = read_bench_service_pid(root)
    if pid is None:
        return
    try:
        if process_is_alive(pid):
            os.killpg(pid, signal.SIGTERM)
            deadline = time.time() + 15
            while time.time() < deadline:
                if not process_is_alive(pid):
                    break
                time.sleep(0.2)
            if process_is_alive(pid):
                os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        bench_service_pid_path(root).unlink(missing_ok=True)


def start_bench_managed_service(ctox_bin: Path, root: Path, env: Dict[str, str]) -> None:
    runtime_dir = root / "runtime/airbnb_clone_bench"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    log_path = bench_service_log_path(root)
    env = env.copy()
    env.setdefault("RUST_BACKTRACE", "1")
    if has_systemd_user() and use_systemd_bench_service():
        unit_name = f"{bench_service_unit_name()}-{now_slug().lower()}"
        command = (
            f"cd {root} && exec {ctox_bin} service --foreground >> {log_path} 2>&1"
        )
        result = subprocess.run(
            [
                "systemd-run",
                "--user",
                "--unit",
                unit_name,
                "--collect",
                "bash",
                "-lc",
                command,
            ],
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "failed to start bench-managed systemd service:\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        bench_service_unit_path(root).write_text(f"{unit_name}\n", encoding="utf-8")
        bench_service_pid_path(root).unlink(missing_ok=True)
        return

    bench_service_unit_path(root).unlink(missing_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [str(ctox_bin), "service", "--foreground"],
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
        bench_service_pid_path(root).write_text(f"{proc.pid}\n", encoding="utf-8")


def ensure_bench_managed_service(ctox_bin: Path, root: Path, env: Dict[str, str]) -> Dict[str, Any]:
    stop_bench_managed_service(root)
    start_bench_managed_service(ctox_bin, root, env)
    for _ in range(240):
        try:
            status = run_cmd([str(ctox_bin), "status"], cwd=root, env=env, expect_json=True)
        except Exception:
            status = None
        if status and status.get("running"):
            return status
        pid = read_bench_service_pid(root)
        if pid is not None and not process_is_alive(pid):
            break
        time.sleep(0.5)
    log_excerpt = ""
    log_path = bench_service_log_path(root)
    if log_path.is_file():
        log_excerpt = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
    raise RuntimeError(f"bench-managed CTOX service failed to start\n{log_excerpt}")


def run_cmd(
    cmd: List[str],
    *,
    cwd: Path,
    env: Dict[str, str],
    expect_json: bool = False,
) -> Any:
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    if expect_json:
        return json.loads(proc.stdout)
    return proc.stdout


def service_base_url(ctox_bin: Path, root: Path, env: Dict[str, str]) -> str:
    status = run_cmd([str(ctox_bin), "status"], cwd=root, env=env, expect_json=True)
    listen_addr = status.get("listen_addr") or "127.0.0.1:12435"
    return f"http://{listen_addr}"


def proxy_base_url(root: Path) -> str:
    env_map = load_env_file(root / "runtime/engine.env")
    host = env_map.get("CTOX_PROXY_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = env_map.get("CTOX_PROXY_PORT", "12434").strip() or "12434"
    return f"http://{host}:{port}"


def wait_for_service_http_ready(base_url: str, timeout_seconds: int = 60) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Optional[str] = None
    while time.time() < deadline:
        request = urllib.request.Request(f"{base_url}/ctox/service/status", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status < 500:
                    return
                last_error = f"http {response.status}"
        except urllib.error.URLError as err:
            last_error = str(err)
        time.sleep(1)
    raise RuntimeError(
        f"ctox service HTTP endpoint did not become ready within {timeout_seconds}s; last_error={last_error}"
    )


def wait_for_local_runtime_ready(proxy_url: str, timeout_seconds: int = 180) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_payload: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None
    while time.time() < deadline:
        request = urllib.request.Request(f"{proxy_url}/ctox/telemetry", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
                last_payload = payload
                upstream_base_url = str(payload.get("upstream_base_url") or "")
                if upstream_base_url.startswith("https://api.openai.com"):
                    return payload
                if payload.get("backend_healthy"):
                    return payload
                last_error = f"backend_healthy={payload.get('backend_healthy')}"
        except Exception as err:
            last_error = str(err)
        time.sleep(1)
    raise RuntimeError(
        "ctox local runtime did not become healthy within "
        f"{timeout_seconds}s; last_error={last_error}; last_payload={json.dumps(last_payload or {}, ensure_ascii=True)}"
    )


def ensure_service(
    ctox_bin: Path,
    root: Path,
    env: Dict[str, str],
    *,
    reuse_running_service: bool = False,
) -> Dict[str, Any]:
    status = run_cmd([str(ctox_bin), "status"], cwd=root, env=env, expect_json=True)
    if status.get("running"):
        return status
    if reuse_running_service:
        raise RuntimeError("expected an already running CTOX service, but `ctox status` reported not running")
    return ensure_bench_managed_service(ctox_bin, root, env)


def stop_service_if_running(ctox_bin: Path, root: Path, env: Dict[str, str]) -> None:
    try:
        status = run_cmd([str(ctox_bin), "status"], cwd=root, env=env, expect_json=True)
    except Exception:
        status = {}
    if status.get("running"):
        try:
            run_cmd([str(ctox_bin), "stop"], cwd=root, env=env)
        except Exception:
            pass
    stop_bench_managed_service(root)
    stop_runtime_sidecars(root)


def reset_runtime_dir(root: Path) -> None:
    runtime_dir = root / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for entry in runtime_dir.iterdir():
        if entry.name == "engine.env":
            continue
        if entry.is_symlink() or entry.is_file():
            entry.unlink(missing_ok=True)
            continue
        shutil.rmtree(entry)


def wait_until_idle(
    ctox_bin: Path,
    root: Path,
    env: Dict[str, str],
    *,
    timeout_seconds: int,
) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_status: Optional[Dict[str, Any]] = None
    while time.time() < deadline:
        last_status = run_cmd([str(ctox_bin), "status"], cwd=root, env=env, expect_json=True)
        if last_status.get("running") and not last_status.get("busy") and int(last_status.get("pending_count", 0)) == 0:
            return last_status
        time.sleep(2)
    raise RuntimeError(f"ctox service did not become idle in {timeout_seconds}s; last status={json.dumps(last_status or {}, indent=2)}")


def wait_for_service_activity(
    ctox_bin: Path,
    root: Path,
    env: Dict[str, str],
    *,
    timeout_seconds: int,
) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_status: Optional[Dict[str, Any]] = None
    while time.time() < deadline:
        last_status = run_cmd([str(ctox_bin), "status"], cwd=root, env=env, expect_json=True)
        if last_status.get("busy") or int(last_status.get("pending_count", 0)) > 0:
            return last_status
        time.sleep(1)
    return last_status or {}


def monitor_service_for_idle_failure(
    *,
    ctox_bin: Path,
    root: Path,
    env: Dict[str, str],
    monitor_seconds: int,
    max_idle_gap_seconds: int,
) -> None:
    if monitor_seconds <= 0:
        return
    deadline = time.time() + monitor_seconds
    idle_started_at: Optional[float] = None
    last_status: Optional[Dict[str, Any]] = None
    while time.time() < deadline:
        last_status = run_cmd([str(ctox_bin), "status"], cwd=root, env=env, expect_json=True)
        busy = bool(last_status.get("busy"))
        pending = int(last_status.get("pending_count", 0))
        if busy or pending > 0:
            idle_started_at = None
        else:
            if idle_started_at is None:
                idle_started_at = time.time()
            idle_for = time.time() - idle_started_at
            if idle_for >= max_idle_gap_seconds:
                raise RuntimeError(
                    "service went idle without queued work "
                    f"for {int(idle_for)}s during continuous benchmark monitoring; "
                    f"last status={json.dumps(last_status, indent=2)}"
                )
        time.sleep(2)


def wait_for_report_response(
    *,
    ctox_bin: Path,
    root: Path,
    env: Dict[str, str],
    conversation_id: int,
    cycle: int,
    manifest: Dict[str, Any],
    timeout_seconds: int,
) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds
    db_path = root / "runtime/ctox_lcm.db"
    last_snapshot: Optional[Dict[str, Any]] = None
    while time.time() < deadline:
        last_snapshot = run_cmd(
            [str(ctox_bin), "lcm-dump", str(db_path), str(conversation_id)],
            cwd=root,
            env=env,
            expect_json=True,
        )
        try:
            extract_report_text(last_snapshot, cycle, manifest)
            return last_snapshot
        except RuntimeError:
            time.sleep(2)
    raise RuntimeError(
        f"report response for cycle {cycle} did not appear in time; last snapshot={json.dumps(last_snapshot or {}, indent=2)[:2000]}"
    )


def post_service_chat(base_url: str, prompt: str) -> None:
    data = json.dumps({"prompt": prompt}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/ctox/service/chat",
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status >= 300:
                raise RuntimeError(f"service rejected prompt with status {response.status}")
    except urllib.error.URLError as err:
        raise RuntimeError(f"failed to submit chat prompt: {err}") from err


def setup_workspace(run_dir: Path, manifest: Dict[str, Any], capability_reference: Optional[Dict[str, Any]]) -> Path:
    workspace = run_dir / "workspace"
    (workspace / "apps/web").mkdir(parents=True, exist_ok=True)
    (workspace / "apps/api").mkdir(parents=True, exist_ok=True)
    (workspace / "docs").mkdir(parents=True, exist_ok=True)
    (workspace / "ops/progress").mkdir(parents=True, exist_ok=True)
    (workspace / "ops/runbooks").mkdir(parents=True, exist_ok=True)
    (workspace / "ops/incidents").mkdir(parents=True, exist_ok=True)
    (workspace / "bench_state").mkdir(parents=True, exist_ok=True)

    phase_lines = []
    for phase in manifest["phases"]:
        phase_lines.append(f"## {phase['phase_id']} {phase['name']}")
        phase_lines.append("Capability groups:")
        for name in phase["capability_groups"]:
            phase_lines.append(f"- {name}")
        phase_lines.append("Acceptance examples:")
        for example in phase["acceptance_examples"]:
            phase_lines.append(f"- {example}")
        phase_lines.append("")

    (workspace / "README.md").write_text(
        textwrap.dedent(
            f"""\
            # {manifest['workspace_slug'].capitalize()} Benchmark Workspace

            This workspace is owned by the CTOX Airbnb clone benchmark.

            Mission:
            - Build and operate a credible Airbnb-style marketplace clone.
            - Keep the product and operations roadmap coherent under ongoing sidequests.
            - Maintain explicit progress and operating artifacts.

            Rules:
            """
        )
        + "\n".join(f"- {rule}" for rule in manifest["workspace_rules"])
        + "\n",
        encoding="utf-8",
    )
    (workspace / "docs/architecture.md").write_text("# Architecture\n\nStart documenting the system architecture here.\n", encoding="utf-8")
    (workspace / "docs/backlog.md").write_text("# Backlog\n\nTrack the main roadmap and deferred work here.\n", encoding="utf-8")
    (workspace / "ops/progress/progress-latest.md").write_text(
        "# Progress Report\n\nNo report yet.\n",
        encoding="utf-8",
    )
    (workspace / "bench_state/mission.md").write_text(
        textwrap.dedent(
            f"""\
            # Benchmark Mission

            Objective:
            {manifest['objective']}

            Required artifacts:
            """
        )
        + "\n".join(f"- {artifact}" for artifact in manifest["required_artifacts"])
        + "\n",
        encoding="utf-8",
    )
    (workspace / "bench_state/capability_phases.md").write_text(
        "# Capability Phases\n\n" + "\n".join(phase_lines),
        encoding="utf-8",
    )
    if capability_reference:
        summary_lines = [
            "# Airbnb Capability Reference",
            "",
            f"Source: {capability_reference['source_path']}",
            f"Boundary: {capability_reference['analysis_boundary']}",
            f"Evidence count: {capability_reference['evidence_count']}",
            f"Audit status: {capability_reference['audit_status']}",
            f"Audit reason: {capability_reference['audit_reason']}",
            "",
            "This reference is the long-horizon mission envelope for the benchmark.",
            "Do not treat it as a short to-do list. Use it to preserve scope and to keep the clone governable over time.",
            "",
            "Capability groups:",
        ]
        for group in capability_reference["capability_groups"]:
            summary_lines.append(f"- {group.get('group_id', '?')}: {group.get('name', '')}")
        summary_lines.extend(["", "Control loops:"])
        for loop in capability_reference["control_loops"]:
            summary_lines.append(f"- {loop.get('loop_id', '?')}: {loop.get('name', '')}")
        summary_lines.extend(["", "Integration capabilities:"])
        for capability in capability_reference["integration_capabilities"]:
            summary_lines.append(f"- {capability.get('integration_capability_id', '?')}: {capability.get('name', '')}")
        summary_lines.extend(["", "Representative capabilities:"])
        for capability in capability_reference["capabilities"][:12]:
            summary_lines.append(f"- {capability.get('capability_id', '?')}: {capability.get('name', '')}")
        summary_lines.extend(["", "Representative skill candidates:"])
        for skill in capability_reference["skill_candidates"][:12]:
            summary_lines.append(f"- {skill.get('skill_candidate_id', '?')}: {skill.get('name', '')}")
        (workspace / "bench_state/airbnb_capability_reference.md").write_text(
            "\n".join(summary_lines) + "\n",
            encoding="utf-8",
        )
        (workspace / "bench_state/airbnb_capability_reference.json").write_text(
            json.dumps(capability_reference["raw_payload"], indent=2) + "\n",
            encoding="utf-8",
        )
    return workspace


def build_initial_prompt(
    workspace: Path,
    manifest: Dict[str, Any],
    capability_reference: Optional[Dict[str, Any]],
) -> str:
    reference_block = ""
    if capability_reference:
        reference_block = textwrap.dedent(
            f"""\

            Additional long-horizon reference:
            - {workspace / 'bench_state/airbnb_capability_reference.md'}
            - {workspace / 'bench_state/airbnb_capability_reference.json'}

            Treat that reference as the capability envelope for the clone:
            - capability groups, integration capabilities, and control loops should emerge over time
            - the audit is intentionally still open (`{capability_reference['audit_status']}`), so the mission is not supposed to finish quickly
            - keep building toward a governable operating system, not a shallow demo
            """
        )
    return textwrap.dedent(
        f"""\
        You are starting a long-horizon benchmark mission.

        Work only inside this workspace:
        {workspace}

        Main objective:
        {manifest['objective']}

        First read these files:
        - {workspace / 'README.md'}
        - {workspace / 'bench_state/mission.md'}
        - {workspace / 'bench_state/capability_phases.md'}
        {reference_block}

        Then do one strong first slice:
        - establish a credible roadmap
        - tighten the architecture direction
        - produce real artifacts in the workspace
        - keep the mission durable with CTOX queue state when more slices are clearly needed

        Keep `ops/progress/progress-latest.md` current as the canonical latest status note.
        """
    )


def build_report_prompt(cycle: int, workspace: Path, manifest: Dict[str, Any]) -> str:
    marker = f"{manifest['report_contract']['marker_prefix']}{cycle}"
    headings = "\n".join(f"- {heading}" for heading in manifest["report_contract"]["required_headings"])
    return textwrap.dedent(
        f"""\
        {marker}

        Benchmark progress report is due now.

        First update this file:
        {workspace / 'ops/progress/progress-latest.md'}

        Then reply in chat using these exact headings:
        {headings}

        Requirements:
        - Name concrete artifacts or paths touched.
        - Distinguish main mission progress from sidequest handling.
        - State the biggest risks or blockers honestly.
        - State the next slice you intend to execute.
        - Mention which capability areas moved this cycle.
        """
    )


def preflight_runtime(
    *,
    ctox_bin: Path,
    root: Path,
    env: Dict[str, str],
    conversation_id: int,
) -> Dict[str, Any]:
    queue = run_cmd(
        [
            str(ctox_bin),
            "queue",
            "list",
            "--status",
            "pending",
            "--status",
            "leased",
            "--status",
            "blocked",
            "--limit",
            "128",
        ],
        cwd=root,
        env=env,
        expect_json=True,
    )
    snapshot = run_cmd(
        [str(ctox_bin), "lcm-dump", str(root / "runtime/ctox_lcm.db"), str(conversation_id)],
        cwd=root,
        env=env,
        expect_json=True,
    )
    return {
        "open_queue_count": int(queue.get("count", 0)),
        "conversation_message_count": len(snapshot.get("messages", [])),
        "conversation_summary_count": len(snapshot.get("summaries", [])),
    }


def capture_cycle_state(
    *,
    ctox_bin: Path,
    root: Path,
    env: Dict[str, str],
    run_dir: Path,
    cycle: int,
    conversation_id: int,
) -> Dict[str, Path]:
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    status_path = artifacts / f"status_cycle_{cycle:02d}.json"
    queue_path = artifacts / f"queue_cycle_{cycle:02d}.json"
    lcm_path = artifacts / f"lcm_cycle_{cycle:02d}.json"
    health_path = artifacts / f"context_health_cycle_{cycle:02d}.json"

    status = run_cmd([str(ctox_bin), "status"], cwd=root, env=env, expect_json=True)
    queue = run_cmd(
        [
            str(ctox_bin),
            "queue",
            "list",
            "--status",
            "pending",
            "--status",
            "leased",
            "--status",
            "blocked",
            "--limit",
            "128",
        ],
        cwd=root,
        env=env,
        expect_json=True,
    )
    lcm = run_cmd(
        [str(ctox_bin), "lcm-dump", str(root / "runtime/ctox_lcm.db"), str(conversation_id)],
        cwd=root,
        env=env,
        expect_json=True,
    )
    health = run_cmd(
        [
            str(ctox_bin),
            "context-health",
            str(root / "runtime/ctox_lcm.db"),
            str(conversation_id),
            f"bench-cycle-{cycle}",
            "131072",
        ],
        cwd=root,
        env=env,
        expect_json=True,
    )

    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    queue_path.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
    lcm_path.write_text(json.dumps(lcm, indent=2) + "\n", encoding="utf-8")
    health_path.write_text(json.dumps(health, indent=2) + "\n", encoding="utf-8")
    return {
        "status": status_path,
        "queue": queue_path,
        "lcm": lcm_path,
        "health": health_path,
    }


def extract_report_text(snapshot: Dict[str, Any], cycle: int, manifest: Dict[str, Any]) -> str:
    marker = f"{manifest['report_contract']['marker_prefix']}{cycle}"
    messages = snapshot.get("messages", [])
    report_user_seq = None
    for message in messages:
        if message.get("role") == "user" and marker in message.get("content", ""):
            report_user_seq = int(message["seq"])
    if report_user_seq is None:
        raise RuntimeError(f"report marker {marker} not found in LCM snapshot")
    for message in messages:
        if int(message.get("seq", -1)) > report_user_seq and message.get("role") == "assistant":
            return message.get("content", "")
    raise RuntimeError(f"assistant report after marker {marker} not found")


def workspace_metrics(workspace: Path) -> Dict[str, int]:
    file_count = 0
    code_files = 0
    report_files = 0
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        file_count += 1
        if path.suffix in {".rs", ".ts", ".tsx", ".js", ".jsx", ".py", ".sql", ".sh"}:
            code_files += 1
        if path.parts[-2:] and "progress" in path.parts:
            report_files += 1
    return {
        "file_count": file_count,
        "code_file_count": code_files,
        "progress_file_count": report_files,
    }


def normalize_report_line(line: str) -> str:
    normalized = line.strip().lower()
    for token in ("**", "__", "`"):
        normalized = normalized.replace(token, "")
    while normalized.startswith(("#", "-", "*", " ")):
        normalized = normalized[1:].lstrip()
    return normalized.strip()


def score_report(
    *,
    report_text: str,
    manifest: Dict[str, Any],
    workspace: Path,
    health: Dict[str, Any],
    queue_snapshot: Dict[str, Any],
    previous_report_text: Optional[str],
    previous_metrics: Optional[Dict[str, int]],
) -> Dict[str, Any]:
    lowered = report_text.lower()
    lines = [normalize_report_line(line) for line in report_text.splitlines()]
    score = 100
    findings: List[str] = []
    headings_missing = []
    for heading in manifest["report_contract"]["required_headings"]:
        heading_lower = heading.lower()
        heading_present = any(
            line == heading_lower
            or line.startswith(f"{heading_lower}:")
            for line in lines
        )
        if not heading_present:
            headings_missing.append(heading)
    if headings_missing:
        score -= min(25, 4 * len(headings_missing))
        findings.append(f"missing report headings: {', '.join(headings_missing)}")

    artifact_hits = sum(1 for token in ["/", ".md", ".rs", ".ts", "apps/", "docs/", "ops/"] if token in report_text)
    if artifact_hits < 2:
        score -= 12
        findings.append("report names too few concrete artifacts or paths")

    capability_terms = []
    for phase in manifest["phases"]:
        capability_terms.append(phase["name"].lower())
        capability_terms.extend(group.lower() for group in phase["capability_groups"])
        capability_terms.append(phase["phase_id"].lower())
    capability_hits = sum(1 for term in capability_terms if term in lowered)
    if capability_hits == 0:
        score -= 10
        findings.append("report does not name moved capability areas")

    next_present = any(
        line == "next"
        or line.startswith("next:")
        for line in lines
    ) or "next step" in lowered
    if not next_present:
        score -= 10
        findings.append("report does not define a next slice clearly")

    if "risk" not in lowered and "block" not in lowered:
        score -= 8
        findings.append("report does not state risks or blockers clearly")

    if "sidequest" not in lowered and "queue" not in lowered and "incident" not in lowered:
        score -= 8
        findings.append("report does not show sidequest handling explicitly")

    metrics = workspace_metrics(workspace)
    if metrics["file_count"] < 6:
        score -= 10
        findings.append("workspace still has too little artifact growth")
    if metrics["progress_file_count"] == 0:
        score -= 10
        findings.append("progress artifacts are missing from workspace")

    if previous_report_text:
        similarity = difflib.SequenceMatcher(a=previous_report_text, b=report_text).ratio()
        if similarity > 0.92:
            score -= 20
            findings.append(f"report is near-duplicate of previous cycle ({similarity:.2f} similarity)")
        elif similarity > 0.84:
            score -= 10
            findings.append(f"report is too similar to previous cycle ({similarity:.2f} similarity)")
    else:
        similarity = None

    if previous_metrics:
        if metrics["file_count"] <= previous_metrics["file_count"] and metrics["code_file_count"] <= previous_metrics["code_file_count"]:
            score -= 8
            findings.append("workspace metrics show little or no structural progress since previous cycle")

    health_status = str(health.get("status", "")).lower()
    health_score = int(health.get("overall_score", 0))
    if health_status == "critical":
        score -= 25
        findings.append("context health is critical")
    elif health_status not in {"healthy", ""}:
        score -= 12
        findings.append(f"context health is degraded ({health_status})")
    if health.get("repair_recommended"):
        score -= 10
        findings.append("context health recommends repair")

    warning_codes = [warning.get("code") for warning in health.get("warnings", []) if warning.get("code")]
    if warning_codes:
        score -= min(20, 4 * len(warning_codes))
        findings.append(f"context health warnings present: {', '.join(warning_codes[:5])}")

    pending_tasks = int(queue_snapshot.get("count", 0))
    if pending_tasks > 12:
        score -= 10
        findings.append(f"queue pressure is high ({pending_tasks} open queue items)")

    score = max(0, min(100, score))
    if score >= 85:
        verdict = "strong"
    elif score >= 70:
        verdict = "acceptable"
    elif score >= 55:
        verdict = "warning"
    else:
        verdict = "major_flaw"

    return {
        "score": score,
        "verdict": verdict,
        "findings": findings,
        "workspace_metrics": metrics,
        "context_health_score": health_score,
        "context_health_status": health_status,
        "context_warning_codes": warning_codes,
        "report_similarity_to_previous": similarity,
        "report_excerpt": report_text[:800],
    }


def inject_queue_sidequests(
    *,
    ctox_bin: Path,
    root: Path,
    env: Dict[str, str],
    manifest: Dict[str, Any],
    cycle: int,
) -> List[str]:
    created = []
    span = max(1, max(int(item["cycle"]) for item in manifest["sidequests"]))
    cycle_slot = ((cycle - 1) % span) + 1
    for item in manifest["sidequests"]:
        if int(item["cycle"]) != cycle_slot:
            continue
        run_cmd(
            [
                str(ctox_bin),
                "queue",
                "add",
                "--title",
                item["title"],
                "--prompt",
                item["prompt"],
                "--thread-key",
                item["thread_key"],
                "--priority",
                item["priority"],
            ],
            cwd=root,
            env=env,
            expect_json=True,
        )
        created.append(item["title"])
    return created


def inject_owner_hints(base_url: str, manifest: Dict[str, Any], cycle: int) -> List[str]:
    sent = []
    span = max(1, max(int(item["cycle"]) for item in manifest["owner_hints"]))
    cycle_slot = ((cycle - 1) % span) + 1
    for hint in manifest["owner_hints"]:
        if int(hint["cycle"]) != cycle_slot:
            continue
        post_service_chat(base_url, hint["message"])
        sent.append(hint["title"])
    return sent


def run_benchmark(args: argparse.Namespace) -> int:
    root = Path(args.ctox_root).resolve()
    note_phase(root, "run_benchmark:start")
    source_root = repo_root()
    sync_repo_source(source_root, root)
    note_phase(root, "run_benchmark:sync_repo_source")
    ensure_bench_runtime_env(root, source_root, args.model)
    note_phase(root, "run_benchmark:ensure_bench_runtime_env")
    manifest = load_manifest(root)
    capability_reference = load_capability_reference(root)
    ctox_bin = ensure_ctox_bin(root, force_rebuild=root != source_root)
    note_phase(root, "run_benchmark:ensure_ctox_bin")
    env = os.environ.copy()
    env["CTOX_ROOT"] = str(root)
    env["CODEX_HOME"] = str((root / "runtime/codex_home").resolve())

    if args.reset_runtime:
        note_phase(root, "run_benchmark:reset_runtime:stop_service")
        stop_service_if_running(ctox_bin, root, env)
        note_phase(root, "run_benchmark:reset_runtime:reset_runtime_dir")
        reset_runtime_dir(root)
        note_phase(root, "run_benchmark:reset_runtime:ensure_bench_runtime_env")
        ensure_bench_runtime_env(root, source_root, args.model)

    run_root = root / "runtime/airbnb_clone_bench/runs"
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / now_slug()
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace = setup_workspace(run_dir, manifest, capability_reference)

    meta = {
        "root": str(root),
        "run_dir": str(run_dir),
        "workspace": str(workspace),
        "cycles": args.cycles,
        "continuous": args.continuous,
        "max_runtime_seconds": args.max_runtime_seconds,
        "max_idle_gap_seconds": args.max_idle_gap_seconds,
        "report_interval_seconds": args.report_interval_seconds,
        "conversation_id": args.conversation_id,
        "model": args.model,
        "capability_reference_source": capability_reference["source_path"] if capability_reference else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    if args.prepare_only:
        print(json.dumps({"ok": True, "prepared_only": True, "run_dir": str(run_dir), "workspace": str(workspace)}, indent=2))
        return 0

    note_phase(root, "run_benchmark:ensure_service")
    ensure_service(
        ctox_bin,
        root,
        env,
        reuse_running_service=args.reuse_running_service,
    )
    base_url = service_base_url(ctox_bin, root, env)
    proxy_url = proxy_base_url(root)
    note_phase(root, "run_benchmark:wait_for_http_ready")
    wait_for_service_http_ready(base_url)
    note_phase(root, "run_benchmark:wait_for_local_runtime_ready")
    wait_for_local_runtime_ready(proxy_url)
    note_phase(root, "run_benchmark:preflight_runtime")
    preflight = preflight_runtime(
        ctox_bin=ctox_bin,
        root=root,
        env=env,
        conversation_id=args.conversation_id,
    )
    (run_dir / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    if not args.allow_dirty_runtime and (
        preflight["open_queue_count"] > 0 or preflight["conversation_message_count"] > 0
    ):
        raise RuntimeError(
            "benchmark root is not clean; use an isolated CTOX_ROOT or rerun with --allow-dirty-runtime"
        )
    note_phase(root, "run_benchmark:post_initial_prompt")
    post_service_chat(base_url, build_initial_prompt(workspace, manifest, capability_reference))
    note_phase(root, "run_benchmark:wait_until_idle_after_initial_prompt")
    wait_until_idle(ctox_bin, root, env, timeout_seconds=args.idle_timeout_seconds)

    previous_report_text = None
    previous_metrics = None
    cycle_summaries = []
    benchmark_started_at = time.time()
    cycle = 1
    idle_failure_detected = False
    idle_failure_reason = None

    while True:
        if not args.continuous and cycle > args.cycles:
            break
        if args.continuous and args.max_runtime_seconds > 0:
            elapsed = time.time() - benchmark_started_at
            if elapsed >= args.max_runtime_seconds:
                break
        sidequests = inject_queue_sidequests(ctox_bin=ctox_bin, root=root, env=env, manifest=manifest, cycle=cycle)
        hints = inject_owner_hints(base_url, manifest, cycle)
        if cycle > 1 and args.report_interval_seconds > 0:
            if args.continuous:
                try:
                    monitor_service_for_idle_failure(
                        ctox_bin=ctox_bin,
                        root=root,
                        env=env,
                        monitor_seconds=args.report_interval_seconds,
                        max_idle_gap_seconds=args.max_idle_gap_seconds,
                    )
                except RuntimeError as err:
                    idle_failure_detected = True
                    idle_failure_reason = str(err)
                    break
            else:
                time.sleep(args.report_interval_seconds)
        if sidequests or hints:
            note_phase(root, f"run_benchmark:cycle:{cycle}:wait_for_service_activity")
            wait_for_service_activity(
                ctox_bin,
                root,
                env,
                timeout_seconds=min(args.idle_timeout_seconds, 30),
            )

        note_phase(root, f"run_benchmark:cycle:{cycle}:post_report_prompt")
        post_service_chat(base_url, build_report_prompt(cycle, workspace, manifest))
        note_phase(root, f"run_benchmark:cycle:{cycle}:wait_for_report_response")
        wait_for_report_response(
            ctox_bin=ctox_bin,
            root=root,
            env=env,
            conversation_id=args.conversation_id,
            cycle=cycle,
            manifest=manifest,
            timeout_seconds=args.idle_timeout_seconds,
        )

        captured = capture_cycle_state(
            ctox_bin=ctox_bin,
            root=root,
            env=env,
            run_dir=run_dir,
            cycle=cycle,
            conversation_id=args.conversation_id,
        )
        snapshot = json.loads(captured["lcm"].read_text())
        health = json.loads(captured["health"].read_text())
        queue_state = json.loads(captured["queue"].read_text())
        report_text = extract_report_text(snapshot, cycle, manifest)
        report_path = run_dir / "artifacts" / f"report_cycle_{cycle:02d}.txt"
        report_path.write_text(report_text + "\n", encoding="utf-8")

        evaluation = score_report(
            report_text=report_text,
            manifest=manifest,
            workspace=workspace,
            health=health,
            queue_snapshot=queue_state,
            previous_report_text=previous_report_text,
            previous_metrics=previous_metrics,
        )
        evaluation.update(
            {
                "cycle": cycle,
                "sidequests_injected": sidequests,
                "owner_hints_sent": hints,
                "report_path": str(report_path),
            }
        )
        eval_path = run_dir / "artifacts" / f"report_eval_cycle_{cycle:02d}.json"
        eval_path.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")

        previous_report_text = report_text
        previous_metrics = evaluation["workspace_metrics"]
        cycle_summaries.append(
            {
                "cycle": cycle,
                "score": evaluation["score"],
                "verdict": evaluation["verdict"],
                "context_health_status": evaluation["context_health_status"],
                "context_health_score": evaluation["context_health_score"],
                "findings": evaluation["findings"],
            }
        )

        if args.stop_on_major_flaw and evaluation["verdict"] == "major_flaw":
            break
        cycle += 1

    summary = {
        "run_dir": str(run_dir),
        "workspace": str(workspace),
        "cycles_completed": len(cycle_summaries),
        "major_flaw_detected": any(item["verdict"] == "major_flaw" for item in cycle_summaries) or idle_failure_detected,
        "idle_failure_detected": idle_failure_detected,
        "idle_failure_reason": idle_failure_reason,
        "cycle_summaries": cycle_summaries,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def self_test() -> int:
    manifest = load_manifest(repo_root())
    temp_root = repo_root() / "runtime/airbnb_clone_bench/self_test"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)
    workspace = setup_workspace(temp_root, manifest, None)
    (workspace / "apps/web/app.tsx").write_text("export const ready = true;\n", encoding="utf-8")
    (workspace / "apps/api/server.ts").write_text("export const server = true;\n", encoding="utf-8")
    (workspace / "ops/progress/progress-latest.md").write_text(
        "# Progress Report\n\nMission: keep moving.\nCompleted: architecture and backlog.\nArtifacts: docs/architecture.md, docs/backlog.md.\nSidequests: queue triage.\nRisks: payouts and trust.\nNext: booking flow.\nCapabilities: Marketplace Core, Booking Operations.\n",
        encoding="utf-8",
    )
    report = textwrap.dedent(
        """\
        Mission: Build and operate the clone while keeping the roadmap coherent.
        Completed: Updated docs/architecture.md, docs/backlog.md, apps/web/app.tsx, and apps/api/server.ts.
        Artifacts: docs/architecture.md, docs/backlog.md, ops/progress/progress-latest.md, apps/web/app.tsx.
        Sidequests: Processed a queue sidequest about cancellation policy and kept the main roadmap intact.
        Risks: Payments and trust flows still need deeper work.
        Next: Implement booking lifecycle and incident runbooks.
        Capabilities: Marketplace Core, Booking Operations.
        """
    )
    evaluation = score_report(
        report_text=report,
        manifest=manifest,
        workspace=workspace,
        health={"status": "healthy", "overall_score": 92, "warnings": [], "repair_recommended": False},
        queue_snapshot={"count": 2},
        previous_report_text=None,
        previous_metrics=None,
    )
    if evaluation["score"] < 80 or evaluation["verdict"] not in {"strong", "acceptable"}:
        print(json.dumps(evaluation, indent=2))
        return 1

    snapshot = {
        "messages": [
            {"seq": 1, "role": "user", "content": "start"},
            {"seq": 2, "role": "assistant", "content": "ok"},
            {"seq": 3, "role": "user", "content": f"{manifest['report_contract']['marker_prefix']}2\nreport please"},
            {"seq": 4, "role": "assistant", "content": report},
        ]
    }
    extracted = extract_report_text(snapshot, 2, manifest)
    if "Capabilities:" not in extracted:
        return 1
    print(json.dumps({"ok": True, "self_test": True, "score": evaluation["score"]}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CTOX Airbnb clone long-horizon benchmark.")
    parser.add_argument("--ctox-root", default=str(repo_root()))
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--max-runtime-seconds", type=int, default=0)
    parser.add_argument("--max-idle-gap-seconds", type=int, default=180)
    parser.add_argument("--report-interval-seconds", type=int, default=3600)
    parser.add_argument("--idle-timeout-seconds", type=int, default=900)
    parser.add_argument("--conversation-id", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-dirty-runtime", action="store_true")
    parser.add_argument("--reset-runtime", action="store_true")
    parser.add_argument("--reuse-running-service", action="store_true")
    parser.add_argument("--stop-on-major-flaw", action="store_true", default=False)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.ctox_root).resolve()
    install_bench_signal_handlers(root)
    if not args.reuse_running_service:
        atexit.register(lambda: stop_bench_managed_service(root))
    if args.self_test:
        return self_test()
    try:
        return run_benchmark(args)
    except Exception:
        bench_error_log_path(root).write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    sys.exit(main())
