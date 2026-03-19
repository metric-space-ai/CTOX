#!/usr/bin/env python3
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PORT = int(os.environ.get("CTO_MOCK_PORT", "12391"))
STATE_FILE = Path(os.environ.get("CTO_MOCK_STATE_FILE", "/tmp/cto_mock_kleinhirn_state.json"))
MODEL_ID = os.environ.get("CTO_MOCK_MODEL_ID", "openai/gpt-oss-20b")


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {"mode": "healthy"}
    return {"mode": "healthy"}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def current_mode(*, probe: bool) -> str:
    state = load_state()
    temporary_key = "temporaryReadyModes" if probe else "temporaryChatModes"
    temporary = state.get(temporary_key) or []
    if temporary:
        mode = temporary.pop(0)
        state[temporary_key] = temporary
        save_state(state)
        return str(mode)
    return str(state.get("mode", "healthy"))


def ready_payload() -> bytes:
    body = {
        "id": "mock-ready",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "READY"},
                "finish_reason": "stop",
            }
        ],
    }
    return json.dumps(body).encode("utf-8")


def content_payload(content: str) -> bytes:
    body = {
        "id": "mock-chat",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    return json.dumps(body).encode("utf-8")


def completion_payload(content: str) -> bytes:
    body = {
        "id": "mock-completion",
        "object": "text_completion",
        "choices": [
            {
                "index": 0,
                "text": content,
                "finish_reason": "stop",
            }
        ],
    }
    return json.dumps(body).encode("utf-8")


def responses_payload(content: str) -> bytes:
    body = {
        "id": "mock-response",
        "object": "response",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                    }
                ],
            }
        ],
    }
    return json.dumps(body).encode("utf-8")


def healthy_json(reply: str, checkpoint: str = "bounded step ok") -> str:
    return json.dumps(
        {
            "taskStatus": "done",
            "nextMode": "review",
            "checkpointSummary": checkpoint,
            "reply": reply,
        }
    )


def reprioritize_json(user_text: str) -> str | None:
    try:
        payload = json.loads(user_text)
    except Exception:
        return None
    if payload.get("trigger") != "reprioritize_selection":
        return None
    candidates = payload.get("candidateTasks") or []
    if not isinstance(candidates, list) or not candidates:
        return json.dumps(
            {
                "checkpointSummary": "Keine Kandidaten sichtbar.",
            }
        )
    selected = None
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("trustLevel") == "owner":
            selected = candidate
            break
    if selected is None:
        selected = candidates[0]
    selected_id = selected.get("id") if isinstance(selected, dict) else None
    return json.dumps(
        {
            "selectedTaskId": selected_id,
            "checkpointSummary": f"Ich waehle Task {selected_id} als naechsten bounded Fokus.",
        }
    )


def infer_task_kind(user_text: str) -> str:
    for line in user_text.splitlines():
        if line.startswith("Art: "):
            return line.split("Art: ", 1)[1].strip()
    return ""


class Handler(BaseHTTPRequestHandler):
    server_version = "cto-mock-kleinhirn/1.0"

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))

    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL_ID,
                            "object": "model",
                            "owned_by": "cto-mock",
                        }
                    ],
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/health":
            mode = current_mode(probe=False)
            body = json.dumps({"status": "ok", "mode": mode}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/v1/responses", "/v1/completions"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {}
        user_text = extract_user_text(payload)
        reprioritize = reprioritize_json(user_text)
        if reprioritize is not None:
            body = render_payload(self.path, reprioritize)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        task_kind = infer_task_kind(user_text)
        mode = current_mode(probe=False)

        if "Reply with READY" in user_text:
            mode = current_mode(probe=True)
            if mode == "timeout":
                time.sleep(float(os.environ.get("CTO_MOCK_TIMEOUT_SECS", "25")))
            elif mode == "error":
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b'{"error":"mock error"}')
                return
            elif mode == "malformed":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"not-json")
                return
            elif mode in ("overflow", "overflow_on_probe"):
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error":"maximum context length exceeded"}')
                return
            body = render_payload(self.path, "READY")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        mode = current_mode(probe=False)
        if mode == "timeout":
            time.sleep(float(os.environ.get("CTO_MOCK_TIMEOUT_SECS", "25")))
            return
        if mode == "error":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error":"mock error"}')
            return
        if mode == "malformed":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"not-json")
            return
        if mode in ("overflow", "overflow_on_task"):
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":"maximum context length exceeded"}')
            return

        if mode == "delegate":
            content = json.dumps(
                {
                    "taskStatus": "done",
                    "nextMode": "delegate",
                    "checkpointSummary": "Delegation is the best bounded next step.",
                    "reply": "Ich delegiere diesen Schritt.",
                    "delegateWorkerKind": "specialist_worker",
                    "delegateContractTitle": "Delegierte Umsetzung aus Mock-Kleinhirn",
                    "delegateContractDetail": "Fuehre bounded Umsetzung aus und melde fuer Review zurueck.",
                    "delegateRequestNote": "Arbeite autonom und bounded.",
                }
            )
        elif mode == "historical_research":
            content = json.dumps(
                {
                    "taskStatus": "continue",
                    "nextMode": "historical_research",
                    "checkpointSummary": "Ich brauche gezielte historische Nachladung vor dem naechsten bounded Schritt.",
                    "reply": "Mir fehlt eine alte Festlegung und ich will sie gezielt nachladen.",
                    "contextAction": "expand_history",
                    "contextConcern": "Fruehere Owner-Entscheidung zum Branding ist noch unscharf.",
                    "historyResearchQuery": "Finde fruehere Owner-Aussagen zu BIOS, Branding und Superpassword.",
                }
            )
        elif mode == "question_compaction":
            content = json.dumps(
                {
                    "taskStatus": "continue",
                    "nextMode": "historical_research",
                    "checkpointSummary": "Die aktuelle Verdichtung wirkt unsauber; ich will die Rohhistorie gezielt gegenpruefen.",
                    "reply": "Ich hinterfrage die aktuelle Kompaktierung.",
                    "contextAction": "question_compaction",
                    "contextConcern": "Eine verdichtete Fassung wirkt widerspruechlich.",
                    "historyResearchQuery": "Hole die rohen Stellen zur widerspruechlichen Festlegung erneut.",
                }
            )
        elif mode == "blocked":
            content = json.dumps(
                {
                    "taskStatus": "blocked",
                    "nextMode": "blocked",
                    "checkpointSummary": "Mock-Kleinhirn blockiert diese Aufgabe bewusst.",
                    "reply": "Ich bin an dieser Stelle blockiert.",
                }
            )
        else:
            if task_kind == "homepage_bridge":
                content = json.dumps(
                    {
                        "taskStatus": "done",
                        "nextMode": "review",
                        "checkpointSummary": "Ich habe die BIOS-Homepage sichtbar weiter aufgebaut.",
                        "reply": "Die BIOS-Homepage wurde als erster Vertrauenspfad weiter ausgeformt.",
                        "homepageTitle": "CTO-Agent BIOS Bridge",
                        "homepageHeadline": "Der Agent baut seine BIOS-Oberflaeche waehrend des laufenden Infinity Loops weiter aus.",
                        "homepageIntro": "Diese Homepage wird aktiv vom Agenten selbst weitergebaut, waehrend er lebt.",
                        "homepageCommunicationNote": "Der Besitzer kann hier spaeter komfortabler mit dem Agenten sprechen als nur im Terminal.",
                        "homepageTerminalFallbackNote": "Wenn die Homepage versagt, bleibt das Terminal die Systemebene.",
                        "execCommand": [
                            "sh",
                            "-lc",
                            "mkdir -p runtime/agent-artifacts && printf 'homepage bridge built\\n' > runtime/agent-artifacts/homepage-bootstrap.txt"
                        ],
                        "execJustification": "Lege ein sichtbares bounded Artefakt zur Homepage-Bootstrap-Arbeit an."
                    }
                )
            else:
                content = healthy_json("Mock-Kleinhirn bounded step completed.")

        body = render_payload(self.path, content)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def extract_user_text(payload: dict) -> str:
    if isinstance(payload.get("prompt"), str):
        return str(payload.get("prompt") or "")
    input_items = payload.get("input")
    if isinstance(input_items, list):
        chunks = []
        for item in input_items:
            if not isinstance(item, dict):
                continue
            if item.get("role") == "user":
                content = item.get("content")
                if isinstance(content, str):
                    chunks.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text")
                            if isinstance(text, str):
                                chunks.append(text)
        return "\n".join(chunks)
    messages = payload.get("messages") or []
    return "\n".join(
        item.get("content", "")
        for item in messages
        if isinstance(item, dict) and item.get("role") == "user"
    )


def render_payload(path: str, content: str) -> bytes:
    if path == "/v1/responses":
        return responses_payload(content)
    if path == "/v1/completions":
        return completion_payload(content)
    return content_payload(content)


def main() -> int:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        save_state({"mode": "healthy"})
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(json.dumps({"status": "ok", "port": PORT, "stateFile": str(STATE_FILE)}), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
