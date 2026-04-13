#[path = "../harness/mod.rs"]
mod harness;

use harness::TestRoot;
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::net::TcpListener;
use std::process::Command;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tiny_http::{Header, Method, Response, Server, StatusCode};

#[test]
fn proxy_load_and_switch_matrix_passes_when_enabled() {
    let models = match env::var("CTOX_PROXY_E2E_MODELS") {
        Ok(value) => value,
        Err(_) => return,
    };
    let model_list = models
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    if model_list.is_empty() {
        return;
    }

    let mut command = Command::new("python3");
    command.arg("scripts/ctox_proxy_validate.py");
    for model in &model_list {
        command.arg("--model").arg(model);
    }
    if let Ok(preset) = env::var("CTOX_PROXY_E2E_PRESET") {
        if !preset.trim().is_empty() {
            command.arg("--preset").arg(preset);
        }
    }
    if let Ok(timeout) = env::var("CTOX_PROXY_E2E_TIMEOUT_SECS") {
        if !timeout.trim().is_empty() {
            command.arg("--timeout-secs").arg(timeout);
        }
    }

    let output = command
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("failed to execute proxy validator");
    assert!(
        output.status.success(),
        "proxy validator failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn gemma4_proxy_roundtrip_matches_codex_responses_live() {
    let root = TestRoot::new("gemma4-proxy-live");
    let upstream_port = free_port();
    let proxy_port = free_port();
    let codex_home = root.path("runtime/codex_home");
    fs::create_dir_all(&codex_home).expect("failed to create CODEX_HOME");
    fs::write(
        root.path("runtime/engine.env"),
        format!(
            "CTOX_ACTIVE_MODEL=google/gemma-4-31B-it\n\
CTOX_CHAT_MODEL_BASE=google/gemma-4-31B-it\n\
CTOX_CHAT_MODEL=google/gemma-4-31B-it\n\
CTOX_CHAT_SOURCE=local\n\
CTOX_DISABLE_AUXILIARY_BACKENDS=1\n\
CTOX_ENGINE_PORT={upstream_port}\n\
CTOX_PROXY_HOST=127.0.0.1\n\
CTOX_PROXY_PORT={proxy_port}\n\
CODEX_HOME={codex_home}\n",
            upstream_port = upstream_port,
            proxy_port = proxy_port,
            codex_home = codex_home.display()
        ),
    )
    .expect("failed to write runtime env");

    let captured_requests = Arc::new(Mutex::new(Vec::<Value>::new()));
    let server = Server::http(("127.0.0.1", upstream_port)).expect("failed to bind mock upstream");
    let server_thread = spawn_gemma_mock_upstream(server, captured_requests.clone());

    let proxy_log_path = root.path("runtime/proxy.log");
    let proxy_log = fs::File::create(&proxy_log_path).expect("failed to create proxy log");
    let proxy_log_err = proxy_log
        .try_clone()
        .expect("failed to clone proxy log handle");
    let mut proxy = Command::new(env!("CARGO_BIN_EXE_ctox"))
        .arg("serve-responses-proxy")
        .env("CTOX_ROOT", root.path(""))
        .stdout(Stdio::from(proxy_log))
        .stderr(Stdio::from(proxy_log_err))
        .spawn()
        .expect("failed to launch proxy");

    wait_for_proxy_ready(proxy_port);

    let models_text = ureq::get(&format!("http://127.0.0.1:{proxy_port}/v1/models"))
        .call()
        .expect("proxy /v1/models failed")
        .into_string()
        .expect("failed to read /v1/models text");
    assert!(
        models_text.contains("google/gemma-4-31B-it"),
        "unexpected models payload: {models_text}"
    );

    let response = ureq::post(&format!("http://127.0.0.1:{proxy_port}/v1/responses"))
        .set("content-type", "application/json")
        .send_string(
            &json!({
                "model": "google/gemma-4-31B-it",
                "instructions": "You are a careful assistant.",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type":"input_text","text":"Check the weather tool."}]
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_weather_1",
                        "name": "weather.lookup",
                        "arguments": "{\"city\":\"Berlin\"}"
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_weather_1",
                        "output": "{\"temp_c\":17}"
                    }
                ],
                "tools": [{
                    "type": "function",
                    "name": "weather.lookup",
                    "description": "Weather lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"}
                        },
                        "required": ["city"]
                    }
                }],
                "reasoning": {"effort":"high"},
                "max_output_tokens": 64
            })
            .to_string(),
        )
        .expect("proxy /v1/responses failed")
        .into_string()
        .expect("failed to read responses body");
    let response_json: Value =
        serde_json::from_str(&response).expect("proxy response was not valid json");
    assert_eq!(response_json["object"], "response");
    assert_eq!(response_json["model"], "google/gemma-4-31B-it");
    assert_eq!(
        response_json["reasoning"],
        "Need a weather lookup before answering."
    );
    assert_eq!(response_json["output_text"], "Weather looks mild.");
    let output = response_json["output"]
        .as_array()
        .expect("missing output array");
    assert_eq!(output.len(), 2);
    assert_eq!(output[0]["type"], "message");
    assert_eq!(output[0]["content"][0]["text"], "Weather looks mild.");
    assert_eq!(output[1]["type"], "function_call");
    assert_eq!(output[1]["name"], "weather.lookup");
    assert_eq!(output[1]["arguments"], "{\"city\":\"Berlin\"}");

    let captured = captured_requests
        .lock()
        .expect("captured request lock poisoned");
    assert_eq!(captured.len(), 1, "expected exactly one upstream request");
    let upstream_request = &captured[0];
    assert_eq!(upstream_request["model"], "google/gemma-4-31B-it");
    assert_eq!(upstream_request["enable_thinking"], true);
    assert_eq!(upstream_request["reasoning_effort"], "high");
    let messages = upstream_request["messages"]
        .as_array()
        .expect("missing upstream messages");
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[2]["role"], "assistant");
    assert_eq!(messages[2]["tool_calls"][0]["id"], "call_weather_1");
    assert_eq!(
        messages[2]["tool_calls"][0]["function"]["name"],
        "weather.lookup"
    );
    assert_eq!(messages[3]["role"], "tool");
    assert_eq!(messages[3]["name"], "weather.lookup");
    assert_eq!(messages[3]["tool_call_id"], "call_weather_1");
    assert_eq!(messages[3]["content"], "{\"temp_c\":17}");
    drop(captured);

    let _ = ureq::get(&format!("http://127.0.0.1:{upstream_port}/shutdown")).call();
    let _ = proxy.kill();
    let _ = proxy.wait();
    server_thread.join().expect("mock upstream thread failed");
}

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("failed to bind ephemeral port");
    let port = listener
        .local_addr()
        .expect("failed to read local addr")
        .port();
    drop(listener);
    port
}

fn json_header() -> Header {
    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
        .expect("invalid content-type header")
}

fn spawn_gemma_mock_upstream(
    server: Server,
    captured_requests: Arc<Mutex<Vec<Value>>>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        while let Ok(mut request) = server.recv() {
            let method = request.method().clone();
            let url = request.url().to_string();
            match (method, url.as_str()) {
                (Method::Get, "/health") => {
                    let response = Response::from_string("ok").with_status_code(StatusCode(200));
                    let _ = request.respond(response);
                }
                (Method::Get, "/v1/models") => {
                    let payload = json!({
                        "object": "list",
                        "data": [{"id":"google/gemma-4-31B-it","object":"model"}]
                    });
                    let response = Response::from_string(payload.to_string())
                        .with_status_code(StatusCode(200))
                        .with_header(json_header());
                    let _ = request.respond(response);
                }
                (Method::Post, "/v1/chat/completions") => {
                    let mut body = String::new();
                    request
                        .as_reader()
                        .read_to_string(&mut body)
                        .expect("failed to read upstream request body");
                    let parsed: Value =
                        serde_json::from_str(&body).expect("upstream request was not json");
                    captured_requests
                        .lock()
                        .expect("captured request lock poisoned")
                        .push(parsed);
                    let payload = json!({
                        "id":"gemma-chatcmpl-1",
                        "object":"chat.completion",
                        "created": 1_744_000_000u64,
                        "model":"google/gemma-4-31B-it",
                        "choices":[{
                            "index":0,
                            "message":{
                                "role":"assistant",
                                "content":"<|channel>thought\nNeed a weather lookup before answering.\n<channel|>Weather looks mild.<|tool_call>call:weather.lookup{\"city\":\"Berlin\"}<tool_call|>"
                            },
                            "finish_reason":"tool_calls"
                        }],
                        "usage":{"prompt_tokens":32,"completion_tokens":12,"total_tokens":44}
                    });
                    let response = Response::from_string(payload.to_string())
                        .with_status_code(StatusCode(200))
                        .with_header(json_header());
                    let _ = request.respond(response);
                }
                (Method::Get, "/shutdown") => {
                    let response = Response::from_string("bye").with_status_code(StatusCode(200));
                    let _ = request.respond(response);
                    break;
                }
                _ => {
                    let response =
                        Response::from_string("not found").with_status_code(StatusCode(404));
                    let _ = request.respond(response);
                }
            }
        }
    })
}

fn wait_for_proxy_ready(proxy_port: u16) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if let Ok(response) =
            ureq::get(&format!("http://127.0.0.1:{proxy_port}/ctox/telemetry")).call()
        {
            if let Ok(text) = response.into_string() {
                if let Ok(payload) = serde_json::from_str::<Value>(&text) {
                    if payload["active_model"] == "google/gemma-4-31B-it"
                        && payload["backend_healthy"] == true
                    {
                        return;
                    }
                }
            }
        }
        thread::sleep(Duration::from_millis(250));
    }
    panic!("proxy did not become ready on port {proxy_port}");
}
