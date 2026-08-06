// ref: internal/runtime/executor/antigravity_executor_request.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::Value;
use sha2::{Digest, Sha256};

/// Applies the upstream Antigravity envelope metadata and schema sanitation
/// needed by `generateContent`. Reasoning replay stays in its own mirrored
/// module so request normalization remains independently testable.
pub fn prepare_antigravity_generate_body(
    translated_body: &[u8],
    model: &str,
    project_id: &str,
) -> Result<Vec<u8>, AntigravityRequestError> {
    if model.trim().is_empty() {
        return Err(AntigravityRequestError::MissingModel);
    }
    if project_id.trim().is_empty() || project_id.chars().any(char::is_control) {
        return Err(AntigravityRequestError::InvalidProjectId);
    }
    let mut root: Value = serde_json::from_slice(translated_body)
        .map_err(|_| AntigravityRequestError::InvalidJson)?;
    sanitize_antigravity_request_schemas(
        &mut root,
        model.to_ascii_lowercase().contains("claude")
            || model.contains("gemini-3-pro")
            || model.contains("gemini-3.1-pro"),
    );
    let object = root
        .as_object_mut()
        .ok_or(AntigravityRequestError::InvalidObject)?;
    object.insert("model".to_owned(), Value::String(model.to_owned()));
    object.insert(
        "userAgent".to_owned(),
        Value::String("antigravity".to_owned()),
    );
    object.insert(
        "requestType".to_owned(),
        Value::String(
            object
                .get("requestType")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .unwrap_or(if model.contains("image") {
                    "image_gen"
                } else {
                    "agent"
                })
                .to_owned(),
        ),
    );
    object.insert(
        "project".to_owned(),
        Value::String(project_id.trim().to_owned()),
    );
    object.insert(
        "requestId".to_owned(),
        Value::String(format!("agent-{}", uuid::Uuid::new_v4())),
    );

    let request = object
        .get_mut("request")
        .and_then(Value::as_object_mut)
        .ok_or(AntigravityRequestError::MissingRequest)?;
    request.remove("safetySettings");
    let session_id = request
        .get("sessionId")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| stable_session_id(request));
    request.insert("sessionId".to_owned(), Value::String(session_id));

    if model.to_ascii_lowercase().contains("claude") {
        let tool_config = request
            .entry("toolConfig".to_owned())
            .or_insert_with(|| serde_json::json!({}));
        if !tool_config.is_object() {
            *tool_config = serde_json::json!({});
        }
        tool_config["functionCallingConfig"]["mode"] = Value::String("VALIDATED".to_owned());
    } else if let Some(generation) = request
        .get_mut("generationConfig")
        .and_then(Value::as_object_mut)
    {
        generation.remove("maxOutputTokens");
    }

    serde_json::to_vec(&root).map_err(|_| AntigravityRequestError::InvalidJson)
}

const DECLARATION_SCHEMA_KEYS: &[&str] = &[
    "parameters",
    "parametersJsonSchema",
    "parameters_json_schema",
    "response",
    "responseJsonSchema",
    "response_json_schema",
];
const GENERATION_SCHEMA_KEYS: &[&str] = &[
    "responseSchema",
    "responseJsonSchema",
    "response_schema",
    "response_json_schema",
];
const UNSUPPORTED_SCHEMA_KEYS: &[&str] = &[
    "minLength",
    "maxLength",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "pattern",
    "minItems",
    "maxItems",
    "uniqueItems",
    "format",
    "default",
    "examples",
    "$schema",
    "$defs",
    "definitions",
    "const",
    "$ref",
    "$id",
    "additionalProperties",
    "propertyNames",
    "patternProperties",
    "$comment",
    "enumDescriptions",
    "enumTitles",
    "prefill",
    "deprecated",
];

fn sanitize_antigravity_request_schemas(root: &mut Value, antigravity_tool_schema: bool) {
    let Some(request) = root.get_mut("request").and_then(Value::as_object_mut) else {
        return;
    };
    if let Some(tools) = request.get_mut("tools").and_then(Value::as_array_mut) {
        for tool in tools {
            let Some(tool) = tool.as_object_mut() else {
                continue;
            };
            for container in ["functionDeclarations", "function_declarations"] {
                let Some(declarations) = tool.get_mut(container).and_then(Value::as_array_mut)
                else {
                    continue;
                };
                for declaration in declarations {
                    let Some(declaration) = declaration.as_object_mut() else {
                        continue;
                    };
                    if let Some(schema) = declaration.remove("parametersJsonSchema") {
                        declaration.insert("parameters".to_owned(), schema);
                    }
                    for key in DECLARATION_SCHEMA_KEYS {
                        if let Some(schema) =
                            declaration.get_mut(*key).filter(|value| value.is_object())
                        {
                            clean_schema(
                                schema,
                                SchemaMode {
                                    flatten_unions: true,
                                    force_enum_string: true,
                                    remove_title: !antigravity_tool_schema,
                                    add_placeholder: antigravity_tool_schema,
                                },
                                true,
                            );
                        }
                    }
                }
            }
        }
    }
    for container in ["generationConfig", "generation_config"] {
        let Some(generation) = request.get_mut(container).and_then(Value::as_object_mut) else {
            continue;
        };
        for key in GENERATION_SCHEMA_KEYS {
            if let Some(schema) = generation.get_mut(*key).filter(|value| value.is_object()) {
                clean_schema(
                    schema,
                    SchemaMode {
                        flatten_unions: false,
                        force_enum_string: false,
                        remove_title: false,
                        add_placeholder: false,
                    },
                    false,
                );
            }
        }
    }
}

#[derive(Clone, Copy)]
struct SchemaMode {
    flatten_unions: bool,
    force_enum_string: bool,
    remove_title: bool,
    add_placeholder: bool,
}

fn clean_schema(schema: &mut Value, mode: SchemaMode, nested_root: bool) {
    let Some(object) = schema.as_object_mut() else {
        return;
    };
    if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
        let name = reference.rsplit('/').next().unwrap_or(reference);
        *schema = serde_json::json!({"type":"object","description":format!("See: {name}")});
        return;
    }
    if mode.flatten_unions {
        for union_key in ["anyOf", "oneOf"] {
            let union = object.get(union_key).and_then(Value::as_array).cloned();
            if let Some(union) = union.filter(|items| !items.is_empty()) {
                let selected = union
                    .into_iter()
                    .max_by_key(schema_score)
                    .unwrap_or_else(|| serde_json::json!({"type":"string"}));
                let description = object.get("description").cloned();
                *schema = selected;
                if let Some(description) = description {
                    if let Some(selected) = schema.as_object_mut() {
                        selected
                            .entry("description".to_owned())
                            .or_insert(description);
                    }
                }
                clean_schema(schema, mode, nested_root);
                return;
            }
        }
    }
    let object = schema.as_object_mut().expect("schema object retained");
    if let Some(constant) = object.remove("const") {
        object
            .entry("enum".to_owned())
            .or_insert_with(|| Value::Array(vec![constant]));
    }
    if let Some(types) = object.get("type").and_then(Value::as_array) {
        let selected = types
            .iter()
            .filter_map(Value::as_str)
            .find(|kind| *kind != "null")
            .unwrap_or("string")
            .to_owned();
        object.insert("type".to_owned(), Value::String(selected));
    }
    if let Some(values) = object.get_mut("enum").and_then(Value::as_array_mut) {
        for value in values {
            if !value.is_string() {
                let encoded = match &*value {
                    Value::Null => String::new(),
                    Value::Bool(value) => value.to_string(),
                    Value::Number(value) => value.to_string(),
                    other => serde_json::to_string(other).unwrap_or_default(),
                };
                *value = Value::String(encoded);
            }
        }
        if mode.force_enum_string {
            object.insert("type".to_owned(), Value::String("string".to_owned()));
        }
    }
    let mut nullable_properties = Vec::new();
    if let Some(properties) = object.get_mut("properties").and_then(Value::as_object_mut) {
        for (name, property) in properties.iter_mut() {
            if property
                .get("type")
                .and_then(Value::as_array)
                .is_some_and(|types| types.iter().any(|kind| kind.as_str() == Some("null")))
            {
                nullable_properties.push(name.clone());
            }
            clean_schema(property, mode, true);
        }
    }
    if let Some(items) = object.get_mut("items") {
        clean_schema(items, mode, true);
    }
    for key in UNSUPPORTED_SCHEMA_KEYS {
        object.remove(*key);
    }
    object.retain(|key, _| !key.starts_with("x-"));
    if mode.remove_title {
        object.remove("nullable");
        object.remove("title");
    }
    let valid_properties = object
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| {
            properties
                .keys()
                .cloned()
                .collect::<std::collections::HashSet<_>>()
        });
    if let (Some(required), Some(valid)) = (
        object.get_mut("required").and_then(Value::as_array_mut),
        valid_properties,
    ) {
        required.retain(|name| {
            name.as_str().is_some_and(|name| {
                valid.contains(name) && !nullable_properties.iter().any(|nullable| nullable == name)
            })
        });
        if required.is_empty() {
            object.remove("required");
        }
    }
    if mode.add_placeholder && nested_root {
        let has_required = object
            .get("required")
            .and_then(Value::as_array)
            .is_some_and(|required| !required.is_empty());
        let properties = object
            .entry("properties".to_owned())
            .or_insert_with(|| serde_json::json!({}));
        if let Some(properties) = properties.as_object_mut() {
            if properties.is_empty() {
                properties.insert(
                    "reason".to_owned(),
                    serde_json::json!({"type":"string","description":"Brief explanation of why you are calling this tool"}),
                );
                object.insert("required".to_owned(), serde_json::json!(["reason"]));
            } else if !has_required {
                properties
                    .entry("_".to_owned())
                    .or_insert_with(|| serde_json::json!({"type":"boolean"}));
                object.insert("required".to_owned(), serde_json::json!(["_"]));
            }
        }
    }
}

fn schema_score(value: &Value) -> u8 {
    match value.get("type").and_then(Value::as_str) {
        Some("object") => 3,
        Some("array") => 2,
        Some("null") | None => u8::from(value.get("properties").is_some()) * 3,
        Some(_) => 1,
    }
}

fn stable_session_id(request: &serde_json::Map<String, Value>) -> String {
    let text = request
        .get("contents")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|content| content.get("role").and_then(Value::as_str) == Some("user"))
        .and_then(|content| content.pointer("/parts/0/text"))
        .and_then(Value::as_str);
    if let Some(text) = text.filter(|value| !value.is_empty()) {
        let digest = Sha256::digest(text.as_bytes());
        let number = i64::from_be_bytes(digest[..8].try_into().expect("SHA-256 prefix")) & i64::MAX;
        return format!("-{number}");
    }
    let mut random = [0_u8; 8];
    getrandom::fill(&mut random).expect("operating-system random source");
    let number = i64::from_be_bytes(random) & i64::MAX;
    format!("-{number}")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntigravityRequestError {
    InvalidJson,
    InvalidObject,
    MissingModel,
    InvalidProjectId,
    MissingRequest,
}

#[cfg(feature = "antigravity-http-transport")]
mod native {
    use std::future::Future;
    use std::pin::Pin;
    use std::time::Duration;

    use futures_util::StreamExt;
    use tokio::sync::mpsc;
    use wreq::header::{ACCEPT, AUTHORIZATION, CONNECTION, CONTENT_TYPE, RETRY_AFTER, USER_AGENT};
    use wreq::{Client, Proxy};

    use super::super::antigravity_executor::{
        AntigravityGenerateRequest, AntigravityGenerateResponse, AntigravityGenerateStreamResponse,
        AntigravityGenerateStreamingTransport, AntigravityGenerateTransport,
        AntigravityGenerateTransportFailure, ANTIGRAVITY_USER_AGENT,
    };

    const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

    #[derive(Clone)]
    pub struct AntigravityGenerateHttpTransport {
        client: Client,
    }

    impl AntigravityGenerateHttpTransport {
        pub fn new(proxy_url: Option<&str>) -> Result<Self, AntigravityTransportBuildError> {
            let mut builder = Client::builder()
                .http1_only()
                .connect_timeout(CONNECT_TIMEOUT)
                .retry(wreq::retry::Policy::never())
                .redirect(wreq::redirect::Policy::none());
            match proxy_url.map(str::trim).filter(|value| !value.is_empty()) {
                Some(proxy_url) => {
                    builder = builder.proxy(
                        Proxy::all(proxy_url)
                            .map_err(|_| AntigravityTransportBuildError::InvalidProxy)?,
                    );
                }
                None => builder = builder.no_proxy(),
            }
            Ok(Self {
                client: builder
                    .build()
                    .map_err(|_| AntigravityTransportBuildError::Client)?,
            })
        }
    }

    impl AntigravityGenerateTransport for AntigravityGenerateHttpTransport {
        fn execute<'a>(
            &'a self,
            request: &'a AntigravityGenerateRequest,
            timeout: Duration,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            AntigravityGenerateResponse,
                            AntigravityGenerateTransportFailure,
                        >,
                    > + Send
                    + 'a,
            >,
        > {
            Box::pin(async move {
                let response = self
                    .client
                    .post(request.url())
                    .header(CONTENT_TYPE, "application/json")
                    .header(CONNECTION, "close")
                    .header(USER_AGENT, ANTIGRAVITY_USER_AGENT)
                    .header(
                        AUTHORIZATION,
                        format!("Bearer {}", request.access_token().expose_secret()),
                    )
                    .timeout(timeout)
                    .body(request.body().to_vec())
                    .send()
                    .await
                    .map_err(classify_transport_error)?;
                let status = response.status().as_u16();
                let retry_after = response
                    .headers()
                    .get(RETRY_AFTER)
                    .and_then(|value| value.to_str().ok())
                    .map(ToOwned::to_owned);
                let body = response
                    .bytes()
                    .await
                    .map_err(classify_transport_error)?
                    .to_vec();
                Ok(AntigravityGenerateResponse::new(status, retry_after, body))
            })
        }
    }

    impl AntigravityGenerateStreamingTransport for AntigravityGenerateHttpTransport {
        fn execute_stream<'a>(
            &'a self,
            request: &'a AntigravityGenerateRequest,
            timeout: Duration,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            AntigravityGenerateStreamResponse,
                            AntigravityGenerateTransportFailure,
                        >,
                    > + Send
                    + 'a,
            >,
        > {
            Box::pin(async move {
                let response = self
                    .client
                    .post(request.url())
                    .header(CONTENT_TYPE, "application/json")
                    .header(ACCEPT, "text/event-stream")
                    .header(CONNECTION, "close")
                    .header(USER_AGENT, ANTIGRAVITY_USER_AGENT)
                    .header(
                        AUTHORIZATION,
                        format!("Bearer {}", request.access_token().expose_secret()),
                    )
                    .timeout(timeout)
                    .body(request.body().to_vec())
                    .send()
                    .await
                    .map_err(classify_transport_error)?;
                let status = response.status().as_u16();
                let retry_after = response
                    .headers()
                    .get(RETRY_AFTER)
                    .and_then(|value| value.to_str().ok())
                    .map(ToOwned::to_owned);
                let (sender, receiver) = mpsc::channel(8);
                if (200..300).contains(&status) {
                    let mut bytes = response.bytes_stream();
                    tokio::spawn(async move {
                        while let Some(chunk) = bytes.next().await {
                            let chunk = chunk
                                .map(|value| value.to_vec())
                                .map_err(classify_transport_error);
                            let terminal = chunk.is_err();
                            if sender.send(chunk).await.is_err() || terminal {
                                break;
                            }
                        }
                    });
                }
                Ok(AntigravityGenerateStreamResponse::new(
                    status,
                    retry_after,
                    receiver,
                ))
            })
        }
    }

    fn classify_transport_error(error: wreq::Error) -> AntigravityGenerateTransportFailure {
        if error.is_timeout() {
            AntigravityGenerateTransportFailure::Timeout
        } else if error.is_connect() {
            AntigravityGenerateTransportFailure::Connect
        } else {
            AntigravityGenerateTransportFailure::Protocol
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum AntigravityTransportBuildError {
        InvalidProxy,
        Client,
    }
}

#[cfg(feature = "antigravity-http-transport")]
pub use native::{AntigravityGenerateHttpTransport, AntigravityTransportBuildError};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepares_project_request_metadata_and_model_policy() {
        let body = prepare_antigravity_generate_body(
            br#"{"project":"","model":"old","request":{"contents":[{"role":"user","parts":[{"text":"hello"}]}],"safetySettings":[{}],"generationConfig":{"maxOutputTokens":100}}}"#,
            "gemini-3-flash-agent",
            "project-1",
        )
        .unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["project"], "project-1");
        assert_eq!(value["model"], "gemini-3-flash-agent");
        assert_eq!(value["userAgent"], "antigravity");
        assert_eq!(value["requestType"], "agent");
        assert!(value["requestId"].as_str().unwrap().starts_with("agent-"));
        assert_eq!(value["request"]["sessionId"], "-3238736544897475342");
        assert!(value["request"].get("safetySettings").is_none());
        assert!(value["request"]["generationConfig"]
            .get("maxOutputTokens")
            .is_none());
    }

    #[test]
    fn claude_enables_validated_function_calling() {
        let body = prepare_antigravity_generate_body(
            br#"{"request":{"contents":[]}}"#,
            "claude-opus-4-6-thinking",
            "project-1",
        )
        .unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            value["request"]["toolConfig"]["functionCallingConfig"]["mode"],
            "VALIDATED"
        );
    }

    #[test]
    fn schema_cleaning_never_mutates_function_call_history() {
        let payload = serde_json::json!({"request":{
            "contents":[{"role":"model","parts":[{"functionCall":{"name":"write","args":{
                "title":"kept","format":"markdown","default":"x","pattern":"p","const":"c",
                "deprecated":false,"examples":"e","additionalProperties":"ap","x-custom":"keep"
            }}}]}],
            "tools":[{"functionDeclarations":[{"name":"write","parametersJsonSchema":{
                "type":"object","required":["title"],"$id":"drop",
                "properties":{"title":{"type":"string","minLength":3,"title":"schema-title"}}
            }}]}]
        }});
        let body = prepare_antigravity_generate_body(
            &serde_json::to_vec(&payload).unwrap(),
            "gemini-3-flash-agent",
            "project-1",
        )
        .unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        let args = &value["request"]["contents"][0]["parts"][0]["functionCall"]["args"];
        for key in [
            "title",
            "format",
            "default",
            "pattern",
            "const",
            "deprecated",
            "examples",
            "additionalProperties",
            "x-custom",
        ] {
            assert!(
                args.get(key).is_some(),
                "history argument {key} was removed"
            );
        }
        let declaration = &value["request"]["tools"][0]["functionDeclarations"][0];
        assert!(declaration.get("parametersJsonSchema").is_none());
        let schema = &declaration["parameters"];
        assert!(schema.get("$id").is_none());
        assert!(schema["properties"]["title"].get("minLength").is_none());
        assert!(schema["properties"]["title"].get("title").is_none());
        assert!(schema["properties"].get("title").is_some());
        assert_eq!(schema["required"], serde_json::json!(["title"]));
    }

    #[test]
    fn claude_tool_placeholders_do_not_leak_into_response_schemas() {
        let payload = serde_json::json!({"request":{
            "tools":[{"functionDeclarations":[{"name":"tool","parameters":{
                "type":"object","properties":{"optional":{"type":"string"}}
            }}]}],
            "generationConfig":{"responseSchema":{"type":"object","properties":{
                "empty":{"type":"object"},
                "action":{"anyOf":[{"type":"object","properties":{"name":{"type":"string"}}},{"type":"null"}]},
                "conviction":{"type":"number","enum":[0.25,0.5,1]}
            }}}
        }});
        let body = prepare_antigravity_generate_body(
            &serde_json::to_vec(&payload).unwrap(),
            "claude-opus-4-6-thinking",
            "project-1",
        )
        .unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        let tool = &value["request"]["tools"][0]["functionDeclarations"][0]["parameters"];
        assert_eq!(tool["required"], serde_json::json!(["_"]));
        assert_eq!(tool["properties"]["_"]["type"], "boolean");
        let response = &value["request"]["generationConfig"]["responseSchema"];
        assert!(response["properties"]["empty"].get("required").is_none());
        assert_eq!(
            response["properties"]["action"]["anyOf"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(response["properties"]["conviction"]["type"], "number");
        assert!(response["properties"]["conviction"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .all(Value::is_string));
    }

    #[cfg(feature = "antigravity-http-transport")]
    #[tokio::test]
    async fn native_stream_bootstraps_before_upstream_tail() {
        use std::time::Duration;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;
        use tokio::sync::oneshot;

        use crate::internal::auth::antigravity::SecretString;
        use crate::internal::runtime::executor::antigravity_executor::{
            AntigravityGenerateRequest, AntigravityGenerateStreamingTransport,
            AntigravityResponsesStream, AntigravityUpstreamTarget,
        };

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (tail_sender, tail_receiver) = oneshot::channel::<()>();
        let (request_sender, request_receiver) = oneshot::channel::<String>();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let count = socket.read(&mut buffer).await.unwrap();
                bytes.extend_from_slice(&buffer[..count]);
                let Some(header_end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") else {
                    continue;
                };
                let head = String::from_utf8_lossy(&bytes[..header_end + 4]);
                let length = head
                    .lines()
                    .find_map(|line| {
                        line.strip_prefix("content-length: ")
                            .or_else(|| line.strip_prefix("Content-Length: "))
                            .and_then(|value| value.trim().parse::<usize>().ok())
                    })
                    .unwrap_or(0);
                if bytes.len() >= header_end + 4 + length {
                    break;
                }
            }
            request_sender
                .send(String::from_utf8_lossy(&bytes).into_owned())
                .unwrap();
            socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n").await.unwrap();
            let first = b"data: {\"response\":{\"responseId\":\"native-stream\",\"createTime\":\"2026-08-03T12:34:56Z\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"first\"}]}}]}}\n\n";
            socket
                .write_all(format!("{:X}\r\n", first.len()).as_bytes())
                .await
                .unwrap();
            socket.write_all(first).await.unwrap();
            socket.write_all(b"\r\n").await.unwrap();
            socket.flush().await.unwrap();
            tail_receiver.await.unwrap();
            let tail = b"data: {\"response\":{\"candidates\":[{\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":1,\"candidatesTokenCount\":1,\"totalTokenCount\":2}}}\n\n";
            socket
                .write_all(format!("{:X}\r\n", tail.len()).as_bytes())
                .await
                .unwrap();
            socket.write_all(tail).await.unwrap();
            socket.write_all(b"\r\n0\r\n\r\n").await.unwrap();
        });

        let target = AntigravityUpstreamTarget::new(format!("http://{address}")).unwrap();
        let request = AntigravityGenerateRequest::new_stream(
            &target,
            SecretString::new("access-stream").unwrap(),
            br#"{"request":{"contents":[]}}"#.to_vec(),
        );
        let transport = AntigravityGenerateHttpTransport::new(None).unwrap();
        let upstream = transport
            .execute_stream(&request, Duration::from_secs(3))
            .await
            .unwrap();
        let mut stream = AntigravityResponsesStream::new(
            upstream,
            br#"{"request":{"model":"gemini-3-flash-agent"}}"#.to_vec(),
            request.body().to_vec(),
        );
        tokio::time::timeout(Duration::from_secs(1), stream.bootstrap())
            .await
            .unwrap()
            .unwrap();
        let first_event = stream.next_event().await.unwrap().unwrap();
        assert!(String::from_utf8_lossy(&first_event).contains("response.created"));
        let captured = request_receiver.await.unwrap();
        assert!(captured.starts_with("POST /v1internal:streamGenerateContent HTTP/1.1"));
        assert!(captured
            .to_ascii_lowercase()
            .contains("accept: text/event-stream"));
        tail_sender.send(()).unwrap();
        let mut completed = false;
        while let Some(event) = stream.next_event().await {
            if String::from_utf8_lossy(&event.unwrap()).contains("response.completed") {
                completed = true;
            }
        }
        assert!(completed);
    }
}
