//! OpenAI Chat Completions adapter.
//!
//! Also handles `azure-openai` and `openai-compatible` — they share the
//! OpenAI ABI and only differ in URL pattern / auth header. The
//! [`OpenAiAdapter::new_*`] constructors set the right knobs.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};

use mcpg_backend_llm_shared::normalized::{
    AudioFormat, AudioSource, ContentPart, FileSource, FinishReason, ImageDetail, ImageSource,
    Message, MessageContent, NormalizedChatRequest, NormalizedChatResponse, Role, TokenUsage,
    ToolCall, ToolChoiceWire,
};
use mcpg_backend_llm_shared::{
    ChatProviderAdapter, NormalizedStreamEvent, ProviderError, StreamEventReceiver,
};

/// Provider variants that share the OpenAI ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiVariant {
    /// `Authorization: Bearer <key>` + path `/chat/completions`.
    OpenAi,
    /// `api-key: <key>` + path is operator-supplied (Azure includes the
    /// deployment + api-version in the base_url).
    Azure,
    /// `Authorization: Bearer <key>` + path `/chat/completions`,
    /// operator-supplied base URL. Auth header optional (some
    /// self-hosted endpoints accept any/none).
    Compatible,
}

pub struct OpenAiAdapter {
    client: Client,
    base_url: String,
    api_key: Arc<str>,
    variant: OpenAiVariant,
    /// Static label for metrics — distinct from `variant` so observers
    /// see `azure-openai` vs `openai-compatible` separately.
    label: &'static str,
}

impl OpenAiAdapter {
    pub fn new(
        variant: OpenAiVariant,
        label: &'static str,
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        connect_timeout: Duration,
    ) -> Result<Self, ProviderError> {
        let client = Client::builder()
            .user_agent("mcpg-plugin-backend-llm-openai/1.0")
            .connect_timeout(connect_timeout)
            // The per-call timeout is applied on the request itself
            // (so the engine can use a shorter window for retries).
            .build()
            .map_err(|e| ProviderError::Network {
                message: format!("build http client: {e}"),
            })?;
        let base_url = base_url.into();
        if base_url.is_empty() {
            return Err(ProviderError::BadRequest {
                message: "base_url is empty".into(),
            });
        }
        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_owned(),
            api_key: Arc::from(api_key.into()),
            variant,
            label,
        })
    }

    /// URL for `chat/completions`. For OpenAI / openai-compatible the
    /// path is appended; for Azure the operator's `base_url` already
    /// includes the deployment + `api-version=…`, so we don't add a
    /// path.
    fn endpoint_url(&self) -> String {
        match self.variant {
            OpenAiVariant::Azure => self.base_url.clone(),
            _ => format!("{}/chat/completions", self.base_url),
        }
    }

    fn build_headers(&self) -> Result<HeaderMap, ProviderError> {
        let mut h = HeaderMap::new();
        h.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let key = self.api_key.as_ref();
        match self.variant {
            OpenAiVariant::Azure => {
                let v = HeaderValue::from_str(key).map_err(|_| ProviderError::BadRequest {
                    message: "api_key contains characters not allowed in HTTP headers".into(),
                })?;
                h.insert(HeaderName::from_static("api-key"), v);
            }
            OpenAiVariant::OpenAi | OpenAiVariant::Compatible => {
                if !key.is_empty() {
                    let v = HeaderValue::from_str(&format!("Bearer {key}")).map_err(|_| {
                        ProviderError::BadRequest {
                            message: "api_key contains characters not allowed in HTTP headers"
                                .into(),
                        }
                    })?;
                    h.insert(AUTHORIZATION, v);
                }
            }
        }
        Ok(h)
    }
}

#[async_trait]
impl ChatProviderAdapter for OpenAiAdapter {
    fn label(&self) -> &'static str {
        self.label
    }

    async fn chat_completion(
        &self,
        request: &NormalizedChatRequest,
        timeout: Duration,
    ) -> Result<NormalizedChatResponse, ProviderError> {
        let body = encode_request(request);
        let headers = self.build_headers()?;
        let url = self.endpoint_url();

        let resp = self
            .client
            .post(&url)
            .headers(headers)
            .timeout(timeout)
            .json(&body)
            .send()
            .await
            .map_err(map_reqwest_error)?;

        let status = resp.status();
        let bytes = resp.bytes().await.map_err(|e| ProviderError::Network {
            message: format!("read response body: {e}"),
        })?;

        if !status.is_success() {
            return Err(map_status_error(status, &bytes));
        }

        let value: Value =
            serde_json::from_slice(&bytes).map_err(|e| ProviderError::Malformed {
                message: format!("response is not JSON: {e}"),
            })?;

        decode_response(&value)
    }

    async fn stream_chat_completion(
        &self,
        request: &NormalizedChatRequest,
        timeout: Duration,
    ) -> Result<StreamEventReceiver, ProviderError> {
        let mut body = encode_request(request);
        // Enable SSE streaming + ask for final usage in the same stream
        // (`stream_options.include_usage`). Without the latter, OpenAI
        // omits usage from the streamed body — surfacing it requires
        // a second non-streaming call.
        if let Value::Object(obj) = &mut body {
            obj.insert("stream".into(), Value::Bool(true));
            obj.insert("stream_options".into(), json!({"include_usage": true}));
        }
        let headers = self.build_headers()?;
        let url = self.endpoint_url();

        let resp = self
            .client
            .post(&url)
            .headers(headers)
            .timeout(timeout)
            .json(&body)
            .send()
            .await
            .map_err(map_reqwest_error)?;

        let status = resp.status();
        if !status.is_success() {
            // Drain the body so the error message includes the upstream
            // detail (some 4xx responses describe what's wrong).
            let bytes = resp.bytes().await.unwrap_or_default();
            return Err(map_status_error(status, &bytes));
        }

        let (tx, rx) =
            tokio::sync::mpsc::channel::<Result<NormalizedStreamEvent, ProviderError>>(32);
        let mut byte_stream = resp.bytes_stream();

        tokio::spawn(async move {
            let mut buffer: Vec<u8> = Vec::new();
            // tool_calls accumulate across multiple delta chunks. Map
            // index → (id, name, args_text). When a new index appears
            // OR finish_reason arrives, we flush the previous one(s)
            // as `ToolCallReady`.
            let mut current_tool_calls: std::collections::BTreeMap<
                u64,
                (Option<String>, String, String),
            > = std::collections::BTreeMap::new();
            let mut final_finish_reason: Option<FinishReason> = None;
            let mut final_usage: TokenUsage = TokenUsage::default();
            let mut stream_error: Option<ProviderError> = None;

            'outer: while let Some(chunk_res) = byte_stream.next().await {
                let chunk = match chunk_res {
                    Ok(c) => c,
                    Err(e) => {
                        stream_error = Some(ProviderError::Network {
                            message: format!("read sse chunk: {e}"),
                        });
                        break 'outer;
                    }
                };
                buffer.extend_from_slice(&chunk);

                // Process complete SSE events (`\n\n` separated).
                while let Some(boundary) = find_event_boundary(&buffer) {
                    let event_bytes = buffer.drain(..boundary).collect::<Vec<u8>>();
                    // `find_event_boundary` returns the byte before the
                    // separator; advance past the boundary bytes (\n\n or
                    // \r\n\r\n) so the next iteration parses the next event.
                    let _ = strip_boundary_prefix(&mut buffer);

                    let event_text = match std::str::from_utf8(&event_bytes) {
                        Ok(s) => s,
                        Err(_) => continue, // ignore non-utf8 (shouldn't happen)
                    };

                    // SSE event may have multiple `data:` lines.
                    let data_lines: Vec<&str> = event_text
                        .lines()
                        .filter_map(|line| line.strip_prefix("data:"))
                        .map(|s| s.trim_start_matches(' '))
                        .collect();
                    if data_lines.is_empty() {
                        continue;
                    }
                    let data_payload = data_lines.join("\n");
                    if data_payload.trim() == "[DONE]" {
                        break 'outer;
                    }
                    let event: Value = match serde_json::from_str(&data_payload) {
                        Ok(v) => v,
                        Err(_) => continue, // skip malformed lines
                    };

                    // Extract usage if present (OpenAI emits a final
                    // chunk with `choices: []` and `usage` object when
                    // stream_options.include_usage is set).
                    if let Some(u) = event.get("usage") {
                        final_usage = decode_usage(Some(u));
                    }

                    let Some(choice) = event
                        .get("choices")
                        .and_then(|c| c.as_array())
                        .and_then(|a| a.first())
                    else {
                        continue;
                    };

                    if let Some(delta) = choice.get("delta") {
                        // Text content delta.
                        if let Some(content) = delta.get("content").and_then(|v| v.as_str())
                            && !content.is_empty()
                            && tx
                                .send(Ok(NormalizedStreamEvent::TextDelta(content.to_owned())))
                                .await
                                .is_err()
                        {
                            // Receiver dropped; stop.
                            return;
                        }

                        // tool_calls delta — array, each element has an
                        // `index`. The first delta for a given index
                        // includes `id`, `function.name`; subsequent
                        // deltas append to `function.arguments`.
                        if let Some(arr) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                            for tc_delta in arr {
                                let Some(idx) = tc_delta.get("index").and_then(|v| v.as_u64())
                                else {
                                    continue;
                                };
                                let entry = current_tool_calls.entry(idx).or_insert((
                                    None,
                                    String::new(),
                                    String::new(),
                                ));
                                if let Some(id) = tc_delta.get("id").and_then(|v| v.as_str()) {
                                    entry.0 = Some(id.to_owned());
                                }
                                if let Some(func) = tc_delta.get("function") {
                                    if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                                        entry.1 = name.to_owned();
                                    }
                                    if let Some(args) =
                                        func.get("arguments").and_then(|v| v.as_str())
                                    {
                                        entry.2.push_str(args);
                                    }
                                }
                            }
                        }
                    }

                    if let Some(reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                        final_finish_reason = Some(match reason {
                            "stop" => FinishReason::Stop,
                            "tool_calls" => FinishReason::ToolCalls,
                            "length" => FinishReason::Length,
                            "content_filter" => FinishReason::ContentFilter,
                            _ => FinishReason::Other,
                        });
                    }
                }
            }

            // Flush any accumulated tool_calls as ToolCallReady events.
            for (_idx, (id_opt, name, args_str)) in current_tool_calls {
                if name.is_empty() {
                    continue;
                }
                let arguments: Value = serde_json::from_str(&args_str).unwrap_or(Value::Null);
                let id = id_opt.unwrap_or_default();
                if tx
                    .send(Ok(NormalizedStreamEvent::ToolCallReady(ToolCall {
                        id,
                        name,
                        arguments,
                    })))
                    .await
                    .is_err()
                {
                    return;
                }
            }

            if let Some(err) = stream_error {
                let _ = tx.send(Err(err)).await;
                return;
            }

            let reason = final_finish_reason.unwrap_or(FinishReason::Other);
            let _ = tx
                .send(Ok(NormalizedStreamEvent::Finish {
                    reason,
                    usage: final_usage,
                }))
                .await;
        });

        Ok(rx)
    }
}

/// Find the end-of-event boundary (`\n\n` or `\r\n\r\n`) in `buf`.
/// Returns the index where the event payload ends (boundary excluded).
fn find_event_boundary(buf: &[u8]) -> Option<usize> {
    if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
        return Some(pos);
    }
    buf.windows(2).position(|w| w == b"\n\n")
}

/// After draining up to the boundary index, advance past the
/// boundary bytes themselves (2 or 4 bytes).
fn strip_boundary_prefix(buf: &mut Vec<u8>) -> usize {
    if buf.starts_with(b"\r\n\r\n") {
        buf.drain(..4);
        4
    } else if buf.starts_with(b"\n\n") {
        buf.drain(..2);
        2
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Request encoding
// ---------------------------------------------------------------------------

fn encode_request(req: &NormalizedChatRequest) -> Value {
    let mut body = serde_json::Map::new();
    body.insert("model".into(), Value::String(req.model.clone()));
    body.insert("messages".into(), encode_messages(&req.messages));

    if let Some(schema) = &req.response_schema {
        body.insert(
            "response_format".into(),
            json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": req.strict_response,
                    "schema": schema,
                }
            }),
        );
    }

    if !req.tools.is_empty() {
        let tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                })
            })
            .collect();
        body.insert("tools".into(), Value::Array(tools));
        body.insert(
            "tool_choice".into(),
            match req.tool_choice {
                ToolChoiceWire::Auto => Value::String("auto".into()),
                ToolChoiceWire::Required => Value::String("required".into()),
                ToolChoiceWire::None => Value::String("none".into()),
            },
        );
    }

    if let Some(t) = req.temperature {
        body.insert("temperature".into(), json!(t));
    }
    if let Some(t) = req.top_p {
        body.insert("top_p".into(), json!(t));
    }
    if let Some(n) = req.max_completion_tokens {
        body.insert("max_completion_tokens".into(), json!(n));
    }
    if let Some(s) = req.seed {
        body.insert("seed".into(), json!(s));
    }

    Value::Object(body)
}

fn encode_messages(messages: &[Message]) -> Value {
    let arr: Vec<Value> = messages
        .iter()
        .map(|m| match m.role {
            // System prompts are text-only by spec; flatten any
            // accidentally-Parts variant via `as_text` to be defensive.
            Role::System => json!({ "role": "system", "content": m.content.as_text() }),
            // User messages may carry multimodal `Parts`; encode them
            // as OpenAI content-part array, fall through to plain
            // string for the common text case.
            Role::User => match &m.content {
                MessageContent::Text(s) => json!({ "role": "user", "content": s }),
                MessageContent::Parts(parts) => {
                    json!({ "role": "user", "content": encode_user_parts(parts) })
                }
            },
            Role::Assistant => {
                let mut obj = serde_json::Map::new();
                obj.insert("role".into(), Value::String("assistant".into()));
                let text = m.content.as_text();
                if !text.is_empty() {
                    obj.insert("content".into(), Value::String(text));
                } else {
                    // Some providers reject `content: ""` on assistant
                    // messages that emit only tool_calls; null is the
                    // OpenAI-canonical placeholder.
                    obj.insert("content".into(), Value::Null);
                }
                if !m.tool_calls.is_empty() {
                    let calls: Vec<Value> = m
                        .tool_calls
                        .iter()
                        .map(|c| {
                            json!({
                                "id": c.id,
                                "type": "function",
                                "function": {
                                    "name": c.name,
                                    "arguments": c.arguments.to_string(),
                                }
                            })
                        })
                        .collect();
                    obj.insert("tool_calls".into(), Value::Array(calls));
                }
                Value::Object(obj)
            }
            Role::Tool => json!({
                "role": "tool",
                "tool_call_id": m.tool_call_id.clone().unwrap_or_default(),
                "content": m.content.as_text(),
            }),
        })
        .collect();
    Value::Array(arr)
}

/// Encode a [`MessageContent::Parts`] user message as OpenAI's
/// content-part array. OpenAI accepts:
///
/// - `{type: "text", text: "..."}` for prose.
/// - `{type: "image_url", image_url: {url, detail}}` where `url` is
///   a public HTTP(S) URL or a `data:<mime>;base64,...` data-URL.
/// - `{type: "input_audio", input_audio: {data, format}}` for audio
///   inputs (requires the audio-capable model variants).
/// - `{type: "file", file: {file_data: "data:<mime>;base64,..."}}`
///   for documents — currently only the assistants API surface
///   ships this; chat-completions fails for non-image files. We
///   still emit the structure so operators can opt in once OpenAI
///   ships file inputs on chat.
///
/// `mcpg-resource://` sources are unexpected here — the engine's
/// pre-encode resolver should have replaced them with `Base64`
/// already. If one slips through (resolution failure), we encode it
/// as text so the model gets the unresolved URI rather than a
/// silent encoding failure.
fn encode_user_parts(parts: &[ContentPart]) -> Value {
    let mut out: Vec<Value> = Vec::with_capacity(parts.len());
    for p in parts {
        match p {
            ContentPart::Text(s) => {
                out.push(json!({"type": "text", "text": s}));
            }
            ContentPart::Image(img) => match &img.source {
                ImageSource::Url(u) => {
                    let mut iu = serde_json::Map::new();
                    iu.insert("url".into(), Value::String(u.clone()));
                    if let Some(d) = img.detail.as_ref() {
                        iu.insert(
                            "detail".into(),
                            Value::String(image_detail_label(*d).into()),
                        );
                    }
                    out.push(json!({"type": "image_url", "image_url": Value::Object(iu)}));
                }
                ImageSource::Base64 { mime_type, data } => {
                    let url = format!("data:{mime_type};base64,{data}");
                    let mut iu = serde_json::Map::new();
                    iu.insert("url".into(), Value::String(url));
                    if let Some(d) = img.detail.as_ref() {
                        iu.insert(
                            "detail".into(),
                            Value::String(image_detail_label(*d).into()),
                        );
                    }
                    out.push(json!({"type": "image_url", "image_url": Value::Object(iu)}));
                }
                ImageSource::McpResource(uri) => {
                    out.push(json!({
                        "type": "text",
                        "text": format!("[unresolved image resource: {uri}]"),
                    }));
                }
            },
            ContentPart::Audio(au) => match &au.source {
                AudioSource::Url(u) => {
                    out.push(json!({
                        "type": "text",
                        "text": format!("[audio url: {u}]"),
                    }));
                }
                AudioSource::Base64 { data } => {
                    out.push(json!({
                        "type": "input_audio",
                        "input_audio": {
                            "data": data,
                            "format": audio_format_label(au.format),
                        }
                    }));
                }
                AudioSource::McpResource(uri) => {
                    out.push(json!({
                        "type": "text",
                        "text": format!("[unresolved audio resource: {uri}]"),
                    }));
                }
            },
            ContentPart::File(f) => {
                let data_url = match &f.source {
                    FileSource::Base64 { data } => {
                        Some(format!("data:{};base64,{data}", f.mime_type))
                    }
                    FileSource::Url(u) => Some(u.clone()),
                    FileSource::McpResource(_) => None,
                };
                if let Some(url) = data_url {
                    let mut file = serde_json::Map::new();
                    file.insert("file_data".into(), Value::String(url));
                    if let Some(name) = f.filename.as_ref() {
                        file.insert("filename".into(), Value::String(name.clone()));
                    }
                    out.push(json!({"type": "file", "file": Value::Object(file)}));
                } else {
                    out.push(json!({
                        "type": "text",
                        "text": "[unresolved file resource]",
                    }));
                }
            }
        }
    }
    Value::Array(out)
}

fn image_detail_label(d: ImageDetail) -> &'static str {
    match d {
        ImageDetail::Auto => "auto",
        ImageDetail::High => "high",
        ImageDetail::Low => "low",
    }
}

fn audio_format_label(f: AudioFormat) -> &'static str {
    match f {
        AudioFormat::Mp3 => "mp3",
        AudioFormat::Wav => "wav",
        AudioFormat::Flac => "flac",
        AudioFormat::Ogg => "ogg",
        AudioFormat::Aac => "aac",
        AudioFormat::Pcm => "pcm",
    }
}

// ---------------------------------------------------------------------------
// Response decoding
// ---------------------------------------------------------------------------

fn decode_response(value: &Value) -> Result<NormalizedChatResponse, ProviderError> {
    let choice = value
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| ProviderError::Malformed {
            message: "response has no choices[0]".into(),
        })?;

    let message = choice
        .get("message")
        .ok_or_else(|| ProviderError::Malformed {
            message: "choices[0].message missing".into(),
        })?;

    let content = message
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or_default()
        .to_owned();

    let mut tool_calls = Vec::new();
    if let Some(arr) = message.get("tool_calls").and_then(|c| c.as_array()) {
        for tc in arr {
            let id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_owned();
            let func = tc.get("function").ok_or_else(|| ProviderError::Malformed {
                message: "tool_call without function".into(),
            })?;
            let name = func
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_owned();
            let raw_args = func
                .get("arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("{}");
            // Arguments arrive as a JSON-encoded string. Decode here so
            // the engine sees a real Value.
            let arguments: Value =
                serde_json::from_str(raw_args).map_err(|e| ProviderError::Malformed {
                    message: format!("tool_call '{name}' has malformed arguments: {e}"),
                })?;
            tool_calls.push(ToolCall {
                id,
                name,
                arguments,
            });
        }
    }

    let finish_reason = match choice.get("finish_reason").and_then(|v| v.as_str()) {
        Some("stop") => FinishReason::Stop,
        Some("tool_calls") => FinishReason::ToolCalls,
        Some("length") => FinishReason::Length,
        Some("content_filter") => FinishReason::ContentFilter,
        _ => FinishReason::Other,
    };

    let usage = decode_usage(value.get("usage"));

    Ok(NormalizedChatResponse {
        content,
        tool_calls,
        finish_reason,
        usage,
    })
}

fn decode_usage(value: Option<&Value>) -> TokenUsage {
    let Some(u) = value else {
        return TokenUsage::default();
    };
    let input = u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let output = u
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let cached = u
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    TokenUsage {
        input_tokens: input,
        output_tokens: output,
        cached_input_tokens: cached,
    }
}

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

fn map_status_error(status: reqwest::StatusCode, body: &[u8]) -> ProviderError {
    let message = body_excerpt(body);
    let code = status.as_u16();
    if code == 429 {
        return ProviderError::RateLimited { message };
    }
    if code == 401 || code == 403 {
        return ProviderError::AuthFailed { message };
    }
    if code == 413 {
        return ProviderError::ContextLimit { message };
    }
    if code == 400 {
        // Heuristic: OpenAI returns 400 with `code: "context_length_exceeded"`
        // for token-window overflow.
        if message.contains("context_length")
            || message.contains("maximum context")
            || message.contains("token limit")
        {
            return ProviderError::ContextLimit { message };
        }
        return ProviderError::BadRequest { message };
    }
    if (500..600).contains(&code) {
        return ProviderError::Server { message };
    }
    ProviderError::Server { message }
}

fn map_reqwest_error(err: reqwest::Error) -> ProviderError {
    if err.is_timeout() {
        return ProviderError::Network {
            message: format!("timeout: {err}"),
        };
    }
    if err.is_connect() {
        return ProviderError::Network {
            message: format!("connect failed: {err}"),
        };
    }
    if err.is_request() || err.is_body() || err.is_decode() {
        return ProviderError::Network {
            message: format!("transport: {err}"),
        };
    }
    ProviderError::Network {
        message: err.to_string(),
    }
}

/// Trim the body to a bounded preview suitable for error messages.
/// We never log the full body — it may include the request payload
/// (and thus user input).
fn body_excerpt(body: &[u8]) -> String {
    const MAX: usize = 512;
    let s = String::from_utf8_lossy(body);
    if s.len() <= MAX {
        s.into_owned()
    } else {
        format!("{}…[truncated]", &s[..MAX])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcpg_backend_llm_shared::normalized::{Message, ToolDef};

    #[test]
    fn encode_request_minimal_text() {
        let req = NormalizedChatRequest {
            model: "gpt-4o-mini".into(),
            messages: vec![Message::system("you are helpful"), Message::user("hi")],
            response_schema: None,
            strict_response: true,
            tools: vec![],
            tool_choice: ToolChoiceWire::Auto,
            temperature: Some(0.5),
            top_p: None,
            max_completion_tokens: Some(64),
            seed: None,
        };
        let body = encode_request(&req);
        assert_eq!(body["model"], json!("gpt-4o-mini"));
        assert_eq!(body["messages"][0]["role"], json!("system"));
        assert_eq!(body["messages"][1]["role"], json!("user"));
        assert_eq!(body["temperature"], json!(0.5));
        assert_eq!(body["max_completion_tokens"], json!(64));
        // No tools / no response_format / no top_p / no seed.
        assert!(body.get("tools").is_none());
        assert!(body.get("response_format").is_none());
        assert!(body.get("top_p").is_none());
        assert!(body.get("seed").is_none());
    }

    #[test]
    fn encode_request_with_response_schema() {
        let schema = json!({"type": "object", "properties": {"x": {"type": "string"}}});
        let req = NormalizedChatRequest {
            model: "gpt-4o-mini".into(),
            messages: vec![Message::user("hi")],
            response_schema: Some(schema.clone()),
            strict_response: true,
            tools: vec![],
            tool_choice: ToolChoiceWire::Auto,
            temperature: None,
            top_p: None,
            max_completion_tokens: None,
            seed: None,
        };
        let body = encode_request(&req);
        assert_eq!(body["response_format"]["type"], json!("json_schema"));
        assert_eq!(
            body["response_format"]["json_schema"]["strict"],
            json!(true)
        );
        assert_eq!(body["response_format"]["json_schema"]["schema"], schema);
    }

    #[test]
    fn encode_request_with_tools_emits_function_defs() {
        let req = NormalizedChatRequest {
            model: "x".into(),
            messages: vec![Message::user("y")],
            response_schema: None,
            strict_response: false,
            tools: vec![ToolDef {
                name: "linear.fetch_issue".into(),
                description: "Fetch a Linear issue".into(),
                parameters: json!({"type": "object"}),
            }],
            tool_choice: ToolChoiceWire::Required,
            temperature: None,
            top_p: None,
            max_completion_tokens: None,
            seed: None,
        };
        let body = encode_request(&req);
        assert_eq!(body["tools"][0]["type"], json!("function"));
        assert_eq!(
            body["tools"][0]["function"]["name"],
            json!("linear.fetch_issue")
        );
        assert_eq!(body["tool_choice"], json!("required"));
    }

    #[test]
    fn assistant_message_with_only_tool_calls_serializes_null_content() {
        let msg = Message::assistant_tool_calls(vec![ToolCall {
            id: "call_1".into(),
            name: "x".into(),
            arguments: json!({"a": 1}),
        }]);
        let v = encode_messages(&[msg]);
        assert_eq!(v[0]["role"], json!("assistant"));
        assert_eq!(v[0]["content"], Value::Null);
        // arguments serialize to a JSON-encoded string (OpenAI quirk).
        assert_eq!(
            v[0]["tool_calls"][0]["function"]["arguments"],
            json!("{\"a\":1}")
        );
    }

    #[test]
    fn decode_response_text_only() {
        let raw = json!({
            "choices": [{
                "message": { "role": "assistant", "content": "hello back" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 3 }
        });
        let r = decode_response(&raw).unwrap();
        assert_eq!(r.content, "hello back");
        assert!(r.tool_calls.is_empty());
        assert_eq!(r.finish_reason, FinishReason::Stop);
        assert_eq!(r.usage.input_tokens, 10);
        assert_eq!(r.usage.output_tokens, 3);
    }

    #[test]
    fn decode_response_with_tool_calls() {
        let raw = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "fetch",
                            "arguments": "{\"id\":42}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 5, "completion_tokens": 8 }
        });
        let r = decode_response(&raw).unwrap();
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0].id, "call_1");
        assert_eq!(r.tool_calls[0].name, "fetch");
        assert_eq!(r.tool_calls[0].arguments, json!({"id": 42}));
        assert_eq!(r.finish_reason, FinishReason::ToolCalls);
    }

    #[test]
    fn decode_response_rejects_malformed_tool_args() {
        let raw = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "c",
                        "type": "function",
                        "function": {
                            "name": "f",
                            "arguments": "not json"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let err = decode_response(&raw).unwrap_err();
        assert!(matches!(err, ProviderError::Malformed { .. }));
    }

    #[test]
    fn map_status_429_is_rate_limited() {
        let e = map_status_error(reqwest::StatusCode::from_u16(429).unwrap(), b"slow down");
        assert!(matches!(e, ProviderError::RateLimited { .. }));
    }

    #[test]
    fn map_status_401_is_auth_failed() {
        let e = map_status_error(reqwest::StatusCode::from_u16(401).unwrap(), b"bad key");
        assert!(matches!(e, ProviderError::AuthFailed { .. }));
    }

    #[test]
    fn map_status_400_with_context_marker_is_context_limit() {
        let e = map_status_error(
            reqwest::StatusCode::from_u16(400).unwrap(),
            b"This model's maximum context length is 8192 tokens.",
        );
        assert!(matches!(e, ProviderError::ContextLimit { .. }));
    }

    #[test]
    fn map_status_500_is_server() {
        let e = map_status_error(reqwest::StatusCode::from_u16(500).unwrap(), b"oops");
        assert!(matches!(e, ProviderError::Server { .. }));
    }

    #[test]
    fn body_excerpt_truncates_long_bodies() {
        let body = vec![b'x'; 1000];
        let s = body_excerpt(&body);
        assert!(s.ends_with("[truncated]"));
        assert!(s.len() < body.len());
    }

    #[test]
    fn provider_error_retryability_matches_design() {
        assert!(
            ProviderError::RateLimited {
                message: "x".into()
            }
            .is_retryable()
        );
        assert!(
            ProviderError::Server {
                message: "x".into()
            }
            .is_retryable()
        );
        assert!(
            ProviderError::Network {
                message: "x".into()
            }
            .is_retryable()
        );
        assert!(
            !ProviderError::AuthFailed {
                message: "x".into()
            }
            .is_retryable()
        );
        assert!(
            !ProviderError::BadRequest {
                message: "x".into()
            }
            .is_retryable()
        );
        assert!(
            !ProviderError::ContextLimit {
                message: "x".into()
            }
            .is_retryable()
        );
    }

    // ----- Multimodal user-parts encoding -----

    #[test]
    fn encode_user_image_url_emits_image_url_part() {
        use mcpg_backend_llm_shared::normalized::{
            ContentPart, ImageContent, ImageDetail, ImageSource,
        };
        let parts = vec![
            ContentPart::Text("describe".into()),
            ContentPart::Image(ImageContent {
                source: ImageSource::Url("https://ex.com/a.png".into()),
                detail: Some(ImageDetail::High),
            }),
        ];
        let r = NormalizedChatRequest {
            model: "gpt-4o".into(),
            messages: vec![Message::user_parts(parts)],
            response_schema: None,
            strict_response: false,
            tools: vec![],
            tool_choice: ToolChoiceWire::Auto,
            temperature: None,
            top_p: None,
            max_completion_tokens: None,
            seed: None,
        };
        let body = encode_request(&r);
        let arr = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(arr[0]["type"], "text");
        assert_eq!(arr[1]["type"], "image_url");
        assert_eq!(arr[1]["image_url"]["url"], "https://ex.com/a.png");
        assert_eq!(arr[1]["image_url"]["detail"], "high");
    }

    #[test]
    fn encode_user_image_base64_emits_data_url() {
        use mcpg_backend_llm_shared::normalized::{ContentPart, ImageContent, ImageSource};
        let parts = vec![ContentPart::Image(ImageContent {
            source: ImageSource::Base64 {
                mime_type: "image/png".into(),
                data: "aGVsbG8=".into(),
            },
            detail: None,
        })];
        let r = NormalizedChatRequest {
            model: "gpt-4o".into(),
            messages: vec![Message::user_parts(parts)],
            response_schema: None,
            strict_response: false,
            tools: vec![],
            tool_choice: ToolChoiceWire::Auto,
            temperature: None,
            top_p: None,
            max_completion_tokens: None,
            seed: None,
        };
        let body = encode_request(&r);
        let url = body["messages"][0]["content"][0]["image_url"]["url"]
            .as_str()
            .unwrap();
        assert!(url.starts_with("data:image/png;base64,"));
        assert!(url.ends_with("aGVsbG8="));
    }

    #[test]
    fn encode_user_unresolved_resource_falls_back_to_text() {
        use mcpg_backend_llm_shared::normalized::{ContentPart, ImageContent, ImageSource};
        let parts = vec![ContentPart::Image(ImageContent {
            source: ImageSource::McpResource("mcpg-resource://hash:abc".into()),
            detail: None,
        })];
        let r = NormalizedChatRequest {
            model: "gpt-4o".into(),
            messages: vec![Message::user_parts(parts)],
            response_schema: None,
            strict_response: false,
            tools: vec![],
            tool_choice: ToolChoiceWire::Auto,
            temperature: None,
            top_p: None,
            max_completion_tokens: None,
            seed: None,
        };
        let body = encode_request(&r);
        let arr = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(arr[0]["type"], "text");
        assert!(arr[0]["text"].as_str().unwrap().contains("hash:abc"));
    }
}
