//! `BackendPlugin` impls for OpenAI + Azure OpenAI chat completions.
//!
//! Both delegate to the shared
//! [`mcpg_backend_llm_shared::ChatEngine`]. The only difference is
//! how each constructs the underlying [`crate::OpenAiAdapter`]:
//! OpenAI passes `OpenAiVariant::OpenAi` with the public default
//! base URL; Azure passes `OpenAiVariant::Azure` with the
//! operator-supplied URL.
//!
//! Both plugins store an installed
//! [`HostHandle`] in a `OnceLock` and route per-call observability
//! through the unified host surface: per-execute span at
//! `llm_{openai,azure_openai}.execute`, latency histogram + call
//! counter with bounded `outcome` + `model` labels, and an audit
//! event per upstream call (`dev.mcpg.llm.{openai,azure_openai}.
//! {completion,failure}`) with model + token + cost details when
//! known. Streaming completions emit the triad at stream-end with
//! aggregated token counts. The pre-existing internal `tracing` +
//! `metrics::*` calls remain wired in both modes (coexistence with
//! the triad floor is intentional).

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock, RwLock};

use mcpg_backend_llm_shared::chat_config::ResponseFormatMode;
use mcpg_backend_llm_shared::template::Templates;
use mcpg_backend_llm_shared::{
    ChatEngine, ChatProviderAdapter, ProviderError, build_child_tool_defs, compile_validator,
    resolve_api_key,
};
use mcpg_plugin_protocol::{
    BackendChunk, BackendChunkStream, BackendError, BackendHost, BackendPlugin, BackendRequest,
    BackendResponse, PluginManifest, async_trait, firstparty_manifest, types::PluginIdentity,
};
use mcpg_plugin_sdk::HostHandle;
use serde_json::Value;
use tracing::{Instrument, debug, info_span, warn};

use crate::adapter::{OpenAiAdapter, OpenAiVariant};
use crate::config::{AzureOpenaiChatSpec, OpenAiChatSpec};
use crate::host_handle_obs::{OpenAiKind, UsageSnapshot, emit_chat_observability, open_span};

/// `BackendPlugin` for `kind: "openai.chat"`.
pub struct OpenAiChatPlugin {
    manifest: PluginManifest,
    engines: Arc<RwLock<BTreeMap<String, Arc<ChatEngine>>>>,
    /// Unified host-observability handle.
    /// `OnceLock` because the boot path installs it exactly once via
    /// [`OpenAiChatPlugin::set_host_handle`] after construction.
    /// Test paths that build the plugin without wiring a host leave
    /// the slot empty; `host_handle()` returns `None` and the triad
    /// short-circuits to a no-op (internal `tracing` / `metrics::*`
    /// floor still carries the load).
    host_handle: OnceLock<HostHandle>,
}

impl std::fmt::Debug for OpenAiChatPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiChatPlugin").finish()
    }
}

impl Default for OpenAiChatPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAiChatPlugin {
    pub fn new() -> Self {
        Self {
            manifest: firstparty_manifest! {
                id: "dev.mcpg.backend.openai.chat",
                name: "OpenAI Chat Completions",
                class: Backend,
            },
            engines: Arc::new(RwLock::new(BTreeMap::new())),
            host_handle: OnceLock::new(),
        }
    }

    #[doc(hidden)]
    pub fn registered_profile_count(&self) -> usize {
        self.engines.read().unwrap().len()
    }

    /// Install the unified [`HostHandle`]
    /// surface. The gateway calls this exactly once at boot, after
    /// [`OpenAiChatPlugin::new`] but before any `execute()` traffic
    /// is dispatched.
    ///
    /// Idempotent — a second call returns `false`. The returned
    /// `bool` indicates whether the handle was installed (`true`) or
    /// the slot was already occupied (`false`).
    pub fn set_host_handle(&self, host: HostHandle) -> bool {
        self.host_handle.set(host).is_ok()
    }

    /// Borrow the installed unified host surface, if any. Returns
    /// `None` in test harnesses that constructed the plugin without
    /// calling [`OpenAiChatPlugin::set_host_handle`].
    fn host_handle(&self) -> Option<&HostHandle> {
        self.host_handle.get()
    }
}

#[async_trait]
impl BackendPlugin for OpenAiChatPlugin {
    fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }

    fn kind(&self) -> &str {
        "openai.chat"
    }

    async fn register_profile(
        &self,
        backend_name: &str,
        spec: &Value,
        host: Arc<dyn BackendHost>,
    ) -> Result<(), BackendError> {
        let parsed: OpenAiChatSpec =
            serde_json::from_value(spec.clone()).map_err(|e| BackendError::InvalidSpec {
                message: format!("openai_chat spec: {e}"),
            })?;
        parsed.validate().map_err(|e| BackendError::InvalidSpec {
            message: e.to_string(),
        })?;

        let api_key = resolve_api_key(&parsed.api_key)?;
        let base_url = parsed.resolved_base_url().to_owned();
        let connect_timeout = parsed.chat.connect_timeout();

        let adapter = OpenAiAdapter::new(
            OpenAiVariant::OpenAi,
            "openai",
            base_url,
            api_key,
            connect_timeout,
        )
        .map_err(|e: ProviderError| BackendError::InvalidSpec {
            message: format!("build openai adapter: {e}"),
        })?;
        let adapter: Arc<dyn ChatProviderAdapter> = Arc::new(adapter);

        let templates = Templates::compile(&parsed.chat.prompt.system, &parsed.chat.prompt.user)
            .map_err(|e| BackendError::InvalidSpec {
                message: format!("template: {e}"),
            })?;

        let (validator, raw_output_schema) = if matches!(
            parsed.chat.response_format.mode,
            ResponseFormatMode::JsonSchema
        ) {
            let schema_value = spec.get("output_schema").cloned();
            if let Some(schema) = schema_value {
                let v = compile_validator(&schema).map_err(|e| BackendError::InvalidSpec {
                    message: e.to_string(),
                })?;
                (Some(v), Some(schema))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        let child_tool_defs = build_child_tool_defs(&parsed.chat.tools, |_name| None);

        let engine = ChatEngine {
            backend_name: backend_name.to_owned(),
            adapter,
            templates,
            validator,
            raw_output_schema,
            spec: parsed.chat,
            host,
            child_tool_defs,
            child_tool_validators: Vec::new(),
        };

        self.engines
            .write()
            .map_err(|_| BackendError::InvalidSpec {
                message: "engine map poisoned".into(),
            })?
            .insert(backend_name.to_owned(), Arc::new(engine));

        Ok(())
    }

    async fn execute(
        &self,
        backend_name: &str,
        request: BackendRequest,
    ) -> Result<BackendResponse, BackendError> {
        let engine = lookup_engine(&self.engines, backend_name)?;
        let args = decode_payload(&request.payload)?;
        let model = engine.spec.model.clone();
        let identity = request.identity.clone();
        let request_id = request.request_id.clone();

        // Wrap the engine call in a per-plugin span
        // so traces from this plugin attribute to its plugin id and
        // operators can route via
        // `plugins[].observability.traces`. Adapter +
        // model labels also flow into the existing call metrics.
        let internal_span = info_span!(
            "openai_chat_execute",
            plugin_id = "dev.mcpg.backend.llm.openai",
            binding = %backend_name,
            model = %model,
        );
        // Open the host span BEFORE engine
        // dispatch so the span window covers the full upstream
        // call. Dropped explicitly AFTER the triad emission so
        // span_end lands last.
        let host_span = open_span(self.host_handle(), OpenAiKind::OpenAi, backend_name, &model);

        let started = std::time::Instant::now();
        let result = engine
            .execute(&args, &request.request_id, request.session_id.as_deref())
            .instrument(internal_span)
            .await;
        let elapsed = started.elapsed();

        emit_call_metrics(
            backend_name,
            engine.adapter.label(),
            &model,
            result.is_ok(),
            elapsed,
        );

        match &result {
            Ok(_) => debug!(
                binding = %backend_name,
                model = %model,
                elapsed_ms = %elapsed.as_millis(),
                "openai chat call succeeded"
            ),
            Err(e) => warn!(
                binding = %backend_name,
                model = %model,
                error = %e,
                "openai chat call failed"
            ),
        }

        emit_chat_observability(
            self.host_handle(),
            OpenAiKind::OpenAi,
            backend_name,
            &model,
            &request_id,
            identity.as_ref(),
            elapsed,
            result.as_ref().map(|_| ()),
            None,
        )
        .await;
        drop(host_span);

        let value = result?;
        let payload = serde_json::to_vec(&value).map_err(|e| BackendError::Transport {
            message: format!("serialize response: {e}"),
        })?;
        Ok(BackendResponse {
            payload,
            truncated: false,
        })
    }

    async fn execute_streaming(
        &self,
        backend_name: &str,
        request: BackendRequest,
    ) -> Result<BackendChunkStream, BackendError> {
        let engine = lookup_engine(&self.engines, backend_name)?;
        let args = decode_payload(&request.payload)?;
        wrap_streaming(
            self.host_handle().cloned(),
            OpenAiKind::OpenAi,
            backend_name.to_owned(),
            engine.spec.model.clone(),
            request.identity.clone(),
            request.request_id.clone(),
            engine.execute_streaming(args, request.request_id, request.session_id),
        )
    }
}

/// `BackendPlugin` for `kind: "azure_openai.chat"`. Wire format
/// identical to OpenAI; URL pattern + auth header differ. The
/// underlying adapter is the same `OpenAiAdapter` with
/// `OpenAiVariant::Azure`.
pub struct AzureOpenaiChatPlugin {
    manifest: PluginManifest,
    engines: Arc<RwLock<BTreeMap<String, Arc<ChatEngine>>>>,
    /// See [`OpenAiChatPlugin::host_handle`].
    host_handle: OnceLock<HostHandle>,
}

impl std::fmt::Debug for AzureOpenaiChatPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureOpenaiChatPlugin").finish()
    }
}

impl Default for AzureOpenaiChatPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl AzureOpenaiChatPlugin {
    pub fn new() -> Self {
        Self {
            manifest: firstparty_manifest! {
                id: "dev.mcpg.backend.azure_openai.chat",
                name: "Azure OpenAI Chat Completions",
                class: Backend,
            },
            engines: Arc::new(RwLock::new(BTreeMap::new())),
            host_handle: OnceLock::new(),
        }
    }

    #[doc(hidden)]
    pub fn registered_profile_count(&self) -> usize {
        self.engines.read().unwrap().len()
    }

    /// Install the unified [`HostHandle`]
    /// surface. See [`OpenAiChatPlugin::set_host_handle`].
    pub fn set_host_handle(&self, host: HostHandle) -> bool {
        self.host_handle.set(host).is_ok()
    }

    fn host_handle(&self) -> Option<&HostHandle> {
        self.host_handle.get()
    }
}

#[async_trait]
impl BackendPlugin for AzureOpenaiChatPlugin {
    fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }

    fn kind(&self) -> &str {
        "azure_openai.chat"
    }

    async fn register_profile(
        &self,
        backend_name: &str,
        spec: &Value,
        host: Arc<dyn BackendHost>,
    ) -> Result<(), BackendError> {
        let parsed: AzureOpenaiChatSpec =
            serde_json::from_value(spec.clone()).map_err(|e| BackendError::InvalidSpec {
                message: format!("azure_openai_chat spec: {e}"),
            })?;
        parsed.validate().map_err(|e| BackendError::InvalidSpec {
            message: e.to_string(),
        })?;

        let api_key = resolve_api_key(&parsed.api_key)?;
        let connect_timeout = parsed.chat.connect_timeout();

        let adapter = OpenAiAdapter::new(
            OpenAiVariant::Azure,
            "azure-openai",
            parsed.base_url.clone(),
            api_key,
            connect_timeout,
        )
        .map_err(|e: ProviderError| BackendError::InvalidSpec {
            message: format!("build azure adapter: {e}"),
        })?;
        let adapter: Arc<dyn ChatProviderAdapter> = Arc::new(adapter);

        let templates = Templates::compile(&parsed.chat.prompt.system, &parsed.chat.prompt.user)
            .map_err(|e| BackendError::InvalidSpec {
                message: format!("template: {e}"),
            })?;

        let (validator, raw_output_schema) = if matches!(
            parsed.chat.response_format.mode,
            ResponseFormatMode::JsonSchema
        ) {
            let schema_value = spec.get("output_schema").cloned();
            if let Some(schema) = schema_value {
                let v = compile_validator(&schema).map_err(|e| BackendError::InvalidSpec {
                    message: e.to_string(),
                })?;
                (Some(v), Some(schema))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        let child_tool_defs = build_child_tool_defs(&parsed.chat.tools, |_name| None);

        let engine = ChatEngine {
            backend_name: backend_name.to_owned(),
            adapter,
            templates,
            validator,
            raw_output_schema,
            spec: parsed.chat,
            host,
            child_tool_defs,
            child_tool_validators: Vec::new(),
        };

        self.engines
            .write()
            .map_err(|_| BackendError::InvalidSpec {
                message: "engine map poisoned".into(),
            })?
            .insert(backend_name.to_owned(), Arc::new(engine));

        Ok(())
    }

    async fn execute(
        &self,
        backend_name: &str,
        request: BackendRequest,
    ) -> Result<BackendResponse, BackendError> {
        let engine = lookup_engine(&self.engines, backend_name)?;
        let args = decode_payload(&request.payload)?;
        let model = engine.spec.model.clone();
        let identity = request.identity.clone();
        let request_id = request.request_id.clone();

        let host_span = open_span(self.host_handle(), OpenAiKind::Azure, backend_name, &model);
        let started = std::time::Instant::now();
        let result = engine
            .execute(&args, &request.request_id, request.session_id.as_deref())
            .await;
        let elapsed = started.elapsed();
        emit_call_metrics(
            backend_name,
            engine.adapter.label(),
            &model,
            result.is_ok(),
            elapsed,
        );
        emit_chat_observability(
            self.host_handle(),
            OpenAiKind::Azure,
            backend_name,
            &model,
            &request_id,
            identity.as_ref(),
            elapsed,
            result.as_ref().map(|_| ()),
            None,
        )
        .await;
        drop(host_span);

        let value = result?;
        let payload = serde_json::to_vec(&value).map_err(|e| BackendError::Transport {
            message: format!("serialize response: {e}"),
        })?;
        Ok(BackendResponse {
            payload,
            truncated: false,
        })
    }

    async fn execute_streaming(
        &self,
        backend_name: &str,
        request: BackendRequest,
    ) -> Result<BackendChunkStream, BackendError> {
        let engine = lookup_engine(&self.engines, backend_name)?;
        let args = decode_payload(&request.payload)?;
        wrap_streaming(
            self.host_handle().cloned(),
            OpenAiKind::Azure,
            backend_name.to_owned(),
            engine.spec.model.clone(),
            request.identity.clone(),
            request.request_id.clone(),
            engine.execute_streaming(args, request.request_id, request.session_id),
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers shared by the two plugins in this crate.
// ---------------------------------------------------------------------------

fn lookup_engine(
    engines: &Arc<RwLock<BTreeMap<String, Arc<ChatEngine>>>>,
    backend_name: &str,
) -> Result<Arc<ChatEngine>, BackendError> {
    engines
        .read()
        .map_err(|_| BackendError::InvalidSpec {
            message: "engine map poisoned".into(),
        })?
        .get(backend_name)
        .cloned()
        .ok_or_else(|| BackendError::ProfileNotFound {
            backend_name: backend_name.to_owned(),
        })
}

fn decode_payload(payload: &[u8]) -> Result<Value, BackendError> {
    if payload.is_empty() {
        return Ok(serde_json::json!({}));
    }
    serde_json::from_slice(payload).map_err(|e| BackendError::InvalidSpec {
        message: format!("execute payload was not JSON: {e}"),
    })
}

fn emit_call_metrics(
    backend_name: &str,
    provider_label: &str,
    model: &str,
    ok: bool,
    elapsed: std::time::Duration,
) {
    metrics::counter!(
        "mcpg_llm_calls_total",
        "binding" => backend_name.to_owned(),
        "provider" => provider_label.to_string(),
        "model" => model.to_owned(),
        "status" => if ok { "ok" } else { "error" },
    )
    .increment(1);
    metrics::histogram!(
        "mcpg_llm_call_overall_seconds",
        "binding" => backend_name.to_owned(),
        "provider" => provider_label.to_string(),
        "model" => model.to_owned(),
    )
    .record(elapsed.as_secs_f64());
}

/// Wrap the engine's streaming chunk stream so
/// we can observe end-of-stream + accumulate `BackendChunk::Usage`
/// tokens, then emit the host triad once when the stream terminates
/// (either via `Done` or an `Err` item).
///
/// The wrapper preserves the chunk order exactly — every chunk from
/// the engine is forwarded unchanged. The host triad emission is
/// driven by the stream's terminal item, so a caller that drops the
/// stream early skips the metric / audit step (the SpanGuard's Drop
/// still fires).
#[allow(clippy::too_many_arguments)] // Bounded per-call observability surface.
fn wrap_streaming(
    host: Option<HostHandle>,
    kind: OpenAiKind,
    backend_name: String,
    model: String,
    identity: Option<PluginIdentity>,
    request_id: String,
    inner: BackendChunkStream,
) -> Result<BackendChunkStream, BackendError> {
    use futures::StreamExt;

    let host_for_state = host.clone();
    let span = host_for_state.as_ref().map(|h| {
        h.span(
            kind_span_name(kind),
            serde_json::json!({
                "binding": backend_name.clone(),
                "model": model.clone(),
            }),
        )
    });
    let t0 = std::time::Instant::now();

    struct State {
        inner: BackendChunkStream,
        host: Option<HostHandle>,
        kind: OpenAiKind,
        backend_name: String,
        model: String,
        identity: Option<PluginIdentity>,
        request_id: String,
        t0: std::time::Instant,
        usage: UsageSnapshot,
        terminated: bool,
        last_err: Option<BackendError>,
        // Held for the lifetime of the stream so span_end fires
        // after triad emission.
        _span: Option<mcpg_plugin_sdk::SpanGuard>,
    }

    let init = State {
        inner,
        host,
        kind,
        backend_name,
        model,
        identity,
        request_id,
        t0,
        usage: UsageSnapshot::default(),
        terminated: false,
        last_err: None,
        _span: span,
    };

    let stream = futures::stream::unfold(init, |mut state| async move {
        if state.terminated {
            return None;
        }
        match state.inner.next().await {
            Some(Ok(chunk)) => {
                // Accumulate token usage across iterations.
                if let BackendChunk::Usage {
                    input_tokens,
                    output_tokens,
                    cached_input_tokens,
                } = &chunk
                {
                    state.usage.input_tokens = state
                        .usage
                        .input_tokens
                        .saturating_add(*input_tokens as u64);
                    state.usage.output_tokens = state
                        .usage
                        .output_tokens
                        .saturating_add(*output_tokens as u64);
                    state.usage.cached_input_tokens = state
                        .usage
                        .cached_input_tokens
                        .saturating_add(*cached_input_tokens as u64);
                }
                let is_done = matches!(chunk, BackendChunk::Done(_));
                if is_done {
                    state.terminated = true;
                    let elapsed = state.t0.elapsed();
                    emit_chat_observability(
                        state.host.as_ref(),
                        state.kind,
                        &state.backend_name,
                        &state.model,
                        &state.request_id,
                        state.identity.as_ref(),
                        elapsed,
                        Ok(()),
                        Some(state.usage),
                    )
                    .await;
                }
                Some((Ok(chunk), state))
            }
            Some(Err(err)) => {
                state.terminated = true;
                state.last_err = Some(clone_backend_error(&err));
                let elapsed = state.t0.elapsed();
                emit_chat_observability(
                    state.host.as_ref(),
                    state.kind,
                    &state.backend_name,
                    &state.model,
                    &state.request_id,
                    state.identity.as_ref(),
                    elapsed,
                    Err(state.last_err.as_ref().unwrap()),
                    Some(state.usage),
                )
                .await;
                Some((Err(err), state))
            }
            None => {
                // Stream ended without a terminal Done. The engine
                // produces a Done last in the happy path; if we
                // reach None first the stream was dropped early —
                // skip the triad emission (the SpanGuard's Drop
                // still lands on state drop).
                None
            }
        }
    });
    Ok(Box::pin(stream))
}

fn clone_backend_error(err: &BackendError) -> BackendError {
    match err {
        BackendError::ProfileNotFound { backend_name } => BackendError::ProfileNotFound {
            backend_name: backend_name.clone(),
        },
        BackendError::InvalidSpec { message } => BackendError::InvalidSpec {
            message: message.clone(),
        },
        BackendError::Timeout { timeout_ms } => BackendError::Timeout {
            timeout_ms: *timeout_ms,
        },
        BackendError::Transport { message } => BackendError::Transport {
            message: message.clone(),
        },
    }
}

fn kind_span_name(kind: OpenAiKind) -> &'static str {
    match kind {
        OpenAiKind::OpenAi => "llm_openai.execute_streaming",
        OpenAiKind::Azure => "llm_azure_openai.execute_streaming",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcpg_plugin_protocol::noop_backend_host;

    #[test]
    fn openai_plugin_kind_and_manifest() {
        let p = OpenAiChatPlugin::new();
        assert_eq!(p.kind(), "openai.chat");
        assert_eq!(p.manifest().id, "dev.mcpg.backend.openai.chat");
    }

    #[test]
    fn azure_plugin_kind_and_manifest() {
        let p = AzureOpenaiChatPlugin::new();
        assert_eq!(p.kind(), "azure_openai.chat");
        assert_eq!(p.manifest().id, "dev.mcpg.backend.azure_openai.chat");
    }

    #[tokio::test]
    async fn openai_register_minimal_spec_succeeds() {
        let plugin = OpenAiChatPlugin::new();
        plugin
            .register_profile(
                "summarize",
                &serde_json::json!({
                    "model": "gpt-4o-mini",
                    "api_key": "k",
                    "prompt": { "system": "x", "user": "{{ input.text }}" },
                    "output_schema": { "type": "object", "properties": {"a":{"type":"string"}}, "required": ["a"] }
                }),
                noop_backend_host(),
            )
            .await
            .unwrap();
        assert_eq!(plugin.registered_profile_count(), 1);
    }

    #[tokio::test]
    async fn azure_register_requires_base_url() {
        let plugin = AzureOpenaiChatPlugin::new();
        let err = plugin
            .register_profile(
                "az",
                &serde_json::json!({
                    "base_url": "",
                    "model": "gpt-4o",
                    "api_key": "k",
                    "prompt": { "system": "x", "user": "y" }
                }),
                noop_backend_host(),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, BackendError::InvalidSpec { .. }));
    }

    #[tokio::test]
    async fn execute_unregistered_returns_not_found() {
        let plugin = OpenAiChatPlugin::new();
        let err = plugin
            .execute(
                "missing",
                BackendRequest {
                    payload: vec![],
                    headers: vec![],
                    request_id: "r".into(),
                    session_id: None,
                    identity: None,
                    idempotency: None,
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(err, BackendError::ProfileNotFound { .. }));
    }
}
