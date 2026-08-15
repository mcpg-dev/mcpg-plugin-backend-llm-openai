//! HostHandle observability triad for the
//! OpenAI + Azure OpenAI chat backend plugins.
//!
//! Provides:
//!
//! - [`outcome_label`] — classify a chat-execute outcome into the
//!   bounded label set (`ok` / `client_error` / `server_error` /
//!   `timeout` / `rate_limited` / `auth_failed` / `model_not_found`
//!   / `transport`).
//! - [`audit_action_for_outcome`] — gate audit emission on the
//!   bounded set: success path emits the `.completion` audit (LLM
//!   calls are low-volume + high-value so per-call audit is
//!   appropriate); failure path emits `.failure`.
//! - [`emit_chat_observability`] — single entry point that opens the
//!   per-call span, records the latency histogram + call counter
//!   with bounded `outcome` + `model` labels, and emits the audit
//!   event with model + token + cost details when known.
//!
//! Token counts and cost: the shared `ChatEngine::execute` returns
//! only the validated content (no usage struct), so the non-
//! streaming path cannot extract per-call tokens at the plugin
//! layer. The shared engine already emits
//! `mcpg_llm_call_tokens_{input,output}` and
//! `mcpg_llm_cost_usd_total` via the metrics floor; the host triad
//! is intentional coexistence. The streaming path accumulates
//! `BackendChunk::Usage` events and forwards the totals to
//! [`emit_chat_observability`] so streaming completions DO carry
//! token + cost detail in the audit event.

use std::time::Duration;

use mcpg_backend_llm_shared::cost::{bundled_rate_card, compute_chat_cost_usd};
use mcpg_backend_llm_shared::normalized::TokenUsage;
use mcpg_plugin_protocol::BackendError;
use mcpg_plugin_protocol::audit::{AuditEvent, AuditOutcome};
use mcpg_plugin_protocol::types::PluginIdentity;
use mcpg_plugin_sdk::{HostHandle, SpanGuard};

/// Which OpenAI plugin called us. Drives the audit action name +
/// the metric/span name prefix. Bounded — there are exactly two
/// variants in this crate.
#[derive(Copy, Clone, Debug)]
pub(crate) enum OpenAiKind {
    OpenAi,
    Azure,
}

impl OpenAiKind {
    fn provider_slug(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Azure => "azure_openai",
        }
    }

    fn span_name(self) -> &'static str {
        match self {
            Self::OpenAi => "llm_openai.execute",
            Self::Azure => "llm_azure_openai.execute",
        }
    }

    fn latency_metric(self) -> &'static str {
        match self {
            Self::OpenAi => "mcpg_llm_openai_latency_seconds",
            Self::Azure => "mcpg_llm_azure_openai_latency_seconds",
        }
    }

    fn calls_metric(self) -> &'static str {
        match self {
            Self::OpenAi => "mcpg_llm_openai_calls_total",
            Self::Azure => "mcpg_llm_azure_openai_calls_total",
        }
    }

    fn input_tokens_metric(self) -> &'static str {
        match self {
            Self::OpenAi => "mcpg_llm_openai_input_tokens_total",
            Self::Azure => "mcpg_llm_azure_openai_input_tokens_total",
        }
    }

    fn output_tokens_metric(self) -> &'static str {
        match self {
            Self::OpenAi => "mcpg_llm_openai_output_tokens_total",
            Self::Azure => "mcpg_llm_azure_openai_output_tokens_total",
        }
    }

    fn cost_metric(self) -> &'static str {
        match self {
            Self::OpenAi => "mcpg_llm_openai_cost_usd_micros_total",
            Self::Azure => "mcpg_llm_azure_openai_cost_usd_micros_total",
        }
    }

    fn completion_action(self) -> &'static str {
        match self {
            Self::OpenAi => "dev.mcpg.llm.openai.completion",
            Self::Azure => "dev.mcpg.llm.azure_openai.completion",
        }
    }

    fn failure_action(self) -> &'static str {
        match self {
            Self::OpenAi => "dev.mcpg.llm.openai.failure",
            Self::Azure => "dev.mcpg.llm.azure_openai.failure",
        }
    }

    /// Rate-card provider key. Both OpenAI variants use the same
    /// rate-card entries — Azure deploys the same models under
    /// operator-renamed deployment IDs, but the cost-per-token
    /// matches the public OpenAI rate.
    fn rate_card_provider(self) -> &'static str {
        "openai"
    }
}

/// Classify a chat-execute outcome into the bounded label set. The
/// label space is intentionally closed so operator dashboards can
/// pivot on `outcome` without cardinality blow-up.
pub(crate) fn outcome_label(result: Result<(), &BackendError>) -> &'static str {
    let Err(err) = result else {
        return "ok";
    };
    match err {
        BackendError::ProfileNotFound { .. } => "model_not_found",
        BackendError::Timeout { .. } => "timeout",
        BackendError::InvalidSpec { .. } => "client_error",
        BackendError::Transport { message } => transport_message_label(message),
    }
}

/// Sub-classify a `BackendError::Transport` message string into a
/// bounded label. The shared engine surfaces upstream HTTP failures
/// as `Transport` with the upstream message embedded; we sniff for
/// well-known shapes (429 / 401 / 5xx) to keep operator triage
/// fast. Anything that doesn't match falls back to `transport`.
fn transport_message_label(message: &str) -> &'static str {
    let lower = message.to_ascii_lowercase();
    if lower.contains("429") || lower.contains("rate limit") || lower.contains("rate_limit") {
        "rate_limited"
    } else if lower.contains("401")
        || lower.contains("403")
        || lower.contains("auth")
        || lower.contains("unauthorized")
        || lower.contains("forbidden")
    {
        "auth_failed"
    } else if lower.contains("404")
        || lower.contains("model not found")
        || lower.contains("does not exist")
    {
        "model_not_found"
    } else if lower.contains("500")
        || lower.contains("502")
        || lower.contains("503")
        || lower.contains("504")
        || lower.contains("server error")
    {
        "server_error"
    } else if lower.contains("400") || lower.contains("invalid") {
        "client_error"
    } else if lower.contains("timed out") || lower.contains("timeout") {
        "timeout"
    } else {
        "transport"
    }
}

/// Bounded audit action name for the outcome. LLM calls audit
/// PER upstream call (low-volume, high-value, compliance-relevant).
fn audit_action_for_outcome(kind: OpenAiKind, label: &str) -> &'static str {
    if label == "ok" {
        kind.completion_action()
    } else {
        kind.failure_action()
    }
}

/// Synthetic identity for audit events emitted
/// from system-initiated paths with no caller attribution. Mirrors
/// the L.5 HTTP + L.8 redis pattern so cross-plugin audit search
/// treats system traffic uniformly.
fn synthetic_system_identity(kind: OpenAiKind) -> PluginIdentity {
    PluginIdentity {
        kind: "system".into(),
        trust_level: "verified".into(),
        subject_id: Some(format!("dev.mcpg.backend.{}.chat", kind.provider_slug())),
        auth_provider: None,
        issuer: None,
        roles: vec![],
        groups: vec![],
        scopes: vec![],
        attributes: Default::default(),
    }
}

/// Best-effort RFC 3339 timestamp for audit event `occurred_at`.
fn rfc3339_now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// Per-call token + cost breakdown the streaming path captures via
/// `BackendChunk::Usage` aggregation. `None` for the non-streaming
/// path where shared `ChatEngine::execute` returns only validated
/// content (no usage struct).
#[derive(Default, Debug, Clone, Copy)]
pub(crate) struct UsageSnapshot {
    pub(crate) input_tokens: u64,
    pub(crate) output_tokens: u64,
    pub(crate) cached_input_tokens: u64,
}

/// Open the per-call host span. Caller MUST drop the returned guard
/// AFTER `emit_chat_observability` so the `span_end` event lands
/// after metric + audit emission.
pub(crate) fn open_span(
    host: Option<&HostHandle>,
    kind: OpenAiKind,
    backend_name: &str,
    model: &str,
) -> Option<SpanGuard> {
    host.map(|h| {
        h.span(
            kind.span_name(),
            serde_json::json!({
                "binding": backend_name,
                "model": model,
            }),
        )
    })
}

/// Emit the per-call observability triad: latency histogram + call
/// counter + one audit event per upstream call.
///
/// `usage` is `Some` only on the streaming path (which observes
/// `BackendChunk::Usage` cumulatively); the non-streaming path
/// passes `None` and the engine's existing
/// `mcpg_llm_call_tokens_*` floor metrics carry the per-call token
/// histogram series.
#[allow(clippy::too_many_arguments)] // Bounded per-call observability surface.
pub(crate) async fn emit_chat_observability(
    host: Option<&HostHandle>,
    kind: OpenAiKind,
    backend_name: &str,
    model: &str,
    request_id: &str,
    identity: Option<&PluginIdentity>,
    elapsed: Duration,
    result: Result<(), &BackendError>,
    usage: Option<UsageSnapshot>,
) {
    let Some(host) = host else {
        return;
    };
    let label = outcome_label(result);
    let elapsed_secs = elapsed.as_secs_f64();

    // Bounded labels: outcome (enum) + model (operator-bounded:
    // typical operator declares 5–20 model IDs across their LLM
    // plugins; cardinality stays well under 100).
    host.histogram(
        kind.latency_metric(),
        elapsed_secs,
        &[("outcome", label), ("model", model)],
    );
    host.counter(
        kind.calls_metric(),
        1,
        &[("outcome", label), ("model", model)],
    );

    // Token + cost metrics — emitted only when the streaming path
    // captured a usage snapshot. Non-streaming `execute` returns
    // validated content without tokens; the engine's existing
    // `mcpg_llm_call_tokens_*` histogram series covers the floor.
    let cost_usd_micros: Option<u64> = if let Some(snap) = usage {
        host.counter(
            kind.input_tokens_metric(),
            snap.input_tokens,
            &[("model", model)],
        );
        host.counter(
            kind.output_tokens_metric(),
            snap.output_tokens,
            &[("model", model)],
        );
        let usage_struct = TokenUsage {
            input_tokens: snap.input_tokens.min(u32::MAX as u64) as u32,
            output_tokens: snap.output_tokens.min(u32::MAX as u64) as u32,
            cached_input_tokens: snap.cached_input_tokens.min(u32::MAX as u64) as u32,
        };
        let cost = compute_chat_cost_usd(
            bundled_rate_card(),
            kind.rate_card_provider(),
            model,
            &usage_struct,
        );
        cost.map(|usd| {
            let micros = (usd * 1_000_000.0).round().max(0.0) as u64;
            host.counter(kind.cost_metric(), micros, &[("model", model)]);
            micros
        })
    } else {
        None
    };

    // Audit event per upstream call. The action name flips between
    // `.completion` (success) and `.failure` (any error class).
    // LLM calls are low-volume + high-value, so per-call audit IS
    // the right shape.
    let action = audit_action_for_outcome(kind, label);
    let actor = identity
        .cloned()
        .unwrap_or_else(|| synthetic_system_identity(kind));
    let mut details = serde_json::json!({
        "binding": backend_name,
        "model": model,
        "outcome": label,
        "provider": kind.provider_slug(),
        "duration_ms": elapsed.as_millis() as u64,
        "alias": host.alias(),
    });
    if let Some(snap) = usage {
        let object = details.as_object_mut().expect("json object");
        object.insert(
            "input_tokens".into(),
            serde_json::Value::from(snap.input_tokens),
        );
        object.insert(
            "output_tokens".into(),
            serde_json::Value::from(snap.output_tokens),
        );
        object.insert(
            "total_tokens".into(),
            serde_json::Value::from(snap.input_tokens + snap.output_tokens),
        );
        if snap.cached_input_tokens > 0 {
            object.insert(
                "cached_input_tokens".into(),
                serde_json::Value::from(snap.cached_input_tokens),
            );
        }
        if let Some(micros) = cost_usd_micros {
            object.insert("cost_usd_micros".into(), serde_json::Value::from(micros));
        }
    }
    if let Err(err) = result {
        let object = details.as_object_mut().expect("json object");
        object.insert(
            "error_class".into(),
            serde_json::Value::String(label.to_owned()),
        );
        object.insert(
            "error_message".into(),
            serde_json::Value::String(err.to_string()),
        );
    }
    let outcome_class = if result.is_ok() {
        AuditOutcome::Success
    } else {
        AuditOutcome::Failure
    };
    let event = AuditEvent {
        event_id: format!(
            "llm-{}-{}-{}",
            kind.provider_slug(),
            request_id,
            elapsed.as_nanos()
        ),
        occurred_at: rfc3339_now(),
        actor,
        action: action.to_owned(),
        resource: Some(format!(
            "llm-binding://{}/{}",
            kind.provider_slug(),
            backend_name
        )),
        outcome: outcome_class,
        request_id: Some(request_id.to_owned()),
        node_id: None,
        details,
        prev_event_hash: None,
    };

    // `HostHandle::audit_event` is sync but bridges to an async
    // sink via `Handle::block_on`. Calling it directly from an
    // async worker panics ("Cannot start a runtime from within a
    // runtime"); move it onto a blocking worker so the bridge is
    // safe. An `_async` variant on `HostHandle` would retire
    // this detour.
    let host_for_audit = host.clone();
    if let Err(join_err) = tokio::task::spawn_blocking(move || {
        let _ = host_for_audit.audit_event(event);
    })
    .await
    {
        tracing::debug!(
            target: "mcpg::llm_openai::host_handle",
            error = %join_err,
            "host_handle.audit_event spawn_blocking failed"
        );
    }
}
