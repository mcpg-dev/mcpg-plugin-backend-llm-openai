//! `BackendPlugin` impls for OpenAI + Azure OpenAI embeddings.
//!
//! Both delegate to the shared
//! [`mcpg_backend_llm_shared::EmbeddingEngine`]; only the wire-format
//! adapter ([`crate::OpenAiEmbeddingAdapter`]) is provider-specific.
//! `OpenAiEmbeddingAdapter` is exported `pub` so the sibling
//! `mcpg-plugin-backend-llm-compat` crate can reuse it for arbitrary
//! OpenAI-compatible embeddings endpoints (vLLM, Together, Groq).

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use mcpg_backend_llm_shared::{
    EmbeddingEngine, EmbeddingProviderAdapter, ProviderError, resolve_api_key,
};
use mcpg_plugin_protocol::{
    BackendChunkStream, BackendError, BackendHost, BackendPlugin, BackendRequest, BackendResponse,
    PluginManifest, async_trait, firstparty_manifest,
};
use serde_json::Value;

use crate::adapter::OpenAiVariant;
use crate::config::{AzureOpenaiEmbeddingSpec, OpenAiEmbeddingSpec};
use crate::embedding_adapter::OpenAiEmbeddingAdapter;

/// `BackendPlugin` for `kind: "openai.embedding"`.
pub struct OpenAiEmbeddingPlugin {
    manifest: PluginManifest,
    engines: Arc<RwLock<BTreeMap<String, Arc<EmbeddingEngine>>>>,
}

impl std::fmt::Debug for OpenAiEmbeddingPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiEmbeddingPlugin").finish()
    }
}

impl Default for OpenAiEmbeddingPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAiEmbeddingPlugin {
    pub fn new() -> Self {
        Self {
            manifest: firstparty_manifest! {
                id: "dev.mcpg.backend.openai.embedding",
                name: "OpenAI Embeddings",
                class: Backend,
            },
            engines: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }

    #[doc(hidden)]
    pub fn registered_profile_count(&self) -> usize {
        self.engines.read().unwrap().len()
    }
}

#[async_trait]
impl BackendPlugin for OpenAiEmbeddingPlugin {
    fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }

    fn kind(&self) -> &str {
        "openai.embedding"
    }

    async fn register_profile(
        &self,
        backend_name: &str,
        spec: &Value,
        host: Arc<dyn BackendHost>,
    ) -> Result<(), BackendError> {
        let parsed: OpenAiEmbeddingSpec =
            serde_json::from_value(spec.clone()).map_err(|e| BackendError::InvalidSpec {
                message: format!("openai_embedding spec: {e}"),
            })?;
        parsed.validate().map_err(|e| BackendError::InvalidSpec {
            message: e.to_string(),
        })?;

        let api_key = resolve_api_key(&parsed.api_key)?;
        let base_url = parsed.resolved_base_url().to_owned();
        let connect_timeout = parsed.embedding.connect_timeout();

        let adapter = OpenAiEmbeddingAdapter::new(
            OpenAiVariant::OpenAi,
            "openai",
            base_url,
            api_key,
            connect_timeout,
        )
        .map_err(|e: ProviderError| BackendError::InvalidSpec {
            message: format!("build openai embedding adapter: {e}"),
        })?;
        let adapter: Arc<dyn EmbeddingProviderAdapter> = Arc::new(adapter);

        let engine = EmbeddingEngine {
            backend_name: backend_name.to_owned(),
            adapter,
            spec: parsed.embedding,
            host: host.clone(),
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
        let result = engine.execute(&args).await;
        emit_call_metrics(
            backend_name,
            engine.adapter.label(),
            &engine.spec.model,
            result.is_ok(),
        );
        let value = result?;
        let payload = serde_json::to_vec(&value).map_err(|e| BackendError::Transport {
            message: format!("serialize embedding response: {e}"),
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
        // Embeddings have no streaming surface — fall through to the
        // default impl by re-running execute and emitting one Done
        // chunk. The default `BackendPlugin::execute_streaming`
        // implementation does exactly that, but we override to keep
        // the override semantics consistent with chat (which does
        // override). Could be removed; cheap.
        let resp = self.execute(backend_name, request).await?;
        Ok(Box::pin(futures::stream::once(async move {
            Ok(mcpg_plugin_protocol::BackendChunk::Done(resp))
        })))
    }
}

/// `BackendPlugin` for `kind: "azure_openai.embedding"`. Same wire
/// format as OpenAI; auth header + URL pattern differ. Underlying
/// adapter is the same `OpenAiEmbeddingAdapter` with `Azure`
/// variant.
pub struct AzureOpenaiEmbeddingPlugin {
    manifest: PluginManifest,
    engines: Arc<RwLock<BTreeMap<String, Arc<EmbeddingEngine>>>>,
}

impl std::fmt::Debug for AzureOpenaiEmbeddingPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureOpenaiEmbeddingPlugin").finish()
    }
}

impl Default for AzureOpenaiEmbeddingPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl AzureOpenaiEmbeddingPlugin {
    pub fn new() -> Self {
        Self {
            manifest: firstparty_manifest! {
                id: "dev.mcpg.backend.azure_openai.embedding",
                name: "Azure OpenAI Embeddings",
                class: Backend,
            },
            engines: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }

    #[doc(hidden)]
    pub fn registered_profile_count(&self) -> usize {
        self.engines.read().unwrap().len()
    }
}

#[async_trait]
impl BackendPlugin for AzureOpenaiEmbeddingPlugin {
    fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }

    fn kind(&self) -> &str {
        "azure_openai.embedding"
    }

    async fn register_profile(
        &self,
        backend_name: &str,
        spec: &Value,
        host: Arc<dyn BackendHost>,
    ) -> Result<(), BackendError> {
        let parsed: AzureOpenaiEmbeddingSpec =
            serde_json::from_value(spec.clone()).map_err(|e| BackendError::InvalidSpec {
                message: format!("azure_openai_embedding spec: {e}"),
            })?;
        parsed.validate().map_err(|e| BackendError::InvalidSpec {
            message: e.to_string(),
        })?;

        let api_key = resolve_api_key(&parsed.api_key)?;
        let connect_timeout = parsed.embedding.connect_timeout();

        let adapter = OpenAiEmbeddingAdapter::new(
            OpenAiVariant::Azure,
            "azure-openai",
            parsed.base_url.clone(),
            api_key,
            connect_timeout,
        )
        .map_err(|e: ProviderError| BackendError::InvalidSpec {
            message: format!("build azure embedding adapter: {e}"),
        })?;
        let adapter: Arc<dyn EmbeddingProviderAdapter> = Arc::new(adapter);

        let engine = EmbeddingEngine {
            backend_name: backend_name.to_owned(),
            adapter,
            spec: parsed.embedding,
            host: host.clone(),
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
        let result = engine.execute(&args).await;
        emit_call_metrics(
            backend_name,
            engine.adapter.label(),
            &engine.spec.model,
            result.is_ok(),
        );
        let value = result?;
        let payload = serde_json::to_vec(&value).map_err(|e| BackendError::Transport {
            message: format!("serialize embedding response: {e}"),
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
        let resp = self.execute(backend_name, request).await?;
        Ok(Box::pin(futures::stream::once(async move {
            Ok(mcpg_plugin_protocol::BackendChunk::Done(resp))
        })))
    }
}

// ---------------------------------------------------------------------------
// Helpers shared by the two plugins
// ---------------------------------------------------------------------------

fn lookup_engine(
    engines: &Arc<RwLock<BTreeMap<String, Arc<EmbeddingEngine>>>>,
    backend_name: &str,
) -> Result<Arc<EmbeddingEngine>, BackendError> {
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

fn emit_call_metrics(backend_name: &str, provider_label: &str, model: &str, ok: bool) {
    metrics::counter!(
        "mcpg_embedding_calls_total",
        "binding" => backend_name.to_owned(),
        "provider" => provider_label.to_string(),
        "model" => model.to_owned(),
        "status" => if ok { "ok" } else { "error" },
    )
    .increment(1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcpg_plugin_protocol::noop_backend_host;

    #[test]
    fn openai_embedding_plugin_kind_and_manifest() {
        let p = OpenAiEmbeddingPlugin::new();
        assert_eq!(p.kind(), "openai.embedding");
        assert_eq!(p.manifest().id, "dev.mcpg.backend.openai.embedding");
    }

    #[test]
    fn azure_embedding_plugin_kind_and_manifest() {
        let p = AzureOpenaiEmbeddingPlugin::new();
        assert_eq!(p.kind(), "azure_openai.embedding");
        assert_eq!(p.manifest().id, "dev.mcpg.backend.azure_openai.embedding");
    }

    #[tokio::test]
    async fn openai_embedding_register_minimal_spec_succeeds() {
        let plugin = OpenAiEmbeddingPlugin::new();
        plugin
            .register_profile(
                "embed",
                &serde_json::json!({
                    "model": "text-embedding-3-small",
                    "api_key": "k"
                }),
                noop_backend_host(),
            )
            .await
            .unwrap();
        assert_eq!(plugin.registered_profile_count(), 1);
    }

    #[tokio::test]
    async fn azure_embedding_register_requires_base_url() {
        let plugin = AzureOpenaiEmbeddingPlugin::new();
        let err = plugin
            .register_profile(
                "embed",
                &serde_json::json!({
                    "base_url": "",
                    "model": "text-embedding-3-small",
                    "api_key": "k"
                }),
                noop_backend_host(),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, BackendError::InvalidSpec { .. }));
    }

    #[tokio::test]
    async fn execute_unregistered_returns_not_found() {
        let plugin = OpenAiEmbeddingPlugin::new();
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
