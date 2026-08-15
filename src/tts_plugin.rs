//! `BackendPlugin` impls for OpenAI + Azure OpenAI TTS.

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use mcpg_backend_llm_shared::{ProviderError, TtsEngine, TtsProviderAdapter, resolve_api_key};
use mcpg_plugin_protocol::{
    BackendChunkStream, BackendError, BackendHost, BackendPlugin, BackendRequest, BackendResponse,
    PluginManifest, async_trait, firstparty_manifest,
};
use serde_json::Value;

use crate::adapter::OpenAiVariant;
use crate::config::{AzureOpenaiTtsSpec, OpenAiTtsSpec};
use crate::tts_adapter::OpenAiTtsAdapter;

pub struct OpenAiTtsPlugin {
    manifest: PluginManifest,
    engines: Arc<RwLock<BTreeMap<String, Arc<TtsEngine>>>>,
}

impl std::fmt::Debug for OpenAiTtsPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiTtsPlugin").finish()
    }
}

impl Default for OpenAiTtsPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAiTtsPlugin {
    pub fn new() -> Self {
        Self {
            manifest: firstparty_manifest! {
                id: "dev.mcpg.backend.openai.tts",
                name: "OpenAI Text-to-Speech",
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
impl BackendPlugin for OpenAiTtsPlugin {
    fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }

    fn kind(&self) -> &str {
        "openai.tts"
    }

    async fn register_profile(
        &self,
        backend_name: &str,
        spec: &Value,
        host: Arc<dyn BackendHost>,
    ) -> Result<(), BackendError> {
        let parsed: OpenAiTtsSpec =
            serde_json::from_value(spec.clone()).map_err(|e| BackendError::InvalidSpec {
                message: format!("openai_tts spec: {e}"),
            })?;
        parsed.validate().map_err(|e| BackendError::InvalidSpec {
            message: e.to_string(),
        })?;

        let api_key = resolve_api_key(&parsed.api_key)?;
        let base_url = parsed.resolved_base_url().to_owned();
        let connect_timeout = parsed.tts.connect_timeout();

        let adapter = OpenAiTtsAdapter::new(
            OpenAiVariant::OpenAi,
            "openai",
            base_url,
            api_key,
            connect_timeout,
        )
        .map_err(|e: ProviderError| BackendError::InvalidSpec {
            message: format!("build openai tts adapter: {e}"),
        })?;
        let adapter: Arc<dyn TtsProviderAdapter> = Arc::new(adapter);

        let engine = TtsEngine {
            backend_name: backend_name.to_owned(),
            adapter,
            spec: parsed.tts,
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
        execute_tts(&self.engines, backend_name, request).await
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

pub struct AzureOpenaiTtsPlugin {
    manifest: PluginManifest,
    engines: Arc<RwLock<BTreeMap<String, Arc<TtsEngine>>>>,
}

impl std::fmt::Debug for AzureOpenaiTtsPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureOpenaiTtsPlugin").finish()
    }
}

impl Default for AzureOpenaiTtsPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl AzureOpenaiTtsPlugin {
    pub fn new() -> Self {
        Self {
            manifest: firstparty_manifest! {
                id: "dev.mcpg.backend.azure_openai.tts",
                name: "Azure OpenAI Text-to-Speech",
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
impl BackendPlugin for AzureOpenaiTtsPlugin {
    fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }

    fn kind(&self) -> &str {
        "azure_openai.tts"
    }

    async fn register_profile(
        &self,
        backend_name: &str,
        spec: &Value,
        host: Arc<dyn BackendHost>,
    ) -> Result<(), BackendError> {
        let parsed: AzureOpenaiTtsSpec =
            serde_json::from_value(spec.clone()).map_err(|e| BackendError::InvalidSpec {
                message: format!("azure_openai_tts spec: {e}"),
            })?;
        parsed.validate().map_err(|e| BackendError::InvalidSpec {
            message: e.to_string(),
        })?;

        let api_key = resolve_api_key(&parsed.api_key)?;
        let connect_timeout = parsed.tts.connect_timeout();

        let adapter = OpenAiTtsAdapter::new(
            OpenAiVariant::Azure,
            "azure-openai",
            parsed.base_url.clone(),
            api_key,
            connect_timeout,
        )
        .map_err(|e: ProviderError| BackendError::InvalidSpec {
            message: format!("build azure tts adapter: {e}"),
        })?;
        let adapter: Arc<dyn TtsProviderAdapter> = Arc::new(adapter);

        let engine = TtsEngine {
            backend_name: backend_name.to_owned(),
            adapter,
            spec: parsed.tts,
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
        execute_tts(&self.engines, backend_name, request).await
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

async fn execute_tts(
    engines: &Arc<RwLock<BTreeMap<String, Arc<TtsEngine>>>>,
    backend_name: &str,
    request: BackendRequest,
) -> Result<BackendResponse, BackendError> {
    let engine = engines
        .read()
        .map_err(|_| BackendError::InvalidSpec {
            message: "engine map poisoned".into(),
        })?
        .get(backend_name)
        .cloned()
        .ok_or_else(|| BackendError::ProfileNotFound {
            backend_name: backend_name.to_owned(),
        })?;
    let args: Value = if request.payload.is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_slice(&request.payload).map_err(|e| BackendError::InvalidSpec {
            message: format!("execute payload was not JSON: {e}"),
        })?
    };
    let result = engine
        .execute(&args, &request.request_id, request.session_id.as_deref())
        .await;
    metrics::counter!(
        "mcpg_tts_calls_total",
        "binding" => backend_name.to_owned(),
        "provider" => engine.adapter.label().to_string(),
        "model" => engine.spec.model.clone(),
        "status" => if result.is_ok() { "ok" } else { "error" },
    )
    .increment(1);
    let value = result?;
    let payload = serde_json::to_vec(&value).map_err(|e| BackendError::Transport {
        message: format!("serialize tts response: {e}"),
    })?;
    Ok(BackendResponse {
        payload,
        truncated: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcpg_plugin_protocol::noop_backend_host;

    #[test]
    fn openai_tts_kind_and_manifest() {
        let p = OpenAiTtsPlugin::new();
        assert_eq!(p.kind(), "openai.tts");
        assert_eq!(p.manifest().id, "dev.mcpg.backend.openai.tts");
    }

    #[test]
    fn azure_tts_kind_and_manifest() {
        let p = AzureOpenaiTtsPlugin::new();
        assert_eq!(p.kind(), "azure_openai.tts");
        assert_eq!(p.manifest().id, "dev.mcpg.backend.azure_openai.tts");
    }

    #[tokio::test]
    async fn openai_tts_register_minimal_succeeds() {
        let plugin = OpenAiTtsPlugin::new();
        plugin
            .register_profile(
                "tts",
                &serde_json::json!({
                    "model": "tts-1",
                    "voice": "alloy",
                    "api_key": "k"
                }),
                noop_backend_host(),
            )
            .await
            .unwrap();
        assert_eq!(plugin.registered_profile_count(), 1);
    }

    #[tokio::test]
    async fn azure_tts_register_requires_base_url() {
        let plugin = AzureOpenaiTtsPlugin::new();
        let err = plugin
            .register_profile(
                "tts",
                &serde_json::json!({
                    "base_url": "",
                    "model": "tts-1",
                    "voice": "alloy",
                    "api_key": "k"
                }),
                noop_backend_host(),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, BackendError::InvalidSpec { .. }));
    }
}
