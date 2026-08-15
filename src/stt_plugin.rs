//! `BackendPlugin` impls for OpenAI + Azure OpenAI STT (Whisper).

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use mcpg_backend_llm_shared::{ProviderError, SttEngine, SttProviderAdapter, resolve_api_key};
use mcpg_plugin_protocol::{
    BackendChunkStream, BackendError, BackendHost, BackendPlugin, BackendRequest, BackendResponse,
    PluginManifest, async_trait, firstparty_manifest,
};
use serde_json::Value;

use crate::adapter::OpenAiVariant;
use crate::config::{AzureOpenaiSttSpec, OpenAiSttSpec};
use crate::stt_adapter::OpenAiSttAdapter;

pub struct OpenAiSttPlugin {
    manifest: PluginManifest,
    engines: Arc<RwLock<BTreeMap<String, Arc<SttEngine>>>>,
}

impl std::fmt::Debug for OpenAiSttPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiSttPlugin").finish()
    }
}

impl Default for OpenAiSttPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAiSttPlugin {
    pub fn new() -> Self {
        Self {
            manifest: firstparty_manifest! {
                id: "dev.mcpg.backend.openai.stt",
                name: "OpenAI Speech-to-Text (Whisper)",
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
impl BackendPlugin for OpenAiSttPlugin {
    fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }

    fn kind(&self) -> &str {
        "openai.stt"
    }

    async fn register_profile(
        &self,
        backend_name: &str,
        spec: &Value,
        host: Arc<dyn BackendHost>,
    ) -> Result<(), BackendError> {
        let parsed: OpenAiSttSpec =
            serde_json::from_value(spec.clone()).map_err(|e| BackendError::InvalidSpec {
                message: format!("openai_stt spec: {e}"),
            })?;
        parsed.validate().map_err(|e| BackendError::InvalidSpec {
            message: e.to_string(),
        })?;

        let api_key = resolve_api_key(&parsed.api_key)?;
        let base_url = parsed.resolved_base_url().to_owned();
        let connect_timeout = parsed.stt.connect_timeout();

        let adapter = OpenAiSttAdapter::new(
            OpenAiVariant::OpenAi,
            "openai",
            base_url,
            api_key,
            connect_timeout,
        )
        .map_err(|e: ProviderError| BackendError::InvalidSpec {
            message: format!("build openai stt adapter: {e}"),
        })?;
        let adapter: Arc<dyn SttProviderAdapter> = Arc::new(adapter);

        let engine = SttEngine {
            backend_name: backend_name.to_owned(),
            adapter,
            spec: parsed.stt,
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
        execute_stt(&self.engines, backend_name, request).await
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

pub struct AzureOpenaiSttPlugin {
    manifest: PluginManifest,
    engines: Arc<RwLock<BTreeMap<String, Arc<SttEngine>>>>,
}

impl std::fmt::Debug for AzureOpenaiSttPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureOpenaiSttPlugin").finish()
    }
}

impl Default for AzureOpenaiSttPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl AzureOpenaiSttPlugin {
    pub fn new() -> Self {
        Self {
            manifest: firstparty_manifest! {
                id: "dev.mcpg.backend.azure_openai.stt",
                name: "Azure OpenAI Speech-to-Text",
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
impl BackendPlugin for AzureOpenaiSttPlugin {
    fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }

    fn kind(&self) -> &str {
        "azure_openai.stt"
    }

    async fn register_profile(
        &self,
        backend_name: &str,
        spec: &Value,
        host: Arc<dyn BackendHost>,
    ) -> Result<(), BackendError> {
        let parsed: AzureOpenaiSttSpec =
            serde_json::from_value(spec.clone()).map_err(|e| BackendError::InvalidSpec {
                message: format!("azure_openai_stt spec: {e}"),
            })?;
        parsed.validate().map_err(|e| BackendError::InvalidSpec {
            message: e.to_string(),
        })?;

        let api_key = resolve_api_key(&parsed.api_key)?;
        let connect_timeout = parsed.stt.connect_timeout();

        let adapter = OpenAiSttAdapter::new(
            OpenAiVariant::Azure,
            "azure-openai",
            parsed.base_url.clone(),
            api_key,
            connect_timeout,
        )
        .map_err(|e: ProviderError| BackendError::InvalidSpec {
            message: format!("build azure stt adapter: {e}"),
        })?;
        let adapter: Arc<dyn SttProviderAdapter> = Arc::new(adapter);

        let engine = SttEngine {
            backend_name: backend_name.to_owned(),
            adapter,
            spec: parsed.stt,
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
        execute_stt(&self.engines, backend_name, request).await
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

async fn execute_stt(
    engines: &Arc<RwLock<BTreeMap<String, Arc<SttEngine>>>>,
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
        "mcpg_stt_calls_total",
        "binding" => backend_name.to_owned(),
        "provider" => engine.adapter.label().to_string(),
        "model" => engine.spec.model.clone(),
        "status" => if result.is_ok() { "ok" } else { "error" },
    )
    .increment(1);
    let value = result?;
    let payload = serde_json::to_vec(&value).map_err(|e| BackendError::Transport {
        message: format!("serialize stt response: {e}"),
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
    fn openai_stt_kind_and_manifest() {
        let p = OpenAiSttPlugin::new();
        assert_eq!(p.kind(), "openai.stt");
        assert_eq!(p.manifest().id, "dev.mcpg.backend.openai.stt");
    }

    #[test]
    fn azure_stt_kind_and_manifest() {
        let p = AzureOpenaiSttPlugin::new();
        assert_eq!(p.kind(), "azure_openai.stt");
        assert_eq!(p.manifest().id, "dev.mcpg.backend.azure_openai.stt");
    }

    #[tokio::test]
    async fn openai_stt_register_minimal_succeeds() {
        let plugin = OpenAiSttPlugin::new();
        plugin
            .register_profile(
                "stt",
                &serde_json::json!({
                    "model": "whisper-1",
                    "api_key": "k"
                }),
                noop_backend_host(),
            )
            .await
            .unwrap();
        assert_eq!(plugin.registered_profile_count(), 1);
    }
}
