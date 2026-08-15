//! `BackendPlugin` impls for OpenAI + Azure OpenAI image generation.

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use mcpg_backend_llm_shared::{ImageEngine, ImageProviderAdapter, ProviderError, resolve_api_key};
use mcpg_plugin_protocol::{
    BackendChunkStream, BackendError, BackendHost, BackendPlugin, BackendRequest, BackendResponse,
    PluginManifest, async_trait, firstparty_manifest,
};
use serde_json::Value;

use crate::adapter::OpenAiVariant;
use crate::config::{AzureOpenaiImageSpec, OpenAiImageSpec};
use crate::image_adapter::OpenAiImageAdapter;

/// `BackendPlugin` for `kind: "openai.image"`.
pub struct OpenAiImagePlugin {
    manifest: PluginManifest,
    engines: Arc<RwLock<BTreeMap<String, Arc<ImageEngine>>>>,
}

impl std::fmt::Debug for OpenAiImagePlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiImagePlugin").finish()
    }
}

impl Default for OpenAiImagePlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAiImagePlugin {
    pub fn new() -> Self {
        Self {
            manifest: firstparty_manifest! {
                id: "dev.mcpg.backend.openai.image",
                name: "OpenAI Image Generation",
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
impl BackendPlugin for OpenAiImagePlugin {
    fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }

    fn kind(&self) -> &str {
        "openai.image"
    }

    async fn register_profile(
        &self,
        backend_name: &str,
        spec: &Value,
        host: Arc<dyn BackendHost>,
    ) -> Result<(), BackendError> {
        let parsed: OpenAiImageSpec =
            serde_json::from_value(spec.clone()).map_err(|e| BackendError::InvalidSpec {
                message: format!("openai_image spec: {e}"),
            })?;
        parsed.validate().map_err(|e| BackendError::InvalidSpec {
            message: e.to_string(),
        })?;

        let api_key = resolve_api_key(&parsed.api_key)?;
        let base_url = parsed.resolved_base_url().to_owned();
        let connect_timeout = parsed.image.connect_timeout();

        let adapter = OpenAiImageAdapter::new(
            OpenAiVariant::OpenAi,
            "openai",
            base_url,
            api_key,
            connect_timeout,
        )
        .map_err(|e: ProviderError| BackendError::InvalidSpec {
            message: format!("build openai image adapter: {e}"),
        })?;
        let adapter: Arc<dyn ImageProviderAdapter> = Arc::new(adapter);

        let engine = ImageEngine {
            backend_name: backend_name.to_owned(),
            adapter,
            spec: parsed.image,
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
        execute_image(&self.engines, backend_name, request).await
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

/// `BackendPlugin` for `kind: "azure_openai.image"`.
pub struct AzureOpenaiImagePlugin {
    manifest: PluginManifest,
    engines: Arc<RwLock<BTreeMap<String, Arc<ImageEngine>>>>,
}

impl std::fmt::Debug for AzureOpenaiImagePlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureOpenaiImagePlugin").finish()
    }
}

impl Default for AzureOpenaiImagePlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl AzureOpenaiImagePlugin {
    pub fn new() -> Self {
        Self {
            manifest: firstparty_manifest! {
                id: "dev.mcpg.backend.azure_openai.image",
                name: "Azure OpenAI Image Generation",
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
impl BackendPlugin for AzureOpenaiImagePlugin {
    fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }

    fn kind(&self) -> &str {
        "azure_openai.image"
    }

    async fn register_profile(
        &self,
        backend_name: &str,
        spec: &Value,
        host: Arc<dyn BackendHost>,
    ) -> Result<(), BackendError> {
        let parsed: AzureOpenaiImageSpec =
            serde_json::from_value(spec.clone()).map_err(|e| BackendError::InvalidSpec {
                message: format!("azure_openai_image spec: {e}"),
            })?;
        parsed.validate().map_err(|e| BackendError::InvalidSpec {
            message: e.to_string(),
        })?;

        let api_key = resolve_api_key(&parsed.api_key)?;
        let connect_timeout = parsed.image.connect_timeout();

        let adapter = OpenAiImageAdapter::new(
            OpenAiVariant::Azure,
            "azure-openai",
            parsed.base_url.clone(),
            api_key,
            connect_timeout,
        )
        .map_err(|e: ProviderError| BackendError::InvalidSpec {
            message: format!("build azure image adapter: {e}"),
        })?;
        let adapter: Arc<dyn ImageProviderAdapter> = Arc::new(adapter);

        let engine = ImageEngine {
            backend_name: backend_name.to_owned(),
            adapter,
            spec: parsed.image,
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
        execute_image(&self.engines, backend_name, request).await
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

async fn execute_image(
    engines: &Arc<RwLock<BTreeMap<String, Arc<ImageEngine>>>>,
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
        "mcpg_image_calls_total",
        "binding" => backend_name.to_owned(),
        "provider" => engine.adapter.label().to_string(),
        "model" => engine.spec.model.clone(),
        "status" => if result.is_ok() { "ok" } else { "error" },
    )
    .increment(1);
    let value = result?;
    let payload = serde_json::to_vec(&value).map_err(|e| BackendError::Transport {
        message: format!("serialize image response: {e}"),
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
    fn openai_image_plugin_kind_and_manifest() {
        let p = OpenAiImagePlugin::new();
        assert_eq!(p.kind(), "openai.image");
        assert_eq!(p.manifest().id, "dev.mcpg.backend.openai.image");
    }

    #[test]
    fn azure_image_plugin_kind_and_manifest() {
        let p = AzureOpenaiImagePlugin::new();
        assert_eq!(p.kind(), "azure_openai.image");
        assert_eq!(p.manifest().id, "dev.mcpg.backend.azure_openai.image");
    }

    #[tokio::test]
    async fn openai_image_register_minimal_spec_succeeds() {
        let plugin = OpenAiImagePlugin::new();
        plugin
            .register_profile(
                "img",
                &serde_json::json!({
                    "model": "dall-e-3",
                    "api_key": "k"
                }),
                noop_backend_host(),
            )
            .await
            .unwrap();
        assert_eq!(plugin.registered_profile_count(), 1);
    }

    #[tokio::test]
    async fn azure_image_register_requires_base_url() {
        let plugin = AzureOpenaiImagePlugin::new();
        let err = plugin
            .register_profile(
                "img",
                &serde_json::json!({
                    "base_url": "",
                    "model": "dall-e-3",
                    "api_key": "k"
                }),
                noop_backend_host(),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, BackendError::InvalidSpec { .. }));
    }
}
