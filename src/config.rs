//! Operator-facing config types for OpenAI + Azure OpenAI chat
//! bindings.
//!
//! Each provider crate's spec carries only the provider-specific
//! knobs (`api_key`, `base_url` overrides) and flattens
//! `ChatExecutionSpec` for the common surface. This keeps the
//! operator-facing YAML identical across providers for every field
//! whose semantics are the same.

use mcpg_backend_llm_shared::{
    ApiKeyRef, ChatExecutionSpec, ConfigError, EmbeddingExecutionSpec, ImageExecutionSpec,
    SttExecutionSpec, TtsExecutionSpec,
};
use serde::{Deserialize, Serialize};

/// Spec for `binding_type: openai_chat`.
///
/// Default `base_url`: `https://api.openai.com/v1`. Operators can
/// override (e.g., to point at a forwarding proxy or test fixture).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiChatSpec {
    /// Override for the default `https://api.openai.com/v1`. Most
    /// operators leave this unset.
    #[serde(default)]
    pub base_url: Option<String>,

    pub api_key: ApiKeyRef,

    /// Provider-agnostic execution config (model, prompt, sampling,
    /// tools, retry, guardrails, streaming, response_format,
    /// timeouts).
    #[serde(flatten)]
    pub chat: ChatExecutionSpec,
}

impl OpenAiChatSpec {
    pub const DEFAULT_BASE_URL: &'static str = "https://api.openai.com/v1";

    pub fn validate(&self) -> Result<(), ConfigError> {
        // OpenAI public API has a hard-coded default base URL; no
        // additional provider-level invariants beyond the shared
        // chat-execution validation.
        self.chat.validate()
    }

    /// Resolve the base URL with the OpenAI default applied when
    /// the operator hasn't supplied an override.
    pub fn resolved_base_url(&self) -> &str {
        self.base_url.as_deref().unwrap_or(Self::DEFAULT_BASE_URL)
    }
}

/// Spec for `binding_type: azure_openai_chat`. Azure operators
/// MUST declare a `base_url` (per-deployment URL with embedded
/// `api-version` query string).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureOpenaiChatSpec {
    /// Required. Full Azure URL up to (and including)
    /// `?api-version=…`. Example:
    /// `https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-06`.
    pub base_url: String,

    pub api_key: ApiKeyRef,

    #[serde(flatten)]
    pub chat: ChatExecutionSpec,
}

impl AzureOpenaiChatSpec {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.base_url.trim().is_empty() {
            return Err(ConfigError::InvalidSpec(
                "azure_openai_chat: base_url is required (operator must include the \
                 deployment + ?api-version=… in the URL)"
                    .into(),
            ));
        }
        self.chat.validate()
    }
}

/// Spec for `binding_type: openai_embedding`. Same flatten-passthrough
/// pattern as `OpenAiChatSpec` — operator-side knobs (`base_url`
/// override, `api_key`) plus an embedded
/// [`EmbeddingExecutionSpec`] for the provider-agnostic
/// `model` / `dimensions` / `timeout_ms` / `retry` fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiEmbeddingSpec {
    #[serde(default)]
    pub base_url: Option<String>,

    pub api_key: ApiKeyRef,

    #[serde(flatten)]
    pub embedding: EmbeddingExecutionSpec,
}

impl OpenAiEmbeddingSpec {
    pub const DEFAULT_BASE_URL: &'static str = "https://api.openai.com/v1";

    pub fn validate(&self) -> Result<(), ConfigError> {
        self.embedding.validate()
    }

    pub fn resolved_base_url(&self) -> &str {
        self.base_url.as_deref().unwrap_or(Self::DEFAULT_BASE_URL)
    }
}

/// Spec for `binding_type: azure_openai_embedding`. Azure operators
/// MUST declare a `base_url` (per-deployment embeddings URL with
/// embedded `?api-version=…`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureOpenaiEmbeddingSpec {
    pub base_url: String,
    pub api_key: ApiKeyRef,
    #[serde(flatten)]
    pub embedding: EmbeddingExecutionSpec,
}

impl AzureOpenaiEmbeddingSpec {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.base_url.trim().is_empty() {
            return Err(ConfigError::InvalidSpec(
                "azure_openai_embedding: base_url is required (operator must include the \
                 deployment + ?api-version=… in the URL)"
                    .into(),
            ));
        }
        self.embedding.validate()
    }
}

/// Spec for `binding_type: openai_image`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiImageSpec {
    #[serde(default)]
    pub base_url: Option<String>,
    pub api_key: ApiKeyRef,
    #[serde(flatten)]
    pub image: ImageExecutionSpec,
}

impl OpenAiImageSpec {
    pub const DEFAULT_BASE_URL: &'static str = "https://api.openai.com/v1";

    pub fn validate(&self) -> Result<(), ConfigError> {
        self.image.validate()
    }

    pub fn resolved_base_url(&self) -> &str {
        self.base_url.as_deref().unwrap_or(Self::DEFAULT_BASE_URL)
    }
}

/// Spec for `binding_type: azure_openai_image`. Operator must
/// declare the per-deployment URL with `?api-version=…`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureOpenaiImageSpec {
    pub base_url: String,
    pub api_key: ApiKeyRef,
    #[serde(flatten)]
    pub image: ImageExecutionSpec,
}

impl AzureOpenaiImageSpec {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.base_url.trim().is_empty() {
            return Err(ConfigError::InvalidSpec(
                "azure_openai_image: base_url is required (operator must include the \
                 deployment + ?api-version=… in the URL)"
                    .into(),
            ));
        }
        self.image.validate()
    }
}

/// Spec for `binding_type: openai_tts`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiTtsSpec {
    #[serde(default)]
    pub base_url: Option<String>,
    pub api_key: ApiKeyRef,
    #[serde(flatten)]
    pub tts: TtsExecutionSpec,
}

impl OpenAiTtsSpec {
    pub const DEFAULT_BASE_URL: &'static str = "https://api.openai.com/v1";

    pub fn validate(&self) -> Result<(), ConfigError> {
        self.tts.validate()
    }

    pub fn resolved_base_url(&self) -> &str {
        self.base_url.as_deref().unwrap_or(Self::DEFAULT_BASE_URL)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureOpenaiTtsSpec {
    pub base_url: String,
    pub api_key: ApiKeyRef,
    #[serde(flatten)]
    pub tts: TtsExecutionSpec,
}

impl AzureOpenaiTtsSpec {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.base_url.trim().is_empty() {
            return Err(ConfigError::InvalidSpec(
                "azure_openai_tts: base_url is required".into(),
            ));
        }
        self.tts.validate()
    }
}

/// Spec for `binding_type: openai_stt` (Whisper).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiSttSpec {
    #[serde(default)]
    pub base_url: Option<String>,
    pub api_key: ApiKeyRef,
    #[serde(flatten)]
    pub stt: SttExecutionSpec,
}

impl OpenAiSttSpec {
    pub const DEFAULT_BASE_URL: &'static str = "https://api.openai.com/v1";

    pub fn validate(&self) -> Result<(), ConfigError> {
        self.stt.validate()
    }

    pub fn resolved_base_url(&self) -> &str {
        self.base_url.as_deref().unwrap_or(Self::DEFAULT_BASE_URL)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureOpenaiSttSpec {
    pub base_url: String,
    pub api_key: ApiKeyRef,
    #[serde(flatten)]
    pub stt: SttExecutionSpec,
}

impl AzureOpenaiSttSpec {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.base_url.trim().is_empty() {
            return Err(ConfigError::InvalidSpec(
                "azure_openai_stt: base_url is required".into(),
            ));
        }
        self.stt.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcpg_backend_llm_shared::PromptSpec;
    use serde_json::json;

    fn minimal_chat_exec() -> ChatExecutionSpec {
        ChatExecutionSpec {
            model: "gpt-4o-mini".into(),
            timeout_ms: 30_000,
            connect_timeout_ms: 5_000,
            prompt: PromptSpec {
                system: "you are helpful".into(),
                user: "{{ input.text }}".into(),
                ..Default::default()
            },
            sampling: Default::default(),
            response_format: Default::default(),
            tools: Default::default(),
            streaming: Default::default(),
            retry: Default::default(),
            guardrails: Default::default(),
            cache: Default::default(),
            budget: Default::default(),
        }
    }

    #[test]
    fn openai_default_base_url() {
        let s = OpenAiChatSpec {
            base_url: None,
            api_key: ApiKeyRef::new("k"),
            chat: minimal_chat_exec(),
        };
        assert_eq!(s.resolved_base_url(), "https://api.openai.com/v1");
        s.validate().unwrap();
    }

    #[test]
    fn openai_override_base_url() {
        let s = OpenAiChatSpec {
            base_url: Some("https://example.com/v1".into()),
            api_key: ApiKeyRef::new("k"),
            chat: minimal_chat_exec(),
        };
        assert_eq!(s.resolved_base_url(), "https://example.com/v1");
    }

    #[test]
    fn azure_requires_base_url() {
        let s = AzureOpenaiChatSpec {
            base_url: "  ".into(),
            api_key: ApiKeyRef::new("k"),
            chat: minimal_chat_exec(),
        };
        assert!(s.validate().is_err());

        let s = AzureOpenaiChatSpec {
            base_url: "https://r.openai.azure.com/...".into(),
            api_key: ApiKeyRef::new("k"),
            chat: minimal_chat_exec(),
        };
        s.validate().unwrap();
    }

    #[test]
    fn json_round_trip_openai() {
        let json = json!({
            "model": "gpt-4o-mini",
            "api_key": "k",
            "prompt": { "system": "x", "user": "y" }
        });
        let s: OpenAiChatSpec = serde_json::from_value(json).unwrap();
        assert!(s.base_url.is_none());
        assert_eq!(s.chat.model, "gpt-4o-mini");
        s.validate().unwrap();
    }

    #[test]
    fn json_round_trip_azure() {
        let json = json!({
            "base_url": "https://r.openai.azure.com/...?api-version=2024-08-06",
            "model": "gpt-4o",
            "api_key": "k",
            "prompt": { "system": "x", "user": "y" }
        });
        let s: AzureOpenaiChatSpec = serde_json::from_value(json).unwrap();
        assert_eq!(s.chat.model, "gpt-4o");
        s.validate().unwrap();
    }

    // ----- Embedding specs -----

    #[test]
    fn openai_embedding_default_base_url() {
        let s = OpenAiEmbeddingSpec {
            base_url: None,
            api_key: ApiKeyRef::new("k"),
            embedding: EmbeddingExecutionSpec {
                model: "text-embedding-3-small".into(),
                ..Default::default()
            },
        };
        assert_eq!(s.resolved_base_url(), "https://api.openai.com/v1");
        s.validate().unwrap();
    }

    #[test]
    fn azure_openai_embedding_requires_base_url() {
        let s = AzureOpenaiEmbeddingSpec {
            base_url: "  ".into(),
            api_key: ApiKeyRef::new("k"),
            embedding: EmbeddingExecutionSpec {
                model: "text-embedding-3-small".into(),
                ..Default::default()
            },
        };
        assert!(s.validate().is_err());

        let s = AzureOpenaiEmbeddingSpec {
            base_url: "https://r.openai.azure.com/...".into(),
            api_key: ApiKeyRef::new("k"),
            embedding: EmbeddingExecutionSpec {
                model: "text-embedding-3-small".into(),
                ..Default::default()
            },
        };
        s.validate().unwrap();
    }

    #[test]
    fn openai_embedding_json_round_trip() {
        let v = json!({
            "model": "text-embedding-3-small",
            "api_key": "k",
            "dimensions": 512,
        });
        let s: OpenAiEmbeddingSpec = serde_json::from_value(v).unwrap();
        s.validate().unwrap();
        assert_eq!(s.embedding.model, "text-embedding-3-small");
        assert_eq!(s.embedding.dimensions, Some(512));
    }
}
