//! OpenAI / Azure OpenAI / OpenAI-compatible embedding adapter.
//!
//! Endpoint: `POST {base_url}/embeddings` (OpenAI / compat) or the
//! operator-supplied Azure deployment URL. Same auth header rules
//! as the chat adapter — Azure uses `api-key`, the others use
//! `Authorization: Bearer …`.
//!
//! Wire format:
//!
//! ```json
//! // Request
//! { "model": "text-embedding-3-small",
//!   "input": ["...","..."],
//!   "dimensions": 1536  // optional
//! }
//!
//! // Response
//! { "object": "list",
//!   "data": [
//!     { "object": "embedding", "index": 0, "embedding": [...] },
//!     ...
//!   ],
//!   "usage": { "prompt_tokens": 17, "total_tokens": 17 },
//!   "model": "text-embedding-3-small"
//! }
//! ```

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use mcpg_backend_llm_shared::embedding::{
    EmbeddingProviderAdapter, EmbeddingTokenUsage, NormalizedEmbeddingRequest,
    NormalizedEmbeddingResponse,
};
use mcpg_backend_llm_shared::error::ProviderError;
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};

use crate::adapter::OpenAiVariant;

/// Per-provider hard cap on inputs per request. OpenAI accepts up
/// to 2048 inputs per call across 3-series + ada-002 models. Azure
/// matches OpenAI. Compatible endpoints (vLLM, Together, Groq) are
/// commonly more conservative; we surface the OpenAI ceiling and
/// rely on operator-side `max_batch_size` to ratchet down.
pub const OPENAI_MAX_INPUTS: usize = 2048;

pub struct OpenAiEmbeddingAdapter {
    client: Client,
    base_url: String,
    api_key: Arc<str>,
    variant: OpenAiVariant,
    /// Static metrics label — `openai`, `azure-openai`, `openai-compatible`.
    label: &'static str,
}

impl std::fmt::Debug for OpenAiEmbeddingAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiEmbeddingAdapter")
            .field("variant", &self.variant)
            .field("label", &self.label)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl OpenAiEmbeddingAdapter {
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

    fn endpoint_url(&self) -> String {
        match self.variant {
            // Azure operators set `base_url` to the full deployment
            // URL with `?api-version=…` already appended; we don't
            // add `/embeddings` because the URL points at the
            // embeddings deployment specifically (Azure's path is
            // `/openai/deployments/{deploy}/embeddings`).
            OpenAiVariant::Azure => self.base_url.clone(),
            _ => format!("{}/embeddings", self.base_url),
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
impl EmbeddingProviderAdapter for OpenAiEmbeddingAdapter {
    fn label(&self) -> &'static str {
        self.label
    }

    fn max_batch_size(&self) -> usize {
        OPENAI_MAX_INPUTS
    }

    async fn embed(
        &self,
        request: &NormalizedEmbeddingRequest,
        timeout: Duration,
    ) -> Result<NormalizedEmbeddingResponse, ProviderError> {
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
            .map_err(|e| ProviderError::Network {
                message: format!("send: {e}"),
            })?;

        let status = resp.status();
        let bytes = resp.bytes().await.map_err(|e| ProviderError::Network {
            message: format!("read body: {e}"),
        })?;

        if !status.is_success() {
            return Err(map_status_error(status, &bytes));
        }

        let value: Value =
            serde_json::from_slice(&bytes).map_err(|e| ProviderError::Malformed {
                message: format!("parse response json: {e}"),
            })?;
        decode_response(&value)
    }
}

fn encode_request(request: &NormalizedEmbeddingRequest) -> Value {
    let mut body = json!({
        "model": request.model,
        "input": request.inputs,
    });
    if let Some(d) = request.dimensions {
        body["dimensions"] = json!(d);
    }
    body
}

fn decode_response(value: &Value) -> Result<NormalizedEmbeddingResponse, ProviderError> {
    let data = value
        .get("data")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ProviderError::Malformed {
            message: "response missing `data`".into(),
        })?;

    let mut indexed: Vec<(usize, Vec<f32>)> = Vec::with_capacity(data.len());
    let mut dimensions: u32 = 0;
    for entry in data {
        let index =
            entry
                .get("index")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| ProviderError::Malformed {
                    message: "embedding entry missing `index`".into(),
                })? as usize;
        let arr = entry
            .get("embedding")
            .and_then(|v| v.as_array())
            .ok_or_else(|| ProviderError::Malformed {
                message: "embedding entry missing `embedding`".into(),
            })?;
        let mut vec_f32 = Vec::with_capacity(arr.len());
        for e in arr {
            let f = e.as_f64().ok_or_else(|| ProviderError::Malformed {
                message: "embedding contains non-number".into(),
            })?;
            vec_f32.push(f as f32);
        }
        if dimensions == 0 {
            dimensions = vec_f32.len() as u32;
        } else if vec_f32.len() as u32 != dimensions {
            return Err(ProviderError::Malformed {
                message: "embeddings have inconsistent dimensions".into(),
            });
        }
        indexed.push((index, vec_f32));
    }
    // Provider returns entries in input order in practice, but the
    // spec only guarantees `index` is present — sort by it to be
    // strict.
    indexed.sort_by_key(|(i, _)| *i);
    let embeddings: Vec<Vec<f32>> = indexed.into_iter().map(|(_, v)| v).collect();

    let usage = value
        .get("usage")
        .and_then(|u| u.get("prompt_tokens").and_then(|v| v.as_u64()))
        .map(|n| EmbeddingTokenUsage {
            input_tokens: n as u32,
        });

    Ok(NormalizedEmbeddingResponse {
        embeddings,
        dimensions,
        usage,
    })
}

/// Same status-code → `ProviderError` mapping as the chat adapter.
/// Keep in sync if either side adds a new variant.
fn map_status_error(status: reqwest::StatusCode, body: &[u8]) -> ProviderError {
    let body_str = String::from_utf8_lossy(body).to_string();
    match status.as_u16() {
        401 | 403 => ProviderError::AuthFailed { message: body_str },
        429 => ProviderError::RateLimited { message: body_str },
        400 if body_str.to_lowercase().contains("token") => {
            ProviderError::ContextLimit { message: body_str }
        }
        400..=499 => ProviderError::BadRequest { message: body_str },
        500..=599 => ProviderError::Server { message: body_str },
        _ => ProviderError::Network {
            message: format!("unexpected status {status}: {body_str}"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_request_includes_dimensions_when_set() {
        let r = NormalizedEmbeddingRequest {
            model: "text-embedding-3-small".into(),
            inputs: vec!["hello".into(), "world".into()],
            dimensions: Some(512),
        };
        let body = encode_request(&r);
        assert_eq!(body["model"], "text-embedding-3-small");
        assert_eq!(body["input"][0], "hello");
        assert_eq!(body["input"][1], "world");
        assert_eq!(body["dimensions"], 512);
    }

    #[test]
    fn encode_request_omits_dimensions_when_unset() {
        let r = NormalizedEmbeddingRequest {
            model: "text-embedding-3-small".into(),
            inputs: vec!["hello".into()],
            dimensions: None,
        };
        let body = encode_request(&r);
        assert!(body.get("dimensions").is_none());
    }

    #[test]
    fn decode_response_parses_well_formed_data() {
        let raw = json!({
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]}
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 7, "total_tokens": 7}
        });
        let r = decode_response(&raw).unwrap();
        assert_eq!(r.dimensions, 3);
        assert_eq!(r.embeddings.len(), 2);
        assert_eq!(r.embeddings[0][0], 0.1_f32);
        assert_eq!(r.embeddings[1][2], 0.6_f32);
        assert_eq!(r.usage.unwrap().input_tokens, 7);
    }

    #[test]
    fn decode_response_sorts_by_index() {
        let raw = json!({
            "data": [
                {"index": 1, "embedding": [9.0]},
                {"index": 0, "embedding": [1.0]}
            ]
        });
        let r = decode_response(&raw).unwrap();
        assert_eq!(r.embeddings[0][0], 1.0_f32);
        assert_eq!(r.embeddings[1][0], 9.0_f32);
    }

    #[test]
    fn decode_response_rejects_inconsistent_dimensions() {
        let raw = json!({
            "data": [
                {"index": 0, "embedding": [0.1, 0.2]},
                {"index": 1, "embedding": [0.3, 0.4, 0.5]}
            ]
        });
        let err = decode_response(&raw).unwrap_err();
        assert!(matches!(err, ProviderError::Malformed { .. }));
    }

    #[test]
    fn decode_response_handles_missing_usage() {
        let raw = json!({
            "data": [
                {"index": 0, "embedding": [0.1]}
            ]
        });
        let r = decode_response(&raw).unwrap();
        assert!(r.usage.is_none());
    }

    #[test]
    fn map_status_429_rate_limited() {
        let e = map_status_error(reqwest::StatusCode::from_u16(429).unwrap(), b"slow down");
        assert!(matches!(e, ProviderError::RateLimited { .. }));
    }

    #[test]
    fn map_status_401_auth_failed() {
        let e = map_status_error(reqwest::StatusCode::from_u16(401).unwrap(), b"bad key");
        assert!(matches!(e, ProviderError::AuthFailed { .. }));
    }

    #[test]
    fn map_status_400_token_overflow_is_context_limit() {
        let e = map_status_error(
            reqwest::StatusCode::from_u16(400).unwrap(),
            b"input exceeds maximum token limit",
        );
        assert!(matches!(e, ProviderError::ContextLimit { .. }));
    }

    #[test]
    fn endpoint_url_appends_embeddings_for_openai() {
        let a = OpenAiEmbeddingAdapter::new(
            OpenAiVariant::OpenAi,
            "openai",
            "https://api.openai.com/v1",
            "k",
            Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(a.endpoint_url(), "https://api.openai.com/v1/embeddings");
    }

    #[test]
    fn endpoint_url_keeps_azure_url_as_is() {
        let a = OpenAiEmbeddingAdapter::new(
            OpenAiVariant::Azure,
            "azure-openai",
            "https://r.openai.azure.com/openai/deployments/emb/embeddings?api-version=2024-08-06",
            "k",
            Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(
            a.endpoint_url(),
            "https://r.openai.azure.com/openai/deployments/emb/embeddings?api-version=2024-08-06"
        );
    }
}
