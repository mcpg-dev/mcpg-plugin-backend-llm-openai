//! OpenAI / Azure OpenAI image-generation adapter.
//!
//! Endpoint: `POST {base_url}/images/generations` (OpenAI) or the
//! operator-supplied Azure deployment URL. Same auth header rules
//! as the chat / embedding adapters — Azure uses `api-key`, OpenAI
//! uses `Authorization: Bearer …`.
//!
//! Wire format:
//!
//! ```json
//! // Request
//! { "model": "dall-e-3",
//!   "prompt": "a cat sitting on a chair",
//!   "n": 1,
//!   "size": "1024x1024",
//!   "quality": "standard",
//!   "style": "natural",
//!   "response_format": "b64_json"
//! }
//!
//! // Response
//! { "created": 1700000000,
//!   "data": [
//!     { "b64_json": "iVBORw0KGgo…", "revised_prompt": "..." }
//!   ]
//! }
//! ```
//!
//! Always requests `b64_json` so the engine has bytes in hand for
//! `ContentStore` push without a follow-up HTTP fetch. URL
//! responses are accepted as a fallback (some compat servers
//! ignore `response_format`).

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine;
use mcpg_backend_llm_shared::error::ProviderError;
use mcpg_backend_llm_shared::image::{
    GeneratedImage, ImageProviderAdapter, NormalizedImageRequest, NormalizedImageResponse,
};
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};

use crate::adapter::OpenAiVariant;

pub struct OpenAiImageAdapter {
    client: Client,
    base_url: String,
    api_key: Arc<str>,
    variant: OpenAiVariant,
    label: &'static str,
}

impl std::fmt::Debug for OpenAiImageAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiImageAdapter")
            .field("variant", &self.variant)
            .field("label", &self.label)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl OpenAiImageAdapter {
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
            // Azure URL points at the per-deployment images endpoint
            // already, including `?api-version=…`.
            OpenAiVariant::Azure => self.base_url.clone(),
            _ => format!("{}/images/generations", self.base_url),
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
impl ImageProviderAdapter for OpenAiImageAdapter {
    fn label(&self) -> &'static str {
        self.label
    }

    async fn generate(
        &self,
        request: &NormalizedImageRequest,
        timeout: Duration,
    ) -> Result<NormalizedImageResponse, ProviderError> {
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
        decode_response(&value, &self.client, timeout).await
    }
}

fn encode_request(request: &NormalizedImageRequest) -> Value {
    let mut body = json!({
        "model": request.model,
        "prompt": request.prompt,
        "n": request.n,
        "response_format": "b64_json",
    });
    if let Some(s) = request.size.as_deref() {
        body["size"] = json!(s);
    }
    if let Some(q) = request.quality.as_deref() {
        body["quality"] = json!(q);
    }
    if let Some(st) = request.style.as_deref() {
        body["style"] = json!(st);
    }
    // gpt-image-1 accepts `output_format`. DALL-E 2/3 reject it
    // with a 400; the operator gets a clean error in that case
    // — we don't filter by model name here.
    if let Some(fmt) = request.output_format.as_deref() {
        body["output_format"] = json!(fmt);
    }
    body
}

async fn decode_response(
    value: &Value,
    http_client: &Client,
    fetch_timeout: Duration,
) -> Result<NormalizedImageResponse, ProviderError> {
    let data = value
        .get("data")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ProviderError::Malformed {
            message: "image response missing `data`".into(),
        })?;

    let mut images: Vec<GeneratedImage> = Vec::with_capacity(data.len());
    for entry in data {
        let revised_prompt = entry
            .get("revised_prompt")
            .and_then(|v| v.as_str())
            .map(|s| s.to_owned());
        let mime_type = entry
            .get("mime_type")
            .and_then(|v| v.as_str())
            .unwrap_or("image/png")
            .to_owned();

        if let Some(b64) = entry.get("b64_json").and_then(|v| v.as_str()) {
            let raw = base64::engine::general_purpose::STANDARD
                .decode(b64)
                .map_err(|e| ProviderError::Malformed {
                    message: format!("decode b64_json: {e}"),
                })?;
            images.push(GeneratedImage {
                bytes: bytes::Bytes::from(raw),
                mime_type,
                revised_prompt,
            });
        } else if let Some(url) = entry.get("url").and_then(|v| v.as_str()) {
            // Some compat endpoints ignore `response_format` and
            // return a temporary URL — fetch the bytes inline so the
            // engine still gets a `Bytes` to push into ContentStore.
            let r = http_client
                .get(url)
                .timeout(fetch_timeout)
                .send()
                .await
                .map_err(|e| ProviderError::Network {
                    message: format!("fetch image url: {e}"),
                })?;
            let status = r.status();
            let body = r.bytes().await.map_err(|e| ProviderError::Network {
                message: format!("read image url body: {e}"),
            })?;
            if !status.is_success() {
                return Err(ProviderError::Server {
                    message: format!("upstream image fetch returned {status}"),
                });
            }
            images.push(GeneratedImage {
                bytes: body,
                mime_type,
                revised_prompt,
            });
        } else {
            return Err(ProviderError::Malformed {
                message: "image entry has neither `b64_json` nor `url`".into(),
            });
        }
    }
    Ok(NormalizedImageResponse { images })
}

fn map_status_error(status: reqwest::StatusCode, body: &[u8]) -> ProviderError {
    let body_str = String::from_utf8_lossy(body).to_string();
    match status.as_u16() {
        401 | 403 => ProviderError::AuthFailed { message: body_str },
        429 => ProviderError::RateLimited { message: body_str },
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
    use base64::Engine;

    #[test]
    fn encode_request_includes_optional_fields() {
        let r = NormalizedImageRequest {
            model: "dall-e-3".into(),
            prompt: "a cat".into(),
            n: 1,
            size: Some("1024x1024".into()),
            quality: Some("hd".into()),
            style: Some("vivid".into()),
            seed: None,
            negative_prompt: None,
            output_format: None,
        };
        let body = encode_request(&r);
        assert_eq!(body["model"], "dall-e-3");
        assert_eq!(body["prompt"], "a cat");
        assert_eq!(body["n"], 1);
        assert_eq!(body["size"], "1024x1024");
        assert_eq!(body["quality"], "hd");
        assert_eq!(body["style"], "vivid");
        assert_eq!(body["response_format"], "b64_json");
    }

    #[test]
    fn encode_request_omits_optional_fields_when_unset() {
        let r = NormalizedImageRequest {
            model: "dall-e-3".into(),
            prompt: "a cat".into(),
            n: 1,
            size: None,
            quality: None,
            style: None,
            seed: None,
            negative_prompt: None,
            output_format: None,
        };
        let body = encode_request(&r);
        assert!(body.get("size").is_none());
        assert!(body.get("quality").is_none());
        assert!(body.get("style").is_none());
        assert!(body.get("output_format").is_none());
    }

    #[test]
    fn encode_request_passes_output_format_through() {
        let r = NormalizedImageRequest {
            model: "gpt-image-1".into(),
            prompt: "a cat".into(),
            n: 1,
            size: None,
            quality: None,
            style: None,
            seed: None,
            negative_prompt: None,
            output_format: Some("webp".into()),
        };
        let body = encode_request(&r);
        assert_eq!(body["output_format"], "webp");
    }

    #[tokio::test]
    async fn decode_response_parses_b64_json() {
        let payload = b"hello image bytes";
        let b64 = base64::engine::general_purpose::STANDARD.encode(payload);
        let raw = json!({
            "created": 1700000000,
            "data": [
                { "b64_json": b64, "revised_prompt": "cat sitting on a chair" }
            ]
        });
        let client = Client::new();
        let r = decode_response(&raw, &client, Duration::from_secs(1))
            .await
            .unwrap();
        assert_eq!(r.images.len(), 1);
        assert_eq!(r.images[0].bytes.as_ref(), payload);
        assert_eq!(r.images[0].mime_type, "image/png");
        assert_eq!(
            r.images[0].revised_prompt.as_deref(),
            Some("cat sitting on a chair")
        );
    }

    #[tokio::test]
    async fn decode_response_handles_missing_revised_prompt() {
        let payload = b"x";
        let b64 = base64::engine::general_purpose::STANDARD.encode(payload);
        let raw = json!({
            "data": [{ "b64_json": b64 }]
        });
        let r = decode_response(&raw, &Client::new(), Duration::from_secs(1))
            .await
            .unwrap();
        assert!(r.images[0].revised_prompt.is_none());
    }

    #[tokio::test]
    async fn decode_response_rejects_entry_without_bytes_or_url() {
        let raw = json!({
            "data": [ {} ]
        });
        let err = decode_response(&raw, &Client::new(), Duration::from_secs(1))
            .await
            .unwrap_err();
        assert!(matches!(err, ProviderError::Malformed { .. }));
    }

    #[tokio::test]
    async fn decode_response_rejects_missing_data_field() {
        let raw = json!({});
        let err = decode_response(&raw, &Client::new(), Duration::from_secs(1))
            .await
            .unwrap_err();
        assert!(matches!(err, ProviderError::Malformed { .. }));
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
    fn endpoint_url_appends_images_generations_for_openai() {
        let a = OpenAiImageAdapter::new(
            OpenAiVariant::OpenAi,
            "openai",
            "https://api.openai.com/v1",
            "k",
            Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(
            a.endpoint_url(),
            "https://api.openai.com/v1/images/generations"
        );
    }

    #[test]
    fn endpoint_url_keeps_azure_url_as_is() {
        let a = OpenAiImageAdapter::new(
            OpenAiVariant::Azure,
            "azure-openai",
            "https://r.openai.azure.com/openai/deployments/dalle/images/generations?api-version=2024-02-01",
            "k",
            Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(
            a.endpoint_url(),
            "https://r.openai.azure.com/openai/deployments/dalle/images/generations?api-version=2024-02-01"
        );
    }
}
