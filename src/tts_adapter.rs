//! OpenAI / Azure OpenAI TTS adapter.
//!
//! Endpoint: `POST {base_url}/audio/speech` (OpenAI) or the
//! operator-supplied Azure deployment URL. Request is JSON; the
//! response body is raw audio bytes (Content-Type matches the
//! requested format).

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use mcpg_backend_llm_shared::audio::{
    NormalizedTtsRequest, NormalizedTtsResponse, TtsProviderAdapter,
};
use mcpg_backend_llm_shared::error::ProviderError;
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};

use crate::adapter::OpenAiVariant;

pub struct OpenAiTtsAdapter {
    client: Client,
    base_url: String,
    api_key: Arc<str>,
    variant: OpenAiVariant,
    label: &'static str,
}

impl std::fmt::Debug for OpenAiTtsAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiTtsAdapter")
            .field("variant", &self.variant)
            .field("label", &self.label)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl OpenAiTtsAdapter {
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
            OpenAiVariant::Azure => self.base_url.clone(),
            _ => format!("{}/audio/speech", self.base_url),
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
impl TtsProviderAdapter for OpenAiTtsAdapter {
    fn label(&self) -> &'static str {
        self.label
    }

    async fn synthesize(
        &self,
        request: &NormalizedTtsRequest,
        timeout: Duration,
    ) -> Result<NormalizedTtsResponse, ProviderError> {
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
        // Capture upstream content-type before consuming the body —
        // it's the most reliable MIME source.
        let upstream_mime = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_owned());
        let bytes = resp.bytes().await.map_err(|e| ProviderError::Network {
            message: format!("read body: {e}"),
        })?;

        if !status.is_success() {
            return Err(map_status_error(status, &bytes));
        }

        let mime_type = upstream_mime
            .filter(|m| m.starts_with("audio/"))
            .unwrap_or_else(|| request.format.mime_type().to_owned());

        Ok(NormalizedTtsResponse { bytes, mime_type })
    }
}

fn encode_request(request: &NormalizedTtsRequest) -> Value {
    let format_str = match request.format {
        mcpg_backend_llm_shared::normalized::AudioFormat::Mp3 => "mp3",
        mcpg_backend_llm_shared::normalized::AudioFormat::Wav => "wav",
        mcpg_backend_llm_shared::normalized::AudioFormat::Flac => "flac",
        mcpg_backend_llm_shared::normalized::AudioFormat::Ogg => "opus",
        mcpg_backend_llm_shared::normalized::AudioFormat::Aac => "aac",
        mcpg_backend_llm_shared::normalized::AudioFormat::Pcm => "pcm",
    };
    let mut body = json!({
        "model": request.model,
        "input": request.text,
        "voice": request.voice,
        "response_format": format_str,
    });
    if let Some(s) = request.speed {
        body["speed"] = json!(s);
    }
    body
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
    use mcpg_backend_llm_shared::normalized::AudioFormat;

    #[test]
    fn encode_request_includes_speed_when_set() {
        let r = NormalizedTtsRequest {
            model: "tts-1".into(),
            text: "hello".into(),
            voice: "alloy".into(),
            format: AudioFormat::Mp3,
            speed: Some(1.5),
        };
        let body = encode_request(&r);
        assert_eq!(body["model"], "tts-1");
        assert_eq!(body["input"], "hello");
        assert_eq!(body["voice"], "alloy");
        assert_eq!(body["response_format"], "mp3");
        assert_eq!(body["speed"], 1.5);
    }

    #[test]
    fn encode_request_maps_ogg_to_opus() {
        let r = NormalizedTtsRequest {
            model: "tts-1".into(),
            text: "hello".into(),
            voice: "alloy".into(),
            format: AudioFormat::Ogg,
            speed: None,
        };
        let body = encode_request(&r);
        assert_eq!(body["response_format"], "opus");
    }

    #[test]
    fn endpoint_url_appends_audio_speech_for_openai() {
        let a = OpenAiTtsAdapter::new(
            OpenAiVariant::OpenAi,
            "openai",
            "https://api.openai.com/v1",
            "k",
            Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(a.endpoint_url(), "https://api.openai.com/v1/audio/speech");
    }

    #[test]
    fn map_status_429_rate_limited() {
        let e = map_status_error(reqwest::StatusCode::from_u16(429).unwrap(), b"");
        assert!(matches!(e, ProviderError::RateLimited { .. }));
    }
}
