//! OpenAI / Azure OpenAI STT adapter (Whisper).
//!
//! Endpoint: `POST {base_url}/audio/transcriptions` (OpenAI) or the
//! operator-supplied Azure deployment URL. Multipart body — fields:
//!
//! - `file` — the audio bytes, with a filename hint (extension
//!   derived from MIME so Whisper picks the right decoder).
//! - `model` — `whisper-1`.
//! - `response_format` — `verbose_json` so we get language +
//!   duration alongside the transcript.
//! - `language` — optional ISO-639-1 hint.
//! - `prompt` — optional bias string.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use mcpg_backend_llm_shared::audio::{
    NormalizedSttRequest, NormalizedSttResponse, SttProviderAdapter,
};
use mcpg_backend_llm_shared::error::ProviderError;
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;

use crate::adapter::OpenAiVariant;

pub struct OpenAiSttAdapter {
    client: Client,
    base_url: String,
    api_key: Arc<str>,
    variant: OpenAiVariant,
    label: &'static str,
}

impl std::fmt::Debug for OpenAiSttAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiSttAdapter")
            .field("variant", &self.variant)
            .field("label", &self.label)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl OpenAiSttAdapter {
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
            _ => format!("{}/audio/transcriptions", self.base_url),
        }
    }

    fn build_headers(&self) -> Result<HeaderMap, ProviderError> {
        // Multipart bodies set Content-Type at request build time;
        // we only set the auth header here.
        let mut h = HeaderMap::new();
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
impl SttProviderAdapter for OpenAiSttAdapter {
    fn label(&self) -> &'static str {
        self.label
    }

    async fn transcribe(
        &self,
        request: &NormalizedSttRequest,
        timeout: Duration,
    ) -> Result<NormalizedSttResponse, ProviderError> {
        let url = self.endpoint_url();
        let headers = self.build_headers()?;

        let filename = filename_for_mime(&request.mime_type);
        let part = reqwest::multipart::Part::bytes(request.bytes.to_vec())
            .file_name(filename)
            .mime_str(&request.mime_type)
            .map_err(|e| ProviderError::BadRequest {
                message: format!("invalid mime_type {}: {e}", request.mime_type),
            })?;
        let mut form = reqwest::multipart::Form::new()
            .text("model", request.model.clone())
            .text("response_format", "verbose_json")
            .part("file", part);
        if let Some(lang) = request.language.as_deref() {
            form = form.text("language", lang.to_owned());
        }
        if let Some(prompt) = request.prompt.as_deref() {
            form = form.text("prompt", prompt.to_owned());
        }

        let resp = self
            .client
            .post(&url)
            .headers(headers)
            .timeout(timeout)
            .multipart(form)
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

fn decode_response(value: &Value) -> Result<NormalizedSttResponse, ProviderError> {
    let text = value
        .get("text")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ProviderError::Malformed {
            message: "transcription response missing `text`".into(),
        })?
        .to_owned();
    let language = value
        .get("language")
        .and_then(|v| v.as_str())
        .map(|s| s.to_owned());
    let duration_seconds = value.get("duration").and_then(|v| v.as_f64());
    Ok(NormalizedSttResponse {
        text,
        language,
        duration_seconds,
    })
}

/// Whisper expects a filename so it can pick the right decoder
/// (mp3 / wav / m4a / etc.). Derive a sane extension from the
/// MIME type; fall back to `.bin` for the rare unknown cases.
fn filename_for_mime(mime: &str) -> &'static str {
    match mime {
        "audio/mpeg" | "audio/mp3" => "audio.mp3",
        "audio/wav" | "audio/x-wav" => "audio.wav",
        "audio/flac" => "audio.flac",
        "audio/ogg" | "audio/opus" => "audio.ogg",
        "audio/mp4" | "audio/x-m4a" => "audio.m4a",
        "audio/webm" => "audio.webm",
        _ => "audio.bin",
    }
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
    use serde_json::json;

    #[test]
    fn decode_response_parses_verbose_json() {
        let raw = json!({
            "text": "hello world",
            "language": "en",
            "duration": 1.5
        });
        let r = decode_response(&raw).unwrap();
        assert_eq!(r.text, "hello world");
        assert_eq!(r.language.as_deref(), Some("en"));
        assert_eq!(r.duration_seconds, Some(1.5));
    }

    #[test]
    fn decode_response_handles_basic_json() {
        let raw = json!({"text": "ok"});
        let r = decode_response(&raw).unwrap();
        assert_eq!(r.text, "ok");
        assert!(r.language.is_none());
        assert!(r.duration_seconds.is_none());
    }

    #[test]
    fn decode_response_rejects_missing_text() {
        let err = decode_response(&json!({})).unwrap_err();
        assert!(matches!(err, ProviderError::Malformed { .. }));
    }

    #[test]
    fn filename_for_mime_maps_known_types() {
        assert_eq!(filename_for_mime("audio/mpeg"), "audio.mp3");
        assert_eq!(filename_for_mime("audio/wav"), "audio.wav");
        assert_eq!(filename_for_mime("audio/flac"), "audio.flac");
        assert_eq!(filename_for_mime("audio/ogg"), "audio.ogg");
        assert_eq!(filename_for_mime("audio/x-m4a"), "audio.m4a");
        assert_eq!(filename_for_mime("application/octet-stream"), "audio.bin");
    }

    #[test]
    fn endpoint_url_appends_transcriptions_for_openai() {
        let a = OpenAiSttAdapter::new(
            OpenAiVariant::OpenAi,
            "openai",
            "https://api.openai.com/v1",
            "k",
            Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(
            a.endpoint_url(),
            "https://api.openai.com/v1/audio/transcriptions"
        );
    }

    #[test]
    fn map_status_401_auth_failed() {
        let e = map_status_error(reqwest::StatusCode::from_u16(401).unwrap(), b"");
        assert!(matches!(e, ProviderError::AuthFailed { .. }));
    }
}
