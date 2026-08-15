//! # mcpg-plugin-backend-llm-openai
//!
//! OpenAI + Azure OpenAI chat-completion binding plugins for MCPG.
//!
//! Ships two `BackendPlugin` instances:
//!
//! - [`OpenAiChatPlugin`] (`kind: "openai.chat"`) — public OpenAI API.
//! - [`AzureOpenaiChatPlugin`] (`kind: "azure_openai.chat"`) — Azure
//!   OpenAI deployments.
//!
//! Both delegate execution to the shared
//! [`mcpg_backend_llm_shared::ChatEngine`]; only the wire-format
//! adapter ([`OpenAiAdapter`]) is provider-specific. The adapter is
//! exported `pub` so the sibling `mcpg-plugin-backend-llm-compat`
//! crate can reuse it for arbitrary OpenAI-compatible endpoints.

mod adapter;
/// cdylib sync bridge + `declare_plugin!` export (backend-plugin-migration).
/// Additive: the gateway keeps using the static `new()` + `set_host_handle`
/// path. The `mcpg_plugin_register` FFI symbol is gated behind the
/// `cdylib-export` feature inside the macro expansion. Public so the
/// wrapper types + macro-generated entity modules are part of the
/// crate's public surface (mirrors the nats / kafka pilots, which keep
/// their bridges at crate root) — this also keeps the wrappers from
/// tripping `dead_code` on the default rlib build where neither
/// `cdylib-export` nor `static-firstparty` references them.
pub mod cdylib;
mod config;
mod embedding_adapter;
mod embedding_plugin;
mod host_handle_obs;
mod image_adapter;
mod image_plugin;
mod plugin;
mod stt_adapter;
mod stt_plugin;
mod tts_adapter;
mod tts_plugin;

pub use adapter::{OpenAiAdapter, OpenAiVariant};
pub use config::{
    AzureOpenaiChatSpec, AzureOpenaiEmbeddingSpec, AzureOpenaiImageSpec, AzureOpenaiSttSpec,
    AzureOpenaiTtsSpec, OpenAiChatSpec, OpenAiEmbeddingSpec, OpenAiImageSpec, OpenAiSttSpec,
    OpenAiTtsSpec,
};
pub use embedding_adapter::{OPENAI_MAX_INPUTS, OpenAiEmbeddingAdapter};
pub use embedding_plugin::{AzureOpenaiEmbeddingPlugin, OpenAiEmbeddingPlugin};
pub use image_adapter::OpenAiImageAdapter;
pub use image_plugin::{AzureOpenaiImagePlugin, OpenAiImagePlugin};
pub use plugin::{AzureOpenaiChatPlugin, OpenAiChatPlugin};
pub use stt_adapter::OpenAiSttAdapter;
pub use stt_plugin::{AzureOpenaiSttPlugin, OpenAiSttPlugin};
pub use tts_adapter::OpenAiTtsAdapter;
pub use tts_plugin::{AzureOpenaiTtsPlugin, OpenAiTtsPlugin};
