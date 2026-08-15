# OpenAI + Azure OpenAI Backends — `dev.mcpg.backend.llm.openai`

> class `backend` · `native` · package `mcpg-plugin-backend-llm-openai` · artifact `libmcpg_plugin_backend_llm_openai.so` · Apache-2.0

Exposes the OpenAI platform as MCP capabilities: chat completions, embeddings,
image generation, text-to-speech and speech-to-text — each in a public-API and
an Azure-deployment flavour, ten backend entities in one artifact. A binding
pins one model and one execution policy; the plugin turns a tool call into an
upstream request and hands back a validated result. Reach for it when GPT models
should be reachable as governed, budgeted, audited MCP tools rather than an API
key distributed to every client. Azure operators use the `azure_openai_*` kinds,
which differ only in URL pattern and auth header.

## What it does
- Registers ten backend entities under one cdylib. Each self-describes its
  `BackendPlugin::kind()` at load time, so the gateway dispatches every binding
  to the right one.
- Renders `prompt.system` and `prompt.user` as MiniJinja templates over
  `input.*` (the caller's tool arguments) and `meta.*` (`backend_name`,
  `request_id`, `session_id`, `timestamp_iso8601`).
- Enforces structured chat output through OpenAI's native `json_schema`
  response format, then re-validates the reply binding-side.
- Runs a bounded agentic loop over child MCP tools named in `tools.allowed`,
  refusing any call the model invents outside that list before it leaves the
  plugin.
- Streams incremental chat tokens over SSE and accumulates token usage for the
  terminal event.
- Accepts image, audio and file parts in the user turn, resolving
  `mcpg-resource://` URIs, `data:` URLs, plain URLs and raw base64.
- Pushes generated image and speech bytes into the gateway's content store and
  returns `mcpg-resource://<id>` URIs, so tool results stay small.
- Splits large embedding batches across parallel calls, capped at OpenAI's
  2048-input ceiling.
- Retries rate-limit, 5xx and network failures with exponential backoff, and
  enforces per-binding token and daily-USD budget caps before spending.
- Declares the `network_outbound` capability — required in every mode, since
  every call is an outbound HTTPS request.

| `backend.kind` | Registry kind | Entity id | Surface |
|---|---|---|---|
| `openai_chat` | `openai.chat` | `dev.mcpg.backend.openai.chat` | chat completions |
| `azure_openai_chat` | `azure_openai.chat` | `dev.mcpg.backend.azure_openai.chat` | Azure chat completions |
| `openai_embedding` | `openai.embedding` | `dev.mcpg.backend.openai.embedding` | embeddings |
| `azure_openai_embedding` | `azure_openai.embedding` | `dev.mcpg.backend.azure_openai.embedding` | Azure embeddings |
| `openai_image` | `openai.image` | `dev.mcpg.backend.openai.image` | image generation |
| `azure_openai_image` | `azure_openai.image` | `dev.mcpg.backend.azure_openai.image` | Azure image generation |
| `openai_tts` | `openai.tts` | `dev.mcpg.backend.openai.tts` | text-to-speech |
| `azure_openai_tts` | `azure_openai.tts` | `dev.mcpg.backend.azure_openai.tts` | Azure text-to-speech |
| `openai_stt` | `openai.stt` | `dev.mcpg.backend.openai.stt` | speech-to-text |
| `azure_openai_stt` | `azure_openai.stt` | `dev.mcpg.backend.azure_openai.stt` | Azure speech-to-text |

## Configuration

Load the artifact once from the flat top-level `plugins:` list — all ten
entities come with it — then declare one binding per capability under
`mcp.capabilities.tools[]` (or `.prompts[]` / `.resources[]`), selecting the
entity with `backend.kind`. Everything else inside the `backend:` block is the
plugin's own spec, forwarded verbatim and validated by the plugin at boot, so an
invalid value fails gateway startup rather than the first call.

```yaml
plugins:
  - id: dev.mcpg.backend.llm.openai
    class: backend
    source:
      oci: ghcr.io/mcpg-dev/source-code/plugins/backend-llm-openai:protocol-1

mcp:
  capabilities:
    tools:
      - name: ticket.classify
        description: Classify a support ticket.
        input_schema:
          type: object
          properties:
            body: { type: string }
          required: [body]
        backend:
          kind: openai_chat
          api_key: "${env.OPENAI_API_KEY}"
          model: gpt-4o-mini
          prompt:
            system: You classify support tickets. Answer only as JSON.
            user: "{{ input.body }}"
          sampling:
            temperature: 0
          response_format:
            mode: json_schema
          # Read by the plugin when `response_format.mode: json_schema`.
          output_schema:
            type: object
            properties:
              category: { type: string }
              urgency:  { type: string }
            required: [category, urgency]
```

### Provider fields (every kind)

| Field | Type | Default | Description |
|---|---|---|---|
| `api_key` | string | *(required)* | Sent as `Authorization: Bearer …` on the OpenAI kinds and as the `api-key` header on the Azure kinds. Supply `${env.NAME}` or a `scheme://` URI bound to a `secret_provider` plugin (for example `vault://secret/openai#key`); the gateway substitutes the literal value at config load. An empty resolved value is rejected. |
| `base_url` | string | `https://api.openai.com/v1` on the OpenAI kinds; **required** on every `azure_openai_*` kind | Azure operators put the full per-deployment URL here, including `?api-version=…`. On the OpenAI kinds, override only for a forwarding proxy or a test fixture. |

### Chat execution fields (`openai_chat`, `azure_openai_chat`)

Shared with every other MCPG chat binding, so switching providers means changing
`kind` and `model` — not relearning the schema.

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | *(required)* | Model id, or the model behind the Azure deployment. |
| `prompt.system` | string | *(required)* | System-prompt template. Must be non-empty after trimming. |
| `prompt.user` | string | *(required)* | User-prompt template. Must be non-empty after trimming. |
| `prompt.image_inputs` | string[] | `[]` | Argument names carrying image content (URL, `data:` URL, raw base64, `mcpg-resource://` URI, or an explicit object). An array value fans out to several parts. |
| `prompt.audio_inputs` | string[] | `[]` | Argument names carrying audio; base64 sources become `input_audio` parts. |
| `prompt.file_inputs` | string[] | `[]` | Argument names carrying documents; object values may set `mime_type` and `filename`. |
| `timeout_ms` | integer | `60000` | Per-iteration wall-clock budget upstream, retries included. |
| `connect_timeout_ms` | integer | `5000` | TCP connect timeout, kept separate so a slow-but-connected upstream is not killed early. |
| `sampling.temperature` | number | *(unset)* | Passed through when set. |
| `sampling.top_p` | number | *(unset)* | Passed through when set. |
| `sampling.max_completion_tokens` | integer | *(unset)* | Per-iteration output cap. |
| `sampling.seed` | integer | *(unset)* | Passed through when set. |
| `response_format.mode` | `json_schema` \| `text` | `json_schema` | `text` wraps the reply as `{"text": "…"}` and skips validation. |
| `response_format.strict` | boolean | `true` | Sets OpenAI's `strict` flag on the JSON-schema response format; binding-side validation runs either way. |
| `response_format.on_mismatch` | `error` \| `retry_once` \| `return_raw` | `error` | `return_raw` is legal only with `mode: text`. |
| `tools.allowed` | string[] | `[]` | Names of other bindings in this gateway the model may call. Empty means single-shot. |
| `tools.max_iterations` | integer | `1` when `allowed` is empty, else `5` | Maximum model round-trips. Values above `50` are refused at boot. |
| `tools.tool_choice` | `auto` \| `required` \| `none` | `auto` | Provider-level tool-choice hint. |
| `tools.tool_result_max_bytes` | integer | `16384` | Each child result is truncated to this before re-entering the conversation. |
| `tools.on_iteration_exhausted` | `error` \| `return_partial` | `error` | What happens when the loop runs out of iterations. |
| `retry.max_attempts` | integer | `3` | Attempts per upstream call. |
| `retry.initial_backoff_ms` | integer | `500` | First backoff; must not exceed `max_backoff_ms`. |
| `retry.max_backoff_ms` | integer | `8000` | Backoff ceiling. |
| `retry.retry_on` | list of `rate_limited` \| `server` \| `network` | all three | Failure classes worth retrying. |
| `guardrails.max_output_tokens_per_iteration` | integer | *(unset)* | Hard cap that overrides `sampling.max_completion_tokens`. |
| `cache.enabled` | boolean | `false` | Opt-in response cache. Refused at boot together with a non-empty `tools.allowed`. |
| `cache.ttl_seconds` | integer | `3600000` | Per-entry TTL, in seconds. |
| `budget.tokens_per_call_cap` | integer | `0` (uncapped) | Total input + output tokens across all loop iterations of one call. Checked between iterations, never on the first. |
| `budget.usd_daily_cap` | number | `0` (uncapped) | Aggregate spend for this binding per UTC day, checked before each call. |
| `output_schema` | object | *(unset)* | JSON Schema the reply must satisfy under `mode: json_schema`. Read out of this `backend:` block, not the binding-level field. |

### Embedding fields (`openai_embedding`, `azure_openai_embedding`)

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | *(required)* | Embedding model id. |
| `dimensions` | integer | *(unset)* | Requests reduced vectors; honoured by the 3-series models. |
| `max_batch_size` | integer | provider cap | Per-call batch size, clamped to OpenAI's 2048-input ceiling. Larger inputs split into parallel calls. |
| `timeout_ms` | integer | `10000` | Per-call timeout. |
| `connect_timeout_ms` | integer | `5000` | TCP connect timeout. |
| `retry.max_attempts` | integer | `3` | Attempts per upstream call. |
| `retry.initial_backoff_ms` | integer | `200` | First backoff. |
| `retry.max_backoff_ms` | integer | `2000` | Backoff ceiling. |
| `cache.enabled` | boolean | `false` | Opt-in; `text → vector` is deterministic, so caching is sound. |
| `cache.ttl_seconds` | integer | `86400` | Per-entry TTL, in seconds. |

### Image fields (`openai_image`, `azure_openai_image`)

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | *(required)* | Image model id. |
| `timeout_ms` | integer | `60000` | Per-call timeout. |
| `connect_timeout_ms` | integer | `5000` | TCP connect timeout. |
| `defaults.size` | string | *(unset)* | Default `size`, overridable per call. |
| `defaults.quality` | string | *(unset)* | Default `quality`. |
| `defaults.style` | string | *(unset)* | Default `style`. |
| `defaults.n` | integer | *(unset)* | Default image count; the engine falls back to `1` when neither the binding nor the call sets it. Must be at least `1` when set. |
| `defaults.output_format` | string | *(unset)* | `png` \| `jpeg` \| `webp`; accepted by `gpt-image-1` and rejected upstream by DALL-E. |
| `retry.max_attempts` / `retry.initial_backoff_ms` / `retry.max_backoff_ms` | integer | `3` / `200` / `2000` | Same retry shape as embeddings. |

The shared image spec also carries `defaults.negative_prompt`, and the engine
accepts a per-call `seed`, for parity with the Gemini and Stability image
bindings; the OpenAI image request carries neither.

### Speech fields (`openai_tts`, `azure_openai_tts`, `openai_stt`, `azure_openai_stt`)

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | *(required)* | Speech model id. |
| `voice` | string | *(required on TTS)* | Default voice; a per-call `voice` argument overrides it. |
| `format` | audio format | `mp3` | Default TTS output format. |
| `speed` | number | *(unset)* | Default speed multiplier; must be between `0.25` and `4.0`. |
| `language` | string | *(unset, STT)* | Default ISO-639-1 hint for the recogniser. |
| `max_input_bytes` | integer | `20971520` | STT inline byte cap when the input is a URL or an `mcpg-resource://` URI. |
| `timeout_ms` | integer | `60000` | Per-call timeout. |
| `connect_timeout_ms` | integer | `5000` | TCP connect timeout. |
| `retry.max_attempts` / `retry.initial_backoff_ms` / `retry.max_backoff_ms` | integer | `3` / `200` / `2000` | Same retry shape as embeddings. |

## Operations

Each non-chat kind takes its own per-call argument shape and returns its own
envelope.

| Kind | Arguments | Result |
|---|---|---|
| `*_embedding` | `input` — a string or an array of strings | `{embeddings, dimensions, usage}`; `embeddings` always carries one entry per input |
| `*_image` | `prompt` (required), plus optional `size`, `quality`, `style`, `n`, `output_format` | `{images: [{image_uri, mime_type, revised_prompt?}]}`, always an array |
| `*_tts` | `text` (required), plus optional `voice`, `format`, `speed` | `{audio_uri, mime_type, format}` |
| `*_stt` | `audio` (required — an `mcpg-resource://` URI, an `https://` URL, raw base64, or an object form), plus optional `language`, `prompt` | `{text, language?, duration_seconds?}` |

Image and speech bytes never travel inline. The engine pushes them into the
gateway's content store and returns an `mcpg-resource://<id>` URI that clients
fetch with an MCP `resources/read`. The image adapter always asks OpenAI for
`response_format: b64_json` so it has the bytes in hand without a second fetch.

```yaml
      - name: docs.embed
        description: Embed one or more passages.
        backend:
          kind: openai_embedding
          api_key: "${env.OPENAI_API_KEY}"
          model: text-embedding-3-small
          dimensions: 512
          cache: { enabled: true }

      - name: art.generate
        description: Generate an illustration.
        backend:
          kind: openai_image
          api_key: "${env.OPENAI_API_KEY}"
          model: gpt-image-1
          defaults: { size: "1024x1024", output_format: webp }
```

### Azure deployments

Azure encodes the deployment and the API version in the URL, so `base_url` is
mandatory and complete — the adapter posts to it verbatim instead of appending
a path.

```yaml
        backend:
          kind: azure_openai_chat
          base_url: "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-06"
          api_key: "${env.AZURE_OPENAI_KEY}"
          model: gpt-4o
          prompt:
            system: You are a helpful assistant.
            user: "{{ input.question }}"
```

## Response envelope

Chat bindings under `response_format.mode: json_schema` return the validated
object as-is; a reply that is not valid JSON or does not satisfy the schema
either fails the call or earns one corrective round-trip, per
`response_format.on_mismatch`. Under `mode: text` they return `{"text": "…"}` and
skip validation entirely.

## Security

- The API key is held in a redacting wrapper — `Debug` renders `***`, so it
  cannot leak through logs or error strings. A key that resolves to an empty
  value is rejected at boot rather than producing unauthenticated calls.
- Prompt templates can reference only `input.*` and `meta.*`. There is no
  filesystem loader, no env-var lookup, and the `debug` filter is removed, so a
  template cannot dump gateway state or exfiltrate the context. Undefined
  variables fail loudly instead of rendering empty.
- `tools.allowed` is an explicit allowlist enforced inside the plugin: a tool
  call the model invents that is not on the list never leaves the plugin. The
  gateway refuses a child call that targets the initiating binding itself and
  caps child-invocation depth at 8, on top of `tools.max_iterations`.
- Child tool calls carry no caller identity, and `cred://` credential threading
  is unsupported on that path. They are ungated unless you turn on
  `governance.child_invoke.enforce_gates`, which makes each child call run the
  same policy chain, trust floor, CEL `allow_if` gate and tool-gate chain a
  direct `tools/call` runs.
- Budget caps fail closed: exceeding `budget.usd_daily_cap` refuses the call
  before any upstream request is made. Models absent from the bundled rate card
  cannot accumulate cost, so a USD cap is inert for them.

## Observability

Every chat call opens a span (`llm_openai.execute`, or
`llm_azure_openai.execute` on the Azure kinds) and emits a latency histogram
plus a call counter — `mcpg_llm_openai_latency_seconds` /
`mcpg_llm_openai_calls_total`, and the `mcpg_llm_azure_openai_*` pair for Azure
— both labelled with a bounded `outcome` (`ok`, `rate_limited`, `auth_failed`,
`model_not_found`, `server_error`, `client_error`, `timeout`, `transport`) and
`model`. When token usage is known — the streaming path — it also emits
`mcpg_llm_openai_input_tokens_total`, `mcpg_llm_openai_output_tokens_total` and
`mcpg_llm_openai_cost_usd_micros_total`, priced from the rate card vendored in
`mcpg-backend-llm-shared`. Azure bindings price against the public OpenAI rates.

One audit event lands per chat call at `dev.mcpg.llm.openai.completion` /
`.failure` (`dev.mcpg.llm.azure_openai.*` for Azure), carrying binding, model,
outcome, duration and — when known — token counts and cost in micro-USD. The
embedding, image and speech engines emit their own counters and histograms
(`mcpg_embedding_*`, `mcpg_image_*`, `mcpg_tts_*`, `mcpg_stt_*`).

## MCP surfaces & composition

### As a child tool

Any binding backed by this plugin can appear in another chat binding's
`tools.allowed`, which is how you compose a router model in front of an
expensive one, or let a chat model call an embedding or transcription binding
mid-turn, with no gateway-side orchestration code.

```yaml
        backend:
          kind: openai_chat
          api_key: "${env.OPENAI_API_KEY}"
          model: gpt-4o-mini
          prompt:
            system: Transcribe attachments with `media.transcribe` when present.
            user: "{{ input.question }}"
          tools:
            allowed: [media.transcribe]   # a binding backed by openai_stt
```

### Schemas & annotations

The binding-level `input_schema` is what clients see in `tools/list` and what
the gateway validates arguments against. The `output_schema` *inside* the
`backend:` block is what a chat binding enforces on the model's reply; declare
the binding-level `output_schema` too when you want clients to see the
contract. Mark bindings that only read as side-effect-free:

```yaml
        annotations: { read_only: true, open_world: true }
```

## Build
Default feature set is OFF (avoids `mcpg_plugin_register` linker
collisions in the workspace build); opt in to the cdylib:

```bash
cargo build -p mcpg-plugin-backend-llm-openai --features cdylib-export --release   # → target/release/libmcpg_plugin_backend_llm_openai.so
```

## Sign & load (production)
Sign the artifact, pin/verify via the entry's `signature:` block, and honour
revocations. See <https://mcpg.dev/docs/security/plugin-security>.

## See also
- Backend binding reference: <https://mcpg.dev/docs/reference/backends>
- Full gateway config schema: <https://mcpg.dev/docs/reference/configuration>
- Provider-agnostic engines and shared config types: `libs/plugins/backend/llms/shared`
- Any other OpenAI-ABI endpoint (vLLM, Groq, Together, LocalAI): `libs/plugins/backend/llms/compat`
