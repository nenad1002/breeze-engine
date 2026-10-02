# Breeze service

A local text API and browser workspace for teams with suitable CPU servers.
This guide covers installation, checkpoint setup, API access, and deployment.

## Start here

Python 3.11+ is required for serving. Use an isolated environment from the
repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-serve.txt
.venv/bin/python serve_breeze.py --demo
```

Open <http://127.0.0.1:8080>. Demo responses are **scripted**, do not answer the
submitted content, and are not evidence of model quality or inference speed.
Demo mode does not need model weights, a compiler, or a particular CPU. It is
useful for evaluating the workflow before preparing a deployment.

The workspace includes editable meeting-note, customer-reply, and customer-update
examples. Nothing runs until Send is pressed. Messages, settings, and the access
key remain in the current page's memory; reload or Disconnect clears them.
There is no account database, analytics service, remote font, or CDN dependency.

## Connect a real checkpoint

Native inference requires Linux x86-64, AVX-512 VNNI and VBMI, sufficient RAM, and
an **already-prepared dense Qwen3.5-9B or 27B text-decoder bundle**. Downloading,
exporting, and quantizing models are not part of the service. Verify the model's
licensing for your intended use.

Keep these files together in the bundle directory:

- `model.onnx` and its referenced external weight files.
- The matching `config.json` and `tokenizer.json`.
- `embeddings.npy`: an FP16 or FP32 array of shape `(vocab_size, hidden_size)`.

The decoder must accept `inputs_embeds`, produce `logits`, and retain the full
language head. Packed matrices use 32-element blocks with 4-bit or 8-bit weights;
INT8 weights require symmetric zero point 128. Normalization vectors must match
the exported graph's already-offset RMSNorm convention. Dense decoder shapes,
projections and rotary configuration are checked during loading. Do not mix
embedding tables and decoders from different checkpoints.

When embeddings or tokenizer metadata are missing, [prepare_qwen35.py](prepare_qwen35.py)
can finalize an existing decoder bundle from a local original checkpoint.
It prints a plan by default; add `--execute` to validate, copy matching metadata,
and extract FP16 embeddings. Extraction also requires `safetensors`.

```bash
.venv/bin/python build_kernel.py
.venv/bin/python serve_breeze.py --model /srv/models/qwen35/model.onnx --check
.venv/bin/python serve_breeze.py \
  --model /srv/models/qwen35/model.onnx \
  --model-id team-assistant --threads 48 --max-seq 4096 \
  --max-output-tokens 512 --chunk-size 128 --max-pending 4
```

The thread count is an example, not a recommended setting for every host.
`--check` examines prerequisites without loading weights; successful startup
then checks the full decoder contract while loading. It is not a quality test.
The decoder, configuration, tokenizer and embeddings must be available locally.
Build the native library on the deployment CPU; do not copy a host-specific
binary to a less capable machine.

### Access and deployment

- Binding defaults to **127.0.0.1**. Without a key, any local process that can
  reach that listener can use it. Loopback is not per-user authentication.
- Set `BREEZE_API_KEY` through your environment/secret manager for bearer access.
  Use a randomly generated value of at least 24 ASCII characters. It is not a
  command-line option and is not embedded in the browser's example code.
- Binding off loopback requires that key. Add exact `--allowed-host` values for
  deployment hostnames. Wildcards and hostname-plus-port entries are rejected.
- API/status/schema/metrics routes require the key when configured. Minimal
  liveness/readiness and the static workspace are public. Cross-origin API use
  is rejected; arbitrary websites cannot use the unauthenticated local API.
- Put shared deployments behind TLS, a firewall, and an access gateway with
  appropriate rate limits and identities. A single bearer key is **not** RBAC,
  SSO, tenant isolation, an audit trail, or a compliance certification.
- Only loopback reverse proxies are trusted for forwarded connection metadata.
  Preserve `Host`, forward the external scheme, disable response buffering,
  and use proxy timeouts above the service deadline. For non-loopback proxies,
  arrange loopback forwarding or explicitly review trust configuration before
  adapting the launcher. Do not trust arbitrary forwarded headers.
- [deploy/Dockerfile](deploy/Dockerfile), [deploy/compose.yaml](deploy/compose.yaml),
  and [deploy/breeze.service](deploy/breeze.service) are deployment templates.
  Container startup compiles the native library on the target host. The Compose
  port is loopback-only, and the model mount is read-only. Never put model weights
  or keys in the image. Review paths, host settings, memory and shutdown budgets.

For Compose, provide `BREEZE_MODEL_DIR` (directory containing the prepared
decoder) and `BREEZE_API_KEY` externally. No secret file is committed. The image
requires network access only during package installation; model execution uses
local files. A service startup can still fail if CPU flags or model contracts
do not match.

## API contract

The API implements a **small chat-completion-shaped subset**, not full protocol
or SDK compatibility. Unsupported parameters are rejected instead of ignored.

| Route | Purpose | Access |
|---|---|---|
| `GET /` | Browser workspace | Public |
| `GET /healthz` | Event-loop liveness | Public |
| `GET /readyz` | Loaded and accepting requests | Public; 503 if unavailable |
| `GET /v1/models` | Configured public model ID | Bearer when configured |
| `POST /v1/chat/completions` | JSON or SSE text generation | Bearer when configured |
| `GET /api/status` | Capabilities, bounded queue and counters | Bearer when configured |
| `GET /api/schema` | Machine-readable request schema | Bearer when configured |
| `GET /metrics` | Aggregate Prometheus-format metrics | Bearer when configured |

Example request body:

```json
{
  "model": "team-assistant",
  "messages": [
    {"role": "system", "content": "Be concise. Do not invent missing details."},
    {"role": "user", "content": "Draft a short welcome message for a new teammate."}
  ],
  "max_tokens": 128,
  "temperature": 0,
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

- Use the configured model ID; demo always uses `breeze-demo`.
- Text-only content. An optional first system message is followed by alternating
  user and assistant turns, ending with a user. Send history with every request.
- Greedy decoding only: `temperature=0`, `top_p=1`, `n=1` if specified. No tools,
  images, custom stop sequences, sampling, embeddings API, or JSON constraints.
- Non-thinking prompt template. This is not a reasoning-token streaming API.
- At most 64 messages, 65,536 combined content characters, and 128 KiB encoded
  request body. Body receipt has a 10-second limit. Control-token delimiters and
  NUL characters are rejected. This prevents template delimiter injection; it
  does not solve prompt injection or make untrusted source instructions safe.
- Input tokens plus requested output must fit the context allocation. Requests
  are rejected, never silently truncated. Output defaults to 256 tokens; set it
  explicitly if the operator has configured a lower output limit.
- SSE uses `data: {JSON}` frames with text deltas, a finish chunk, optional usage,
  then `data: [DONE]`. Midstream failures are error frames; discard partial
  conversation history. Errors before headers use HTTP status codes. The
  browser only commits completed turn pairs to subsequent request history.
- Errors have `error.message`, `error.type`, `error.code`, and `request_id`.
  Typical statuses: 400 validation/context, 401 key, 404 model, 413 body limit,
  429 queue full (`Retry-After: 1`), 503 unavailable, 504 soft deadline.

The workspace's Connection & API view creates a Python standard-library client
example using the actual origin and model ID, without exposing the entered key.

## Operational behavior

**One resident model, one native owner thread, one active request.** Model load,
tokenization, inference, and close stay on that worker. A bounded admission
queue accepts a small number of requests; `--max-pending` includes the active
request. A full queue returns 429. Multiple users can submit sequential work,
but there is no continuous batching or parallel generation.

Each request re-prefills its own history from position zero. No server session
or cache is shared between conversations. This trades throughput for a simple
and testable isolation boundary. Larger histories increase prefill latency.
Do not run unrelated Breeze model calls in the serving process: native settings
remain process-global. Do not add multiple web workers; each would load its own
weights and compete for CPU/memory. Separate instances require explicit capacity
planning and routing.

The deadline includes queue time. Cancellation and deadlines are **cooperative**:
checked between prefill chunks, decode steps, and output-queue waits. An in-flight
native call cannot be interrupted safely. Shutdown waits for it before freeing
the model. A hard hang needs process supervision; HTTP timeout is not a hard CPU
execution deadline. Disconnected queued clients relinquish admission, and slow
stream consumers face bounded buffering rather than unlimited text retention.

Unexpected inference errors mark the service unready until restarted. Context
and input errors do not. Health is not a synthetic model-generation probe.

### Privacy and observability

Breeze does not write prompts, responses, conversation histories, or API keys to
disk. Access logging is disabled by the launcher. Failure logs contain only
request IDs and exception class names. Counters contain totals, not content.
The application still holds active requests and model state in RAM; host swap,
core dumps, browser extensions, proxies and operator instrumentation have their
own retention/security implications. No claim of whole-system zero retention.

Responses include `usage` and a `breeze` timing extension: queue time, wall time,
time to first text, prefill time, decode-only time and decode steps. Model load
is startup work, not per-request timing. Demo counts are word-based estimates
and inference timings/rates are null. Do not use demo or one-token smoke timing
as a throughput benchmark.

## Validate a deployment

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -p no:cacheprovider tests -q
node --test tests/test_web.mjs
```

Service tests use deterministic fake backends and do not download or load model
weights. Native numerical tests still require the built library. Verify a real
checkpoint separately with a short request, then two unrelated conversations,
a cancelled stream, and your application's own representative documents. Check
quality and latency on the deployment hardware before serving application traffic.

The opt-in Chromium test installs separately with
[requirements-browser.txt](requirements-browser.txt). Install Chromium for
Playwright, then set `BREEZE_BROWSER_TESTS=1` when running
[tests/test_service_browser.py](tests/test_service_browser.py). It starts a
temporary loopback demo server, tests the authenticated desktop/mobile workflow,
and shuts the server down. It does not leave a serving process running.
