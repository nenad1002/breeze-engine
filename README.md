# Breeze

**Your server. Your workspace.**

Created and maintained by **Nenad Banfic**. Project attribution is recorded in
[NOTICE](NOTICE); separately attributed third-party components retain their
original notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Breeze turns a compatible CPU server into a local text-assistant service:
a browser workspace for summaries and reviewed drafts, a streaming text API
for your applications, and the native Python inference library underneath.
It loads prepared quantized ONNX checkpoints directly; no hosted inference
dependency is needed to process a request.

Start with the workspace below or the [deployment and API guide](SERVICE.md).

## Try the workspace

Use Python 3.11 or newer. Run these commands from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-serve.txt
.venv/bin/python serve_breeze.py --demo
```

Open <http://127.0.0.1:8080>. The preview is clearly labeled **scripted**, needs
no model, and is not a performance or answer-quality demonstration. The
workspace includes streaming, multi-turn chat, editable work templates, stop,
copy, connection controls, and real service status. It uses only local assets.

If port 8080 is occupied, add `--port 8081` and open that port instead. For a
remote VS Code workspace, forward the chosen port in the Ports panel.

For native inference, build the kernel and supply a local dense Qwen3.5 bundle.
The optional service keeps one model loaded with one active worker, bounded
admission, bearer-key access, health checks and aggregate metrics. See
[SERVICE.md](SERVICE.md) for CPU requirements, authentication, deployment and API
limits. There is no automatic download or model conversion.

## Model APIs

| Public class | Model | Scope |
|---|---|---|
| `Qwen35CpuModel` | Dense Qwen3.5-9B / 27B | Text-only prefill and greedy decoding |
| `Phi35CpuModel` | Phi-3.5 | Experimental batch-one prefill, 1-8 tokens only |
| `Qwen35GpuModel` | Dense Qwen3.5-9B | Experimental CUDA prefill and decoding |
| `Qwen35ReferenceModel` | Qwen3.5 | Graph-based development reference |
| `InferenceSession` | Compatible operator graphs | Graph-based development reference |

The native CPU backend requires Linux/x86-64 with AVX-512 VNNI and VBMI. Build
the library on the target machine. CPU models use INT4/INT8 packed weights;
Qwen3.5 defaults to INT16 two-pass activation quantization. Only one active
sequence per model is supported. Native settings are process-global, so model
calls and configuration changes must be serialized, even across model instances.

## Setup

Native development also requires a C++17 compiler with OpenMP (such as `g++`)
and a CPU with the features listed above. JavaScript tests require Node.js 20
or newer; Node.js is not needed to serve the playground. The optional browser
test setup is documented in [SERVICE.md](SERVICE.md).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python build_kernel.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest tests -q
node --test tests/test_web.mjs
```

No model is downloaded, loaded, or executed during setup. Generated native
libraries, bytecode, checkpoints, and measurement artifacts are not tracked.
Run the native build and model tools from this source checkout. The shared
library is compiled for the deployment CPU. For a minimal library-only
environment, install `requirements-cpu.txt`.

## Python example

Use an already-prepared local model bundle. The configuration, tokenizer, and
embedding table are loaded from the same directory as the decoder.

```python
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "48"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from tokenizers import Tokenizer
from breeze import Qwen35CpuModel, cpu_backend
from breeze.chat import chat_prompt

model_dir = Path("models/qwen35-27b-cpu")
tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
prompt_ids = tokenizer.encode(
    chat_prompt("Explain Python generators briefly.", no_thinking=True)
).ids
eos_ids = {
    token_id
    for marker in ("<|im_end|>", "<|endoftext|>")
    if (token_id := tokenizer.token_to_id(marker)) is not None
}

cpu_backend.set_threads(48)
with Qwen35CpuModel(model_dir / "model.onnx", max_seq=4096) as model:
    output_ids = model.generate(prompt_ids, max_new_tokens=128, eos_ids=eos_ids)
    print(tokenizer.decode(output_ids, skip_special_tokens=True))
```

- `generate()` accepts prompt token IDs and returns generated token IDs using
  greedy decoding. It starts a fresh sequence on each call; it is not a stream.
- `prefill()` accepts token IDs, bounds temporary buffers through chunking, and
  returns the last position's logits.
- `embed()` maps token IDs to contiguous FP32 embeddings.
- `run()` accepts embeddings and an explicit `past_len`; it returns logits and
  updates cached state. Passing `past_len=0` resets the sequence.
- `close()` releases the Qwen CPU model's resources; the context manager calls
  it automatically.

Tokenization, application routing, document retrieval, and chat history are
owned by the library caller. The optional service supplies text tokenization,
request routing and a browser client; retrieval and business-system integration
remain application responsibilities. Importing `breeze` does not load web dependencies.

## Command-line tools

- [serve_breeze.py](serve_breeze.py): local JSON/SSE inference service and browser
  workspace; `--demo` previews the UX and `--check` verifies prerequisites.
- [generate_qwen35.py](generate_qwen35.py): standalone CPU generation with
  progressive terminal output and separate load/prefill/decode timing.
- [generate_qwen35_gpu.py](generate_qwen35_gpu.py): optional dense-9B CUDA
  generation; compilation and graph capture require explicit flags.
- [benchmark_qwen35_cpu.py](benchmark_qwen35_cpu.py): deterministic fixed-token
  prefill/decode workloads, warmup rounds, latency statistics, and saved logits.
- [benchmark_qwen35_prefill.py](benchmark_qwen35_prefill.py): exactly one measured
  prefill call with full logit capture.
- [prepare_qwen35.py](prepare_qwen35.py): local bundle finalization and FP16
  embedding extraction. An already-quantized decoder is required; this utility
  does not export, quantize, or download model weights.
- [benchmark_cpu_matmul.py](benchmark_cpu_matmul.py) and
  [benchmark_gpu_matmul.py](benchmark_gpu_matmul.py): explicit kernel experiments.
- [profile_graph.py](profile_graph.py): per-operator graph profiling.
- [measure_process.py](measure_process.py): Linux child-process resource sampling.

Each tool provides `--help` without loading a checkpoint or allocating GPU
memory. Heavy inference and preparation run only when explicitly invoked.

## Relative generation performance

Measured relative generation throughput (Breeze / baseline):

| Compared with | Breeze throughput |
|---|---:|
| ONNX Runtime (ORT) | **1.62x** |
| llama.cpp | **1.01x** |

## Layout

- [breeze/](breeze/): model APIs, graph loader, packed-weight bridge, and numerical
  operators. Backend imports are lazy at the package boundary.
- [kernel/](kernel/): quantized matrix multiplication and model-specific native
  decoders.
- [tests/](tests/): small numerical, API, and command-line tests; no model
  downloads or full-checkpoint inference.
- [deploy/](deploy/): container and service-manager deployment templates.
- [breeze/web/](breeze/web/): self-contained browser workspace; no frontend build.
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md): applicable attribution and
  license notices for derived operator algorithms.

The Qwen CPU implementation lives in [breeze/qwen35_cpu.py](breeze/qwen35_cpu.py).
Use the public package imports rather than implementation-specific module paths.

## License

The project is licensed under the [Apache License 2.0](LICENSE). Third-party
attribution is preserved in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Model weights are not included and remain subject to their own license.