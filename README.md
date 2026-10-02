# Breeze

**Native CPU inference for quantized language models.**

Created and maintained by **Nenad Banfic**. Project attribution is recorded in
[NOTICE](NOTICE); separately attributed third-party components retain their
original notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Breeze is a CPU inference engine that combines native C++ kernels with a small
Python API. It loads prepared quantized ONNX checkpoints directly and runs
prefill and autoregressive decoding locally, without a GPU or hosted inference
service.

- **Packed INT4/INT8 weights** with AVX-512 VNNI matrix multiplication.
- **Two-pass INT16 activations** by default for the Qwen CPU decoder.
- **Native decoder execution** with OpenMP and resident attention, convolution,
  and recurrent state across decode steps.

Use Breeze directly from Python or through the command-line tools. The HTTP API
and browser playground are optional interfaces to the same CPU engine.

## Setup

Requirements: Python 3.11+, Linux/x86-64 with AVX-512 VNNI and VBMI, and a C++17
compiler with OpenMP (such as `g++`). Build on the machine that will run inference.
Run these commands from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.txt
.venv/bin/python build_kernel.py
```

The CPU library needs NumPy, ONNX, and tokenizers; web and GPU dependencies are
optional. Setup does not download or convert model weights. Supply a prepared
local decoder bundle with matching configuration, tokenizer, and embeddings.
See [checkpoint setup](SERVICE.md#connect-a-real-checkpoint) for the file contract.
Generated libraries, checkpoints, and measurement artifacts are not tracked.

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
owned by the library caller. Importing `breeze` does not load web or GPU backends.

## Model APIs

| Public class | Model | Scope |
|---|---|---|
| `Qwen35CpuModel` | Dense Qwen3.5-9B / 27B | Text-only prefill and greedy decoding |
| `Phi35CpuModel` | Phi-3.5 | Experimental batch-one prefill, 1-8 tokens only |
| `Qwen35ReferenceModel` | Qwen3.5 | Graph-based development reference |
| `InferenceSession` | Compatible operator graphs | Graph-based development reference |
| `Qwen35GpuModel` | Dense Qwen3.5-9B | Optional experimental CUDA prefill and decoding |

Only one active sequence per CPU model is supported. Native settings are
process-global, so model calls and configuration changes must be serialized,
even across model instances.

## Command-line tools

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

## Optional HTTP API and playground

[serve_breeze.py](serve_breeze.py) exposes streaming chat and a browser interface:

```bash
.venv/bin/python -m pip install -r requirements-serve.txt
.venv/bin/python serve_breeze.py --model models/qwen35-27b-cpu/model.onnx
```

Open <http://127.0.0.1:8080>. Use `--port 8081` if needed and forward the port for
remote workspaces. `--demo` replaces `--model` with a scripted, model-free preview.
See the [deployment and API guide](SERVICE.md) for authentication and configuration.

## Tests

After building the native library:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest tests -q
node --test tests/test_web.mjs
```

The tests use small fixtures rather than pretrained models. JavaScript tests
require Node.js 20+; the CPU engine does not. Optional browser-test setup is in
[SERVICE.md](SERVICE.md#validate-a-deployment).

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