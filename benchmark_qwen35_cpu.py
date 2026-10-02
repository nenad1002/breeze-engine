"""Breeze CPU prefill/decode timing on deterministic fixed-token workloads.

Run one thread configuration per process. Replay fixed token IDs regardless of
predictions or EOS. These are throughput workloads, not answer-quality evaluations.
Loading, tokenization, embedding gathers, logit inspection, and result writes are
excluded from forward timers. Warmup is reported separately from measured rounds.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np


def workload_ids(tokenizer, prompt_length, decode_steps):
    if prompt_length < 1 or decode_steps < 1:
        raise ValueError("Prompt length and decode steps must be positive")
    prefix = tokenizer.encode("<|im_start|>user\n").ids
    suffix = tokenizer.encode(
        "\nSummarize the passage in three sentences.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ).ids
    passage = tokenizer.encode(
        "A research team measures a language model on a CPU. The model reads "
        "a prompt and processes one new token at a time. Quantized weights "
        "reduce memory traffic. The team records latency and memory use, "
        "repeats each measurement, and checks the numerical outputs. "
    ).ids
    continuation = tokenizer.encode(
        "The study measures CPU inference using repeated, controlled workloads. "
        "It separates model loading, prompt processing, and token decoding. "
        "Numerical validation is reported separately from throughput. "
    ).ids
    body_length = prompt_length - len(prefix) - len(suffix)
    if body_length < 1:
        raise ValueError(f"Prompt length must exceed {len(prefix) + len(suffix)}")
    if not passage or not continuation:
        raise ValueError("Tokenizer must produce nonempty passage and continuation IDs")
    prompt = prefix + (passage * ((body_length + len(passage) - 1) // len(passage)))[:body_length] + suffix
    decode = (continuation * ((decode_steps + len(continuation) - 1) // len(continuation)))[:decode_steps]
    return prompt, decode


def run_workload(model, prompt, continuation, chunk_size, capture=False,
                 capture_every=16, progress_every=0):
    for name, value, minimum in (("chunk_size", chunk_size, 1),
                                 ("capture_every", capture_every, 1),
                                 ("progress_every", progress_every, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if len(prompt) == 0 or len(continuation) == 0:
        raise ValueError("Prompt and continuation must be nonempty")
    prefill_seconds = 0.0
    logits = None
    for offset in range(0, len(prompt), chunk_size):
        chunk = prompt[offset:offset + chunk_size]
        start = time.perf_counter()
        logits = model.run(chunk, past_len=offset)
        prefill_seconds += time.perf_counter() - start
        if not np.isfinite(logits).all():
            raise RuntimeError("Non-finite prefill logits")
    rows = [logits[-1].copy()] if capture else []
    positions = [len(prompt) - 1] if capture else []
    predicted = [int(logits[-1].argmax())]
    samples = []
    for index, embedding in enumerate(continuation):
        past_len = len(prompt) + index
        start = time.perf_counter()
        logits = model.run(embedding, past_len=past_len)
        samples.append(time.perf_counter() - start)
        if not np.isfinite(logits).all():
            raise RuntimeError("Non-finite decode logits")
        predicted.append(int(logits[-1].argmax()))
        if capture and (index == 0 or (index + 1) % capture_every == 0 or index + 1 == len(continuation)):
            rows.append(logits[-1].copy())
            positions.append(len(prompt) + index)
        if progress_every and ((index + 1) % progress_every == 0 or index + 1 == len(continuation)):
            print(
                f"  decode {index + 1}/{len(continuation)}: "
                f"{(index + 1) / sum(samples):.3f} tok/s; "
                f"last step {samples[-1]:.4f}s", flush=True,
            )
    result = {
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": len(prompt) / prefill_seconds,
        "decode_seconds": sum(samples),
        "decode_step_seconds": samples,
        "decode_tokens_per_second": len(samples) / sum(samples),
        "decode_step_median_ms": float(np.median(samples) * 1000),
        "decode_step_p95_ms": float(np.quantile(samples, 0.95) * 1000),
        "predicted_token_ids": predicted,
    }
    return result, rows, positions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", help="HF config path or 9b/27b preset")
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[64, 512])
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--capture-every", type=int, default=16, help="Save logits every N decode steps in the final round")
    parser.add_argument("--progress-every", type=int, default=0, help="Print progress every N decode steps; 0 disables")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.threads, args.decode_steps, args.chunk_size, args.rounds, *args.prompt_lengths) < 1 or args.warmup < 0:
        parser.error("Sizes/rounds/threads must be positive and warmup nonnegative")
    if len(set(args.prompt_lengths)) != len(args.prompt_lengths):
        parser.error("Prompt lengths must be distinct")
    if args.capture_every < 1 or args.progress_every < 0:
        parser.error("Capture interval must be positive and progress interval nonnegative")
    if os.environ.get("I4_QWEN_MAXL") is not None:
        parser.error("Unset I4_QWEN_MAXL; partial-layer runs are not benchmarks")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Set CUDA_VISIBLE_DEVICES='' for these CPU-only benchmarks")
    logits_path = args.output.with_suffix(".logits.npz")
    if args.output.exists() or logits_path.exists():
        parser.error("Output exists; choose a new result path")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ.setdefault("OMP_WAIT_POLICY", "active")
    os.environ.setdefault("OMP_PROC_BIND", "spread")
    os.environ.setdefault("OMP_PLACES", "cores")
    from tokenizers import Tokenizer
    from breeze.qwen35_config import QwenConfig
    import onnx

    args.model = args.model.absolute()
    config = QwenConfig.resolve(args.model, args.config)
    tokenizer_path = args.tokenizer or args.model.parent / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    embeddings_path = args.embeddings or args.model.parent / "embeddings.npy"
    table = np.load(embeddings_path, mmap_mode="r")
    if table.size == 0 or table.shape != (config.vocab_size, config.hidden_size) or table.dtype not in (np.float16, np.float32):
        parser.error("Embedding table must be nonempty with the configured shape and float16/float32 dtype")
    workloads = []
    for count in args.prompt_lengths:
        try:
            prompt_ids, decode_ids = workload_ids(tokenizer, count, args.decode_steps)
        except ValueError as exc:
            parser.error(str(exc))
        prompt = np.ascontiguousarray(table[prompt_ids], dtype=np.float32)
        continuation = [np.ascontiguousarray(table[[token]], dtype=np.float32) for token in decode_ids]
        workloads.append((prompt_ids, decode_ids, prompt, continuation))
    graph = onnx.load(args.model, load_external_data=False)
    bit_counts = Counter(int(onnx.helper.get_attribute_value(attr)) for node in graph.graph.node
                         if node.op_type == "MatMulNBits" for attr in node.attribute if attr.name == "bits")
    del graph
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv if argv is None else [sys.argv[0], *argv],
        "engine": "breeze",
        "model": str(args.model),
        "source_graph_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matmul_weight_bits_counts": dict(bit_counts),
        "threads": args.threads,
        "cpu_affinity_before_runtime": sorted(os.sched_getaffinity(0)),
        "numa_maps_first_lines": Path("/proc/self/numa_maps").read_text().splitlines()[:3],
        "cpu_only": True,
        "protocol": "Fixed real-token replay, batch one, no EOS stopping; warmup excluded; one Breeze thread configuration per process",
        "timing_scope": "model.run only; excludes load, tokenization, embedding gather, logit inspection and result writes",
        "chunk_size": args.chunk_size,
        "decode_steps": args.decode_steps,
        "warmup_rounds": args.warmup,
        "measured_rounds": args.rounds,
        "capture_every": args.capture_every,
        "progress_every": args.progress_every,
        "numpy_version": np.__version__,
        "logits_path": str(logits_path),
    }
    from breeze import Qwen35CpuModel
    from breeze import cpu_backend as ib
    ib.set_threads(args.threads)
    metadata["activation_mode"] = "int16-two-pass"
    metadata["kernel_sha256"] = hashlib.sha256(Path(ib._SO).read_bytes()).hexdigest()
    print(f"[breeze] loading; {args.threads} threads; weights {dict(bit_counts)}", flush=True)
    start = time.perf_counter()
    model = Qwen35CpuModel(args.model, config=config, hi_prec=True,
                          max_seq=max(args.prompt_lengths) + args.decode_steps + 16)
    try:
        metadata["load_seconds"] = time.perf_counter() - start
        print(f"[breeze] loaded in {metadata['load_seconds']:.2f}s", flush=True)
        cases, saved_logits = [], {}
        for prompt_ids, decode_ids, prompt, continuation in workloads:
            case = {"prompt_tokens": len(prompt_ids), "prompt_token_ids": prompt_ids,
                    "decode_token_ids": decode_ids, "warmup": [], "rounds": []}
            for trial in range(args.warmup + args.rounds):
                warmup = trial < args.warmup
                label = f"warmup {trial + 1}" if warmup else f"round {trial - args.warmup + 1}"
                print(f"[breeze] pp{len(prompt_ids)}/decode{args.decode_steps} {label} starting", flush=True)
                result, rows, positions = run_workload(model, prompt, continuation, args.chunk_size,
                                                       capture=trial == args.warmup + args.rounds - 1,
                                                       capture_every=args.capture_every,
                                                       progress_every=args.progress_every)
                case["warmup" if warmup else "rounds"].append(result)
                print(f"[breeze] {label}: pp={result['prefill_seconds']:.4f}s; "
                      f"decode={result['decode_tokens_per_second']:.3f} tok/s", flush=True)
                if rows:
                    saved_logits[f"pp{len(prompt_ids)}"] = np.stack(rows)
                    case["sampled_logit_positions"] = positions
            case["prefill_median_seconds"] = float(np.median([r["prefill_seconds"] for r in case["rounds"]]))
            case["prefill_median_tokens_per_second"] = len(prompt_ids) / case["prefill_median_seconds"]
            case["decode_median_tokens_per_second"] = float(np.median([r["decode_tokens_per_second"] for r in case["rounds"]]))
            case["decode_pooled_tokens_per_second"] = args.decode_steps * args.rounds / sum(r["decode_seconds"] for r in case["rounds"])
            cases.append(case)
        metadata.update(cases=cases, peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                        completed_utc=datetime.now(timezone.utc).isoformat())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with logits_path.open("xb") as handle:
            np.savez(handle, **saved_logits)
        with args.output.open("x") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
        print(f"[breeze] saved {args.output}; peak RSS {metadata['peak_rss_gib']:.2f} GiB", flush=True)
    finally:
        model.close()


if __name__ == "__main__":
    main()