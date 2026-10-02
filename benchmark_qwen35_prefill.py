"""Measure one Breeze CPU prefill forward call and save every token's logits.

Run one thread configuration per process. No warmup, generation loop, or model
inference occurs outside the single measured run call.
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--pack-threads", type=int, help="Breeze packing threads; defaults to --threads")
    parser.add_argument("--max-seq", type=int, default=1024)
    parser.add_argument("--prompt", default="What is the capital of France? Answer in one short sentence.")
    args = parser.parse_args(argv)
    if min(args.threads, args.max_seq) <= 0:
        parser.error("Thread and context counts must be positive")
    if args.pack_threads is not None and args.pack_threads <= 0:
        parser.error("--pack-threads must be positive")
    if not args.prompt.strip():
        parser.error("Prompt must be nonempty")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Set CUDA_VISIBLE_DEVICES='' for CPU-only timing")
    if os.environ.get("I4_QWEN_MAXL") is not None:
        parser.error("Unset I4_QWEN_MAXL; partial-layer runs are not benchmarks")
    logits_path = args.output.with_suffix(".logits.npy")
    if args.output.exists() or logits_path.exists():
        parser.error("Output exists; choose a new result path")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ.setdefault("OMP_PROC_BIND", "spread")
    os.environ.setdefault("OMP_PLACES", "cores")
    os.environ.setdefault("OMP_WAIT_POLICY", "active")

    import onnx
    from tokenizers import Tokenizer
    from breeze.qwen35_config import QwenConfig
    from breeze.chat import chat_prompt

    args.model = args.model.absolute()
    config = QwenConfig.resolve(args.model)
    tokenizer = Tokenizer.from_file(str(args.model.parent / "tokenizer.json"))
    token_ids = tokenizer.encode(chat_prompt(args.prompt, no_thinking=True)).ids
    if not token_ids or len(token_ids) > args.max_seq:
        parser.error("Prompt must be nonempty and within the configured context")
    table = np.load(args.model.parent / "embeddings.npy", mmap_mode="r")
    if table.size == 0 or table.shape != (config.vocab_size, config.hidden_size) or table.dtype not in (np.float16, np.float32):
        parser.error("Embedding table must be nonempty with the configured shape and float16/float32 dtype")
    embeddings = np.ascontiguousarray(table[token_ids], dtype=np.float32)
    graph = onnx.load(args.model, load_external_data=False)
    bits = Counter()
    for node in graph.graph.node:
        if node.op_type == "MatMulNBits":
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            bits[int(attrs["bits"])] += 1
    del graph

    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "breeze",
        "model": str(args.model),
        "graph_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(embeddings.tobytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_executable": sys.executable,
        "cpu_only": True,
        "threads": args.threads,
        "cpu_affinity_before_runtime": sorted(os.sched_getaffinity(0)),
        "host_load_before": list(os.getloadavg()),
        "matmul_weight_bits_counts": dict(bits),
        "prompt": args.prompt,
        "thinking": False,
        "prompt_token_ids": token_ids,
        "prompt_tokens": len(token_ids),
        "max_seq": args.max_seq,
        "forward_calls": 1,
        "warmup_calls": 0,
        "decode_steps": 0,
        "timing_scope": "Single first model.run call; excludes loading, tokenization, embedding gather, logit inspection and disk writes. Not steady-state decode throughput.",
    }
    from breeze import Qwen35CpuModel
    from breeze import cpu_backend as ib
    report.update(activation_mode="int16-two-pass",
                  packing_threads=args.pack_threads or args.threads,
                  prepack_backend="fused-native" if ib._native_prepack is not None else "numpy",
                  prepack_source_sha256=hashlib.sha256(Path(ib.__file__).read_bytes()).hexdigest(),
                  kernel_sha256=hashlib.sha256(Path(ib._SO).read_bytes()).hexdigest())

    print(f"[breeze] loading {config.num_hidden_layers} layers, {sum(bits.values())} matmuls", flush=True)
    model = None
    start = time.perf_counter()
    try:
        ib.set_threads(report["packing_threads"])
        try:
            model = Qwen35CpuModel(args.model, config=config, hi_prec=True, max_seq=args.max_seq)
        finally:
            ib.set_threads(args.threads)
        report["load_seconds"] = time.perf_counter() - start
        print(f"[breeze] loaded in {report['load_seconds']:.3f}s; one {len(token_ids)}-token forward starting", flush=True)
        start = time.perf_counter()
        logits = model.run(embeddings, past_len=0)
        report["forward_seconds"] = time.perf_counter() - start
        if logits.shape != (len(token_ids), config.vocab_size) or not np.isfinite(logits).all():
            raise RuntimeError("Invalid output logits")
        report.update(
            logits_shape=list(logits.shape),
            next_token_id=int(logits[-1].argmax()),
            next_token_text=tokenizer.decode([int(logits[-1].argmax())]),
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            host_load_after=list(os.getloadavg()),
            completed_utc=datetime.now(timezone.utc).isoformat(),
            logits_path=str(logits_path),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with logits_path.open("xb") as handle:
            np.save(handle, logits)
        with args.output.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(json.dumps({k: report[k] for k in ("engine", "prompt_tokens", "load_seconds", "forward_seconds", "next_token_id", "next_token_text", "peak_rss_gib")}, indent=2), flush=True)
    finally:
        if model is not None:
            model.close()


if __name__ == "__main__":
    main()