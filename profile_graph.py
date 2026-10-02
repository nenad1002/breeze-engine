"""Profile Breeze graph operators for a supplied Phi-3.5 reference graph.

Input contract: batch=1; input_ids and attention_mask are int64 [1, tokens].
There are 32 layers, each with empty float16 past key/value tensors shaped
[1, 32, 0, 96] (batch, KV heads, past length, head dimension). This is repeated
prefill with empty caches, not autoregressive decoding or a generic model input
adapter. Token IDs must belong to the supplied model's vocabulary.
"""

import argparse
from collections import defaultdict
import os
from pathlib import Path
from time import perf_counter


def positive_int(value):
    value = int(value)
    if not 0 < value <= 2**31 - 1:
        raise argparse.ArgumentTypeError("must be a positive signed 32-bit integer")
    return value


def token_id(value):
    value = int(value)
    if not 0 <= value < 2**63:
        raise argparse.ArgumentTypeError("must be a nonnegative signed 64-bit token ID")
    return value


def make_feeds(token_ids):
    """Create only the explicitly documented Phi-3.5 reference inputs."""
    import numpy as np

    feeds = {"input_ids": np.array([token_ids], dtype=np.int64),
             "attention_mask": np.ones((1, len(token_ids)), dtype=np.int64)}
    for layer in range(32):
        for kind in ("key", "value"):
            feeds[f"past_key_values.{layer}.{kind}"] = np.zeros((1, 32, 0, 96), np.float16)
    return feeds


def profile_session(session, ops, feeds, *, iterations=1, warmup=1, clock=perf_counter):
    """Profile a duck-typed session/registry; no model or backend is imported here.

    Temporarily patches the process-global registry: do not run concurrently
    with other graph sessions. Times are inclusive (nested graph ops can overlap)
    and include instrumentation overhead. Warmup has separate counters.
    """
    if type(iterations) is not int or iterations <= 0 or type(warmup) is not int or warmup <= 0:
        raise ValueError("iterations and warmup must be positive integers")
    times = defaultdict(float)
    counts = defaultdict(int)
    original = ops.get

    def timed(op_type):
        function = original(op_type)

        def wrapped(*args, **kwargs):
            start = clock()
            try:
                return function(*args, **kwargs)
            finally:
                times[op_type] += clock() - start
                counts[op_type] += 1

        return wrapped

    ops.get = timed
    try:
        start = clock()
        for _ in range(warmup):
            session.run(feeds)
        warmup_wall = clock() - start
        warmup_times, warmup_counts = dict(times), dict(counts)
        times.clear()
        counts.clear()
        start = clock()
        for _ in range(iterations):
            session.run(feeds)
        wall = clock() - start
        return {"times": dict(times), "counts": dict(counts), "wall_seconds": wall,
                "warmup_times": warmup_times, "warmup_counts": warmup_counts,
                "warmup_wall_seconds": warmup_wall}
    finally:
        ops.get = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Phi-3.5 reference ONNX graph")
    parser.add_argument("--token-ids", type=token_id, nargs="+", required=True)
    parser.add_argument("--iterations", type=positive_int, default=1)
    parser.add_argument("--warmup", type=positive_int, default=1)
    parser.add_argument("--threads", type=positive_int, help="set CPU/BLAS threads before imports")
    args = parser.parse_args(argv)
    if not args.model.is_file():
        parser.error("--model must name an existing file")
    if args.threads is not None:
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = str(args.threads)

    from breeze import cpu_backend, ops
    from breeze.graph_session import InferenceSession

    if args.threads is not None:
        cpu_backend.set_threads(args.threads)
    session = InferenceSession(str(args.model))
    result = profile_session(session, ops, make_feeds(args.token_ids),
                             iterations=args.iterations, warmup=args.warmup)
    print(f"Warmup: {args.warmup} runs, {result['warmup_wall_seconds'] * 1000:.3f} ms (excluded)")
    print(f"Measured: {args.iterations} runs, {result['wall_seconds'] * 1000:.3f} ms total; "
          f"{result['wall_seconds'] * 1000 / args.iterations:.3f} ms/run")
    print("Per-op inclusive totals (nested ops may overlap):")
    for name in sorted(result["times"], key=result["times"].get, reverse=True):
        print(f"  {name:34s} {result['times'][name] * 1000:9.3f} ms  x{result['counts'][name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
