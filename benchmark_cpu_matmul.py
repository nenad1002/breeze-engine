"""Synthetic INT4/INT8 CPU microbenchmark; no model is needed.

Only repeated native matmul calls (and Python dispatch) are timed. Packing,
input/output allocation and pointer conversion are outside the timed region.
"""

import argparse
import ctypes
import os
from time import perf_counter


def positive_int(value):
    value = int(value)
    if not 0 < value <= 2**31 - 1:
        raise argparse.ArgumentTypeError("must be a positive signed 32-bit integer")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def benchmark(backend, *, n, k, rows, threads, iterations, warmup, bits=4,
              seed=0, clock=perf_counter):
    """Time one run, releasing its packed handle even if a native call fails."""
    import numpy as np

    rng = np.random.default_rng(seed)
    blocks = k // 32
    weights = rng.integers(0, 256, (n, blocks, 4 * bits), dtype=np.uint8)
    scales = (rng.random(n * blocks, dtype=np.float32) * 0.02 + 0.001).astype(np.float16)
    zeros = (rng.integers(0, 256, n * ((blocks + 1) // 2), dtype=np.uint8)
             if bits == 4 else None)
    a = rng.standard_normal((rows, k), dtype=np.float32)
    out = np.empty((rows, n), dtype=np.float32)
    backend.set_threads(threads)
    handle = backend.prepack(weights, scales, zeros, k, n, bits=bits)
    try:
        if not handle:
            raise MemoryError("CPU prepack returned a null handle")
        # Use the bridge's native entry point to exclude its output allocation.
        native = backend._lib.i4_matmul
        args = (ctypes.c_void_p(handle), a.ctypes.data_as(ctypes.c_void_p),
                rows, out.ctypes.data_as(ctypes.c_void_p))
        for _ in range(warmup):
            native(*args)
        start = clock()
        for _ in range(iterations):
            native(*args)
        return (clock() - start) / iterations
    finally:
        backend.free(handle)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=positive_int, default=512, help="output columns; multiple of 16")
    parser.add_argument("--k", type=positive_int, default=1024, help="input columns; multiple of 32")
    parser.add_argument("--rows", type=positive_int, default=1)
    parser.add_argument("--threads", type=positive_int, nargs="+", default=[1])
    parser.add_argument("--iterations", type=positive_int, default=20)
    parser.add_argument("--warmup", type=nonnegative_int, default=3)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    args = parser.parse_args(argv)
    if args.n % 16 or args.k % 32:
        parser.error("--n must be divisible by 16 and --k by 32")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.threads[0])
    from breeze import cpu_backend

    if not cpu_backend.available():
        parser.error("Breeze CPU kernel is unavailable")
    for threads in args.threads:
        seconds = benchmark(cpu_backend, n=args.n, k=args.k, rows=args.rows,
                            threads=threads, iterations=args.iterations,
                            warmup=args.warmup, bits=args.bits, seed=args.seed)
        print(f"INT{args.bits} N={args.n} K={args.k} rows={args.rows} "
              f"threads={threads}: {seconds * 1000:.3f} ms/native call")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

