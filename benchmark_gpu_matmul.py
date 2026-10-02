"""Experimental synthetic W8A16 GPU benchmark with per-32-weight scales.

Torch and Triton are imported only after CLI validation. --tune explicitly
enables the full configuration grid for the requested shape. GPU timings include
output allocation, split-K zeroing and conversion, unlike the CPU microbenchmark.
"""

import argparse
from functools import lru_cache
from itertools import product


TILE_VALUES = ((16,), (64, 128, 256), (32, 64), (4, 8), (2, 3, 4), (1, 2, 4, 8, 16))
DEFAULT_CONFIG = (16, 128, 32, 4, 3, 1)


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


def validate_config(k, n, rows, config):
    """Validate tile/split constraints without importing any GPU packages.

    M and N tails are masked. K must contain whole 32-weight scale groups and
    each split must contain whole BLOCK_K tiles (and therefore scale groups).
    """
    if any(type(v) is not int or not 0 < v <= 2**31 - 1 for v in (k, n, rows)):
        raise ValueError("K, N and rows must be positive signed 32-bit integers")
    if len(config) != len(TILE_VALUES) or any(
            type(v) is not int or v not in allowed
            for v, allowed in zip(config, TILE_VALUES)):
        raise ValueError("unsupported tile/warp/stage/split configuration")
    if k % 32:
        raise ValueError("K must be divisible by the scale block size 32")
    bk, split = config[2], config[5]
    if k % split or (k // split) % bk:
        raise ValueError("K must be divisible by split_k * block_k")
    return k // split


def configuration_grid():
    """Return the explicit experimental grid, including possibly invalid splits."""
    return product(*TILE_VALUES)


@lru_cache(maxsize=1)
def load_backend():
    # Triton's compiler resolves language symbols from the kernel's globals.
    # Populate this only on use; importing this script still needs no Triton.
    global tl
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def gemm_i8_kernel(a_ptr, w_ptr, s_ptr, y_ptr, M, K, N, ksplit,
                       sam, sak, swk, swn, ssb, ssn, sym, syn,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_k = tl.program_id(2)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        n_mask = offs_n < N
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        k_lo = pid_k * ksplit
        for k0 in range(k_lo, k_lo + ksplit, BLOCK_K):
            k_off = k0 + tl.arange(0, BLOCK_K)
            k_mask = (k_off < K) & (k_off < k_lo + ksplit)
            a = tl.load(a_ptr + offs_m[:, None] * sam + k_off[None, :] * sak,
                        mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            w = tl.load(w_ptr + k_off[:, None] * swk + offs_n[None, :] * swn,
                        mask=k_mask[:, None] & n_mask[None, :], other=0).to(tl.float16)
            # Scales are [K // 32, N], NOT one scale per BLOCK_K tile.
            s = tl.load(s_ptr + (k_off[:, None] // 32) * ssb + offs_n[None, :] * ssn,
                        mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float16)
            w = ((w - 128.0) * s).to(tl.float16)
            acc += tl.dot(a, w, out_dtype=tl.float32)
        if ksplit >= K:
            tl.store(y_ptr + offs_m[:, None] * sym + offs_n[None, :] * syn,
                     acc.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])
        else:
            tl.atomic_add(y_ptr + offs_m[:, None] * sym + offs_n[None, :] * syn,
                          acc, mask=m_mask[:, None] & n_mask[None, :])

    return torch, triton, gemm_i8_kernel


def pack(qweight, scales, k, n):
    """Copy byte weights [N, K/32, 32] and scales into GPU K-major layout."""
    import numpy as np

    torch, _, _ = load_backend()
    q = np.ascontiguousarray(qweight).reshape(n, k)
    s = np.asarray(scales, dtype=np.float16).reshape(n, k // 32)
    w = torch.from_numpy(np.ascontiguousarray(q.T)).to("cuda")
    sc = torch.from_numpy(np.ascontiguousarray(s.T)).to("cuda")
    return w, sc


def run_gemm(a, w, sc, n, k, config):
    """Run one validated experimental configuration, including output setup."""
    rows = a.shape[0]
    ksplit = validate_config(k, n, rows, config)
    torch, triton, kernel = load_backend()
    bm, bn, bk, warps, stages, split = config
    y = (torch.zeros((rows, n), device=a.device, dtype=torch.float32) if split > 1
         else torch.empty((rows, n), device=a.device, dtype=torch.float16))
    grid = (triton.cdiv(rows, bm), triton.cdiv(n, bn), split)
    kernel[grid](a, w, sc, y, rows, k, n, ksplit,
                 a.stride(0), a.stride(1), w.stride(0), w.stride(1),
                 sc.stride(0), sc.stride(1), y.stride(0), y.stride(1),
                 BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                 num_warps=warps, num_stages=stages)
    return y.to(torch.float16) if split > 1 else y


def benchmark_calls(call, torch, iterations, warmup):
    """Return CUDA-event microseconds/call, excluding compilation warmup."""
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=positive_int, default=1024)
    parser.add_argument("--n", type=positive_int, default=512)
    parser.add_argument("--rows", type=positive_int, default=1)
    parser.add_argument("--iterations", type=positive_int, default=20)
    parser.add_argument("--warmup", type=nonnegative_int, default=3)
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument("--tune", action="store_true", help="try the full tile/split grid")
    names = ("block-m", "block-n", "block-k", "warps", "stages", "split-k")
    for name, allowed, default in zip(names, TILE_VALUES, DEFAULT_CONFIG):
        parser.add_argument(f"--{name}", type=int, choices=allowed, default=default)
    args = parser.parse_args(argv)
    config = tuple(getattr(args, name.replace("-", "_")) for name in names)
    try:
        validate_config(args.k, args.n, args.rows, config)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        torch, _, _ = load_backend()
    except ImportError as exc:
        parser.error(f"GPU benchmark requires Torch and Triton: {exc}")
    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required")

    import numpy as np
    from triton.compiler.errors import CompilationError
    from triton.runtime.errors import OutOfResources
    from breeze.quant import dequantize_matmul_nbits

    rng = np.random.default_rng(args.seed)
    qw = rng.integers(0, 256, (args.n, args.k // 32, 32), dtype=np.uint8)
    # Round scales once so the reference and kernel use identical scales.
    scales = (rng.random((args.n, args.k // 32)) * 0.01 + 0.001).astype(np.float16)
    dense = dequantize_matmul_nbits(qw, scales, None, bits=8, block_size=32,
                                   K=args.k, N=args.n)
    wf = torch.from_numpy(dense).to("cuda", torch.float16)
    a = torch.from_numpy(rng.standard_normal((args.rows, args.k)).astype(np.float16)).to("cuda") * 0.1
    w, sc = pack(qw, scales, args.k, args.n)
    ref = a @ wf
    out = torch.empty_like(ref)
    fp16 = benchmark_calls(lambda: torch.mm(a, wf, out=out), torch,
                           args.iterations, args.warmup)
    print(f"K={args.k} N={args.n} rows={args.rows}: FP16 {fp16:.3f} us/call")
    best = None
    for cfg in configuration_grid() if args.tune else (config,):
        try:
            validate_config(args.k, args.n, args.rows, cfg)
        except ValueError as exc:
            print(f"SKIPPED {cfg}: {exc}")
            continue
        try:
            got = run_gemm(a, w, sc, args.n, args.k, cfg)
            torch.cuda.synchronize()
            # This is a correctness check, not evidence of hardware validation here.
            torch.testing.assert_close(got, ref, rtol=0.03, atol=0.01)
            elapsed = benchmark_calls(lambda: run_gemm(a, w, sc, args.n, args.k, cfg),
                                      torch, args.iterations, args.warmup)
        except (CompilationError, OutOfResources, RuntimeError) as exc:
            if not args.tune:
                raise
            print(f"SKIPPED {cfg}: {type(exc).__name__}: {exc}")
            continue
        print(f"INT8 {cfg}: {elapsed:.3f} us/call; correctness passed")
        if best is None or elapsed < best[0]:
            best = (elapsed, cfg)
    if best is None:
        print("No GPU configurations completed.")
        return 1
    print(f"Best tested configuration: {best[1]} ({best[0]:.3f} us/call)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
