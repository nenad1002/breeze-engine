"""Experimental Triton int8 dequant-GEMV for symmetric MatMulNBits, zp=128.

Keeps weights as int8 and dequantizes on the fly (W8A16: weight int8, activation
fp16). Packed weight storage is smaller than the dense fp16 path. Performance
and numerical agreement require validation on the target GPU and workload;
dequantization rounding and split-K reduction order can change the result.

Layout for coalescing: weight stored ``[K, N]`` (k-major, N contiguous) and
scales ``[nblk, N]``. Each program owns BLOCK_N contiguous outputs; the inner
[BLOCK_K, BLOCK_N] tile loads BLOCK_N contiguous int8 per k -> full cache lines.
"""
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_i8_kernel(a_ptr, w_ptr, s_ptr, y_ptr,
                    M, K, N, nblk, ksplit,
                    sam, sak, swk, swn, ssb, ssn, sym, syn,
                    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)      # [BLOCK_N]
    n_mask = n_off < N
    k_start = pid_k * ksplit
    k_end = k_start + ksplit
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in range(k_start, k_end, BLOCK_K):
        blk = k0 // BLOCK_K                              # BLOCK_K == block_size (32)
        k_off = k0 + tl.arange(0, BLOCK_K)               # [BLOCK_K]
        a = tl.load(a_ptr + pid_m * sam + k_off * sak).to(tl.float32)         # [BLOCK_K]
        w = tl.load(w_ptr + k_off[:, None] * swk + n_off[None, :] * swn,
                    mask=n_mask[None, :], other=0).to(tl.float32)             # [BLOCK_K, BLOCK_N]
        s = tl.load(s_ptr + blk * ssb + n_off * ssn, mask=n_mask, other=0.0)  # [BLOCK_N]
        acc += tl.sum((w - 128.0) * a[:, None], axis=0) * s.to(tl.float32)
    tl.atomic_add(y_ptr + pid_m * sym + n_off * syn, acc, mask=n_mask)


def gemv_i8(a, w_i8, scale, N, K, nblk, block_size=32, block_n=128, split_k=8, num_warps=4):
    """a [M, K] fp16 @ dequant(w_i8 [K, N], scale [nblk, N]) -> [M, N] fp16."""
    a = a.contiguous()
    M = a.shape[0]
    y = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    while K % (split_k * block_size) != 0 and split_k > 1:
        split_k //= 2
    ksplit = K // split_k
    grid = (triton.cdiv(N, block_n), split_k, M)
    _gemv_i8_kernel[grid](a, w_i8, scale, y, M, K, N, nblk, ksplit,
                          a.stride(0), a.stride(1), w_i8.stride(0), w_i8.stride(1),
                          scale.stride(0), scale.stride(1), y.stride(0), y.stride(1),
                          BLOCK_N=block_n, BLOCK_K=block_size, GROUP=1, num_warps=num_warps)
    return y.to(torch.float16)


def pack_i8(qweight, scales, K, N, block_size=32, dev="cuda"):
    """Raw MatMulNBits int8 weight -> (uint8 [K,N] gpu, fp16 scale [nblk,N] gpu)."""
    nblk = (K + block_size - 1) // block_size
    q = np.ascontiguousarray(qweight).reshape(N, nblk * block_size)[:, :K]     # [N, K]
    s = np.ascontiguousarray(scales, dtype=np.float32).reshape(N, nblk)        # [N, nblk]
    w_i8 = torch.from_numpy(np.ascontiguousarray(q.T, np.uint8)).to(dev)       # [K, N]
    scale = torch.from_numpy(np.ascontiguousarray(s.T, np.float32)).to(dev, torch.float16)  # [nblk, N]
    return w_i8, scale, N, K, nblk
