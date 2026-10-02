"""ctypes bridge to Breeze's compiled INT4/INT8 CPU kernels.

Converts ONNX MatMulNBits weights into the kernel's layout and drives the
fused AVX-512 dequant+FMA matmul.
"""
import ctypes
import os

import numpy as np

_SO = os.environ.get("BREEZE_KERNEL", os.path.join(os.path.dirname(__file__), "_breeze_cpu.so"))
_lib = None
_native_prepack = None
try:
    _lib = ctypes.CDLL(_SO)
    _lib.i4_prepack.restype = ctypes.c_void_p
    _lib.i4_prepack.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _lib.i4_matmul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    _lib.i4_free.argtypes = [ctypes.c_void_p]
    _lib.i4_set_threads.argtypes = [ctypes.c_int]
    _lib.i4_set_hi_prec.argtypes = [ctypes.c_int]
    _lib.i4_linear_attention.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 9 + [ctypes.c_float]
    _lib.i4_causal_conv.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 5
    # Full Qwen3.5 C++ forward
    _lib.i4_qwen_new.restype = ctypes.c_void_p
    _lib.i4_qwen_new.argtypes = [ctypes.c_int] * 13 + [ctypes.c_float, ctypes.c_float, ctypes.c_int]
    _lib.i4_qwen_set_embed.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.i4_qwen_set_rotary.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _lib.i4_qwen_set_linear.argtypes = [ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 15
    _lib.i4_qwen_set_full.argtypes = [ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 11
    _lib.i4_qwen_set_final.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _lib.i4_qwen_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    _lib.i4_qwen_forward.restype = ctypes.c_int
    _lib.i4_qwen_free.argtypes = [ctypes.c_void_p]
    # Full-forward model API (kernel/phi35_decoder.cpp).
    _lib.i4_model_new.restype = ctypes.c_void_p
    _lib.i4_model_new.argtypes = [ctypes.c_int] * 5 + [ctypes.c_int, ctypes.c_float,
                                                       ctypes.c_float, ctypes.c_int]
    _lib.i4_model_set_embed.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.i4_model_set_rotary.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _lib.i4_model_set_layer.argtypes = [ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 7
    _lib.i4_model_set_final.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _lib.i4_model_free.argtypes = [ctypes.c_void_p]
    _lib.i4_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
except (OSError, AttributeError):
    _lib = None

if _lib is not None:
    # Optional symbol: old kernels retain the NumPy fallback until rebuilt.
    _native_prepack = getattr(_lib, "i4_prepack_onnx", None)
    if _native_prepack is not None:
        _native_prepack.restype = ctypes.c_void_p
        _native_prepack.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4


def available():
    return _lib is not None


def set_threads(n):
    """Set the process-global native thread count.

    Configure before inference; concurrent mutation is not thread-safe and
    callers must serialize setting changes with native work.
    """
    if _lib is not None:
        _lib.i4_set_threads(int(n))


def set_hi_prec(on):
    """Select process-global int16 two-pass (~15-bit) versus int8 activation.

    Configure before inference; concurrent mutation is not thread-safe and
    callers must serialize setting changes with native work.
    """
    if _lib is not None:
        _lib.i4_set_hi_prec(1 if on else 0)


def _p(a):
    return a.ctypes.data_as(ctypes.c_void_p) if a is not None else None


def linear_attention(q, k, v, past, decay, beta, *, Hq, Hkv, dk, dv, n_k,
                     decay_per_key_dim, beta_per_head, scale):
    """GatedDeltaNet linear attention via the compiled kernel."""
    q = np.ascontiguousarray(q, np.float32); k = np.ascontiguousarray(k, np.float32)
    v = np.ascontiguousarray(v, np.float32)
    decay = np.ascontiguousarray(decay, np.float32); beta = np.ascontiguousarray(beta, np.float32)
    if past is not None:
        past = np.ascontiguousarray(past, np.float32)
    B, T = q.shape[0], q.shape[1]
    output = np.empty((B, T, max(Hq, Hkv) * dv), np.float32)
    present = np.empty((B, Hkv, dk, dv), np.float32)
    _lib.i4_linear_attention(_p(q), _p(k), _p(v), _p(past), _p(decay), _p(beta),
                             _p(output), _p(present),
                             int(B), int(T), int(Hq), int(Hkv), int(dk), int(dv), int(n_k),
                             int(decay_per_key_dim), int(beta_per_head), float(scale))
    return output, present


def causal_conv(x, w, bias, past, silu):
    """Depthwise causal 1D conv with carry state via the compiled kernel."""
    x = np.ascontiguousarray(x, np.float32); w = np.ascontiguousarray(w, np.float32)
    B, C, L = x.shape
    K = w.shape[-1]
    if bias is not None:
        bias = np.ascontiguousarray(bias, np.float32)
    if past is not None:
        past = np.ascontiguousarray(past, np.float32)
    output = np.empty((B, C, L), np.float32)
    present = np.empty((B, C, K - 1), np.float32)
    _lib.i4_causal_conv(_p(x), _p(w), _p(bias), _p(past), _p(output), _p(present),
                        int(B), int(C), int(L), int(K), int(silu))
    return output, present


def prepack(qweight, scales, qzeros, K, N, bits=4, block_size=32):
    """Rearrange ONNX MatMulNBits weights into the kernel's VNNI int8 layout.

    Produces packed weights (kgroup layout), per-block fp32 scales, per-block
    int8 zero-points (int4 only), and the int32 bias correction 128*sum(w-zp).
    Supports 4-bit (nibble-packed, zp default 8) and 8-bit (byte, zp default 128).
    New kernels transform directly into native buffers without full-tensor
    unpacking/dtype-expansion temporaries. Older kernels use the NumPy path.
    """
    if bits not in (4, 8) or block_size != 32:
        raise ValueError("VNNI prepack requires 4/8-bit weights and block_size=32")
    if (not isinstance(N, (int, np.integer)) or not isinstance(K, (int, np.integer))
            or isinstance(N, (bool, np.bool_)) or isinstance(K, (bool, np.bool_))
            or N > np.iinfo(np.int32).max or K > np.iinfo(np.int32).max):
        raise ValueError("Prepack dimensions must be signed 32-bit integers")
    N, K = int(N), int(K)
    if N <= 0 or K <= 0 or N % 16 or K % 32:
        raise ValueError(f"need N%16==0,K%32==0 (got N={N},K={K})")
    qweight = np.asarray(qweight)
    if qweight.dtype != np.uint8:
        raise ValueError("MatMulNBits packed weights must be uint8")
    nblk = K // 32
    if qweight.size != N * nblk * (4 * bits):
        raise ValueError("Incorrect packed weight size for the supplied dimensions")
    scales = np.asarray(scales)
    if scales.size != N * nblk:
        raise ValueError("Incorrect scale count for the supplied dimensions")
    if qzeros is not None and np.size(qzeros):
        qzeros = np.asarray(qzeros)
        if qzeros.dtype != np.uint8:
            raise ValueError("Packed zero points must be uint8")
        if qzeros.size != N * (nblk if bits == 8 else (nblk + 1) // 2):
            raise ValueError("Incorrect zero-point count for the supplied dimensions")
        if bits == 8 and np.any(qzeros != 128):
            raise ValueError("INT8 VNNI weights require symmetric zero point 128")
        qzeros = np.ascontiguousarray(qzeros) if bits == 4 else None
    else:
        qzeros = None
    if _lib is None:
        raise RuntimeError("CPU kernel unavailable; run build_kernel.py first")
    if _native_prepack is None:
        return _prepack_numpy(qweight, scales, qzeros, K, N, bits, block_size)

    qw = np.ascontiguousarray(qweight)
    scale_dtype = np.float16 if scales.dtype == np.float16 else np.float32
    sc = np.require(scales, dtype=scale_dtype, requirements=["C", "A"])
    handle = _native_prepack(_p(qw), _p(sc), _p(qzeros), N, K, int(bits), sc.itemsize * 8)
    if not handle:
        raise MemoryError(f"Could not allocate packed {bits}-bit weights (N={N}, K={K})")
    return handle


def _prepack_numpy(qweight, scales, qzeros, K, N, bits=4, block_size=32):
    """Legacy layout oracle/fallback; callers validate the input contract first."""
    nblk = K // 32
    ntiles = N // 16

    def tile16(x):  # [N, nblk] -> [ntiles, nblk, 16]
        return np.ascontiguousarray(x.reshape(ntiles, 16, nblk).transpose(0, 2, 1))

    sc = tile16(np.asarray(scales, dtype=np.float32).reshape(N, nblk))

    if bits == 8:
        # qweight: [N, nblk, 32] uint8. Symmetric zp=128 -> signed b = w-128.
        w = np.ascontiguousarray(qweight).reshape(N, nblk, 32).astype(np.int16)
        b_s8 = (w - 128).astype(np.int8)
        # kgroup layout: v[col*4+k] per (tile, block, kgroup of 4 k) -> [nt,nblk,8,64]
        Bv = np.ascontiguousarray(
            b_s8.reshape(ntiles, 16, nblk, 8, 4).transpose(0, 2, 3, 1, 4).reshape(ntiles, nblk, 8, 64))
        colsum = b_s8.astype(np.int32).sum(axis=2)              # sum_k b  [N, nblk]
        corr = tile16((128 * colsum).astype(np.int32))
        return _lib.i4_prepack(
            Bv.ctypes.data_as(ctypes.c_void_p),
            sc.ctypes.data_as(ctypes.c_void_p),
            None,
            corr.ctypes.data_as(ctypes.c_void_p),
            int(N), int(K), int(nblk), 8)

    qw = np.ascontiguousarray(qweight).reshape(N, nblk, 16)

    w = np.empty((N, nblk, 32), dtype=np.uint8)     # w[n, blk, kk] in 0..15
    w[:, :, 0::2] = qw & 0x0F
    w[:, :, 1::2] = qw >> 4

    if qzeros is not None and np.size(qzeros) > 0:
        packed_cols = (nblk + 1) // 2
        qz = np.ascontiguousarray(qzeros).reshape(N, packed_cols)
        zpf = np.empty((N, packed_cols * 2), dtype=np.uint8)
        zpf[:, 0::2] = qz & 0x0F
        zpf[:, 1::2] = qz >> 4
        zp = zpf[:, :nblk].astype(np.int32)         # [N, nblk]
    else:
        zp = np.full((N, nblk), 8, dtype=np.int32)

    # Weights in VNNI kgroup layout: v[col*4 + k] per (tile, block, kgroup of 4 k).
    w2 = w.reshape(ntiles, 16, nblk, 8, 4).transpose(0, 2, 3, 1, 4).reshape(ntiles, nblk, 8, 64)
    Bv = np.ascontiguousarray((w2[..., 0::2] | (w2[..., 1::2] << 4)).astype(np.uint8))  # [nt,nblk,8,32]

    zp_t = tile16(zp.astype(np.int8))
    colsum = w.astype(np.int32).sum(axis=2) - 32 * zp        # sum_k (w - zp)  [N, nblk]
    corr = tile16((128 * colsum).astype(np.int32))

    handle = _lib.i4_prepack(
        Bv.ctypes.data_as(ctypes.c_void_p),
        sc.ctypes.data_as(ctypes.c_void_p),
        zp_t.ctypes.data_as(ctypes.c_void_p),
        corr.ctypes.data_as(ctypes.c_void_p),
        int(N), int(K), int(nblk), 4)
    return handle


def matmul(handle, a2d, N):
    """a2d: contiguous fp32 [M, K]. Returns fp32 [M, N]."""
    a2d = np.ascontiguousarray(a2d, dtype=np.float32)
    M = a2d.shape[0]
    out = np.empty((M, N), dtype=np.float32)
    _lib.i4_matmul(ctypes.c_void_p(handle), a2d.ctypes.data_as(ctypes.c_void_p),
                   int(M), out.ctypes.data_as(ctypes.c_void_p))
    return out


def free(handle):
    if _lib is not None and handle:
        _lib.i4_free(ctypes.c_void_p(handle))


if __name__ == "__main__":
    # Self-test: whole chain (prepack + kernel) vs the verified numpy dequant.
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from quant import dequantize_matmul_nbits

    rng = np.random.default_rng(0)
    N, K = 512, 3072
    nblk = K // 32
    for has_zp in (True, False):
        qweight = rng.integers(0, 256, size=(N, nblk, 16), dtype=np.uint8)
        scales = (rng.random((N * nblk,), dtype=np.float32) * 0.02 + 0.001).astype(np.float16)
        qzeros = rng.integers(0, 256, size=(N * ((nblk + 1) // 2),), dtype=np.uint8) if has_zp else None

        W_ref = dequantize_matmul_nbits(qweight, scales, qzeros, bits=4, block_size=32, K=K, N=N)
        M = 5
        A = rng.standard_normal((M, K), dtype=np.float32)
        C_ref = A @ W_ref

        h = prepack(qweight, scales, qzeros, K, N)
        C = matmul(h, A, N)
        free(h)

        a = C.ravel(); bb = C_ref.ravel()
        rel = np.max(np.abs(C - C_ref)) / (np.max(np.abs(C_ref)) + 1e-9)
        cos = float(a @ bb / (np.linalg.norm(a) * np.linalg.norm(bb)))
        print(f"has_zp={has_zp}: rel={rel:.2e} cosine={cos:.6f} "
              f"{'OK' if cos > 0.999 else 'FAIL'}")
