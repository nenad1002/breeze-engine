"""INT4 (MatMulNBits) dequantization.

Reproduces the com.microsoft ``MatMulNBits`` weight decode so that a plain
``A @ W`` matches the fused int4 kernel.
"""
import numpy as np


def dequantize_matmul_nbits(qweight, scales, qzeros, *, bits, block_size, K, N):
    """Dequantize a MatMulNBits weight into a dense fp32 ``[K, N]`` matrix.

    Layout (4-bit):
      qweight : uint8 ``[N, n_blocks, block_size // 2]``  (2 nibbles/byte, low nibble first)
      scales  : fp16  ``[N * n_blocks]``  (row-major ``[N, n_blocks]``)
      qzeros  : uint8 ``[N * ceil(n_blocks/2)]`` or ``None``  (packed 4-bit; default zp = 8)
    Layout (8-bit):
      qweight : uint8 ``[N, n_blocks, block_size]``  (1 byte/weight)
      qzeros  : uint8 ``[N, n_blocks]`` or ``None``  (default zp = 128)

    Dequant: ``w[n, b, j] = (q - zero_point[n, b]) * scale[n, b]``.
    Returns ``W`` with ``Y = A @ W`` equal to ``MatMulNBits(A) = A @ dequant(B).T``.
    """
    if bits not in (4, 8):
        raise NotImplementedError(f"MatMulNBits bits={bits} not supported (only 4, 8).")
    n_blocks = (K + block_size - 1) // block_size
    scales = np.asarray(scales, dtype=np.float32).reshape(N, n_blocks)

    if bits == 8:
        q = np.ascontiguousarray(qweight).reshape(N, n_blocks, block_size).astype(np.float32)
        if qzeros is not None and np.size(qzeros) > 0:
            zp = np.ascontiguousarray(qzeros).reshape(N, n_blocks).astype(np.float32)
        else:
            zp = np.full((N, n_blocks), 128.0, dtype=np.float32)
        w = (q - zp[:, :, None]) * scales[:, :, None]
        w = w.reshape(N, n_blocks * block_size)[:, :K]
        return np.ascontiguousarray(w.T)

    qweight = np.ascontiguousarray(qweight).reshape(N, n_blocks, block_size // 2)

    # Unpack two 4-bit values per byte: even index = low nibble, odd index = high nibble.
    q = np.empty((N, n_blocks, block_size), dtype=np.uint8)
    q[:, :, 0::2] = qweight & 0x0F
    q[:, :, 1::2] = qweight >> 4
    q = q.astype(np.float32)

    if qzeros is not None and np.size(qzeros) > 0:
        packed_cols = (n_blocks + 1) // 2
        qzeros = np.ascontiguousarray(qzeros).reshape(N, packed_cols)
        zp = np.empty((N, packed_cols * 2), dtype=np.uint8)
        zp[:, 0::2] = qzeros & 0x0F
        zp[:, 1::2] = qzeros >> 4
        zp = zp[:, :n_blocks].astype(np.float32)
    else:
        zp = np.full((N, n_blocks), 8.0, dtype=np.float32)

    w = (q - zp[:, :, None]) * scales[:, :, None]   # [N, n_blocks, block_size]
    w = w.reshape(N, n_blocks * block_size)[:, :K]  # [N, K]
    return np.ascontiguousarray(w.T)                # [K, N]
