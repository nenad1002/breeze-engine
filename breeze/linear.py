"""Linear (matmul) backend.

Selects an available CPU GEMM:
    * ``torch`` fp16 -> stores half as many weight bytes as fp32. Hardware kernel
        selection and performance depend on the torch build and host CPU.
  * fp32 BLAS -> portable fallback.

Weights are stored in the backend's native form so each forward is one GEMM call.
"""
import os

import numpy as np

try:
    import torch

    # Leave capacity for NumPy's BLAS pool used by the other operators.
    torch.set_num_threads(max(1, (os.cpu_count() or 8) // 2))
    _TORCH = True
except Exception:  # noqa: BLE001 - torch is optional
    torch = None
    _TORCH = False


def backend():
    return "torch-fp16" if _TORCH else "numpy-fp32"


def prepare_weight(w_fp32):
    """Store a dequantized ``[K, N]`` fp32 weight in the backend's native form."""
    w = np.ascontiguousarray(w_fp32)
    if _TORCH:
        return torch.from_numpy(w).to(torch.float16)
    return w


def linear(a_fp32, w):
    """Compute ``a @ w`` for a ``[.., K]`` activation, returning fp32 ``[.., N]``."""
    m = a_fp32.reshape(-1, a_fp32.shape[-1])
    n = int(w.shape[1])
    if _TORCH:
        t = torch.from_numpy(np.ascontiguousarray(m, dtype=np.float16))
        out = torch.matmul(t, w).to(torch.float32).numpy()  # fp16 AMX, fp32 accumulate
    else:
        out = m @ w
    return out.reshape(*a_fp32.shape[:-1], n)
