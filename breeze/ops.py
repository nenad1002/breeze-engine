"""Operator kernels for the INT4 Phi-3.5 ONNX graph.

Numerical operators use float32 intermediates for stability. Matrix products
use the configured native or numerical backend.
"""
import numpy as np

from . import cpu_backend, linear
from .loader import ONNX_TO_NP

_REGISTRY = {}


def op(name):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def get(op_type):
    if op_type not in _REGISTRY:
        raise NotImplementedError(f"Operator {op_type!r} is not implemented in breeze.")
    return _REGISTRY[op_type]


def _f32(x):
    return np.asarray(x).astype(np.float32, copy=False)


# ---------------------------------------------------------------- tensor / shape

@op("Constant")
def _constant(ctx, node, inputs):
    return [node.attrs["value"]]


@op("Shape")
def _shape(ctx, node, inputs):
    return [np.array(inputs[0].shape, dtype=np.int64)]


@op("Gather")
def _gather(ctx, node, inputs):
    data, indices = inputs[0], inputs[1]
    axis = node.attrs.get("axis", 0)
    return [np.take(data, indices.astype(np.int64), axis=axis)]


@op("Cast")
def _cast(ctx, node, inputs):
    return [inputs[0].astype(ONNX_TO_NP[node.attrs["to"]])]


@op("Greater")
def _greater(ctx, node, inputs):
    return [np.greater(inputs[0], inputs[1])]


@op("Sub")
def _sub(ctx, node, inputs):
    return [inputs[0] - inputs[1]]


@op("Mul")
def _mul(ctx, node, inputs):
    return [inputs[0] * inputs[1]]


@op("Sigmoid")
def _sigmoid(ctx, node, inputs):
    return [1.0 / (1.0 + np.exp(-_f32(inputs[0])))]


@op("ReduceSum")
def _reduce_sum(ctx, node, inputs):
    axes = None
    if len(inputs) > 1 and inputs[1] is not None:
        axes = tuple(int(a) for a in np.ravel(inputs[1]))
    keepdims = bool(node.attrs.get("keepdims", 1))
    return [np.sum(inputs[0], axis=axes, keepdims=keepdims)]


@op("If")
def _if(ctx, node, inputs):
    cond = bool(np.ravel(inputs[0])[0])
    branch = node.subgraphs["then_branch" if cond else "else_branch"]
    return ctx.run_graph(branch, {})


# ---------------------------------------------------------------- normalization

def _rms_norm(x, weight, eps):
    x = _f32(x)
    inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x * inv) * _f32(weight)


@op("SimplifiedLayerNormalization")
def _simplified_layernorm(ctx, node, inputs):
    eps = float(node.attrs.get("epsilon", 1e-5))
    return [_rms_norm(inputs[0], inputs[1], eps)]


@op("SkipSimplifiedLayerNormalization")
def _skip_simplified_layernorm(ctx, node, inputs):
    eps = float(node.attrs.get("epsilon", 1e-5))
    total = _f32(inputs[0]) + _f32(inputs[1])            # x + skip
    if len(inputs) > 3 and inputs[3] is not None:        # optional bias
        total = total + _f32(inputs[3])
    normed = _rms_norm(total, inputs[2], eps)
    # outputs: [output, mean (unused), inv_std (unused), input_skip_bias_sum]
    return [normed, None, None, total]


# ---------------------------------------------------------------- matmul (int4)

@op("MatMulNBits")
def _matmul_nbits(ctx, node, inputs):
    # entry is ("i4", handle, N) / ("linear", w, N) / ("dense", W_fp32[K,N], N).
    kind, w, N = ctx.weight_cache[node.name]
    a = _f32(inputs[0])
    a2d = a.reshape(-1, a.shape[-1])
    if kind == "i4":
        out2d = cpu_backend.matmul(w, a2d, N)
    elif kind == "dense":
        out2d = a2d @ w
    else:
        out2d = linear.linear(a2d, w)
    return [out2d.reshape(*a.shape[:-1], N)]


# ---------------------------------------------------------------- attention

def _apply_rope(x, cos_cache, sin_cache, pos, interleaved):
    # x: [b, H, s, hs];  caches: [max_seq, hs/2]
    cos = _f32(cos_cache)[pos][None, None]               # [1, 1, s, hs/2]
    sin = _f32(sin_cache)[pos][None, None]
    if interleaved:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


@op("GroupQueryAttention")
def _group_query_attention(ctx, node, inputs):
    # Qwen3.5 full-attention layers pass SEPARATE, already-roped q/k/v (do_rotary=0);
    # Phi-3.5 passes a single packed qkv with rotary done inside.
    if int(node.attrs.get("do_rotary", 0)) == 0 and inputs[1] is not None and inputs[2] is not None:
        return _gqa_separate(node, inputs)
    # torch path fuses rotary + attention (SDPA) into a few calls, avoiding the
    # ~480 small NumPy dispatches this op otherwise costs over 32 layers.
    if linear._TORCH:
        return _gqa_torch(node, inputs)
    return _gqa_numpy(node, inputs)


def _gqa_separate(node, inputs):
    q, k, v = _f32(inputs[0]), _f32(inputs[1]), _f32(inputs[2])
    H = int(node.attrs["num_heads"])
    KV = int(node.attrs["kv_num_heads"])
    b, s, _ = q.shape
    hs = q.shape[2] // H
    scale = float(node.attrs.get("scale", 0.0)) or 1.0 / np.sqrt(hs)
    past_k, past_v = inputs[3], inputs[4]

    q = q.reshape(b, s, H, hs).transpose(0, 2, 1, 3)
    k = k.reshape(b, s, KV, hs).transpose(0, 2, 1, 3)
    v = v.reshape(b, s, KV, hs).transpose(0, 2, 1, 3)

    past_s = past_k.shape[2] if (past_k is not None and np.size(past_k) > 0) else 0
    if past_s > 0:
        k = np.concatenate([_f32(past_k), k], axis=2)
        v = np.concatenate([_f32(past_v), v], axis=2)
    present_k, present_v = k, v

    if KV < H:
        rep = H // KV
        k = np.repeat(k, rep, axis=1)
        v = np.repeat(v, rep, axis=1)

    total = k.shape[2]
    scores = (q @ np.swapaxes(k, -1, -2)) * scale
    qpos = (past_s + np.arange(s)).reshape(s, 1)
    kpos = np.arange(total).reshape(1, total)
    scores = np.where(kpos <= qpos, scores, -np.inf)
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=-1, keepdims=True)
    out = (p @ v).transpose(0, 2, 1, 3).reshape(b, s, H * hs)
    return [out, present_k, present_v]


def _gqa_numpy(node, inputs):
    qkv = _f32(inputs[0])
    past_k, past_v = inputs[3], inputs[4]
    cos_cache, sin_cache = inputs[7], inputs[8]
    H = int(node.attrs["num_heads"])
    KV = int(node.attrs["kv_num_heads"])
    interleaved = bool(node.attrs.get("rotary_interleaved", 0))
    b, s, hidden = qkv.shape
    hs = hidden // (H + 2 * KV)
    scale = float(node.attrs.get("scale", 0.0)) or 1.0 / np.sqrt(hs)

    q = qkv[..., :H * hs].reshape(b, s, H, hs).transpose(0, 2, 1, 3)
    k = qkv[..., H * hs:(H + KV) * hs].reshape(b, s, KV, hs).transpose(0, 2, 1, 3)
    v = qkv[..., (H + KV) * hs:].reshape(b, s, KV, hs).transpose(0, 2, 1, 3)

    past_s = past_k.shape[2] if (past_k is not None and np.size(past_k) > 0) else 0
    pos = past_s + np.arange(s)

    q = _apply_rope(q, cos_cache, sin_cache, pos, interleaved)
    k = _apply_rope(k, cos_cache, sin_cache, pos, interleaved)

    if past_s > 0:
        k = np.concatenate([_f32(past_k), k], axis=2)
        v = np.concatenate([_f32(past_v), v], axis=2)
    present_k, present_v = k, v

    if KV < H:                                           # grouped-query expansion
        rep = H // KV
        k = np.repeat(k, rep, axis=1)
        v = np.repeat(v, rep, axis=1)

    total = k.shape[2]
    scores = (q @ np.swapaxes(k, -1, -2)) * scale        # [b, H, s, total]
    qpos = pos.reshape(s, 1)
    kpos = np.arange(total).reshape(1, total)
    scores = np.where(kpos <= qpos, scores, -np.inf)     # causal mask
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=-1, keepdims=True)
    out = p @ v                                          # [b, H, s, hs]
    out = out.transpose(0, 2, 1, 3).reshape(b, s, H * hs)
    return [out, present_k, present_v]


def _rope_torch(x, cos, sin, interleaved):
    cos, sin = cos[None, None], sin[None, None]
    if interleaved:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = linear.torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return linear.torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def _gqa_torch(node, inputs):
    torch = linear.torch
    H = int(node.attrs["num_heads"])
    KV = int(node.attrs["kv_num_heads"])
    interleaved = bool(node.attrs.get("rotary_interleaved", 0))
    qkv_np = inputs[0]
    b, s, hidden = qkv_np.shape
    hs = hidden // (H + 2 * KV)
    scale = float(node.attrs.get("scale", 0.0)) or 1.0 / np.sqrt(hs)

    qkv = torch.from_numpy(np.ascontiguousarray(qkv_np, dtype=np.float32))
    q = qkv[..., :H * hs].reshape(b, s, H, hs).permute(0, 2, 1, 3)
    k = qkv[..., H * hs:(H + KV) * hs].reshape(b, s, KV, hs).permute(0, 2, 1, 3)
    v = qkv[..., (H + KV) * hs:].reshape(b, s, KV, hs).permute(0, 2, 1, 3)

    past_k, past_v = inputs[3], inputs[4]
    past_s = past_k.shape[2] if (past_k is not None and np.size(past_k) > 0) else 0
    pos = past_s + np.arange(s)
    cos = torch.from_numpy(np.ascontiguousarray(inputs[7][pos], dtype=np.float32))
    sin = torch.from_numpy(np.ascontiguousarray(inputs[8][pos], dtype=np.float32))

    q = _rope_torch(q, cos, sin, interleaved)
    k = _rope_torch(k, cos, sin, interleaved)

    if past_s > 0:
        k = torch.cat([torch.from_numpy(np.ascontiguousarray(past_k, dtype=np.float32)), k], dim=2)
        v = torch.cat([torch.from_numpy(np.ascontiguousarray(past_v, dtype=np.float32)), v], dim=2)
    present_k, present_v = k, v

    if KV < H:
        rep = H // KV
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    if past_s == 0:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    else:
        total = k.shape[2]
        qpos = torch.arange(s).reshape(s, 1) + past_s
        kpos = torch.arange(total).reshape(1, total)
        mask = torch.where(kpos <= qpos, 0.0, float("-inf"))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)

    out = out.permute(0, 2, 1, 3).reshape(b, s, H * hs)
    return [out.numpy(), present_k.numpy(), present_v.numpy()]


# ================================================================ Qwen3.5 ops
from .qwen35_ops import causal_conv_with_state as _cconv, linear_attention as _lin_attn


@op("Add")
def _add(ctx, node, inputs):
    return [inputs[0] + inputs[1]]


@op("Div")
def _div(ctx, node, inputs):
    return [inputs[0] / inputs[1]]


@op("Sqrt")
def _sqrt(ctx, node, inputs):
    return [np.sqrt(_f32(inputs[0]))]


@op("Reciprocal")
def _reciprocal(ctx, node, inputs):
    return [1.0 / _f32(inputs[0])]


@op("Neg")
def _neg(ctx, node, inputs):
    return [-inputs[0]]


@op("Exp")
def _exp(ctx, node, inputs):
    return [np.exp(_f32(inputs[0]))]


@op("Softplus")
def _softplus(ctx, node, inputs):
    return [np.logaddexp(0.0, _f32(inputs[0])).astype(np.float32)]


@op("Tanh")
def _tanh(ctx, node, inputs):
    return [np.tanh(_f32(inputs[0]))]


@op("QuickGelu")
def _quickgelu(ctx, node, inputs):
    alpha = float(node.attrs.get("alpha", 1.702))
    x = _f32(inputs[0])
    return [x * (1.0 / (1.0 + np.exp(-alpha * x)))]


@op("Transpose")
def _transpose(ctx, node, inputs):
    perm = node.attrs.get("perm")
    return [np.transpose(inputs[0], perm)]


@op("Reshape")
def _reshape(ctx, node, inputs):
    shape = [int(s) for s in np.ravel(inputs[1])]
    if not int(node.attrs.get("allowzero", 0)):
        shape = [inputs[0].shape[i] if s == 0 else s for i, s in enumerate(shape)]
    return [np.reshape(inputs[0], shape)]


@op("Concat")
def _concat(ctx, node, inputs):
    axis = int(node.attrs.get("axis", 0))
    return [np.concatenate(list(inputs), axis=axis)]


@op("Split")
def _split(ctx, node, inputs):
    axis = int(node.attrs.get("axis", 0))
    x = inputs[0]
    if len(inputs) > 1 and inputs[1] is not None:
        sizes = [int(s) for s in np.ravel(inputs[1])]
    elif "split" in node.attrs:
        sizes = [int(s) for s in np.ravel(node.attrs["split"])]
    else:
        n = int(node.attrs.get("num_outputs", len(node.outputs)))
        return list(np.array_split(x, n, axis=axis))
    idx = np.cumsum(sizes)[:-1]
    return list(np.split(x, idx, axis=axis))


@op("Range")
def _range(ctx, node, inputs):
    start, limit, delta = np.ravel(inputs[0])[0], np.ravel(inputs[1])[0], np.ravel(inputs[2])[0]
    return [np.arange(start, limit, delta)]


@op("Where")
def _where(ctx, node, inputs):
    return [np.where(inputs[0], inputs[1], inputs[2])]


@op("Squeeze")
def _squeeze(ctx, node, inputs):
    if len(inputs) > 1 and inputs[1] is not None:
        axes = tuple(int(a) for a in np.ravel(inputs[1]))
    else:
        axes = node.attrs.get("axes")
        axes = tuple(axes) if axes is not None else None
    return [np.squeeze(inputs[0], axes)]


@op("Unsqueeze")
def _unsqueeze(ctx, node, inputs):
    if len(inputs) > 1 and inputs[1] is not None:
        axes = [int(a) for a in np.ravel(inputs[1])]
    else:
        axes = list(node.attrs.get("axes", []))
    x = inputs[0]
    for ax in sorted(axes):
        x = np.expand_dims(x, ax)
    return [x]


@op("Expand")
def _expand(ctx, node, inputs):
    shape = tuple(int(s) for s in np.ravel(inputs[1]))
    target = np.broadcast_shapes(inputs[0].shape, shape)
    return [np.broadcast_to(inputs[0], target).copy()]


@op("Equal")
def _equal(ctx, node, inputs):
    return [np.equal(inputs[0], inputs[1])]


@op("RotaryEmbedding")
def _rotary_embedding_op(ctx, node, inputs):
    x = _f32(inputs[0])
    pos = np.asarray(inputs[1]).astype(np.int64)
    cos = _f32(inputs[2])
    sin = _f32(inputs[3])
    num_heads = int(node.attrs.get("num_heads", 0))
    rotary_dim = int(node.attrs.get("rotary_embedding_dim", 0))
    interleaved = bool(node.attrs.get("interleaved", 0))

    was3d = x.ndim == 3
    if was3d:                                    # [B, S, H*hs]
        B, S, hidden = x.shape
        hs = hidden // num_heads
        x = x.reshape(B, S, num_heads, hs)
    else:                                        # [B, H, S, hs]
        B, H, S, hs = x.shape
        num_heads = H
        x = x.transpose(0, 2, 1, 3)              # -> [B, S, H, hs]
    if rotary_dim == 0:
        rotary_dim = hs
    half = rotary_dim // 2

    pos2 = pos.reshape(B, S) if pos.size == B * S else pos.reshape(-1)[:S][None].repeat(B, 0)
    c = cos[pos2][:, :, None, :half]             # [B, S, 1, half]
    s = sin[pos2][:, :, None, :half]

    xr = x[..., :rotary_dim]
    xp = x[..., rotary_dim:]
    if interleaved:
        x1, x2 = xr[..., 0::2], xr[..., 1::2]
        o = np.empty_like(xr)
        o[..., 0::2] = x1 * c - x2 * s
        o[..., 1::2] = x1 * s + x2 * c
    else:
        x1, x2 = xr[..., :half], xr[..., half:]
        o = np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)
    out = np.concatenate([o, xp], axis=-1)       # [B, S, H, hs]
    if was3d:
        return [out.reshape(B, S, num_heads * hs).astype(np.float32)]
    return [out.transpose(0, 2, 1, 3).astype(np.float32)]


@op("LinearAttention")
def _linear_attention_op(ctx, node, inputs):
    def opt(i):
        v = inputs[i] if len(inputs) > i else None
        return None if v is None or np.size(v) == 0 else _f32(v)
    q, k, v = _f32(inputs[0]), _f32(inputs[1]), _f32(inputs[2])
    past, decay, beta = opt(3), opt(4), opt(5)
    Hq = int(node.attrs["q_num_heads"])
    Hkv = int(node.attrs["kv_num_heads"])
    rule = node.attrs.get("update_rule", "gated_delta")
    scale = float(node.attrs.get("scale", 0.0))
    dk = q.shape[2] // Hq
    if scale == 0.0:
        scale = 1.0 / np.sqrt(dk)

    if cpu_backend.available() and rule == "gated_delta":
        dv = v.shape[2] // Hkv
        n_k = k.shape[2] // dk
        dpkd = 1 if (decay is not None and decay.shape[2] == Hkv * dk) else 0
        bph = 1 if (beta is not None and beta.shape[2] == Hkv) else 0
        return list(cpu_backend.linear_attention(
            q, k, v, past, decay, beta, Hq=Hq, Hkv=Hkv, dk=dk, dv=dv, n_k=n_k,
            decay_per_key_dim=dpkd, beta_per_head=bph, scale=scale))

    out, present = _lin_attn(q, k, v, past, decay, beta, q_num_heads=Hq,
                             kv_num_heads=Hkv, update_rule=rule, scale=scale)
    return [out, present]


@op("CausalConvWithState")
def _causal_conv_op(ctx, node, inputs):
    bias = inputs[2] if len(inputs) > 2 and inputs[2] is not None else None
    past = inputs[3] if (len(inputs) > 3 and inputs[3] is not None and np.size(inputs[3]) > 0) else None
    act = node.attrs.get("activation", "none")
    silu = 1 if act in ("silu", "swish") else 0
    x = _f32(inputs[0])
    w = _f32(inputs[1])
    if cpu_backend.available():
        return list(cpu_backend.causal_conv(
            x, w, None if bias is None else _f32(bias),
            None if past is None else _f32(past), silu))
    out, present = _cconv(x, w, None if bias is None else _f32(bias),
                          None if past is None else _f32(past), act)
    return [out, present]

