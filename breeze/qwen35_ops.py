"""Qwen3.5 causal-convolution and gated-delta operators, computing in float32.

Derived from MIT-licensed CPU operator algorithms by Microsoft Corporation.
See THIRD_PARTY_NOTICES.md for the applicable copyright and license notice.
"""
import numpy as np


def causal_conv_with_state(x, weight, bias=None, past_state=None, activation="none"):
    """Depthwise causal 1D conv with carry state (ndim=1).

    x:          (B, C, L)
    weight:     (C, 1, K)  depthwise
    bias:       (C,)       optional
    past_state: (B, C, K-1) optional (zeros if None)
    returns (output (B, C, L), present_state (B, C, K-1))
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    B, C, L = x.shape
    K = weight.shape[2]
    pad = K - 1
    w = np.asarray(weight, dtype=np.float32).reshape(C, K)

    if past_state is None:
        state = np.zeros((B, C, pad), dtype=np.float32)
    else:
        state = np.ascontiguousarray(past_state, dtype=np.float32)

    # virtual window = [past_state | x]  -> (B, C, pad + L)
    virtual = np.concatenate([state, x], axis=2)

    # depthwise conv: out[..., l] = sum_k w[:, k] * virtual[..., l + k]
    out = np.zeros((B, C, L), dtype=np.float32)
    for k in range(K):
        out += w[None, :, k, None] * virtual[:, :, k:k + L]
    if bias is not None:
        out += np.asarray(bias, dtype=np.float32)[None, :, None]
    if activation in ("silu", "swish"):
        out = out / (1.0 + np.exp(-out))

    present = virtual[:, :, L:L + pad].copy()  # last K-1 of [past | x]
    return out.astype(np.float32), present.astype(np.float32)


def linear_attention(query, key, value, past_state=None, decay=None, beta=None,
                     *, q_num_heads, kv_num_heads, update_rule="gated_delta",
                     scale=0.0):
    """GatedDeltaNet linear attention (gated delta rule).

    query: (B, T, H_q*d_k)   key: (B, T, n_k*d_k)   value: (B, T, H_kv*d_v)
    past_state: (B, H_kv, d_k, d_v) opt (zeros if None)
    decay: (B, T, H_kv*d_k) per-key-dim  or (B, T, H_kv) per-head  (log-space g)
    beta:  (B, T, H_kv) or (B, T, 1)
    returns (output (B, T, max(H_q,H_kv)*d_v), present_state (B, H_kv, d_k, d_v))
    """
    query = np.ascontiguousarray(query, dtype=np.float32)
    key = np.ascontiguousarray(key, dtype=np.float32)
    value = np.ascontiguousarray(value, dtype=np.float32)
    B, T, qh = query.shape
    d_k = qh // q_num_heads
    n_k = key.shape[2] // d_k
    d_v = value.shape[2] // kv_num_heads
    if scale == 0.0:
        scale = 1.0 / np.sqrt(d_k)

    needs_decay = update_rule in ("gated", "gated_delta")
    needs_beta = update_rule in ("delta", "gated_delta")
    needs_retrieval = needs_beta
    decay_per_key_dim = decay is not None and decay.shape[2] == kv_num_heads * d_k
    beta_per_head = beta is not None and beta.shape[2] == kv_num_heads

    out_hidden = max(q_num_heads, kv_num_heads) * d_v
    output = np.zeros((B, T, out_hidden), dtype=np.float32)

    if past_state is None:
        S = np.zeros((B, kv_num_heads, d_k, d_v), dtype=np.float32)
    else:
        S = np.array(past_state, dtype=np.float32)  # copy (written in place)

    kv_per_k = kv_num_heads // n_k
    hpg = q_num_heads // kv_num_heads if q_num_heads >= kv_num_heads else 0

    if decay is not None:
        decay = np.asarray(decay, dtype=np.float32)
    if beta is not None:
        beta = np.asarray(beta, dtype=np.float32)

    # Fast path: inverse GQA (q<kv, e.g. Qwen3.5), vectorized over kv heads.
    if hpg == 0 and update_rule == "gated_delta":
        kidx = np.arange(kv_num_heads) // kv_per_k
        qidx = np.arange(kv_num_heads) * q_num_heads // kv_num_heads
        for b in range(B):
            s = S[b]                                             # [H, dk, dv]
            kb = key[b].reshape(T, n_k, d_k)[:, kidx, :]         # [T, H, dk]
            qb = query[b].reshape(T, q_num_heads, d_k)[:, qidx, :]
            vb = value[b].reshape(T, kv_num_heads, d_v)          # [T, H, dv]
            if decay_per_key_dim:
                gb = np.exp(decay[b].reshape(T, kv_num_heads, d_k))[:, :, :, None]
            else:
                gb = np.exp(decay[b].reshape(T, kv_num_heads))[:, :, None, None]
            bb = (beta[b].reshape(T, kv_num_heads) if beta_per_head
                  else np.broadcast_to(beta[b].reshape(T, 1), (T, kv_num_heads)))
            for t in range(T):
                s *= gb[t]
                retrieved = np.matmul(kb[t][:, None, :], s)[:, 0, :]     # [H, dv]
                delta = bb[t][:, None] * (vb[t] - retrieved)
                s += kb[t][:, :, None] * delta[:, None, :]
                out_t = np.matmul(qb[t][:, None, :], s)[:, 0, :]         # [H, dv]
                output[b, t] = (scale * out_t).reshape(-1)
            S[b] = s
        return output, S

    for b in range(B):
        for h_kv in range(kv_num_heads):
            h_k = h_kv // kv_per_k
            s = S[b, h_kv]  # view (d_k, d_v), updated in place
            for t in range(T):
                kt = key[b, t, h_k * d_k:(h_k + 1) * d_k]
                vt = value[b, t, h_kv * d_v:(h_kv + 1) * d_v]

                if needs_decay:
                    if decay_per_key_dim:
                        gt = decay[b, t, h_kv * d_k:(h_kv + 1) * d_k]
                        s *= np.exp(gt)[:, None]
                    else:
                        s *= np.exp(decay[b, t, h_kv])

                if needs_retrieval:
                    retrieved = s.T @ kt          # (d_v,)
                if needs_beta:
                    bt = beta[b, t, h_kv] if beta_per_head else beta[b, t, 0]
                    delta = bt * (vt - retrieved)
                    s += np.outer(kt, delta)
                else:
                    s += np.outer(kt, vt)

                if hpg > 0:
                    for g in range(hpg):
                        h_q = h_kv * hpg + g
                        qt = query[b, t, h_q * d_k:(h_q + 1) * d_k]
                        output[b, t, h_q * d_v:(h_q + 1) * d_v] = scale * (qt @ s)
                else:
                    h_q = h_kv * q_num_heads // kv_num_heads
                    qt = query[b, t, h_q * d_k:(h_q + 1) * d_k]
                    output[b, t, h_kv * d_v:(h_kv + 1) * d_v] = scale * (qt @ s)
            S[b, h_kv] = s
    return output, S


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # ---- CausalConvWithState: prefill == step-by-step decode with carry ----
    B, C, L, K = 1, 8, 6, 4
    x = rng.standard_normal((B, C, L), dtype=np.float32)
    w = rng.standard_normal((C, 1, K), dtype=np.float32)
    bias = rng.standard_normal((C,), dtype=np.float32)
    full, st_full = causal_conv_with_state(x, w, bias, None, "silu")
    # replay one token at a time, carrying state
    st = None
    outs = []
    for t in range(L):
        o, st = causal_conv_with_state(x[:, :, t:t + 1], w, bias, st, "silu")
        outs.append(o)
    step = np.concatenate(outs, axis=2)
    print("conv prefill==decode:", np.allclose(full, step, atol=1e-5),
          "state match:", np.allclose(st_full, st, atol=1e-5))

    # ---- LinearAttention: Qwen3.5 shapes, prefill == decode carry ----
    Hq, Hkv, dk, dv, n_k = 16, 32, 128, 128, 16
    T = 5
    q = rng.standard_normal((1, T, Hq * dk), dtype=np.float32) * 0.1
    k = rng.standard_normal((1, T, n_k * dk), dtype=np.float32).reshape(1, T, n_k, dk) * 0.1
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    k = k.reshape(1, T, n_k * dk)
    v = rng.standard_normal((1, T, Hkv * dv), dtype=np.float32) * 0.1
    g = -np.abs(rng.standard_normal((1, T, Hkv * dk), dtype=np.float32)) * 0.1
    beta = 1.0 / (1.0 + np.exp(-rng.standard_normal((1, T, Hkv), dtype=np.float32)))
    kw = dict(q_num_heads=Hq, kv_num_heads=Hkv, update_rule="gated_delta", scale=1.0)
    out_full, S_full = linear_attention(q, k, v, None, g, beta, **kw)
    S = None
    outs = []
    for t in range(T):
        o, S = linear_attention(q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1], S,
                                g[:, t:t + 1], beta[:, t:t + 1], **kw)
        outs.append(o)
    step = np.concatenate(outs, axis=1)
    print("LA   prefill==decode:", np.allclose(out_full, step, atol=1e-4),
          "state match:", np.allclose(S_full, S, atol=1e-4),
          "out shape:", out_full.shape)
