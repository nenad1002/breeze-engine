"""Dense Qwen3.5 (9B/27B) text inference using the shared C++ VNNI kernels.

No per-op Python dispatch: prepacks every matmul, hands the C++ side all
weights + cos/sin caches, and runs the configured hybrid decoder in one call.
Batch 1; KV/conv/recurrent state is carried across calls for decode.
"""
import ctypes
from collections import Counter
from pathlib import Path

import numpy as np

from . import cpu_backend as ib
from .loader import load_graph
from .qwen35_config import QwenConfig
from .qwen35_weights import bind_weights


def _p(a):
    return a.ctypes.data_as(ctypes.c_void_p)


class Qwen35CpuModel:
    """Dense Qwen3.5 (9B/27B) text decoder using Breeze's native CPU backend."""

    def __init__(self, model_path, hi_prec=True, verbose=True, max_seq=4096, embed_path=None,
                 config=None):
        self._keep = []          # keep numpy arrays alive
        self._handles = []       # keep prepack handles alive
        self.m = None
        self._past_len = 0
        self.hi_prec = bool(hi_prec)
        if not ib.available():
            raise RuntimeError("CPU kernel unavailable; run build_kernel.py first")
        abi = getattr(ib._lib, "i4_qwen_abi_version", None)
        if abi is None or abi() < 2:
            raise RuntimeError("Rebuild the CPU kernel with build_kernel.py (Qwen ABI v2 required)")
        g = load_graph(model_path, mmap_external=True)
        c = self.config = QwenConfig.resolve(model_path, config, g)
        if type(max_seq) is not int or not 0 < max_seq <= c.max_position_embeddings:
            raise ValueError("max_seq must be positive and within the model's native context limit")
        self.max_seq = max_seq
        self.H, self.L, self.vocab = c.hidden_size, c.num_hidden_layers, c.vocab_size
        bound, vectors = bind_weights(g, c)  # check ALL shapes before native allocations
        if embed_path is None:
            candidate = Path(model_path).parent / "embeddings.npy"
            embed_path = candidate if candidate.is_file() else None
        self._embed = np.load(embed_path, mmap_mode="r") if embed_path else None
        if self._embed is not None and (self._embed.shape != (self.vocab, self.H)
                                       or self._embed.dtype not in (np.float16, np.float32)):
            raise ValueError(f"Embedding table must be float16/float32 [{self.vocab}, {self.H}]")

        self.m = ib._lib.i4_qwen_new(
            self.L, self.H, self.vocab, c.intermediate_size, c.linear_num_key_heads,
            c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim,
            c.linear_conv_kernel_dim, c.num_attention_heads, c.num_key_value_heads,
            c.head_dim, c.rotary_dim, ctypes.c_float(c.head_dim ** -0.5),
            ctypes.c_float(c.rms_norm_eps), max_seq)
        if not self.m:
            raise MemoryError("Could not allocate Qwen decoder")
        try:
            self._build(g, bound, vectors, verbose)
        except Exception:
            self.close()
            raise

    def _build(self, g, bound, vectors, verbose):
        c = self.config
        handles = {}
        uses = Counter(name for node in bound.values() for name in node.inputs[1:4] if name)
        for fragment, node in bound.items():
            qz = g.initializers[node.inputs[3]] if len(node.inputs) > 3 and node.inputs[3] else None
            h = ib.prepack(g.initializers[node.inputs[1]], g.initializers[node.inputs[2]],
                           qz, int(node.attrs["K"]), int(node.attrs["N"]),
                           bits=int(node.attrs["bits"]), block_size=int(node.attrs["block_size"]))
            if not h:
                raise MemoryError(f"Could not prepack {node.name}")
            self._handles.append(h)
            handles[fragment] = h
            # Release each external mapping after its last use (shared weights
            # are legal); the graph itself never materializes the entire checkpoint.
            for name in node.inputs[1:4]:
                if name:
                    uses[name] -= 1
                    if uses[name] == 0:
                        g.initializers.pop(name, None)
            qz = None
            if verbose and len(handles) % 32 == 0:
                print(f"[qwen35_cpu] prepacked {len(handles)}/{len(bound)} matmuls", flush=True)

        def MM(substr):
            return ctypes.c_void_p(handles[substr])

        def vec(name):
            value = vectors[name]
            if value is None:
                return None  # optional zero conv bias
            a = np.array(value, dtype=np.float32, order="C", copy=True)
            self._keep.append(a)
            return _p(a)

        # ---- inputs_embeds provided at run() time (no embedding table here) ----

        # ---- rotary cos/sin [max_pos, rotary_dim/2] ----
        theta, rotary_dim = c.rope_theta, c.rotary_dim
        inv_freq = 1.0 / (theta ** (np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim))
        pos = np.arange(self.max_seq, dtype=np.float64)
        freqs = np.outer(pos, inv_freq)
        cos = np.ascontiguousarray(np.cos(freqs), dtype=np.float32)
        sin = np.ascontiguousarray(np.sin(freqs), dtype=np.float32)
        self._keep += [cos, sin]
        ib._lib.i4_qwen_set_rotary(self.m, _p(cos), _p(sin))

        # ---- per-layer ----
        for l in range(self.L):
            p = f"model.layers.{l}"
            is_full = c.layer_types[l] == "full_attention"
            if not is_full:
                ib._lib.i4_qwen_set_linear(
                    self.m, l,
                    MM(f"layers.{l}/linear_attn/in_proj_qkv"), MM(f"layers.{l}/linear_attn/in_proj_z"),
                    MM(f"layers.{l}/linear_attn/in_proj_b"), MM(f"layers.{l}/linear_attn/in_proj_a"),
                    MM(f"layers.{l}/linear_attn/out_proj"),
                    MM(f"layers.{l}/mlp/gate_proj"), MM(f"layers.{l}/mlp/up_proj"), MM(f"layers.{l}/mlp/down_proj"),
                    vec(f"{p}.input_layernorm.weight"), vec(f"{p}.linear_attn.conv1d.weight"),
                    vec(f"{p}.linear_attn.conv1d.bias"), vec(f"{p}.linear_attn.neg_exp_A"),
                    vec(f"{p}.linear_attn.dt_bias"), vec(f"{p}.linear_attn.norm.weight"),
                    vec(f"{p}.post_attention_layernorm.weight"))
            else:
                ib._lib.i4_qwen_set_full(
                    self.m, l,
                    MM(f"layers.{l}/attn/q_proj"), MM(f"layers.{l}/attn/k_proj"),
                    MM(f"layers.{l}/attn/v_proj"), MM(f"layers.{l}/attn/o_proj"),
                    MM(f"layers.{l}/mlp/gate_proj"), MM(f"layers.{l}/mlp/up_proj"), MM(f"layers.{l}/mlp/down_proj"),
                    vec(f"{p}.input_layernorm.weight"), vec(f"{p}.attn.q_norm.layernorm.weight"),
                    vec(f"{p}.attn.k_norm.layernorm.weight"), vec(f"{p}.post_attention_layernorm.weight"))

        ib._lib.i4_qwen_set_final(self.m, vec(f"model.layers.{self.L}.final_norm_layernorm.weight"),
                                  MM("lm_head"))
        if verbose:
            print(f"[qwen35_cpu] built ({self.L} layers, hidden={self.H}, "
                  f"{len(self._handles)} matmuls, max_seq={self.max_seq})", flush=True)

    def run(self, inputs_embeds, past_len=0):
        if not self.m:
            raise RuntimeError("Model is closed")
        e = np.asarray(inputs_embeds)
        if e.ndim == 3 and e.shape[0] == 1:
            e = e[0]
        if e.ndim != 2 or e.shape[1] != self.H or e.shape[0] == 0:
            raise ValueError(f"inputs_embeds must have shape [seq, {self.H}] (batch one)")
        s = e.shape[0]
        if type(past_len) is not int or past_len < 0 or past_len + s > self.max_seq:
            raise ValueError(f"Sequence exceeds max_seq={self.max_seq}, or past_len is invalid")
        if past_len != 0 and past_len != self._past_len:
            raise ValueError(f"State contains {self._past_len} tokens, not {past_len}; use 0 to reset")
        e = np.ascontiguousarray(e, dtype=np.float32)
        logits = np.empty((s, self.vocab), dtype=np.float32)
        # The kernel's activation mode is process-global. Reapply for sequential
        # instances; concurrent calls on this backend are not supported.
        ib.set_hi_prec(self.hi_prec)
        status = ib._lib.i4_qwen_forward(self.m, _p(e), int(s), past_len, _p(logits))
        if status != 0:
            raise RuntimeError(f"Native Qwen forward rejected the call (status={status})")
        self._past_len = past_len + s
        return logits

    def embed(self, token_ids):
        if self._embed is None:
            raise ValueError("No embedding table; pass embed_path or prepare embeddings.npy")
        ids = np.asarray(token_ids)
        if ids.ndim != 1 or ids.size == 0 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("token_ids must be a nonempty integer vector")
        if ids.min() < 0 or ids.max() >= self.vocab:
            raise ValueError("Token ID is outside the vocabulary")
        return np.ascontiguousarray(self._embed[ids], dtype=np.float32)

    def prefill(self, prompt_ids, chunk_size=256):
        """Bound temporary activations/logits; return only last-token logits."""
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        ids = list(prompt_ids)
        if not ids or len(ids) > self.max_seq:
            raise ValueError("Prompt is empty or exceeds max_seq")
        last = None
        for start in range(0, len(ids), chunk_size):
            last = self.run(self.embed(ids[start:start + chunk_size]), past_len=start)[-1:].copy()
        return last

    def generate(self, prompt_ids, max_new_tokens=64, eos_ids=(), chunk_size=256):
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if max_new_tokens == 0:
            return []
        ids = list(prompt_ids)
        if len(ids) + max_new_tokens - 1 > self.max_seq:
            raise ValueError("Prompt and requested decode steps exceed max_seq")
        logits = self.prefill(ids, chunk_size)
        past = len(ids)
        out = []
        for step in range(max_new_tokens):
            nxt = int(logits[-1].argmax())
            out.append(nxt)
            if nxt in eos_ids or step + 1 == max_new_tokens:
                break
            logits = self.run(self.embed([nxt]), past_len=past)
            past += 1
        return out

    def close(self):
        if getattr(self, "m", None):
            ib._lib.i4_qwen_free(self.m)
            self.m = None
        for h in getattr(self, "_handles", ()):
            ib.free(h)
        self._handles = []
        self._keep = []
        self._embed = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


# Compatibility class name within this architecture-specific module.
QwenCppModel = Qwen35CpuModel
