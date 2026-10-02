"""Build and run the whole Phi-3.5 forward in C++ (kernel/phi35_decoder.cpp).

Every matmul, rmsnorm, GQA and SiLU runs in compiled code with no per-op Python
overhead, and the 161 int4 matmuls execute back-to-back so the memory subsystem
stays streaming. Prefill only (past_len == 0), batch == 1, 1 to 8 tokens.
"""
import ctypes

import numpy as np

from . import cpu_backend as ib
from .loader import load_graph


def _f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


class Phi35CpuModel:
    def __init__(self, model_path, verbose=True):
        assert ib.available(), "int4 kernel (.so) not available"
        g = load_graph(model_path)
        self.g = g
        self._handles = []      # keep prepack handles alive
        self._keep = []         # keep numpy arrays alive (embed/norms/rotary)

        # --- config from a layer-0 qkv node + GQA node ---
        L = 32
        gqa = next(n for n in g.nodes if n.op_type == "GroupQueryAttention")
        nh = int(gqa.attrs["num_heads"])
        scale = float(gqa.attrs["scale"])
        qkv0 = g.initializers["model.layers.0.attn.qkv_proj.MatMulNBits.qweight"]
        H = 3072
        hs = H // nh
        inter = int(g.initializers[
            "model.layers.0.mlp.gate_proj.MatMulNBits.qweight"].shape[0])
        vocab = int(g.initializers["model.embed_tokens.weight"].shape[0])
        eps = 9.999999747378752e-06
        self.H, self.vocab, self.L = H, vocab, L

        self.m = ib._lib.i4_model_new(L, H, nh, hs, inter, vocab,
                                      ctypes.c_float(eps), ctypes.c_float(scale), hs)

        # --- embed (fp16 -> pass raw uint16) ---
        embed = np.ascontiguousarray(
            g.initializers["model.embed_tokens.weight"].view(np.uint16))
        self._keep.append(embed)
        ib._lib.i4_model_set_embed(self.m, embed.ctypes.data_as(ctypes.c_void_p))

        # --- rotary caches: evaluate the If subgraph once ---
        cos_cache, sin_cache = self._eval_rotary()
        cos = _f32(cos_cache); sin = _f32(sin_cache)
        self._keep += [cos, sin]
        ib._lib.i4_model_set_rotary(self.m, cos.ctypes.data_as(ctypes.c_void_p),
                                    sin.ctypes.data_as(ctypes.c_void_p))

        # --- per-layer weights ---
        def prepack_named(prefix):
            qw = g.initializers[prefix + ".qweight"]
            sc = g.initializers[prefix + ".scales"]
            qz = g.initializers.get(prefix + ".qzeros")
            N = int(qw.shape[0]); K = int(np.prod(qw.shape[1:])) * 2
            h = ib.prepack(qw, sc, qz, K, N)
            self._handles.append(h)
            return h

        def norm(name):
            w = _f32(g.initializers[name])
            self._keep.append(w)
            return w.ctypes.data_as(ctypes.c_void_p)

        for l in range(L):
            p = f"model.layers.{l}"
            qkv = prepack_named(f"{p}.attn.qkv_proj.MatMulNBits")
            o = prepack_named(f"{p}.attn.o_proj.MatMulNBits")
            gate = prepack_named(f"{p}.mlp.gate_proj.MatMulNBits")
            up = prepack_named(f"{p}.mlp.up_proj.MatMulNBits")
            down = prepack_named(f"{p}.mlp.down_proj.MatMulNBits")
            ib._lib.i4_model_set_layer(
                self.m, l, qkv, o, gate, up, down,
                norm(f"{p}.input_layernorm.weight"),
                norm(f"{p}.post_attention_layernorm.weight"))

        # --- final norm + lm_head ---
        lm = ib.prepack(
            g.initializers["lm_head.MatMul.weight_Q4"],
            g.initializers["lm_head.MatMul.weight_scales"],
            None, 3072, vocab)
        self._handles.append(lm)
        ib._lib.i4_model_set_final(
            self.m, norm("model.layers.32.final_norm_layernorm.weight"), lm)

        if verbose:
            print(f"[phi35_cpu] built ({len(self._handles)} matmuls, C++ forward)")

    def _eval_rotary(self):
        """Run the If subgraph that emits cos_cache/sin_cache."""
        from .ops import get as get_op

        node = next(n for n in self.g.nodes if n.op_type == "If")

        class _Ctx:
            weight_cache = {}
        env = dict(self.g.initializers)
        # cond input to If
        cond = env.get(node.inputs[0])
        if cond is None:
            # produced by a Constant/Greater chain; default False branch is the
            # long-context path — evaluate whichever the graph would pick.
            cond = np.asarray(False)
        branch = node.subgraphs["then_branch" if bool(np.asarray(cond).ravel()[0])
                                else "else_branch"]
        benv = dict(branch.initializers)
        for n in branch.nodes:
            invals = [benv.get(i) if i != "" else None for i in n.inputs]
            if any(v is None for v in invals if v is not None) or \
               any(iv is None and inp != "" for iv, inp in zip(invals, n.inputs)):
                # missing dependency (references outer scope) — skip gracefully
                pass
            outs = get_op(n.op_type)(_Ctx(), n, invals)
            for name, val in zip(n.outputs, outs):
                if name != "":
                    benv[name] = val
        return benv[branch.outputs[0]], benv[branch.outputs[1]]

    def run(self, input_ids):
        values = np.asarray(input_ids)
        if values.ndim == 2 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError("input_ids must be an integer vector or a batch-one matrix")
        if not 1 <= values.size <= 8:
            raise ValueError("The experimental Phi CPU path supports 1 to 8 prefill tokens")
        if values.min() < 0 or values.max() >= self.vocab:
            raise ValueError("Token ID is outside the vocabulary")
        ids = np.ascontiguousarray(values, dtype=np.int64)
        s = ids.shape[0]
        logits = np.empty((s, self.vocab), dtype=np.float32)
        ib._lib.i4_forward(self.m, ids.ctypes.data_as(ctypes.c_void_p), int(s),
                           logits.ctypes.data_as(ctypes.c_void_p))
        return logits

    def __del__(self):
        try:
            for h in self._handles:
                ib.free(h)
            if getattr(self, "m", None):
                ib._lib.i4_model_free(self.m)
        except Exception:
            pass


Phi35Model = Phi35CpuModel
