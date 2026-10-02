"""Experimental dense Qwen3.5-9B forward on a CUDA GPU using torch.

Dequantizes every matmul weight to fp16 on the GPU and runs the hybrid decoder
in torch. GatedDeltaNet recurrence, attention, and normalization use torch ops.
Provides prefill and incremental decode with state carry, plus optional static
decode buffers and graph capture. This path stores dense fp16 weights; it does
not use the separate experimental quantized Triton kernels.
"""
import time

import numpy as np
import torch

from .loader import load_graph
from .quant import dequantize_matmul_nbits

DEV = "cuda"
FP = torch.float16


class Qwen35GpuModel:
    def __init__(self, model_path, verbose=True):
        t0 = time.time()
        g = load_graph(model_path)
        self.H, self.vocab, self.L, self.inter = 4096, 248320, 32, 12288
        self.Hk, self.Hv, self.hk, self.hv = 16, 32, 128, 128
        self.key_dim, self.value_dim = self.Hk * self.hk, self.Hv * self.hv
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_k = 4
        self.nh, self.nkv, self.hd = 16, 4, 256
        self.rot = 64
        self.scale = self.hd ** -0.5
        self.eps = 1e-6

        # dequantize every matmul -> fp16 [K,N] on GPU, keyed by node name
        self.mm = {}
        for n in g.nodes:
            if n.op_type != "MatMulNBits":
                continue
            qz = g.initializers[n.inputs[3]] if len(n.inputs) > 3 and n.inputs[3] != "" else None
            W = dequantize_matmul_nbits(g.initializers[n.inputs[1]], g.initializers[n.inputs[2]], qz,
                                        bits=int(n.attrs["bits"]), block_size=int(n.attrs["block_size"]),
                                        K=int(n.attrs["K"]), N=int(n.attrs["N"]))
            self.mm[n.name] = torch.from_numpy(W).to(DEV, FP)
            g.initializers.pop(n.inputs[1], None); g.initializers.pop(n.inputs[2], None)

        def MM(sub):
            for k, v in self.mm.items():
                if sub in k:
                    return v
            raise KeyError(sub)

        def vec(name, dt=torch.float32):
            return torch.from_numpy(np.ascontiguousarray(g.initializers[name], np.float32)).to(DEV, dt)

        self.layers = []
        for l in range(self.L):
            p = f"model.layers.{l}"
            if l % 4 != 3:
                d = dict(kind="lin",
                         in_qkv=MM(f"layers.{l}/linear_attn/in_proj_qkv"), in_z=MM(f"layers.{l}/linear_attn/in_proj_z"),
                         in_b=MM(f"layers.{l}/linear_attn/in_proj_b"), in_a=MM(f"layers.{l}/linear_attn/in_proj_a"),
                         out=MM(f"layers.{l}/linear_attn/out_proj"),
                         mg=MM(f"layers.{l}/mlp/gate_proj"), mu=MM(f"layers.{l}/mlp/up_proj"), md=MM(f"layers.{l}/mlp/down_proj"),
                         in_ln=vec(f"{p}.input_layernorm.weight"),
                         conv_w=vec(f"{p}.linear_attn.conv1d.weight").reshape(self.conv_dim, self.conv_k),
                         conv_b=vec(f"{p}.linear_attn.conv1d.bias"),
                         neg_exp_A=vec(f"{p}.linear_attn.neg_exp_A"), dt_bias=vec(f"{p}.linear_attn.dt_bias"),
                         gnorm=vec(f"{p}.linear_attn.norm.weight"), post_ln=vec(f"{p}.post_attention_layernorm.weight"))
            else:
                d = dict(kind="full",
                         q=MM(f"layers.{l}/attn/q_proj"), k=MM(f"layers.{l}/attn/k_proj"),
                         v=MM(f"layers.{l}/attn/v_proj"), o=MM(f"layers.{l}/attn/o_proj"),
                         mg=MM(f"layers.{l}/mlp/gate_proj"), mu=MM(f"layers.{l}/mlp/up_proj"), md=MM(f"layers.{l}/mlp/down_proj"),
                         in_ln=vec(f"{p}.input_layernorm.weight"), q_norm=vec(f"{p}.attn.q_norm.layernorm.weight"),
                         k_norm=vec(f"{p}.attn.k_norm.layernorm.weight"), post_ln=vec(f"{p}.post_attention_layernorm.weight"))
            self.layers.append(d)
        self.final_ln = vec("model.layers.32.final_norm_layernorm.weight")
        self.lmhead = MM("lm_head")

        theta, half = 1e7, self.rot // 2
        inv = 1.0 / (theta ** (np.arange(0, self.rot, 2) / self.rot))
        pos = np.arange(8192)
        fr = np.outer(pos, inv)
        self.cos = torch.from_numpy(np.cos(fr)).to(DEV, torch.float32)   # [P, half]
        self.sin = torch.from_numpy(np.sin(fr)).to(DEV, torch.float32)
        self.reset()
        if verbose:
            print(f"[qwen-gpu] built in {time.time()-t0:.1f}s ({len(self.mm)} matmuls, fp16 on {DEV})")

    def reset(self):
        self.state = [None] * self.L

    def _rms(self, x, w):
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * w

    def _forward(self, emb, past):
        # emb: [T, H] fp32 on GPU
        T = emb.shape[0]
        h = emb
        for l, d in enumerate(self.layers):
            hn = self._rms(h, d["in_ln"]).to(FP)
            if d["kind"] == "lin":
                mix = hn @ d["in_qkv"]                        # [T, conv_dim] fp16
                z = (hn @ d["in_z"]).float()
                bproj = (hn @ d["in_b"]).float(); aproj = (hn @ d["in_a"]).float()
                # causal depthwise conv over channels with state
                st = self.state[l]
                cprev = st[0] if st is not None else torch.zeros(self.conv_dim, self.conv_k - 1, device=DEV)
                xt = mix.float().t()                         # [conv_dim, T]
                win = torch.cat([cprev, xt], dim=1)          # [conv_dim, pad+T]
                out = d["conv_b"][:, None].clone().expand(-1, T).clone()
                for k in range(self.conv_k):
                    out = out + d["conv_w"][:, k:k+1] * win[:, k:k+T]
                conv = torch.nn.functional.silu(out)          # [conv_dim, T]
                new_conv = win[:, T:T + self.conv_k - 1].contiguous()
                mix = conv.t()                                # [T, conv_dim]
                q = mix[:, :self.key_dim].reshape(T, self.Hk, self.hk)
                k = mix[:, self.key_dim:2*self.key_dim].reshape(T, self.Hk, self.hk)
                v = mix[:, 2*self.key_dim:].reshape(T, self.Hv, self.hv)
                q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + self.eps) * (self.hk ** -0.5)
                k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + self.eps)
                beta = torch.sigmoid(bproj)                   # [T, Hv]
                g = d["neg_exp_A"] * torch.nn.functional.softplus(aproj + d["dt_bias"])  # [T, Hv]
                # inverse GQA: expand q,k (Hk=16) to Hv=32
                q = q.repeat_interleave(self.Hv // self.Hk, dim=1)   # [T, Hv, hk]
                k = k.repeat_interleave(self.Hv // self.Hk, dim=1)
                S = st[1] if st is not None else torch.zeros(self.Hv, self.hk, self.hv, device=DEV)
                outs = torch.empty(T, self.Hv, self.hv, device=DEV)
                eg = torch.exp(g)                              # [T, Hv]
                for t in range(T):
                    S = S * eg[t][:, None, None]
                    kt = k[t]; vt = v[t]                        # [Hv, hk], [Hv, hv]
                    retr = torch.einsum("hij,hi->hj", S, kt)   # [Hv, hv]
                    delta = beta[t][:, None] * (vt - retr)
                    S = S + kt[:, :, None] * delta[:, None, :]
                    outs[t] = torch.einsum("hi,hij->hj", q[t], S)
                self.state[l] = (new_conv, S)
                # gated RMSNorm per head_v with silu(z)
                a = outs.float()
                a = a * torch.rsqrt(a.pow(2).mean(-1, keepdim=True) + self.eps) * d["gnorm"]
                a = a * torch.nn.functional.silu(z.reshape(T, self.Hv, self.hv))
                mixer = a.reshape(T, self.value_dim).to(FP) @ d["out"]
            else:
                qkv = (hn @ d["q"]).float().reshape(T, self.nh, self.hd * 2)
                q = qkv[..., :self.hd]; gate = qkv[..., self.hd:]
                kk = (hn @ d["k"]).float().reshape(T, self.nkv, self.hd)
                vv = (hn @ d["v"]).float().reshape(T, self.nkv, self.hd)
                q = self._rms(q, d["q_norm"]); kk = self._rms(kk, d["k_norm"])
                cosr = self.cos[past:past+T]; sinr = self.sin[past:past+T]    # [T, half]
                q = self._rope(q, cosr, sinr); kk = self._rope(kk, cosr, sinr)
                st = self.state[l]
                if st is not None:
                    kk = torch.cat([st[0], kk], dim=0); vv = torch.cat([st[1], vv], dim=0)
                self.state[l] = (kk, vv)
                Kt = kk.repeat_interleave(self.nh // self.nkv, dim=1)         # [S, nh, hd]
                Vt = vv.repeat_interleave(self.nh // self.nkv, dim=1)
                qh = q.permute(1, 0, 2); Kh = Kt.permute(1, 2, 0); Vh = Vt.permute(1, 0, 2)
                sc = torch.bmm(qh, Kh) * self.scale                           # [nh, T, S]
                Stot = Kt.shape[0]
                cm = torch.full((T, Stot), float("-inf"), device=DEV)
                idx = torch.arange(Stot, device=DEV)
                cm = torch.where(idx[None, :] <= (past + torch.arange(T, device=DEV))[:, None], 0.0, cm)
                sc = torch.softmax(sc + cm, dim=-1)
                out = torch.bmm(sc, Vh).permute(1, 0, 2).reshape(T, self.nh * self.hd)
                out = out * torch.sigmoid(gate.reshape(T, self.nh * self.hd))
                mixer = out.to(FP) @ d["o"]
            h = h + mixer.float()
            hn = self._rms(h, d["post_ln"]).to(FP)
            g1 = torch.nn.functional.silu((hn @ d["mg"]).float()) * (hn @ d["mu"]).float()
            h = h + (g1.to(FP) @ d["md"]).float()
        hn = self._rms(h, self.final_ln).to(FP)
        return (hn @ self.lmhead).float()

    def _rope(self, x, cos, sin):    # x [T, heads, hd]; rotate first self.rot dims
        half = self.rot // 2
        xr, xp = x[..., :self.rot], x[..., self.rot:]
        x1, x2 = xr[..., :half], xr[..., half:]
        c = cos[:, None, :]; s = sin[:, None, :]
        rot = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
        return torch.cat([rot, xp], dim=-1)

    @torch.no_grad()
    def run(self, inputs_embeds, past_len=0):
        if past_len == 0:
            self.reset()
        e = torch.as_tensor(np.ascontiguousarray(inputs_embeds, np.float32)).reshape(-1, self.H).to(DEV)
        return self._forward(e, past_len).cpu().numpy()

    # ---------------------------------------------------------------- CUDA-graph decode
    def alloc_static(self, max_seq=1024):
        self.max_seq = max_seq
        self.g_inp = torch.zeros(1, self.H, device=DEV)
        self.g_pos = torch.zeros(1, dtype=torch.long, device=DEV)
        self.g_logits = torch.zeros(1, self.vocab, device=DEV)
        self.arange = torch.arange(max_seq, device=DEV)
        self.g_conv, self.g_recur, self.g_kvk, self.g_kvv = [], [], [], []
        for d in self.layers:
            if d["kind"] == "lin":
                self.g_conv.append(torch.zeros(self.conv_dim, self.conv_k - 1, device=DEV))
                self.g_recur.append(torch.zeros(self.Hv, self.hk, self.hv, device=DEV))
                self.g_kvk.append(None); self.g_kvv.append(None)
            else:
                self.g_conv.append(None); self.g_recur.append(None)
                self.g_kvk.append(torch.zeros(self.nkv, max_seq, self.hd, device=DEV))
                self.g_kvv.append(torch.zeros(self.nkv, max_seq, self.hd, device=DEV))
        self.graph = None

    def prime_static(self, T):
        """Copy the dynamic prefill state into the static decode buffers."""
        for l, d in enumerate(self.layers):
            st = self.state[l]
            if d["kind"] == "lin":
                self.g_conv[l].copy_(st[0]); self.g_recur[l].copy_(st[1])
            else:
                self.g_kvk[l].zero_(); self.g_kvv[l].zero_()
                self.g_kvk[l][:, :T].copy_(st[0].permute(1, 0, 2))
                self.g_kvv[l][:, :T].copy_(st[1].permute(1, 0, 2))

    def _rope1(self, x, cos, sin):     # x [heads, hd], cos/sin [1, half]
        half = self.rot // 2
        xr, xp = x[:, :self.rot], x[:, self.rot:]
        x1, x2 = xr[:, :half], xr[:, half:]
        rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return torch.cat([rot, xp], dim=-1)

    def _decode_forward(self):
        """Single-token forward on static buffers (in-place state), CUDA-graph-safe."""
        pos = self.g_pos
        mask = torch.where(self.arange <= pos, torch.zeros((), device=DEV),
                           torch.full((), float("-inf"), device=DEV))     # [max_seq]
        h = self.g_inp
        for l, d in enumerate(self.layers):
            hn = self._rms(h, d["in_ln"]).to(FP)
            if d["kind"] == "lin":
                mix = (hn @ d["in_qkv"]).float()
                z = (hn @ d["in_z"]).float()
                bproj = (hn @ d["in_b"]).float(); aproj = (hn @ d["in_a"]).float()
                win = torch.cat([self.g_conv[l], mix.t()], dim=1)         # [cd, K]
                conv = torch.nn.functional.silu(d["conv_b"] + (d["conv_w"] * win).sum(1))
                self.g_conv[l].copy_(win[:, 1:])
                q = conv[:self.key_dim].reshape(self.Hk, self.hk)
                k = conv[self.key_dim:2*self.key_dim].reshape(self.Hk, self.hk)
                v = conv[2*self.key_dim:].reshape(self.Hv, self.hv)
                q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + self.eps) * (self.hk ** -0.5)
                k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + self.eps)
                beta = torch.sigmoid(bproj)[0]
                g = (d["neg_exp_A"] * torch.nn.functional.softplus(aproj + d["dt_bias"]))[0]
                q = q.repeat_interleave(self.Hv // self.Hk, dim=0)
                k = k.repeat_interleave(self.Hv // self.Hk, dim=0)
                S = self.g_recur[l]
                S.mul_(torch.exp(g)[:, None, None])
                retr = (S * k[:, :, None]).sum(1)
                delta = beta[:, None] * (v - retr)
                S.add_(k[:, :, None] * delta[:, None, :])
                out = (q[:, :, None] * S).sum(1)                          # [Hv, hv]
                a = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps) * d["gnorm"]
                a = a * torch.nn.functional.silu(z.reshape(self.Hv, self.hv))
                mixer = (a.reshape(1, self.value_dim).to(FP) @ d["out"]).float()
            else:
                qkv = (hn @ d["q"]).float().reshape(self.nh, self.hd * 2)
                qh = qkv[:, :self.hd]; gate = qkv[:, self.hd:]
                kk = (hn @ d["k"]).float().reshape(self.nkv, self.hd)
                vv = (hn @ d["v"]).float().reshape(self.nkv, self.hd)
                qh = self._rms(qh, d["q_norm"]); kk = self._rms(kk, d["k_norm"])
                cosp = self.cos.index_select(0, pos); sinp = self.sin.index_select(0, pos)
                qh = self._rope1(qh, cosp, sinp); kk = self._rope1(kk, cosp, sinp)
                self.g_kvk[l].index_copy_(1, pos, kk[:, None, :])
                self.g_kvv[l].index_copy_(1, pos, vv[:, None, :])
                K = self.g_kvk[l].repeat_interleave(self.nh // self.nkv, dim=0)
                V = self.g_kvv[l].repeat_interleave(self.nh // self.nkv, dim=0)
                sc = (qh[:, None, :] * K).sum(-1) * self.scale + mask[None, :]
                sc = torch.softmax(sc, dim=-1)
                out = (sc[:, :, None] * V).sum(1).reshape(1, self.nh * self.hd)
                out = out * torch.sigmoid(gate.reshape(1, self.nh * self.hd))
                mixer = (out.to(FP) @ d["o"]).float()
            h = h + mixer
            hn = self._rms(h, d["post_ln"]).to(FP)
            g1 = torch.nn.functional.silu((hn @ d["mg"]).float()) * (hn @ d["mu"]).float()
            h = h + (g1.to(FP) @ d["md"]).float()
        hn = self._rms(h, self.final_ln).to(FP)
        self.g_logits.copy_(hn @ self.lmhead)
        return self.g_logits

    def compile_forward(self, warmup=6):
        """Fuse the elementwise ops via torch.compile (no internal cudagraph;
        we wrap the fused kernels in our own graph in capture())."""
        self._compiled = torch.compile(self._decode_forward,
                                       mode="max-autotune-no-cudagraphs")
        for _ in range(warmup):
            self._compiled()
        torch.cuda.synchronize()

    def _fwd(self):
        return self._compiled() if getattr(self, "_compiled", None) else self._decode_forward()

    def capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._fwd()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._fwd()

    @torch.no_grad()
    def decode_step(self, embed, pos, use_graph=True):
        if torch.is_tensor(embed):
            self.g_inp.copy_(embed.reshape(1, self.H))
        else:
            self.g_inp.copy_(torch.as_tensor(np.ascontiguousarray(embed, np.float32)).reshape(1, self.H).to(DEV))
        self.g_pos.fill_(pos)
        if use_graph and self.graph is not None:
            self.graph.replay()
        else:
            self._fwd()
        return self.g_logits


QwenGpuModel = Qwen35GpuModel

