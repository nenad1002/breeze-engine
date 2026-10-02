"""Small CPU tests; no model downloads, GPU, or full-checkpoint inference."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import helper, numpy_helper
import pytest

from breeze import Qwen35CpuModel
from breeze import cpu_backend as ib
from breeze.chat import chat_prompt
from breeze.loader import load_graph
from breeze.qwen35_config import QwenConfig
from breeze.qwen35_weights import bind_weights
from breeze.quant import dequantize_matmul_nbits
from breeze.validation import compare_logits


def compact_config(layers=4, ratio=3):
    return QwenConfig(hidden_size=64, num_hidden_layers=layers, intermediate_size=96, vocab_size=128,
                      linear_num_key_heads=16, linear_num_value_heads=16 * ratio,
                      linear_key_head_dim=2, linear_value_head_dim=2,
                      num_attention_heads=2 * ratio, num_key_value_heads=1, head_dim=16)


def synthetic_graph(c, seed=42):
    rng = np.random.default_rng(seed)
    initializers, nodes, dense = {}, [], {}
    for j, (name, (K, N)) in enumerate(c.matmul_shapes().items()):
        bits = 8 if j % 2 else 4
        weight_name, scale_name = name + ".weight", name + ".scale"
        qw = rng.integers(0, 256, (N, K // 32, 32 * bits // 8), dtype=np.uint8)
        sc = np.full((N, K // 32), 0.002 if bits == 4 else 0.0001, np.float32)
        initializers[weight_name], initializers[scale_name] = qw, sc
        nodes.append(SimpleNamespace(name="/model/" + name + "/MatMul", op_type="MatMulNBits",
                                     attrs=dict(K=K, N=N, bits=bits, block_size=32),
                                     inputs=["x", weight_name, scale_name]))
        dense[name] = dequantize_matmul_nbits(qw, sc, None, K=K, N=N, bits=bits, block_size=32)
    for i, kind in enumerate(c.layer_types):
        p = f"model.layers.{i}"
        initializers[p + ".input_layernorm.weight"] = np.ones(c.hidden_size, np.float32)
        initializers[p + ".post_attention_layernorm.weight"] = np.ones(c.hidden_size, np.float32)
        if kind == "linear_attention":
            initializers[p + ".linear_attn.conv1d.weight"] = rng.normal(0, 0.2, (c.conv_dim, 1, 4)).astype(np.float32)
            initializers[p + ".linear_attn.conv1d.bias"] = rng.normal(0, 0.01, c.conv_dim).astype(np.float32)
            initializers[p + ".linear_attn.neg_exp_A"] = -np.ones(c.linear_num_value_heads, np.float32)
            initializers[p + ".linear_attn.dt_bias"] = np.zeros(c.linear_num_value_heads, np.float32)
            initializers[p + ".linear_attn.norm.weight"] = np.ones(c.linear_value_head_dim, np.float32)
        else:
            initializers[p + ".attn.q_norm.layernorm.weight"] = np.ones(c.head_dim, np.float32)
            initializers[p + ".attn.k_norm.layernorm.weight"] = np.ones(c.head_dim, np.float32)
    initializers[f"model.layers.{c.num_hidden_layers}.final_norm_layernorm.weight"] = np.ones(c.hidden_size, np.float32)
    return SimpleNamespace(nodes=nodes, initializers=initializers), dense


def reference_forward(c, graph, weights, embeddings):
    """Independent float32 full-sequence reference on dequantized tiny weights."""
    def rms(x):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + c.rms_norm_eps)

    def silu(x):
        return x / (1 + np.exp(-x))

    def rope(x):
        d = c.rotary_dim
        freq = np.arange(x.shape[0])[:, None] / c.rope_theta ** (np.arange(0, d, 2) / d)
        co, si = np.cos(freq)[:, None], np.sin(freq)[:, None]
        out = x.copy()
        out[..., :d // 2] = x[..., :d // 2] * co - x[..., d // 2:d] * si
        out[..., d // 2:d] = x[..., d // 2:d] * co + x[..., :d // 2] * si
        return out

    h = embeddings.copy()
    T = len(h)
    for i, kind in enumerate(c.layer_types):
        p, v = f"layers.{i}/", f"model.layers.{i}"
        norm = rms(h)
        if kind == "linear_attention":
            mix = norm @ weights[p + "linear_attn/in_proj_qkv"]
            z = (norm @ weights[p + "linear_attn/in_proj_z"]).reshape(T, c.linear_num_value_heads, c.linear_value_head_dim)
            beta = 1 / (1 + np.exp(-(norm @ weights[p + "linear_attn/in_proj_b"])))
            decay = np.exp(-np.logaddexp(0, norm @ weights[p + "linear_attn/in_proj_a"]))
            cw = graph.initializers[v + ".linear_attn.conv1d.weight"][:, 0]
            cb = graph.initializers[v + ".linear_attn.conv1d.bias"]
            padded = np.pad(mix, ((3, 0), (0, 0)))
            mix = silu(sum(padded[k:k + T] * cw[:, k] for k in range(4)) + cb)
            q = mix[:, :c.key_dim].reshape(T, c.linear_num_key_heads, c.linear_key_head_dim)
            k = mix[:, c.key_dim:2 * c.key_dim].reshape(q.shape)
            values = mix[:, 2 * c.key_dim:].reshape(z.shape)
            q = q / np.sqrt((q * q).sum(-1, keepdims=True) + c.rms_norm_eps) / np.sqrt(c.linear_key_head_dim)
            k = k / np.sqrt((k * k).sum(-1, keepdims=True) + c.rms_norm_eps)
            q = q.repeat(c.linear_num_value_heads // c.linear_num_key_heads, axis=1)
            k = k.repeat(c.linear_num_value_heads // c.linear_num_key_heads, axis=1)
            state = np.zeros((c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim), np.float32)
            out = np.empty_like(values)
            for t in range(T):
                state *= decay[t, :, None, None]
                delta = beta[t, :, None] * (values[t] - np.einsum("hi,hij->hj", k[t], state))
                state += k[t, :, :, None] * delta[:, None]
                out[t] = np.einsum("hi,hij->hj", q[t], state)
            mixed = (rms(out) * silu(z)).reshape(T, c.value_dim) @ weights[p + "linear_attn/out_proj"]
        else:
            qg = (norm @ weights[p + "attn/q_proj"]).reshape(T, c.num_attention_heads, 2 * c.head_dim)
            q = rope(rms(qg[..., :c.head_dim]))
            k = rope(rms((norm @ weights[p + "attn/k_proj"]).reshape(T, c.num_key_value_heads, c.head_dim)))
            val = (norm @ weights[p + "attn/v_proj"]).reshape(T, c.num_key_value_heads, c.head_dim)
            k = k.repeat(c.num_attention_heads // c.num_key_value_heads, axis=1)
            val = val.repeat(c.num_attention_heads // c.num_key_value_heads, axis=1)
            scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(c.head_dim)
            scores[:, np.triu_indices(T, 1)[0], np.triu_indices(T, 1)[1]] = -np.inf
            scores = np.exp(scores - scores.max(-1, keepdims=True))
            scores /= scores.sum(-1, keepdims=True)
            out = np.einsum("hts,shd->thd", scores, val) / (1 + np.exp(-qg[..., c.head_dim:]))
            mixed = out.reshape(T, -1) @ weights[p + "attn/o_proj"]
        h = h + mixed
        norm = rms(h)
        h += (silu(norm @ weights[p + "mlp/gate_proj"]) * (norm @ weights[p + "mlp/up_proj"])) @ weights[p + "mlp/down_proj"]
    return rms(h) @ weights["lm_head"]


@pytest.mark.parametrize("name,layers,H,conv", [("9b", 32, 4096, 8192), ("27b", 64, 5120, 10240)])
def test_real_configs(name, layers, H, conv):
    c = QwenConfig.preset(name)
    assert (c.num_hidden_layers, c.hidden_size, c.conv_dim) == (layers, H, conv)
    assert len(c.full_layers) == layers // 4
    assert len(c.matmul_shapes()) == (layers // 4) * 7 + (layers * 3 // 4) * 8 + 1
    assert QwenConfig.from_dict({"text_config": asdict(c)}) == c
    assert all(K % 32 == 0 and N % 16 == 0 for K, N in c.matmul_shapes().values())


def test_bad_configs():
    with pytest.raises(ValueError, match="Incomplete"):
        QwenConfig.from_dict({"hidden_size": 5120})
    with pytest.raises(ValueError, match="multiple"):
        replace(QwenConfig(), linear_num_value_heads=33)
    data = asdict(QwenConfig())
    data["rope_parameters"] = {"rope_type": "yarn"}
    with pytest.raises(ValueError, match="YaRN"):
        QwenConfig.from_dict(data)


@pytest.mark.parametrize("fault", ["shape", "block", "head", "extra", "vector"])
def test_export_contract_rejects(fault):
    c = compact_config()
    graph, _ = synthetic_graph(c)
    if fault == "shape": graph.nodes[0].attrs["K"] += 32
    if fault == "block": graph.nodes[0].attrs["block_size"] = 128
    if fault == "head": graph.nodes.pop()
    if fault == "extra": graph.nodes.append(graph.nodes[0])
    if fault == "vector": graph.initializers.pop("model.layers.4.final_norm_layernorm.weight")
    with pytest.raises(ValueError):
        bind_weights(graph, c)


def test_mmap_external_and_snapshot_symlink(tmp_path):
    arr = np.arange(32, dtype=np.float32).reshape(4, 8)
    tensor = numpy_helper.from_array(arr, "weight")
    model = helper.make_model(helper.make_graph([], "external", [], [], [tensor]))
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    onnx.save_model(model, blobs / "model.onnx", save_as_external_data=True,
                    all_tensors_to_one_file=True, location="model.onnx.data", size_threshold=0)
    snap = tmp_path / "snapshot"
    snap.mkdir()
    for filename in ("model.onnx", "model.onnx.data"):
        (snap / filename).symlink_to(blobs / filename)
    got = load_graph(snap / "model.onnx", mmap_external=True).initializers["weight"]
    assert isinstance(got, np.memmap)
    np.testing.assert_array_equal(got, arr)


@pytest.fixture
def native():
    assert ib.available(), "Build the native kernel before running these tests"
    assert ib._lib.i4_qwen_abi_version() >= 2
    ib.set_threads(2)
    ib.set_hi_prec(True)


@pytest.mark.parametrize("length", [1, 2, 509, 510, 1024])
def test_conv_arbitrary_prefill(native, length):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(1, 3, length)).astype(np.float32)
    past = rng.normal(size=(1, 3, 3)).astype(np.float32)
    w = rng.normal(size=(3, 1, 4)).astype(np.float32)
    bias = rng.normal(size=3).astype(np.float32)
    out, state = ib.causal_conv(x, w, bias, past, True)
    win = np.concatenate([past, x], axis=-1)
    expected = bias[None, :, None] + sum(win[:, :, k:k + length] * w[None, :, 0, k, None] for k in range(4))
    expected /= 1 + np.exp(-expected)
    np.testing.assert_allclose(out, expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_array_equal(state, win[:, :, -3:])


@pytest.mark.parametrize("K,N,bits", [(5120, 48, 8), (17408, 16, 4), (6144, 64, 8), (4096, 32, 4)])
def test_27b_projection_widths(native, K, N, bits):
    rng = np.random.default_rng(3)
    qw = rng.integers(0, 256, (N, K // 32, 32 * bits // 8), dtype=np.uint8)
    scales = rng.uniform(0.001, 0.005, (N, K // 32)).astype(np.float32)
    a = rng.normal(size=(3, K)).astype(np.float32)
    h = ib.prepack(qw, scales, None, K, N, bits=bits)
    try:
        got = ib.matmul(h, a, N)
    finally:
        ib.free(h)
    expected = a @ dequantize_matmul_nbits(qw, scales, None, bits=bits, block_size=32, K=K, N=N)
    assert compare_logits(got, expected, 0.999999)["passed"]


def test_reject_quantization_mismatch():
    qw = np.zeros((16, 1, 32), np.uint8)
    with pytest.raises(ValueError, match="block_size"):
        ib.prepack(qw, np.ones(16), None, 32, 16, block_size=128)
    with pytest.raises(ValueError, match="zero point"):
        ib.prepack(qw, np.ones(16), np.zeros(16, np.uint8), 32, 16, bits=8)


def test_actual_27b_recurrence_dimensions(native):
    c = QwenConfig.preset("27b")
    rng = np.random.default_rng(19)
    q = rng.normal(size=(1, 3, c.key_dim)).astype(np.float32) * .05
    k = rng.normal(size=q.shape).astype(np.float32) * .05
    v = rng.normal(size=(1, 3, c.value_dim)).astype(np.float32)
    g = np.full((1, 3, 48), -.2, np.float32)
    beta = np.full_like(g, .6)
    out, state = ib.linear_attention(q, k, v, None, g, beta, Hq=16, Hkv=48,
                                     dk=128, dv=128, n_k=16, decay_per_key_dim=0,
                                     beta_per_head=1, scale=128 ** -.5)
    qr = q.reshape(3, 16, 128).repeat(3, axis=1)
    kr = k.reshape(3, 16, 128).repeat(3, axis=1)
    vr = v.reshape(3, 48, 128)
    ref_state = np.zeros((48, 128, 128), np.float32)
    ref_out = []
    for t in range(3):
        ref_state *= np.exp(g[0, t, :, None, None])
        delta = beta[0, t, :, None] * (vr[t] - np.einsum("hi,hij->hj", kr[t], ref_state))
        ref_state += kr[t, :, :, None] * delta[:, None]
        ref_out.append(np.einsum("hi,hij->hj", qr[t], ref_state) * 128 ** -.5)
    np.testing.assert_allclose(out, np.array(ref_out).reshape(1, 3, -1), atol=1e-7, rtol=1e-4)
    np.testing.assert_allclose(state[0], ref_state, atol=1e-7, rtol=1e-4)


@pytest.mark.parametrize("layers,ratio", [(4, 2), (64, 3)])
def test_full_native_decoder_and_cached_decode(native, monkeypatch, tmp_path, layers, ratio):
    c = compact_config(layers, ratio)
    graph, weights = synthetic_graph(c)
    rng = np.random.default_rng(9)
    embeddings = rng.normal(size=(c.vocab_size, c.hidden_size)).astype(np.float32)
    path = tmp_path / "embeddings.npy"
    np.save(path, embeddings)
    # Loader/prepack releases initializer mappings; retain independent reference.
    reference_graph = SimpleNamespace(initializers=dict(graph.initializers))
    monkeypatch.setattr("breeze.qwen35_cpu.load_graph", lambda *a, **kw: graph)
    with Qwen35CpuModel(tmp_path / "model.onnx", config=c, embed_path=path, max_seq=8192, verbose=False) as m:
        ids = [1, 2, 3, 4, 5]
        whole = m.run(m.embed(ids))
        ref = reference_forward(c, reference_graph, weights, embeddings[ids])
        assert compare_logits(whole, ref, 0.999999)["passed"]
        assert whole[-1].argmax() == ref[-1].argmax()
        prefix = m.run(m.embed(ids[:3]))
        step1 = m.run(m.embed(ids[3:4]), 3)
        step2 = m.run(m.embed(ids[4:]), 4)
        np.testing.assert_allclose(np.concatenate([prefix, step1, step2]), whole, atol=2e-5, rtol=2e-4)
        np.testing.assert_allclose(m.prefill(ids, chunk_size=2), whole[-1:], atol=2e-5, rtol=2e-4)
        assert any(a.shape == (8192, c.rotary_dim // 2) for a in m._keep)
        with pytest.raises(ValueError, match="State"):
            m.run(m.embed([2]), 1)
        with pytest.raises(ValueError, match="max_seq"):
            m.run(m.embed([2]), 8192)
        with pytest.raises(ValueError, match="shape"):
            m.run(np.zeros((2, 1, c.hidden_size)))
        with pytest.raises(ValueError, match="vocabulary"):
            m.embed([-1])
        assert m.generate(ids, max_new_tokens=0) == []
        assert len(m.generate(ids, max_new_tokens=2)) == 2
        assert m._past_len == len(ids) + 1  # no wasted forward after final token
    m.close()
    with pytest.raises(RuntimeError, match="closed"):
        m.run(embeddings[:1])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("bad_expected", [False, True])
def test_logits_reject_nonfinite(bad, bad_expected):
    actual, expected = np.array([[bad]]), np.ones((1, 1))
    if bad_expected:
        actual, expected = expected, actual
    with pytest.raises(ValueError, match="non-finite"):
        compare_logits(actual, expected, .999)


def test_logits_reject_shape_mismatch():
    with pytest.raises(ValueError, match="shapes"):
        compare_logits(np.ones((2, 3)), np.ones((1, 3)), .999)


def test_logits_numerical_metrics():
    expected = np.array([[1., 0.], [0., 2.]], dtype=np.float32)
    actual = expected * 2
    actual_before, expected_before = actual.copy(), expected.copy()
    result = compare_logits(actual, expected, 1.)
    assert result == {"min_row_cosine": 1., "max_abs": 2.,
                      "last_token_match": True, "passed": True}
    np.testing.assert_array_equal(actual, actual_before)
    np.testing.assert_array_equal(expected, expected_before)
    actual[0] = [0., 1.]
    result = compare_logits(actual, expected, .5)
    assert result["min_row_cosine"] == 0. and not result["passed"]
    assert result["last_token_match"]  # the worst row is not the last row
    result = compare_logits(-expected, expected, 0.)
    assert result["min_row_cosine"] == -1.
    assert not result["passed"] and not result["last_token_match"]


def test_logits_zero_rows():
    zeros = np.zeros((2, 3), dtype=np.float32)
    assert compare_logits(zeros, zeros.copy(), 1.)["passed"]
    nonzero = zeros.copy()
    nonzero[0, 1] = 1.
    for actual, expected in ((zeros, nonzero), (nonzero, zeros)):
        result = compare_logits(actual, expected, .1)
        assert result["min_row_cosine"] == 0. and not result["passed"]


def test_official_chat_suffix():
    assert chat_prompt("hello").endswith("<|im_start|>assistant\n<think>\n")
    assert chat_prompt("hello", True).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")