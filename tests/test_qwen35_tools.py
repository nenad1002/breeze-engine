"""Local CLI checks using mock models/tokenizers only; no native or GPU inference."""
from contextlib import nullcontext
from dataclasses import asdict, replace
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import urllib.request

import numpy as np
import pytest

import breeze
from breeze.chat import chat_prompt
from breeze.qwen35_config import QwenConfig
import generate_qwen35 as cpu
import generate_qwen35_gpu as gpu
import prepare_qwen35 as prepare


def forbidden(*args, **kwargs):
    pytest.fail("Unexpected native/model allocation, subprocess or network access")


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setitem(breeze.__dict__, "Qwen35CpuModel", forbidden)
    monkeypatch.setitem(breeze.__dict__, "Qwen35GpuModel", forbidden)
    monkeypatch.setitem(sys.modules, "breeze.qwen35_cpu", None)
    monkeypatch.setitem(sys.modules, "breeze.qwen35_gpu", None)
    backend = ModuleType("breeze.cpu_backend")
    backend.set_threads = lambda count: None
    monkeypatch.setitem(sys.modules, "breeze.cpu_backend", backend)
    monkeypatch.setitem(breeze.__dict__, "cpu_backend", backend)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.delenv("I4_QWEN_MAXL", raising=False)


class Tokenizer:
    def __init__(self, ids=(1, 2, 3), eos=9):
        self.ids, self.eos = list(ids), eos
        self.texts, self.decodes = [], []

    def encode(self, text):
        self.texts.append(text)
        return SimpleNamespace(ids=self.ids)

    def token_to_id(self, name):
        return self.eos if name == "<|im_end|>" else None

    def decode(self, ids):
        self.decodes.append(list(ids))
        return "".join(str(i) for i in ids if i != self.eos)


def install_tokenizer(monkeypatch, tokenizer):
    module = ModuleType("tokenizers")
    module.Tokenizer = SimpleNamespace(from_file=lambda path: tokenizer)
    monkeypatch.setitem(sys.modules, "tokenizers", module)


def logits(token):
    result = np.zeros((1, 12), dtype=np.float32)
    result[0, token] = 1
    return result


class CpuModel:
    L, H = 32, 4096

    def __init__(self, tokens=(2, 3, 4, 5), fail=False):
        self.tokens, self.calls = iter(tokens), []
        self.closed, self.fail = False, fail

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def embed(self, ids):
        return list(ids)

    def run(self, embeddings, past_len=0):
        self.calls.append((list(embeddings), past_len))
        if self.fail:
            raise RuntimeError("mock forward failure")
        return logits(next(self.tokens))


def cpu_args(tmp_path, *extra):
    return ["--model", str(tmp_path / "model.onnx"), "--config", "9b",
            "--threads", "1", "--chunk-size", "2", "--max-new-tokens", "3",
            "--max-seq", "5", *extra]


def test_cpu_generation_caps_timing_and_chat(monkeypatch, tmp_path, capsys):
    tok, model, options = Tokenizer(), CpuModel(), {}
    install_tokenizer(monkeypatch, tok)

    def factory(path, **kwargs):
        options.update(kwargs)
        return model

    monkeypatch.setitem(breeze.__dict__, "Qwen35CpuModel", factory)
    ticks = iter(range(20))
    monkeypatch.setattr(cpu, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    result_path = tmp_path / "nested" / "result.json"
    report = cpu.main(cpu_args(tmp_path, "--prompt", "hello", "--no-thinking", "--json-out", str(result_path)))
    assert tok.texts == [chat_prompt("hello", True)]
    assert model.calls == [([1, 2], 0), ([3], 2), ([3], 3), ([4], 4)]
    assert report["token_ids"] == [3, 4, 5]
    assert report["decode_steps"] == 2
    assert report["load_seconds"] == report["prefill_seconds"] == 1
    assert report["decode_seconds"] == 2
    assert report["decode_tokens_per_second"] == 1
    assert len(tok.decodes) == 5
    assert options["config"] == QwenConfig.preset("9b")
    assert options["max_seq"] == 5 and options["hi_prec"]
    assert model.closed
    assert json.loads(result_path.read_text()) == report
    assert capsys.readouterr().out.startswith("345\n")


@pytest.mark.parametrize("tokens,expected,calls", [((2, 9), [9], 2), ((2, 3, 9), [3, 9], 3)])
def test_cpu_tokenizer_eos(monkeypatch, tmp_path, tokens, expected, calls):
    model = CpuModel(tokens)
    install_tokenizer(monkeypatch, Tokenizer())
    monkeypatch.setitem(breeze.__dict__, "Qwen35CpuModel", lambda *a, **k: model)
    report = cpu.main(cpu_args(tmp_path))
    assert report["token_ids"] == expected and len(model.calls) == calls
    assert report["decode_steps"] == len(expected) - 1
    assert model.closed


def test_cpu_one_token_has_no_decode(monkeypatch, tmp_path):
    model = CpuModel((2, 3))
    install_tokenizer(monkeypatch, Tokenizer())
    monkeypatch.setitem(breeze.__dict__, "Qwen35CpuModel", lambda *a, **k: model)
    report = cpu.main(cpu_args(tmp_path, "--max-new-tokens", "1", "--max-seq", "3"))
    assert len(model.calls) == 2 and report["decode_steps"] == 0
    assert report["decode_tokens_per_second"] is None and model.closed


def test_cpu_closes_on_forward_failure(monkeypatch, tmp_path):
    model = CpuModel(fail=True)
    install_tokenizer(monkeypatch, Tokenizer())
    monkeypatch.setitem(breeze.__dict__, "Qwen35CpuModel", lambda *a, **k: model)
    with pytest.raises(RuntimeError, match="mock forward"):
        cpu.main(cpu_args(tmp_path))
    assert model.closed


def test_cpu_holds_incomplete_characters(monkeypatch, tmp_path, capsys):
    tok, model = Tokenizer(), CpuModel((2, 3, 4))
    tok.decode = lambda ids: "\ufffd" if len(ids) == 1 else "é"
    install_tokenizer(monkeypatch, tok)
    monkeypatch.setitem(breeze.__dict__, "Qwen35CpuModel", lambda *a, **k: model)
    cpu.main(cpu_args(tmp_path, "--max-new-tokens", "2"))
    output = capsys.readouterr().out
    assert output.startswith("é\n") and "\ufffd" not in output


@pytest.mark.parametrize("flag,value", [("--threads", "0"), ("--chunk-size", "-1"),
                                       ("--max-seq", "0"), ("--max-new-tokens", "0")])
def test_cpu_positive_options(tmp_path, capsys, flag, value):
    with pytest.raises(SystemExit) as error:
        cpu.main(cpu_args(tmp_path, flag, value))
    assert error.value.code == 2 and "positive" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--validate", "--min-cosine", "--require-token-match"])
def test_cpu_removed_options(tmp_path, capsys, flag):
    with pytest.raises(SystemExit):
        cpu.main(cpu_args(tmp_path, flag))
    assert "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.parametrize("ids,max_seq,message", [((), "5", "empty"), ((1, 2, 3), "4", "exceed"),
                                                 ((1,), "262145", "configured context")])
def test_cpu_context_checks(monkeypatch, tmp_path, capsys, ids, max_seq, message):
    install_tokenizer(monkeypatch, Tokenizer(ids))
    monkeypatch.delitem(breeze.__dict__, "Qwen35CpuModel")
    with pytest.raises(SystemExit) as error:
        cpu.main(cpu_args(tmp_path, "--max-seq", max_seq))
    assert error.value.code == 2 and message in capsys.readouterr().err


@pytest.mark.parametrize("symlink", [False, True])
def test_cpu_refuses_existing_result(tmp_path, capsys, symlink):
    result = tmp_path / "result.json"
    if symlink:
        result.symlink_to(tmp_path / "missing")
    else:
        result.write_text("keep")
    with pytest.raises(SystemExit):
        cpu.main(cpu_args(tmp_path, "--json-out", str(result)))
    assert "already exists" in capsys.readouterr().err
    assert result.is_symlink() if symlink else result.read_text() == "keep"


def test_cpu_exclusive_result_write_race(monkeypatch, tmp_path):
    result, model = tmp_path / "result.json", CpuModel()
    install_tokenizer(monkeypatch, Tokenizer())

    def factory(*args, **kwargs):
        result.write_text("concurrent result")
        return model

    monkeypatch.setitem(breeze.__dict__, "Qwen35CpuModel", factory)
    with pytest.raises(FileExistsError):
        cpu.main(cpu_args(tmp_path, "--json-out", str(result)))
    assert result.read_text() == "concurrent result" and model.closed


@pytest.mark.parametrize("module", [cpu, gpu, prepare])
def test_cli_help_and_required_named_options(module, capsys):
    with pytest.raises(SystemExit) as error:
        module.main(["--help"])
    assert error.value.code == 0
    text = capsys.readouterr().out
    if module is prepare:
        assert "--source" in text and "--model-dir" in text and "NOT performed" in text
    else:
        assert "--model" in text and "--max-new-tokens" in text
        assert "--validate" not in text
    with pytest.raises(SystemExit) as error:
        module.main([])
    assert error.value.code == 2


class Tensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def __getitem__(self, key):
        return Tensor(self.array[key])

    def to(self, device):
        return self

    def float(self):
        return self

    def reshape(self, *shape):
        return Tensor(self.array.reshape(*shape))

    def copy_(self, other):
        self.array[...] = other.array

    def fill_(self, value):
        self.array.fill(value)


def fake_torch():
    return SimpleNamespace(no_grad=nullcontext, from_numpy=Tensor,
                           cuda=SimpleNamespace(synchronize=lambda: None, is_available=lambda: True))


class GpuModel:
    H = 2

    def __init__(self, tokens=(3, 4, 5)):
        self.tokens, self.calls, self.primes = iter(tokens), [], []
        self.compiles, self.captures, self.dirty = [], 0, False
        self.g_pos, self.g_inp = Tensor([0]), Tensor(np.zeros((1, self.H)))

    def run(self, embeddings, past_len=0):
        self.calls.append((len(embeddings), past_len, "dynamic"))
        return logits(next(self.tokens))

    def alloc_static(self, max_seq):
        self.max_seq = max_seq

    def prime_static(self, past):
        self.primes.append(past)
        self.dirty = False

    def compile_forward(self, warmup):
        self.compiles.append(warmup)
        self.dirty = True

    def capture(self):
        self.captures += 1
        self.dirty = True

    def decode_step(self, embedding, pos, use_graph):
        assert pos < self.max_seq
        self.calls.append((1, pos, use_graph))
        return logits(next(self.tokens))


@pytest.mark.parametrize("compiled,graph", [(False, False), (True, False), (False, True), (True, True)])
def test_gpu_generation_paths_and_bounds(compiled, graph, capsys):
    model = GpuModel()
    report = gpu.generate(model, np.ones((12, 2), np.float16), Tokenizer(), [1, 2], 3, 4,
                          fake_torch(), compile_decode=compiled, cuda_graph=graph)
    assert report["token_ids"] == [3, 4, 5] and report["decode_steps"] == 2
    assert [call[1] for call in model.calls] == [0, 2, 3]
    if compiled or graph:
        assert model.primes == [2, 2] and not model.dirty
        assert model.compiles == ([1] if compiled else [])
        assert model.captures == int(graph)
        assert model.calls[1:] == [(1, 2, graph), (1, 3, graph)]
    else:
        assert model.primes == [] and report["setup_seconds"] == 0
    assert capsys.readouterr().out == "345\n"


@pytest.mark.parametrize("token,limit", [(9, 3), (3, 1)])
def test_gpu_eos_or_final_token_skips_setup_and_decode(token, limit):
    model = GpuModel((token,))
    report = gpu.generate(model, np.ones((12, 2)), Tokenizer(), [1, 2], limit, 4,
                          fake_torch(), compile_decode=True, cuda_graph=True)
    assert report["token_ids"] == [token] and report["decode_steps"] == 0
    assert model.primes == [] and model.compiles == [] and model.captures == 0
    assert len(model.calls) == 1


def test_gpu_explicit_warmup_is_not_generation():
    model = GpuModel((3, 1, 1, 4))
    report = gpu.generate(model, np.ones((12, 2)), Tokenizer(), [1, 2], 2, 3,
                          fake_torch(), cuda_graph=True, warmup=2)
    assert report["token_ids"] == [3, 4] and report["decode_steps"] == 1
    assert model.calls == [(2, 0, "dynamic"), (1, 2, False), (1, 2, False), (1, 2, True)]
    assert model.primes == [2, 2] and not model.dirty


@pytest.mark.parametrize("ids,limit,capacity", [([], 1, 2), ([1], 0, 2), ([1], 1, 8193), ([1, 2], 3, 3)])
def test_gpu_helper_rejects_invalid_bounds(ids, limit, capacity):
    model = GpuModel()
    with pytest.raises(ValueError):
        gpu.generate(model, np.ones((12, 2)), Tokenizer(), ids, limit, capacity, fake_torch())
    assert model.calls == []


@pytest.mark.parametrize("flags", [["--max-seq", "8193"], ["--max-new-tokens", "0"],
                                   ["--max-seq", "0"], ["--warmup", "-1"], ["--warmup", "1"]])
def test_gpu_parser_guards(flags):
    with pytest.raises(SystemExit) as error:
        gpu.main(["--model", "absent.onnx", *flags])
    assert error.value.code == 2


@pytest.mark.parametrize("config", [QwenConfig.preset("27b"), replace(QwenConfig(), rope_theta=10000.0)])
def test_gpu_rejects_unsupported_config_before_allocation(tmp_path, capsys, config):
    (tmp_path / "config.json").write_text(json.dumps(asdict(config)))
    with pytest.raises(SystemExit):
        gpu.main(["--model", str(tmp_path / "model.onnx")])
    assert "only the dense 9B" in capsys.readouterr().err


@pytest.mark.parametrize("missing_torch", [True, False])
def test_gpu_missing_dependency_or_device(monkeypatch, tmp_path, capsys, missing_torch):
    (tmp_path / "config.json").write_text(json.dumps(asdict(QwenConfig())))
    install_tokenizer(monkeypatch, Tokenizer())
    monkeypatch.setattr(gpu.np, "load", lambda *a, **k: SimpleNamespace(shape=(248320, 4096), dtype=np.float16))
    if not missing_torch:
        torch = fake_torch()
        torch.cuda.is_available = lambda: False
        monkeypatch.setitem(sys.modules, "torch", torch)
    with pytest.raises(SystemExit) as error:
        gpu.main(["--model", str(tmp_path / "model.onnx")])
    assert error.value.code == 2
    assert ("torch installation" if missing_torch else "available CUDA GPU") in capsys.readouterr().err


def local_bundle(tmp_path):
    source, output = tmp_path / "checkpoint", tmp_path / "bundle"
    source.mkdir()
    output.mkdir()
    (source / "config.json").write_text(json.dumps({**asdict(QwenConfig()), "_name_or_path": "local-test-checkpoint"}))
    (source / "tokenizer.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"mock checkpoint")
    (output / "model.onnx").write_bytes(b"mock decoder")
    return source, output


def prepare_args(source, output, *extra):
    return ["--source", str(source), "--model-dir", str(output), *extra]


def test_prepare_dry_plan_has_no_writes_or_heavy_steps(monkeypatch, tmp_path, capsys):
    source, output = local_bundle(tmp_path)
    monkeypatch.setattr(prepare, "extract_embeddings", forbidden)
    monkeypatch.setattr(prepare, "_validate_decoder", forbidden)
    plan = prepare.main(prepare_args(source, output))
    assert plan["source"] == str(source) and plan["model"] == str(output / "model.onnx")
    assert sorted(p.name for p in output.iterdir()) == ["model.onnx"]
    assert "NOT performed" in capsys.readouterr().out


@pytest.mark.parametrize("overwrite", [False, True])
def test_prepare_execute_validates_first_and_writes_manifest(monkeypatch, tmp_path, overwrite):
    source, output = local_bundle(tmp_path)
    if overwrite:
        (output / "embeddings.npy").write_bytes(b"old embeddings")
        (output / "breeze_manifest.json").write_text("old manifest")
    events = []

    def validate(model, config, destinations):
        events.append("validate")
        assert model.read_bytes() == b"mock decoder"
        return {4: 10, 8: 3}

    def extract(src, target, config, **kwargs):
        assert events == ["validate"] and src == source
        assert kwargs["overwrite"] == overwrite
        events.append("extract")
        target.write_bytes(b"mock embeddings")

    monkeypatch.setattr(prepare, "_validate_decoder", validate)
    monkeypatch.setattr(prepare, "extract_embeddings", extract)
    manifest = prepare.main(prepare_args(source, output, "--execute", *(["--overwrite"] if overwrite else [])))
    assert manifest["source_local"] == str(source)
    assert manifest["model_local"] == str(output / "model.onnx")
    assert manifest["source_identity"] == "local-test-checkpoint"
    assert manifest["weight_bit_counts"] == {4: 10, 8: 3} and manifest["matmuls"] == 13
    assert manifest["status"] == "structural-only" and manifest["config"]["hidden_size"] == 4096
    assert json.loads((output / "breeze_manifest.json").read_text())["weight_bit_counts"] == {"4": 10, "8": 3}
    assert (output / "tokenizer.json").read_bytes() == (source / "tokenizer.json").read_bytes()
    assert (output / "model.onnx").read_bytes() == b"mock decoder"
    assert not list(output.glob(".*.tmp"))


@pytest.mark.parametrize("existing", ["embeddings.npy", "breeze_manifest.json"])
def test_prepare_requires_overwrite(monkeypatch, tmp_path, capsys, existing):
    source, output = local_bundle(tmp_path)
    (output / existing).write_text("keep")
    monkeypatch.setattr(prepare, "extract_embeddings", forbidden)
    with pytest.raises(SystemExit):
        prepare.main(prepare_args(source, output, "--execute"))
    assert "--overwrite" in capsys.readouterr().err
    assert (output / existing).read_text() == "keep"


@pytest.mark.parametrize("problem", ["config", "decoder", "metadata", "same-directory"])
def test_prepare_preflight_failures_do_not_extract(monkeypatch, tmp_path, problem):
    source, output = local_bundle(tmp_path)
    if problem == "config":
        (source / "config.json").write_text("{}")
    elif problem == "decoder":
        (output / "model.onnx").unlink()
    elif problem == "metadata":
        (output / "tokenizer.json").write_text("different")
    else:
        output = source
    monkeypatch.setattr(prepare, "extract_embeddings", forbidden)
    with pytest.raises(SystemExit) as error:
        prepare.main(prepare_args(source, output, "--execute", "--overwrite"))
    assert error.value.code == 2 and not (output / "embeddings.npy").exists()


def test_prepare_graph_failure_preserves_existing_outputs(monkeypatch, tmp_path):
    source, output = local_bundle(tmp_path)
    for name in ("embeddings.npy", "breeze_manifest.json"):
        (output / name).write_text("keep")

    def invalid(*args):
        raise ValueError("wrong packed shape")

    monkeypatch.setattr(prepare, "_validate_decoder", invalid)
    monkeypatch.setattr(prepare, "extract_embeddings", forbidden)
    with pytest.raises(SystemExit):
        prepare.main(prepare_args(source, output, "--execute", "--overwrite"))
    assert (output / "embeddings.npy").read_text() == "keep"
    assert (output / "breeze_manifest.json").read_text() == "keep"


def install_small_safetensors(monkeypatch, source, array, dtype="F32", fail=False):
    """Mock slicing over a tiny local safetensors-shaped file; never import torch."""
    name = "model.embed_tokens.weight"
    raw = ((array.astype(np.float32).view(np.uint32) >> 16).astype("<u2").tobytes()
           if dtype == "BF16" else array.astype(np.float32).tobytes())
    header = json.dumps({name: {"dtype": dtype, "shape": list(array.shape), "data_offsets": [0, len(raw)]}}).encode()
    shard = source / "model.safetensors"
    shard.write_bytes(len(header).to_bytes(8, "little") + header + raw)
    slices = []

    class Slice:
        def get_shape(self):
            return array.shape

        def __getitem__(self, rows):
            slices.append(rows)
            if fail:
                raise ValueError("mock slice failure")
            return array[rows]

    class Handle:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def keys(self):
            return [name]

        def get_slice(self, key):
            assert key == name
            return Slice()

    def safe_open(path, framework):
        assert framework == "numpy" and Path(path) == shard
        return Handle()

    module = ModuleType("safetensors")
    module.safe_open = safe_open
    monkeypatch.setitem(sys.modules, "safetensors", module)
    return slices, shard


@pytest.mark.parametrize("dtype", ["F32", "BF16"])
def test_extract_bounded_atomic_fp16_without_torch(monkeypatch, tmp_path, dtype):
    source = tmp_path / "source"
    source.mkdir()
    array = np.ones((4097, 2), dtype=np.float32) * 1.5
    slices, shard = install_small_safetensors(monkeypatch, source, array, dtype)
    original = shard.read_bytes()
    output = tmp_path / "new" / "embeddings.npy"
    config = SimpleNamespace(vocab_size=4097, hidden_size=2)
    prepare.extract_embeddings(source, output, config)
    result = np.load(output, allow_pickle=False)
    assert result.dtype == np.float16 and np.array_equal(result, array)
    if dtype == "F32":
        assert [(s.start, s.stop) for s in slices] == [(0, 4096), (4096, 4097)]
    assert shard.read_bytes() == original and not list(output.parent.glob(".*.tmp"))
    with pytest.raises(FileExistsError):
        prepare.extract_embeddings(source, output, config)
    prepare.extract_embeddings(source, output, config, overwrite=True)
    assert np.array_equal(np.load(output), array)


def test_extract_failure_cleans_unique_temp_and_preserves_output(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    install_small_safetensors(monkeypatch, source, np.ones((2, 2)), fail=True)
    output = tmp_path / "embeddings.npy"
    output.write_bytes(b"keep")
    unrelated = tmp_path / "embeddings.npy.tmp"
    unrelated.write_bytes(b"unrelated")
    with pytest.raises(ValueError, match="mock slice failure"):
        prepare.extract_embeddings(source, output, SimpleNamespace(vocab_size=2, hidden_size=2), overwrite=True)
    assert output.read_bytes() == b"keep" and unrelated.read_bytes() == b"unrelated"
    assert not list(tmp_path.glob(".*.tmp"))


def test_extract_rejects_source_replacement_and_wrong_shape(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _, shard = install_small_safetensors(monkeypatch, source, np.ones((2, 2)))
    config = SimpleNamespace(vocab_size=3, hidden_size=2)
    with pytest.raises(ValueError, match="source"):
        prepare.extract_embeddings(source, shard, config, overwrite=True)
    output = tmp_path / "embeddings.npy"
    with pytest.raises(ValueError, match="shape"):
        prepare.extract_embeddings(source, output, config)
    assert not output.exists()


def test_decoder_contract_uses_real_shape_binder_with_mock_graph(monkeypatch, tmp_path):
    node = SimpleNamespace(name="decoder/lm_head/MatMulNBits", op_type="MatMulNBits",
                           inputs=["x", "packed", "scales"], attrs={"K": 32, "N": 16, "bits": 4, "block_size": 32})
    graph = SimpleNamespace(nodes=[node], inputs=["inputs_embeds"], outputs=["logits"],
                            initializers={"packed": np.zeros(256, np.uint8),
                            "scales": np.ones(16, np.float32),
                            "model.layers.0.final_norm_layernorm.weight": np.ones(32, np.float32)})
    config = SimpleNamespace(matmul_shapes=lambda: {"lm_head": (32, 16)}, hidden_size=32,
                             num_hidden_layers=0, layer_types=())
    onnx = ModuleType("onnx")
    onnx.TensorProto = SimpleNamespace(EXTERNAL=1)
    proto = SimpleNamespace(graph=SimpleNamespace(initializer=[]))
    onnx.load = lambda *a, **k: proto
    loader = ModuleType("breeze.loader")
    loader.load_graph = lambda *a, **k: graph
    monkeypatch.setitem(sys.modules, "onnx", onnx)
    monkeypatch.setitem(sys.modules, "breeze.loader", loader)
    model, embedding = tmp_path / "model.onnx", tmp_path / "embeddings.npy"
    assert prepare._validate_decoder(model, config, [embedding]) == {4: 1}
    graph.inputs = ["input_ids"]
    with pytest.raises(ValueError, match="inputs_embeds"):
        prepare._validate_decoder(model, config, [embedding])
    graph.inputs = ["inputs_embeds"]
    node.attrs["K"] = 64
    with pytest.raises(ValueError, match="expected K=32"):
        prepare._validate_decoder(model, config, [embedding])
    proto.graph.initializer = [SimpleNamespace(data_location=1, external_data=[
        SimpleNamespace(key="location", value="embeddings.npy")])]
    with pytest.raises(ValueError, match="weight file"):
        prepare._validate_decoder(model, config, [embedding])


@pytest.mark.parametrize("shard_name", ["model.safetensors", "../outside.safetensors", "/outside.safetensors"])
def test_extract_local_shard_index(monkeypatch, tmp_path, shard_name):
    source = tmp_path / "checkpoint"
    source.mkdir()
    array = np.ones((2, 2), np.float32)
    install_small_safetensors(monkeypatch, source, array)
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"model.embed_tokens.weight": shard_name}}))
    output = tmp_path / "embeddings.npy"
    config = SimpleNamespace(vocab_size=2, hidden_size=2)
    if shard_name == "model.safetensors":
        prepare.extract_embeddings(source, output, config)
        assert np.array_equal(np.load(output), array)
    else:
        with pytest.raises(ValueError, match="shard"):
            prepare.extract_embeddings(source, output, config)
        assert not output.exists()


def test_gpu_main_routes_local_paths_flags_and_chat(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(asdict(QwenConfig())))
    tok, calls = Tokenizer(), {}
    install_tokenizer(monkeypatch, tok)
    embeddings = SimpleNamespace(shape=(248320, 4096), dtype=np.float16)

    def load(path, **kwargs):
        calls["embeddings"] = path
        return embeddings

    model = GpuModel()

    def factory(path):
        calls["model"] = path
        return model

    def generate(m, emb, tokenizer, ids, limit, capacity, torch, **kwargs):
        assert m is model and emb is embeddings and tokenizer is tok
        assert ids == tok.ids and limit == 2 and capacity == 4
        calls["options"] = kwargs
        return {"token_ids": [3, 4], "text": "34"}

    monkeypatch.setattr(gpu.np, "load", load)
    monkeypatch.setattr(gpu, "generate", generate)
    monkeypatch.setitem(breeze.__dict__, "Qwen35GpuModel", factory)
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    report = gpu.main(["--model", str(tmp_path / "model.onnx"), "--embeddings", str(tmp_path / "custom.npy"),
                       "--prompt", "hello", "--no-thinking", "--max-new-tokens", "2", "--max-seq", "4",
                       "--compile", "--cuda-graph", "--warmup", "2"])
    assert calls["model"] == str(tmp_path / "model.onnx")
    assert calls["embeddings"] == tmp_path / "custom.npy"
    assert calls["options"] == {"compile_decode": True, "cuda_graph": True, "warmup": 2}
    assert tok.texts == [chat_prompt("hello", True)] and report["token_ids"] == [3, 4]