"""Benchmark bookkeeping checks without loading a model or native library."""
import sys
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import benchmark_qwen35_cpu as suite
import benchmark_qwen35_prefill as prefill


class RecordingModel:
    def __init__(self):
        self.past = 0
        self.calls = []

    def run(self, embeddings, past_len=0):
        if past_len == 0:
            self.past = 0
        assert past_len == self.past
        self.calls.append((len(embeddings), past_len))
        self.past += len(embeddings)
        return np.tile(np.arange(6, dtype=np.float32), (len(embeddings), 1))


def inputs(prompt_tokens=64, decode_steps=128):
    return (np.ones((prompt_tokens, 4), dtype=np.float32),
            [np.ones((1, 4), dtype=np.float32) for _ in range(decode_steps)])


def test_full_decode_capture_and_timing(monkeypatch, capsys):
    ticks = iter(range(258))
    monkeypatch.setattr(suite, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    model = RecordingModel()
    result, rows, positions = suite.run_workload(
        model, *inputs(), 128, capture=True, capture_every=1, progress_every=8)
    assert model.calls == [(64, 0)] + [(1, n) for n in range(64, 192)]
    assert positions == list(range(63, 192))
    assert np.stack(rows).shape == (129, 6)
    assert result["predicted_token_ids"] == [5] * 129
    assert result["prefill_seconds"] == 1
    assert result["decode_step_seconds"] == [1] * 128
    assert result["decode_seconds"] == 128
    assert result["decode_tokens_per_second"] == 1
    assert result["decode_step_median_ms"] == 1000
    assert result["decode_step_p95_ms"] == 1000
    output = capsys.readouterr().out
    assert output.count("decode ") == 16
    assert "decode 128/128: 1.000 tok/s; last step 1.0000s" in output


@pytest.mark.parametrize("interval", [1, 2, 16, 100])
def test_capture_interval_with_chunked_prefill(interval, capsys):
    model = RecordingModel()
    result, rows, positions = suite.run_workload(
        model, *inputs(5, 33), 3, capture=True, capture_every=interval)
    expected = [4] + [5 + i for i in range(33)
                      if i == 0 or (i + 1) % interval == 0 or i == 32]
    assert positions == expected
    assert len(rows) == len(expected)
    assert model.calls[:2] == [(3, 0), (2, 3)]
    assert len(result["decode_step_seconds"]) == 33
    assert capsys.readouterr().out == ""


def test_capture_disabled_and_cache_reset():
    model = RecordingModel()
    for _ in range(2):
        result, rows, positions = suite.run_workload(model, *inputs(5, 3), 3)
        assert rows == [] and positions == []
        assert len(result["predicted_token_ids"]) == 4
    assert model.calls[:5] == model.calls[5:]


@pytest.mark.parametrize("flag,value", [("--capture-every", "0"), ("--progress-every", "-1")])
def test_invalid_intervals_fail_before_model_loading(monkeypatch, tmp_path, flag, value):
    monkeypatch.setattr(sys, "argv", [
        "benchmark_qwen35_cpu.py", "--model", str(tmp_path / "absent.onnx"),
        "--output", str(tmp_path / "result.json"), flag, value,
    ])
    with pytest.raises(SystemExit) as error:
        suite.main()
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kwargs", [
    {"chunk_size": 0}, {"chunk_size": -1}, {"chunk_size": 1.5},
    {"chunk_size": True}, {"chunk_size": None},
    {"capture_every": 0}, {"capture_every": -1}, {"capture_every": 1.5},
    {"capture_every": True}, {"capture_every": None},
    {"progress_every": -1}, {"progress_every": 1.5},
    {"progress_every": True}, {"progress_every": None},
])
def test_direct_interval_guards_before_model_or_timer(monkeypatch, kwargs):
    def unexpected_timer():
        pytest.fail("Invalid workloads must fail before starting a timer")

    monkeypatch.setattr(suite, "time", SimpleNamespace(perf_counter=unexpected_timer))
    model = RecordingModel()
    options = {"chunk_size": 3, **kwargs}
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        suite.run_workload(model, *inputs(5, 3), **options)
    assert model.calls == []


@pytest.mark.parametrize("prompt_tokens,decode_steps", [(0, 3), (5, 0), (0, 0)])
def test_direct_empty_workload_guard(monkeypatch, prompt_tokens, decode_steps):
    def unexpected_timer():
        pytest.fail("Empty workloads must fail before starting a timer")

    monkeypatch.setattr(suite, "time", SimpleNamespace(perf_counter=unexpected_timer))
    model = RecordingModel()
    with pytest.raises(ValueError, match="nonempty"):
        suite.run_workload(model, *inputs(prompt_tokens, decode_steps), 3)
    assert model.calls == []


@pytest.mark.parametrize("module,flags,message", [
    (suite, ["--threads", "0"], "positive"),
    (suite, ["--decode-steps", "0"], "positive"),
    (suite, ["--chunk-size", "-1"], "positive"),
    (suite, ["--rounds", "0"], "positive"),
    (suite, ["--warmup", "-1"], "nonnegative"),
    (suite, ["--prompt-lengths", "0"], "positive"),
    (suite, ["--prompt-lengths", "64", "64"], "distinct"),
    (suite, ["--prompt-lengths"], "expected at least one argument"),
    (suite, ["--threads", "1.5"], "invalid int value"),
    (prefill, ["--threads", "0"], "positive"),
    (prefill, ["--max-seq", "0"], "positive"),
    (prefill, ["--pack-threads", "-1"], "positive"),
    (prefill, ["--prompt", "   "], "nonempty"),
    (suite, ["--engine", "breeze"], "unrecognized arguments"),
    (prefill, ["--engine", "breeze"], "unrecognized arguments"),
])
def test_cli_argument_errors_before_loading(monkeypatch, tmp_path, capsys, module, flags, message):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setitem(sys.modules, "breeze", None)
    with pytest.raises(SystemExit) as error:
        module.main(["--model", str(tmp_path / "absent.onnx"),
                     "--output", str(tmp_path / "result.json"), *flags])
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("module", [suite, prefill])
def test_cli_help_has_no_runtime_selection(module, capsys):
    with pytest.raises(SystemExit) as error:
        module.main(["--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "Breeze" in help_text
    assert "--engine" not in help_text
    assert ("--pack-threads" in help_text) == (module is prefill)


@pytest.mark.parametrize("module", [suite, prefill])
@pytest.mark.parametrize("cuda", [None, "0"])
def test_cpu_only_guard(monkeypatch, tmp_path, module, cuda):
    if cuda is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda)
    monkeypatch.setitem(sys.modules, "breeze", None)
    with pytest.raises(SystemExit) as error:
        module.main(["--model", str(tmp_path / "absent.onnx"),
                     "--output", str(tmp_path / "result.json")])
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("module,suffix", [(suite, ".logits.npz"), (prefill, ".logits.npy")])
@pytest.mark.parametrize("existing", ["report", "logits"])
def test_overwrite_guard_before_loading(monkeypatch, tmp_path, module, suffix, existing):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setitem(sys.modules, "breeze", None)
    output = tmp_path / "result.json"
    occupied = output if existing == "report" else output.with_suffix(suffix)
    occupied.write_bytes(b"preserve")
    with pytest.raises(SystemExit) as error:
        module.main(["--model", str(tmp_path / "absent.onnx"), "--output", str(output)])
    assert error.value.code == 2
    assert occupied.read_bytes() == b"preserve"
    assert list(tmp_path.iterdir()) == [occupied]


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    """Small in-memory dependencies; never import the native Breeze backend."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    config = SimpleNamespace(vocab_size=6, hidden_size=4, num_hidden_layers=2)
    models, thread_calls, config_paths, graph_paths = [], [], [], []
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"mock graph")
    (tmp_path / "tokenizer.json").write_text("{}")
    np.save(tmp_path / "embeddings.npy", np.arange(24, dtype=np.float32).reshape(6, 4))
    kernel = tmp_path / "kernel.bin"
    kernel.write_bytes(b"mock kernel")

    class FakeModel(RecordingModel):
        def __init__(self, path, **kwargs):
            super().__init__()
            self.path = path
            self.options = kwargs
            self.closed = False
            models.append(self)

        def close(self):
            self.closed = True

    class FakeTokenizer:
        @classmethod
        def from_file(cls, path):
            assert Path(path).read_text() == "{}"
            return cls()

        def encode(self, text):
            return SimpleNamespace(ids=[1])

        def decode(self, ids):
            return "token"

    def resolve_config(path, preset=None):
        config_paths.append(path)
        return config

    def load_graph(path, *, load_external_data):
        assert load_external_data is False
        graph_paths.append(path)
        nodes = [SimpleNamespace(op_type="MatMulNBits",
                                 attribute=[SimpleNamespace(name="bits", value=bits)])
                 for bits in (4, 4, 8)]
        return SimpleNamespace(graph=SimpleNamespace(node=nodes))

    modules = {}
    for name in ("breeze", "breeze.cpu_backend", "breeze.qwen35_config",
                 "breeze.chat", "onnx", "tokenizers"):
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
    modules["breeze"].__path__ = []
    modules["breeze"].Qwen35CpuModel = FakeModel
    backend = modules["breeze.cpu_backend"]
    backend.set_threads = thread_calls.append
    backend._SO = str(kernel)
    backend.__file__ = __file__
    backend._native_prepack = None
    modules["breeze"].cpu_backend = backend
    modules["breeze.qwen35_config"].QwenConfig = SimpleNamespace(resolve=resolve_config)
    modules["breeze.chat"].chat_prompt = lambda text, no_thinking: text
    modules["onnx"].load = load_graph
    modules["onnx"].helper = SimpleNamespace(get_attribute_value=lambda attr: attr.value)
    modules["tokenizers"].Tokenizer = FakeTokenizer
    return SimpleNamespace(model_path=model_path, config=config, models=models,
                           thread_calls=thread_calls, config_paths=config_paths,
                           graph_paths=graph_paths, model_class=FakeModel,
                           tokenizer_class=FakeTokenizer)


def test_suite_mock_cli_replay_rounds_and_capture(monkeypatch, tmp_path, fake_runtime):
    ticks = iter(range(1000))
    monkeypatch.setattr(suite, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    output = tmp_path / "result.json"
    argv = ["--model", str(fake_runtime.model_path), "--output", str(output),
            "--threads", "1", "--prompt-lengths", "5", "--decode-steps", "3",
            "--chunk-size", "3", "--warmup", "1", "--rounds", "2", "--capture-every", "1"]
    suite.main(argv)
    report = json.loads(output.read_text())
    assert report["engine"] == "breeze"
    assert report["command"][1:] == argv
    assert report["matmul_weight_bits_counts"] == {"4": 2, "8": 1}
    assert report["load_seconds"] == 1
    assert report["warmup_rounds"] == 1 and report["measured_rounds"] == 2
    case, = report["cases"]
    assert len(case["warmup"]) == 1 and len(case["rounds"]) == 2
    assert case["prompt_token_ids"] == [1] * 5
    assert case["decode_token_ids"] == [1] * 3
    assert case["sampled_logit_positions"] == [4, 5, 6, 7]
    assert case["prefill_median_seconds"] == 2
    assert case["decode_pooled_tokens_per_second"] == 1
    for result in case["warmup"] + case["rounds"]:
        assert result["decode_step_seconds"] == [1] * 3
    with np.load(output.with_suffix(".logits.npz")) as captured:
        assert captured.files == ["pp5"]
        assert captured["pp5"].shape == (4, 6)
    model, = fake_runtime.models
    assert model.calls == [(3, 0), (2, 3), (1, 5), (1, 6), (1, 7)] * 3
    assert model.closed
    assert fake_runtime.thread_calls == [1]


@pytest.mark.parametrize("pack_threads", [None, 2])
def test_prefill_mock_cli_one_call_and_symlink(monkeypatch, tmp_path, fake_runtime, pack_threads):
    ticks = iter(range(4))
    monkeypatch.setattr(prefill, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    target = tmp_path / "stored"
    target.mkdir()
    moved = target / "graph.onnx"
    fake_runtime.model_path.rename(moved)
    fake_runtime.model_path.symlink_to(moved)
    output = tmp_path / "result.json"
    args = ["--model", str(fake_runtime.model_path), "--output", str(output), "--threads", "1"]
    if pack_threads is not None:
        args.extend(["--pack-threads", str(pack_threads)])
    prefill.main(args)
    report = json.loads(output.read_text())
    assert report["engine"] == "breeze"
    assert report["matmul_weight_bits_counts"] == {"4": 2, "8": 1}
    assert report["forward_calls"] == 1 and report["warmup_calls"] == 0
    assert report["decode_steps"] == 0
    assert report["load_seconds"] == report["forward_seconds"] == 1
    assert report["packing_threads"] == (pack_threads or 1)
    assert report["max_seq"] == 1024
    assert report["model"] == str(fake_runtime.model_path)
    assert fake_runtime.config_paths == fake_runtime.graph_paths == [fake_runtime.model_path]
    assert np.load(output.with_suffix(".logits.npy")).shape == (1, 6)
    model, = fake_runtime.models
    assert model.calls == [(1, 0)] and model.closed
    assert model.path == fake_runtime.model_path
    assert fake_runtime.thread_calls == [pack_threads or 1, 1]


@pytest.mark.parametrize("module", [suite, prefill])
@pytest.mark.parametrize("value", ["", "1", "64"])
def test_partial_layer_override_rejected(monkeypatch, tmp_path, fake_runtime, module, value):
    monkeypatch.setenv("I4_QWEN_MAXL", value)
    with pytest.raises(SystemExit) as error:
        module.main(["--model", str(fake_runtime.model_path),
                     "--output", str(tmp_path / "result.json"), "--threads", "1"])
    assert error.value.code == 2
    assert fake_runtime.models == []
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("module", [suite, prefill])
@pytest.mark.parametrize("shape,dtype", [((0, 4), np.float32), ((6, 4), np.int32),
                                        ((6, 4), np.float64), ((6, 3), np.float32)])
def test_embedding_validation_before_model_loading(tmp_path, fake_runtime, module, shape, dtype):
    table = np.zeros(shape, dtype=dtype)
    np.save(tmp_path / "embeddings.npy", table)
    fake_runtime.config.vocab_size = shape[0]
    with pytest.raises(SystemExit) as error:
        module.main(["--model", str(fake_runtime.model_path),
                     "--output", str(tmp_path / "result.json"), "--threads", "1"])
    assert error.value.code == 2
    assert fake_runtime.models == []


@pytest.mark.parametrize("module", [suite, prefill])
def test_model_closed_on_forward_failure(monkeypatch, tmp_path, fake_runtime, module):
    def fail_run(self, embeddings, past_len=0):
        raise RuntimeError("Mock forward failure")

    monkeypatch.setattr(fake_runtime.model_class, "run", fail_run)
    with pytest.raises(RuntimeError, match="Mock forward failure"):
        module.main(["--model", str(fake_runtime.model_path),
                     "--output", str(tmp_path / "result.json"), "--threads", "1"])
    assert len(fake_runtime.models) == 1 and fake_runtime.models[0].closed
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("module", [suite, prefill])
def test_empty_tokenization_before_model_loading(monkeypatch, tmp_path, fake_runtime, module):
    monkeypatch.setattr(fake_runtime.tokenizer_class, "encode", lambda self, text: SimpleNamespace(ids=[]))
    with pytest.raises(SystemExit) as error:
        module.main(["--model", str(fake_runtime.model_path),
                     "--output", str(tmp_path / "result.json"), "--threads", "1"])
    assert error.value.code == 2
    assert fake_runtime.models == []


def test_logit_inspection_excluded_from_forward_timers(monkeypatch):
    clock = [0.0]
    original_isfinite = np.isfinite

    class TimedModel(RecordingModel):
        def run(self, embeddings, past_len=0):
            clock[0] += 2.0
            return super().run(embeddings, past_len)

    def inspect(values):
        clock[0] += 100.0
        return original_isfinite(values)

    monkeypatch.setattr(suite, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    monkeypatch.setattr(suite.np, "isfinite", inspect)
    result, rows, positions = suite.run_workload(TimedModel(), *inputs(5, 3), 3, capture=True)
    assert result["prefill_seconds"] == 4
    assert result["decode_step_seconds"] == [2] * 3
    assert result["decode_seconds"] == 6
    assert len(rows) == len(positions) == 3