"""Developer CLI tests: mock native/graph work, no GPU, model or long jobs."""

import ast
import builtins
import inspect
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import benchmark_cpu_matmul as cpu
import benchmark_gpu_matmul as gpu
import measure_process as process
import profile_graph as graph


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("benchmark_cpu_matmul.py", "benchmark_gpu_matmul.py",
           "profile_graph.py", "measure_process.py")


@pytest.fixture(autouse=True)
def no_real_backend(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"breeze", "torch", "triton"}:
            pytest.fail(f"Unexpected backend import: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("help_only", [False, True])
def test_import_and_help_are_inert(script, help_only):
    # A fresh interpreter proves these imports do not even request NumPy/native
    # libraries, regardless of which modules pytest has already imported.
    code = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'breeze', 'torch', 'triton', 'onnx', 'tokenizers'}:
        raise AssertionError('Unexpected import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
path, mode = sys.argv[1:]
sys.argv = [path, '--help']
runpy.run_path(path, run_name='__main__' if mode == 'help' else 'inert_import')
"""
    result = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT / script),
                             "help" if help_only else "import"],
                            capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout if help_only else result.stdout == ""
    assert result.stderr == ""


@pytest.mark.parametrize("module,args", [
    (cpu, ["--n", "17"]), (cpu, ["--k", "33"]),
    (cpu, ["--rows", "0"]), (cpu, ["--threads", "-1"]),
    (cpu, ["--iterations", "0"]), (cpu, ["--warmup", "-1"]),
    (cpu, ["--bits", "3"]), (cpu, ["--n", str(2**31)]),
    (gpu, ["--k", "48"]), (gpu, ["--k", "32", "--block-k", "64"]),
    (gpu, ["--k", "64", "--split-k", "4"]), (gpu, ["--rows", "0"]),
    (gpu, ["--iterations", "0"]), (gpu, ["--warmup", "-1"]),
    (gpu, ["--block-n", "17"]), (gpu, ["--split-k", "3"]),
    (graph, []), (graph, ["--model", "missing"]),
    (graph, ["--model", "missing", "--token-ids", "-1"]),
    (graph, ["--model", "missing", "--token-ids", str(2**63)]),
    (graph, ["--model", "missing", "--token-ids", "1", "--iterations", "0"]),
    (graph, ["--model", "missing", "--token-ids", "1", "--warmup", "0"]),
    (graph, ["--model", "missing", "--token-ids", "1", "--threads", "0"]),
    (graph, ["--model", "missing", "--token-ids", "1"]),
    (process, []), (process, ["--label", "x"]),
    (process, ["--label", "x", "--"]),
    (process, ["--label", "x", "command"]),
    (process, ["--label", " ", "--", "command"]),
    *[(process, ["--label", "x", "--interval", value, "--", "command"])
      for value in ("0", "-1", "nan", "inf", "1e100")],
])
def test_cli_validation_precedes_work(module, args, monkeypatch):
    monkeypatch.setattr(os, "fork", lambda: pytest.fail("Unexpected child"))
    with pytest.raises(SystemExit) as error:
        module.main(args)
    assert error.value.code == 2


class FakeCpu:
    def __init__(self, fail_on=None):
        self.events = []
        self.calls = 0
        self.fail_on = fail_on
        self._lib = SimpleNamespace(i4_matmul=self.native)

    def set_threads(self, threads):
        self.events.append(("threads", threads))

    def prepack(self, weights, scales, zeros, k, n, *, bits):
        self.events.append("prepack")
        self.packed = (weights.copy(), scales.copy(), None if zeros is None else zeros.copy())
        self.shape = (k, n, bits)
        return 123

    def native(self, handle, a, rows, out):
        self.calls += 1
        assert handle.value == 123 and a.value and out.value and rows == 2
        self.events.append("native")
        if self.calls == self.fail_on:
            raise RuntimeError("native failure")

    def free(self, handle):
        self.events.append(("free", handle))


def cpu_run(backend, **kwargs):
    return cpu.benchmark(backend, n=16, k=96, rows=2, threads=1,
                         iterations=3, warmup=2, **kwargs)


@pytest.mark.parametrize("bits", [4, 8])
def test_cpu_native_only_timing_cleanup_and_seed(bits):
    backend = FakeCpu()
    ticks = iter((10.0, 10.6))

    def clock():
        backend.events.append("clock")
        return next(ticks)

    assert cpu_run(backend, bits=bits, clock=clock) == pytest.approx(0.2)
    assert backend.events == [("threads", 1), "prepack", "native", "native", "clock",
                              "native", "native", "native", "clock", ("free", 123)]
    assert backend.shape == (96, 16, bits)
    assert backend.packed[0].shape == (16, 3, 4 * bits)
    assert backend.packed[0].dtype == np.uint8
    assert backend.packed[1].shape == (48,)
    if bits == 4:
        assert backend.packed[2].shape == (32,)
    else:
        assert backend.packed[2] is None
    other = FakeCpu()
    cpu_run(other, bits=bits, clock=lambda: 0)
    for left, right in zip(backend.packed, other.packed):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("fail_on", [1, 3])
def test_cpu_frees_when_warmup_or_measurement_fails(fail_on):
    backend = FakeCpu(fail_on)
    with pytest.raises(RuntimeError, match="native failure"):
        cpu_run(backend)
    assert backend.events[-1] == ("free", 123)


def test_cpu_cli_threads_before_backend_import(monkeypatch):
    original = builtins.__import__
    fake = SimpleNamespace(cpu_backend=SimpleNamespace(available=lambda: True))

    def import_fake(name, *args, **kwargs):
        if name == "breeze":
            assert all(os.environ[key] == "2" for key in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))
            return fake
        return original(name, *args, **kwargs)

    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setattr(builtins, "__import__", import_fake)
    calls = []
    monkeypatch.setattr(cpu, "benchmark", lambda backend, **kw: calls.append(kw) or 0.1)
    assert cpu.main(["--threads", "2", "3", "--bits", "8"]) == 0
    assert [call["threads"] for call in calls] == [2, 3]
    assert all(call["bits"] == 8 for call in calls)


def test_gpu_grid_and_valid_tail_shapes():
    configs = list(gpu.configuration_grid())
    assert len(configs) == len(set(configs)) == 180
    for cfg in configs:
        assert gpu.validate_config(1024, 17, 3, cfg) == 1024 // cfg[5]
    assert gpu.validate_config(32, 1, 1, gpu.DEFAULT_CONFIG) == 32


@pytest.mark.parametrize("k,n,rows,config", [
    (0, 16, 1, gpu.DEFAULT_CONFIG), (32, -1, 1, gpu.DEFAULT_CONFIG),
    (32, 1, True, gpu.DEFAULT_CONFIG), (33, 16, 1, gpu.DEFAULT_CONFIG),
    (32, 16, 1, (16, 128, 64, 4, 3, 1)),
    (96, 16, 1, (16, 128, 32, 4, 3, 2)),
    (128, 16, 1, (16, 128, 32, 4, 3, 3)),
    (128, 16, 1, (16, 128, 32, 4, 3)),
    (128, 16, 1, (16, 128, 32, 1, 3, 1)),
])
def test_gpu_rejects_unsupported_combinations(k, n, rows, config):
    with pytest.raises(ValueError):
        gpu.validate_config(k, n, rows, config)


def test_gpu_kernel_scale_addresses_and_load_masks_without_gpu():
    # Evaluate the kernel's actual address/mask expressions with NumPy arrays.
    # This guards the scale-index regression, but is NOT a Triton execution test.
    tree = ast.parse(inspect.getsource(gpu.load_backend))
    assignments = {node.targets[0].id: node.value for node in ast.walk(tree)
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}

    def expression(node, values):
        return eval(compile(ast.Expression(node), "<kernel-expression>", "eval"),
                    {"__builtins__": {}}, values)

    loads = {name: next(node for node in ast.walk(assignments[name])
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "load") for name in ("a", "s")}
    # w is assigned twice; find its load independently.
    loads["w"] = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute) and node.func.attr == "load"
                      and any(isinstance(child, ast.Name) and child.id == "w_ptr"
                              for child in ast.walk(node)))
    for k, split in ((128, 64), (128, 32), (96, 64)):
        values = {"k_off": np.arange(64, 128), "K": k, "k_lo": 64, "ksplit": split,
                  "offs_n": np.arange(4), "s_ptr": 0, "ssb": 3, "ssn": 1,
                  "m_mask": np.array([True, False]), "n_mask": np.array([True, True, True, False])}
        values["k_mask"] = expression(assignments["k_mask"], values)
        expected_k = np.arange(64, 128) < min(k, 64 + split)
        np.testing.assert_array_equal(values["k_mask"], expected_k)
        offsets = expression(loads["s"].args[0], values)
        expected = np.repeat([2, 3], 32)[:, None] * 3 + np.arange(4)[None, :]
        np.testing.assert_array_equal(offsets, expected)
        for name, load in loads.items():
            mask = next(keyword.value for keyword in load.keywords if keyword.arg == "mask")
            actual = expression(mask, values)
            expected = (values["m_mask"][:, None] & expected_k[None, :] if name == "a"
                        else expected_k[:, None] & values["n_mask"][None, :])
            np.testing.assert_array_equal(actual, expected)


def fake_graph(fail_run=None):
    ops = SimpleNamespace(get=lambda name: lambda *args: [name])
    calls = []

    def run(feeds):
        calls.append(feeds)
        assert ops.get("Add")(None, None, []) == ["Add"]
        assert ops.get("Add")(None, None, []) == ["Add"]
        if len(calls) == fail_run:
            raise RuntimeError("graph failure")
        ops.get("Multiply")(None, None, [])

    return SimpleNamespace(run=run), ops, calls


def test_graph_profile_warmup_reset_and_registry_restore():
    session, ops, calls = fake_graph()
    original = ops.get
    ticks = iter(range(100))
    feeds = {"mock": 1}
    result = graph.profile_session(session, ops, feeds, iterations=3, warmup=2,
                                   clock=lambda: next(ticks))
    assert ops.get is original
    assert calls == [feeds] * 5
    assert result["warmup_counts"] == {"Add": 4, "Multiply": 2}
    assert result["counts"] == {"Add": 6, "Multiply": 3}
    assert result["times"] == {"Add": 6.0, "Multiply": 3.0}
    assert result["warmup_times"] == {"Add": 4.0, "Multiply": 2.0}
    assert result["wall_seconds"] == 19
    assert result["warmup_wall_seconds"] == 13


@pytest.mark.parametrize("fail_run", [1, 2])
def test_graph_restore_on_warmup_or_measured_failure(fail_run):
    session, ops, _ = fake_graph(fail_run)
    original = ops.get
    with pytest.raises(RuntimeError, match="graph failure"):
        graph.profile_session(session, ops, {}, iterations=1, warmup=1)
    assert ops.get is original


def test_graph_restores_registry_when_operator_raises():
    def broken(*args):
        raise ValueError("operator failure")

    ops = SimpleNamespace(get=lambda name: broken)
    original = ops.get
    session = SimpleNamespace(run=lambda feeds: ops.get("Broken")(None, None, []))
    with pytest.raises(ValueError, match="operator failure"):
        graph.profile_session(session, ops, {})
    assert ops.get is original


def test_graph_reference_feeds():
    feeds = graph.make_feeds([1, 2, 3])
    assert len(feeds) == 66
    np.testing.assert_array_equal(feeds["input_ids"], [[1, 2, 3]])
    assert feeds["input_ids"].dtype == feeds["attention_mask"].dtype == np.int64
    assert feeds["attention_mask"].shape == (1, 3)
    for name, value in feeds.items():
        if name.startswith("past_key_values"):
            assert value.shape == (1, 32, 0, 96) and value.dtype == np.float16


def test_graph_cli_mock_session_and_thread_environment(monkeypatch, tmp_path):
    model = tmp_path / "reference.onnx"
    model.touch()
    session, ops, calls = fake_graph()
    threads = []
    original = builtins.__import__

    def import_fake(name, *args, **kwargs):
        if name in {"breeze", "breeze.graph_session"}:
            assert all(os.environ[key] == "2" for key in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))
            if name == "breeze":
                return SimpleNamespace(ops=ops, cpu_backend=SimpleNamespace(set_threads=threads.append))
            return SimpleNamespace(InferenceSession=lambda path: session)
        return original(name, *args, **kwargs)

    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setattr(builtins, "__import__", import_fake)
    assert graph.main(["--model", str(model), "--token-ids", "1", "2", "--threads", "2"]) == 0
    assert len(calls) == 2 and threads == [2]


def test_process_sampler_without_sleep(monkeypatch):
    class Stop:
        def __init__(self):
            self.waits = 0

        def is_set(self):
            return self.waits == 3

        def wait(self, interval):
            assert interval == 0.25
            self.waits += 1

    ticks = iter((0, 100, 300))
    clock = iter((10.0, 11.0, 12.0))
    monkeypatch.setattr(process, "read_cpu_ticks", lambda pid: next(ticks))
    monkeypatch.setattr(process, "read_rss_kb", lambda pid: 1024)
    samples = process.sample_process(123, Stop(), 0.25, 10.0, 100, clock=lambda: next(clock))
    assert samples == [(1.0, 100.0, 1024), (2.0, 200.0, 1024)]


def test_process_proc_parsing_and_errors(monkeypatch):
    rest = ["S"] + ["0"] * 10 + ["12", "34"]
    monkeypatch.setattr(builtins, "open", lambda *a: io.StringIO("123 (name with ) space) " + " ".join(rest)))
    assert process.read_cpu_ticks(123) == 46
    monkeypatch.setattr(builtins, "open", lambda *a: io.StringIO("Name: test\nVmRSS: 2048 kB\n"))
    assert process.read_rss_kb(123) == 2048
    monkeypatch.setattr(builtins, "open", lambda *a: io.StringIO("VmRSS: broken kB\n"))
    with pytest.raises(ValueError):
        process.read_rss_kb(123)


@pytest.mark.parametrize("error", [FileNotFoundError, ProcessLookupError, PermissionError])
def test_process_disappearing_proc_is_expected(monkeypatch, error):
    def missing(*args):
        raise error()

    monkeypatch.setattr(builtins, "open", missing)
    assert process.read_cpu_ticks(123) is None
    assert process.read_rss_kb(123) is None


@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError])
def test_process_wait_failure_reaps_only_child_and_stops_sampler(monkeypatch, failure):
    state = {}
    monkeypatch.setattr(process.os, "fork", lambda: 12345)
    monkeypatch.setattr(process.os, "kill", lambda pid, sig: state.update(killed=(pid, sig)))
    monkeypatch.setattr(process.signal, "signal", lambda *args: signal.default_int_handler)
    waits = []

    def wait(pid):
        waits.append(pid)
        if len(waits) == 1:
            raise failure("wait failure")
        return pid, 0, None

    class Executor:
        def __init__(self, **kwargs):
            pass

        def submit(self, function, pid, stop, *args):
            state["stop"] = stop
            return SimpleNamespace(result=lambda: [])

        def shutdown(self, wait):
            state["joined"] = wait

    monkeypatch.setattr(process, "wait_child", wait)
    monkeypatch.setattr(process, "ThreadPoolExecutor", Executor)
    if failure is KeyboardInterrupt:
        assert process.measure(["mock"], label="test", interval=0.1) == 130
    else:
        with pytest.raises(RuntimeError, match="wait failure"):
            process.measure(["mock"], label="test", interval=0.1)
    assert state["killed"] == (12345, signal.SIGKILL)
    assert waits == [12345, 12345]
    assert state["stop"].is_set() and state["joined"]


def test_process_sampler_bug_is_not_hidden(monkeypatch):
    state = {}
    monkeypatch.setattr(process.os, "fork", lambda: 12345)
    monkeypatch.setattr(process, "wait_child", lambda pid: (pid, 0, None))
    monkeypatch.setattr(process, "terminate_child", lambda pid: pytest.fail("Already reaped"))

    def broken():
        raise ValueError("sampler bug")

    class Executor:
        def __init__(self, **kwargs):
            pass

        def submit(self, function, pid, stop, *args):
            state["stop"] = stop
            return SimpleNamespace(result=broken)

        def shutdown(self, wait):
            state["joined"] = wait

    monkeypatch.setattr(process, "ThreadPoolExecutor", Executor)
    with pytest.raises(ValueError, match="sampler bug"):
        process.measure(["mock"], label="test", interval=0.1)
    assert state["stop"].is_set() and state["joined"]


@pytest.mark.parametrize("child_code,expected", [
    ("import sys; sys.exit(7)", 7),
    ("import os, signal; os.kill(os.getpid(), signal.SIGTERM)", 143),
])
def test_process_fast_subprocess_exit_status(child_code, expected):
    result = subprocess.run([sys.executable, "-B", str(ROOT / "measure_process.py"),
                             "--label", "fast", "--", sys.executable, "-B", "-c", child_code],
                            capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == expected, result.stderr
    assert f"exit={expected}" in result.stderr and "upper-half heuristic" in result.stderr


def test_process_exec_failure(tmp_path):
    result = subprocess.run([sys.executable, "-B", str(ROOT / "measure_process.py"),
                             "--label", "missing", "--", str(tmp_path / "absent-command")],
                            capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 127 and "Could not execute" in result.stderr


def test_process_explicit_argv_not_shell(monkeypatch):
    seen = []
    monkeypatch.setattr(process, "measure", lambda command, **kwargs: seen.append(command) or 7)
    assert process.main(["--label", "argv", "--", "mock", "literal; $HOME", "--help"]) == 7
    assert seen == [["mock", "literal; $HOME", "--help"]]
