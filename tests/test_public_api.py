"""Public CPU API naming and import compatibility; no checkpoint required."""
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def test_qwen35_cpu_model_exports_and_legacy_alias():
    import breeze
    from breeze import Qwen35CpuModel
    from breeze.qwen35_cpu import Qwen35CpuModel as implementation
    from breeze.qwen35_cpu import QwenCppModel

    assert Qwen35CpuModel is implementation is QwenCppModel
    assert Qwen35CpuModel.__name__ == "Qwen35CpuModel"
    assert set(breeze.__all__) == {"Qwen35CpuModel", "Phi35CpuModel", "Qwen35GpuModel",
                                   "Qwen35ReferenceModel", "InferenceSession"}


def test_phi35_cpu_model_exports_and_alias():
    from breeze import Phi35CpuModel
    from breeze.phi35_cpu import Phi35CpuModel as implementation, Phi35Model

    assert Phi35CpuModel is implementation is Phi35Model
    assert Phi35CpuModel.__name__ == "Phi35CpuModel"


@pytest.mark.parametrize("ids", [[], [1] * 9, [-1], [32], [[1], [2]], [1.5], [True]])
def test_phi_rejects_unsupported_input_before_native_call(ids):
    from breeze import Phi35CpuModel

    model = Phi35CpuModel.__new__(Phi35CpuModel)
    model.vocab = 32
    with pytest.raises(ValueError):
        model.run(ids)


@pytest.mark.parametrize("high_precision", [False, True])
@pytest.mark.parametrize("tokens", [1, 8])
def test_phi_tiny_native_forward_with_both_activation_modes(high_precision, tokens):
    import ctypes
    import numpy as np
    from breeze import Phi35CpuModel, cpu_backend

    if not cpu_backend.available():
        pytest.skip("Build the CPU library to test native Phi prefill")
    model = Phi35CpuModel.__new__(Phi35CpuModel)
    model.vocab = 32
    model._handles = []
    model._keep = []
    model.m = None

    def vector(values):
        array = np.ascontiguousarray(values)
        model._keep.append(array)
        return array.ctypes.data_as(ctypes.c_void_p)

    def projection(inputs, outputs, packed=0x88):
        weights = np.full((outputs, inputs // 32, 16), packed, dtype=np.uint8)
        scales = np.full(outputs * (inputs // 32), 0.125, dtype=np.float32)
        handle = cpu_backend.prepack(weights, scales, None, inputs, outputs)
        model._handles.append(handle)
        return handle

    cpu_backend.set_threads(2)
    cpu_backend.set_hi_prec(high_precision)
    try:
        model.m = cpu_backend._lib.i4_model_new(1, 32, 4, 8, 64, 32, 1e-5, 8 ** -0.5, 8)
        assert model.m
        cpu_backend._lib.i4_model_set_embed(model.m, vector(np.full((32, 32), 0.5, np.float16)))
        cpu_backend._lib.i4_model_set_rotary(model.m, vector(np.ones((8, 4), np.float32)),
                                             vector(np.zeros((8, 4), np.float32)))
        norm = vector(np.ones(32, np.float32))
        cpu_backend._lib.i4_model_set_layer(model.m, 0, projection(32, 96), projection(32, 32),
                                            projection(32, 64), projection(32, 64), projection(64, 32),
                                            norm, norm)
        cpu_backend._lib.i4_model_set_final(model.m, norm, projection(32, 32, 0x99))
        actual = model.run([[1] * tokens])
        expected = 32 * 0.125 * 0.5 / np.sqrt(0.25 + 1e-5)
        assert actual.shape == (tokens, 32)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    finally:
        model.__del__()
        model.m = None
        model._handles = []
        cpu_backend.set_hi_prec(False)


def test_unknown_public_attribute():
    import breeze

    with pytest.raises(AttributeError):
        getattr(breeze, "unknown_model")


def test_cpu_import_keeps_optional_backends_lazy():
    code = textwrap.dedent("""
        import sys
        import breeze

        assert "breeze.qwen35_cpu" not in sys.modules
        assert "breeze.phi35_cpu" not in sys.modules
        assert "breeze.cpu_backend" not in sys.modules

        from breeze import Qwen35CpuModel, Phi35CpuModel

        assert Qwen35CpuModel.__name__ == "Qwen35CpuModel"
        assert Phi35CpuModel.__name__ == "Phi35CpuModel"
        assert "breeze.graph_session" not in sys.modules
        assert "breeze.qwen35_reference" not in sys.modules
        assert "breeze.qwen35_gpu" not in sys.modules
        assert "torch" not in sys.modules
        assert "triton" not in sys.modules
    """)
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_optional_model_exports_and_in_module_aliases():
    # Test naming only: a minimal tensor API stub avoids initializing optional
    # libraries or GPU hardware, and the subprocess keeps imports isolated.
    code = textwrap.dedent("""
        import sys
        from types import ModuleType

        tensor_api = ModuleType("torch")
        tensor_api.float16 = object()
        tensor_api.no_grad = lambda: (lambda fn: fn)
        tensor_api.set_num_threads = lambda n: None
        sys.modules["torch"] = tensor_api

        from breeze import InferenceSession, Qwen35GpuModel, Qwen35ReferenceModel
        from breeze.graph_session import InferenceSession as graph_session
        from breeze.qwen35_gpu import Qwen35GpuModel as gpu, QwenGpuModel
        from breeze.qwen35_reference import Qwen35ReferenceModel as reference, QwenSession

        assert InferenceSession is graph_session
        assert Qwen35GpuModel is gpu is QwenGpuModel
        assert Qwen35ReferenceModel is reference is QwenSession
        assert gpu.__name__ == "Qwen35GpuModel"
        assert reference.__name__ == "Qwen35ReferenceModel"
    """)
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr