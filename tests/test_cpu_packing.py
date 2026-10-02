"""Small native-packing regressions; no model downloads or full checkpoints."""
import ctypes

import numpy as np
import pytest

from breeze import cpu_backend as ib


class PackedWeight(ctypes.Structure):
    """Private layout mirrored solely to compare every packed byte in tests."""
    _fields_ = [(name, ctypes.c_int) for name in ("N", "K", "nblk", "ntiles", "bits")] + [
        (name, ctypes.c_void_p) for name in ("B", "scales", "zp", "corr", "B1", "scales1", "zp1", "corr1")
    ]


def packed_bytes(handle, replica=False):
    packed = ctypes.cast(handle, ctypes.POINTER(PackedWeight)).contents
    blocks = packed.N * packed.nblk
    suffix = "1" if replica else ""
    result = {}
    for name, size in (("B", blocks * 4 * packed.bits), ("scales", blocks * 4),
                       ("zp", blocks if packed.bits == 4 else 0), ("corr", blocks * 4)):
        pointer = getattr(packed, name + suffix)
        if size:
            assert pointer and pointer % 64 == 0
            result[name] = ctypes.string_at(pointer, size)
        else:
            assert pointer is None
    return result


def fixture_weights(N, K, bits, dtype, explicit_zeros, strided=False):
    rng = np.random.default_rng(N + K + bits)
    blocks = K // 32
    qw = rng.integers(0, 256, (N, blocks, 4 * bits), dtype=np.uint8)
    qw.reshape(-1)[:8] = [0, 1, 7, 15, 127, 128, 254, 255]
    scales = rng.uniform(0.0001, 0.02, (N, blocks)).astype(dtype)
    zeros = None
    if explicit_zeros:
        zeros = (rng.integers(0, 256, (N, (blocks + 1) // 2), dtype=np.uint8)
                 if bits == 4 else np.full((N, blocks), 128, np.uint8))

    def stride(array):
        backing = np.empty(array.size * 2, array.dtype)
        backing[::2] = array.reshape(-1)
        return backing[::2]

    if strided:
        qw, scales = stride(qw), stride(scales)
        if zeros is not None:
            zeros = stride(zeros)
    return qw, scales, zeros


@pytest.fixture(autouse=True)
def native():
    assert ib.available() and ib._native_prepack is not None, "Build the new native packer first"
    ib.set_threads(2)
    split = ib._lib.i4_set_numa_split
    split.argtypes = [ctypes.c_int]
    split.restype = None
    split(1 << 30)
    yield
    split(48)
    ib.set_threads(2)
    ib.set_hi_prec(False)


@pytest.mark.parametrize("N,K", [(16, 32), (48, 96), (64, 512)])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("explicit_zeros", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_native_matches_legacy_bytes_and_matmul(monkeypatch, N, K, bits, dtype, explicit_zeros, strided):
    qw, sc, qz = fixture_weights(N, K, bits, dtype, explicit_zeros, strided)
    copies = [array.copy() if array is not None else None for array in (qw, sc, qz)]
    fast = ib.prepack(qw, sc, qz, K, N, bits=bits)
    old = None
    try:
        with monkeypatch.context() as patch:
            patch.setattr(ib, "_native_prepack", None)
            old = ib.prepack(qw, sc, qz, K, N, bits=bits)
        assert packed_bytes(fast) == packed_bytes(old)
        rng = np.random.default_rng(5)
        for high_precision in (False, True):
            ib.set_hi_prec(high_precision)
            for rows in (1, 9):  # decode and prefill beyond the eight-row tile
                inputs = rng.normal(size=(rows, K)).astype(np.float32)
                np.testing.assert_array_equal(ib.matmul(fast, inputs, N), ib.matmul(old, inputs, N))
        for actual, expected in zip((qw, sc, qz), copies):
            if actual is not None:
                np.testing.assert_array_equal(actual, expected)
    finally:
        ib.free(fast)
        ib.free(old)


@pytest.mark.parametrize("dtype", [np.float64, ">f2", ">f4"])
@pytest.mark.parametrize("bits", [4, 8])
def test_scale_conversion_and_byte_order(monkeypatch, dtype, bits):
    qw, sc, qz = fixture_weights(32, 96, bits, dtype, False)
    fast = ib.prepack(qw, sc, qz, 96, 32, bits=bits)
    old = None
    try:
        monkeypatch.setattr(ib, "_native_prepack", None)
        old = ib.prepack(qw, sc, qz, 96, 32, bits=bits)
        assert packed_bytes(fast) == packed_bytes(old)
    finally:
        ib.free(fast)
        ib.free(old)


def test_fp16_scales_include_subnormals_and_signed_zero(monkeypatch):
    qw, _, _ = fixture_weights(16, 32, 4, np.float16, False)
    sc = np.array([0, 0x8000, 1, 0x8001, 0x03FF, 0x0400, 0x3C00, 0x7BFF] * 2,
                  dtype=np.uint16).view(np.float16)
    # Native scalar loads must also work for a deliberately unaligned view.
    backing = np.empty(sc.nbytes + 1, dtype=np.uint8)
    unaligned = np.ndarray(sc.shape, dtype=np.float16, buffer=backing, offset=1)
    unaligned[:] = sc
    assert not unaligned.flags.aligned
    fast = ib.prepack(qw, unaligned, None, 32, 16)
    old = None
    try:
        monkeypatch.setattr(ib, "_native_prepack", None)
        old = ib.prepack(qw, sc, None, 32, 16)
        assert packed_bytes(fast) == packed_bytes(old)
    finally:
        ib.free(fast)
        ib.free(old)


@pytest.mark.parametrize("bits", [4, 8])
def test_read_only_mmap_and_owned_output(tmp_path, bits):
    qw, sc, qz = fixture_weights(48, 96, bits, np.float16, True)
    path = tmp_path / "weights.bin"
    path.write_bytes(qw.tobytes())
    mapped = np.memmap(path, mode="r", dtype=np.uint8, shape=qw.shape)
    sc.flags.writeable = qz.flags.writeable = False
    handle = ib.prepack(mapped, sc, qz, 96, 48, bits=bits)
    try:
        before = packed_bytes(handle)
        del mapped
        sc.flags.writeable = qz.flags.writeable = True
        sc[:] = 0
        qz[:] = 0
        assert packed_bytes(handle) == before
        assert path.read_bytes() == qw.tobytes()
    finally:
        ib.free(handle)


@pytest.mark.parametrize("bits", [4, 8])
def test_replica_complete_after_thread_count_change(bits):
    ib._lib.i4_set_numa_split(0)  # all test workers act as node-1 workers
    ib.set_threads(4)
    qw, sc, qz = fixture_weights(80, 96, bits, np.float32, True)
    handle = ib.prepack(qw, sc, qz, 96, 80, bits=bits)
    try:
        assert packed_bytes(handle, replica=True) == packed_bytes(handle)
        inputs = np.ones((9, 96), np.float32)
        ib.set_hi_prec(True)
        reference = ib.matmul(handle, inputs, 80)
        ib.set_threads(1)
        np.testing.assert_array_equal(ib.matmul(handle, inputs, 80), reference)
    finally:
        ib.free(handle)


@pytest.mark.parametrize("fault", ["weights", "scales", "zeros", "zero_dtype", "fractional_dim", "overflow_dim"])
def test_reject_malformed_inputs(fault):
    qw, sc, qz = fixture_weights(16, 32, 4, np.float32, True)
    K, N = 32, 16
    if fault == "weights": qw = qw.reshape(-1)[:-1]
    elif fault == "scales": sc = sc.reshape(-1)[:-1]
    elif fault == "zeros": qz = qz.reshape(-1)[:-1]
    elif fault == "zero_dtype": qz = qz.astype(np.int32)
    elif fault == "fractional_dim": N = 16.0
    elif fault == "overflow_dim": K = 2**32 + 32
    with pytest.raises(ValueError):
        ib.prepack(qw, sc, qz, K, N)


@pytest.mark.parametrize("N,K,bits,scale_bits", [(0, 32, 4, 32), (16, 0, 4, 32),
                                                  (17, 32, 4, 32), (16, 33, 4, 32),
                                                  (16, 32, 3, 32), (16, 32, 4, 64)])
def test_native_rejects_bad_contract(N, K, bits, scale_bits):
    qw, sc, _ = fixture_weights(16, 32, 4, np.float32, False)
    assert not ib._native_prepack(ib._p(qw), ib._p(sc), None, N, K, bits, scale_bits)


def test_native_allocation_failure_is_reported(monkeypatch):
    qw, sc, _ = fixture_weights(16, 32, 4, np.float32, False)
    monkeypatch.setattr(ib, "_native_prepack", lambda *args: None)
    with pytest.raises(MemoryError, match="Could not allocate"):
        ib.prepack(qw, sc, None, 32, 16)


def test_default_does_not_use_numpy_packing(monkeypatch):
    qw, sc, _ = fixture_weights(16, 32, 4, np.float32, False)

    def forbidden(*args):
        raise AssertionError("The native path should not invoke the NumPy packer")

    monkeypatch.setattr(ib, "_prepack_numpy", forbidden)
    handle = ib.prepack(qw, sc, None, 32, 16)
    ib.free(handle)