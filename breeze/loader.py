"""Minimal ONNX loader: parse the protobuf into a lightweight graph IR."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

# ONNX TensorProto dtype -> numpy dtype (subset used by the Phi-3.5 INT4 model).
ONNX_TO_NP = {
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.FLOAT16: np.float16,
    onnx.TensorProto.DOUBLE: np.float64,
    onnx.TensorProto.INT64: np.int64,
    onnx.TensorProto.INT32: np.int32,
    onnx.TensorProto.INT8: np.int8,
    onnx.TensorProto.UINT8: np.uint8,
    onnx.TensorProto.BOOL: np.bool_,
}


def _attr_value(a, subgraphs):
    A = onnx.AttributeProto
    t = a.type
    if t == A.INT:
        return a.i
    if t == A.FLOAT:
        return a.f
    if t == A.STRING:
        return a.s.decode()
    if t == A.INTS:
        return list(a.ints)
    if t == A.FLOATS:
        return list(a.floats)
    if t == A.TENSOR:
        return numpy_helper.to_array(a.t)
    if t == A.GRAPH:
        subgraphs[a.name] = Graph(a.g)
        return a.name
    raise NotImplementedError(f"Attribute type {t} for {a.name!r} not supported.")


class Node:
    __slots__ = ("op_type", "domain", "name", "inputs", "outputs", "attrs", "subgraphs")

    def __init__(self, proto):
        self.op_type = proto.op_type
        self.domain = proto.domain or "ai.onnx"
        self.name = proto.name
        self.inputs = list(proto.input)
        self.outputs = list(proto.output)
        self.subgraphs = {}
        self.attrs = {a.name: _attr_value(a, self.subgraphs) for a in proto.attribute}

    def __repr__(self):
        return f"Node({self.op_type}, {self.name!r})"


class Graph:
    def __init__(self, graph_proto, tensor_loader=numpy_helper.to_array):
        self.initializers = {t.name: tensor_loader(t) for t in graph_proto.initializer}
        init_names = set(self.initializers)
        self.inputs = [i.name for i in graph_proto.input if i.name not in init_names]
        self.outputs = [o.name for o in graph_proto.output]
        self.nodes = [Node(n) for n in graph_proto.node]


def load_graph(path, mmap_external=False):
    """Load an ONNX model (with external data) into a :class:`Graph`."""
    if mmap_external:
        # Keep the snapshot directory: HF files may themselves be symlinks into
        # a separate blobs directory, but external locations are snapshot-relative.
        base = Path(path).absolute().parent

        def load_tensor(t):
            if t.data_location != onnx.TensorProto.EXTERNAL:
                return numpy_helper.to_array(t)
            info = {entry.key: entry.value for entry in t.external_data}
            location = Path(info["location"])
            if location.is_absolute() or ".." in location.parts:
                raise ValueError(f"External tensor escapes model directory: {t.name}")
            external = base / location
            if t.data_type not in ONNX_TO_NP:
                raise ValueError(f"Unsupported external tensor dtype {t.data_type}: {t.name}")
            dtype = np.dtype(ONNX_TO_NP[t.data_type]).newbyteorder("<")
            offset = int(info.get("offset", 0))
            count = int(np.prod(t.dims, dtype=np.int64))
            if offset < 0 or offset + count * dtype.itemsize > external.stat().st_size:
                raise ValueError(f"Truncated external tensor: {t.name}")
            if count == 0:
                return np.empty(tuple(t.dims), dtype=dtype)
            return np.memmap(external, dtype=dtype, mode="r", offset=offset, shape=tuple(t.dims))

        return Graph(onnx.load(path, load_external_data=False).graph, load_tensor)
    model = onnx.load(path)  # loads the external .data file automatically
    return Graph(model.graph)
