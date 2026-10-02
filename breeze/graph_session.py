"""Minimal execution engine for the INT4 Phi-3.5 ONNX graph.

Weights are int4-dequantized once at session construction (the main optimization)
so each forward pass is pure BLAS matmuls.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

from . import cpu_backend, ops
from .linear import backend, prepare_weight
from .loader import load_graph
from .quant import dequantize_matmul_nbits


class _Context:
    def __init__(self, session):
        self.session = session
        self.weight_cache = session.weight_cache

    def run_graph(self, graph, feeds):
        env = dict(graph.initializers)
        env.update(feeds)
        for node in graph.nodes:
            in_vals = [env[n] if n != "" else None for n in node.inputs]
            outs = ops.get(node.op_type)(self, node, in_vals)
            for name, val in zip(node.outputs, outs):
                if name != "":
                    env[name] = val
        return [env[o] for o in graph.outputs]


class InferenceSession:
    """Graph inference session scoped to this model's op set."""

    def __init__(self, model_path, verbose=True):
        t0 = time.time()
        self.graph = load_graph(model_path)
        self.weight_cache = {}
        self._use_i4 = cpu_backend.available()
        self._predequantize()
        name = "c++ int4 (avx512+openmp)" if self._use_i4 else backend()
        if verbose:
            print(f"[breeze] session ready in {time.time() - t0:.1f}s "
                  f"({len(self.weight_cache)} weights, {name} matmul backend)")

    def _predequantize(self):
        g = self.graph
        nodes = [n for n in g.nodes if n.op_type == "MatMulNBits"]

        if self._use_i4:
            # Compiled kernel takes the raw int4 weights (no fp32 expansion). Prepack
            # sequentially so each weight's NUMA first-touch is deterministic.
            for node in nodes:
                qzeros = None
                if len(node.inputs) > 3 and node.inputs[3] != "":
                    qzeros = g.initializers[node.inputs[3]]
                N, K = int(node.attrs["N"]), int(node.attrs["K"])
                handle = cpu_backend.prepack(
                    g.initializers[node.inputs[1]],
                    g.initializers[node.inputs[2]],
                    qzeros, K, N)
                self.weight_cache[node.name] = ("i4", handle, N)
            return

        def work(node):
            qzeros = None
            if len(node.inputs) > 3 and node.inputs[3] != "":
                qzeros = g.initializers[node.inputs[3]]
            w = dequantize_matmul_nbits(
                g.initializers[node.inputs[1]],
                g.initializers[node.inputs[2]],
                qzeros,
                bits=int(node.attrs["bits"]),
                block_size=int(node.attrs["block_size"]),
                K=int(node.attrs["K"]),
                N=int(node.attrs["N"]),
            )
            return node.name, ("linear", prepare_weight(w), int(node.attrs["N"]))

        # int4 unpack is independent per weight; NumPy releases the GIL on the
        # large elementwise ops, so a thread pool parallelizes the load.
        workers = min(len(nodes), (os.cpu_count() or 8), 32)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for name, w in pool.map(work, nodes):
                self.weight_cache[name] = w

    @property
    def input_names(self):
        return list(self.graph.inputs)

    @property
    def output_names(self):
        return list(self.graph.outputs)

    def run(self, feeds):
        """Run the graph. ``feeds`` maps input name -> numpy array.

        Returns a dict of every graph output (``logits`` and the ``present.*`` caches).
        """
        outs = _Context(self).run_graph(self.graph, feeds)
        return dict(zip(self.graph.outputs, outs))
