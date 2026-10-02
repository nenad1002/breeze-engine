"""Qwen3.5-9B reference engine (numpy fp32).

Dequantizes every MatMulNBits weight (int8 + int4) to dense fp32 and runs the
generic ONNX executor. Prefill only, batch 1.
"""
import time
from concurrent.futures import ThreadPoolExecutor

from . import cpu_backend, linear
from .graph_session import _Context
from .loader import load_graph
from .quant import dequantize_matmul_nbits


class Qwen35ReferenceModel:
    def __init__(self, model_path, verbose=True, backend="cpp", max_workers=8):
        t0 = time.time()
        self.graph = load_graph(model_path)
        self.weight_cache = {}
        use_cpp = backend in ("cpp", "cpp_fast", "mixed") and cpu_backend.available()
        # int16 2-pass activation (accurate ~15-bit) for 'cpp'; int8 (fast/lossy)
        # only for the explicit 'cpp_fast' backend.
        self._hi_prec = backend != "cpp_fast"
        if use_cpp:
            cpu_backend.set_hi_prec(self._hi_prec)
            pred = None
            if backend == "mixed":
                # Big feed-forward matmuls (bulk of params, not in the recurrent
                # path) tolerate int8 activations; keep attention + lm_head fp16.
                pred = lambda n: "mlp" in n.name
            self._prepack_cpp(pred)
        else:
            self._dequantize(max_workers, kind="linear" if backend == "torch" else "dense")
        if verbose:
            name = {"cpp": "c++ vnni int16-act", "cpp_fast": "c++ vnni int8-act",
                    "mixed": "c++ mlp + fp16 attn", "torch": linear.backend(),
                    "numpy": "numpy fp32"}.get(backend, backend)
            print(f"[qwen] session ready in {time.time() - t0:.1f}s "
                  f"({len(self.weight_cache)} matmuls, {name})")

    def _prepack_cpp(self, cpp_predicate=None):
        """Prepack every MatMulNBits into the compiled kernel's layout (int8+int4).
        Sequential so each weight's NUMA first-touch is deterministic.
        cpp_predicate(node)->bool selects the C++ kernel; others use torch fp16."""
        g = self.graph
        for node in g.nodes:
            if node.op_type != "MatMulNBits":
                continue
            qz = None
            if len(node.inputs) > 3 and node.inputs[3] != "":
                qz = g.initializers[node.inputs[3]]
            N = int(node.attrs["N"])
            if cpp_predicate is None or cpp_predicate(node):
                handle = cpu_backend.prepack(
                    g.initializers[node.inputs[1]],
                    g.initializers[node.inputs[2]],
                    qz, int(node.attrs["K"]), N, bits=int(node.attrs["bits"]))
                self.weight_cache[node.name] = ("i4", handle, N)
            else:
                W = dequantize_matmul_nbits(
                    g.initializers[node.inputs[1]], g.initializers[node.inputs[2]], qz,
                    bits=int(node.attrs["bits"]), block_size=int(node.attrs["block_size"]),
                    K=int(node.attrs["K"]), N=N)
                self.weight_cache[node.name] = ("linear", linear.prepare_weight(W), N)

    def _dequantize(self, max_workers, kind="dense"):
        g = self.graph
        nodes = [n for n in g.nodes if n.op_type == "MatMulNBits"]

        def work(node):
            qz = None
            if len(node.inputs) > 3 and node.inputs[3] != "":
                qz = g.initializers[node.inputs[3]]
            W = dequantize_matmul_nbits(
                g.initializers[node.inputs[1]],
                g.initializers[node.inputs[2]],
                qz,
                bits=int(node.attrs["bits"]),
                block_size=int(node.attrs["block_size"]),
                K=int(node.attrs["K"]),
                N=int(node.attrs["N"]),
            )
            N = int(node.attrs["N"])
            if kind == "linear":
                return node.name, ("linear", linear.prepare_weight(W), N)
            return node.name, ("dense", W, N)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for name, w in pool.map(work, nodes):
                self.weight_cache[name] = w

    @property
    def input_names(self):
        return list(self.graph.inputs)

    def run(self, feed):
        if cpu_backend.available():
            cpu_backend.set_hi_prec(self._hi_prec)
        outs = _Context(self).run_graph(self.graph, feed)
        return dict(zip(self.graph.outputs, outs))


QwenSession = Qwen35ReferenceModel
