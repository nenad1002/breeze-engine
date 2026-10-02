"""Check the packed graph weight contract before passing pointers to C++."""


def bind_weights(graph, config):
    nodes = [n for n in graph.nodes if n.op_type == "MatMulNBits"]
    bound = {}
    for fragment, (K, N) in config.matmul_shapes().items():
        matches = [n for n in nodes if fragment + "/" in n.name]
        if len(matches) != 1:
            raise ValueError(f"Expected one MatMulNBits for {fragment}; found {len(matches)}. "
                             "Supply a compatible packed graph with the LM head retained.")
        node = matches[0]
        if (node.attrs.get("K"), node.attrs.get("N")) != (K, N):
            raise ValueError(f"{node.name}: expected K={K}, N={N}; wrong model config/export")
        bits = node.attrs.get("bits")
        if bits not in (4, 8) or node.attrs.get("block_size") != 32:
            raise ValueError(f"{node.name}: requires 4/8-bit MatMulNBits, block_size=32")
        if K % 32 or N % 16:
            raise ValueError(f"{node.name}: dimensions do not meet VNNI alignment requirements")
        if any(node.inputs[4:]):
            raise ValueError(f"{node.name}: g_idx/bias inputs are not supported")
        if len(node.inputs) < 3:
            raise ValueError(f"{node.name}: missing packed weights or scales")
        for name, size in ((node.inputs[1], N * K * bits // 8),
                           (node.inputs[2], N * (K // 32))):
            if name not in graph.initializers or graph.initializers[name].size != size:
                raise ValueError(f"{node.name}: incorrect initializer size for {name}")
        if len(node.inputs) > 3 and node.inputs[3]:
            zero = graph.initializers.get(node.inputs[3])
            count = N * ((K // 32 + 1) // 2 if bits == 4 else K // 32)
            if zero is None or zero.size != count:
                raise ValueError(f"{node.name}: incorrect zero-point shape")
        bound[fragment] = node
    if len({n.name for n in bound.values()}) != len(nodes):
        raise ValueError("Export contains extra/duplicate MatMulNBits nodes; unsupported decoder layout")

    vectors = {}

    def vec(name, size, optional=False):
        array = graph.initializers.get(name)
        if array is None and optional:
            vectors[name] = None
            return
        if array is None or array.size != size:
            raise ValueError(f"Missing or incorrectly sized initializer {name} (expected {size})")
        vectors[name] = array

    c = config
    for i, kind in enumerate(c.layer_types):
        p = f"model.layers.{i}"
        vec(p + ".input_layernorm.weight", c.hidden_size)
        vec(p + ".post_attention_layernorm.weight", c.hidden_size)
        if kind == "linear_attention":
            vec(p + ".linear_attn.conv1d.weight", c.conv_dim * c.linear_conv_kernel_dim)
            vec(p + ".linear_attn.conv1d.bias", c.conv_dim, optional=True)
            vec(p + ".linear_attn.neg_exp_A", c.linear_num_value_heads)
            vec(p + ".linear_attn.dt_bias", c.linear_num_value_heads)
            vec(p + ".linear_attn.norm.weight", c.linear_value_head_dim)
        else:
            vec(p + ".attn.q_norm.layernorm.weight", c.head_dim)
            vec(p + ".attn.k_norm.layernorm.weight", c.head_dim)
    vec(f"model.layers.{c.num_hidden_layers}.final_norm_layernorm.weight", c.hidden_size)
    return bound, vectors