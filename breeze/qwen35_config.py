"""Validated text-decoder configuration for dense Qwen3.5 CPU inference."""
from dataclasses import dataclass, fields
import json
from pathlib import Path


@dataclass(frozen=True)
class QwenConfig:
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 12288
    vocab_size: int = 248320
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    max_position_embeddings: int = 262144
    layer_types: tuple = ()

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if f.type is int and (type(value) is not int or value <= 0):
                raise ValueError(f"{f.name} must be a positive integer")
        kinds = self.layer_types or tuple(
            "full_attention" if i % 4 == 3 else "linear_attention"
            for i in range(self.num_hidden_layers))
        object.__setattr__(self, "layer_types", tuple(kinds))
        if len(kinds) != self.num_hidden_layers or any(
                k not in ("linear_attention", "full_attention") for k in kinds):
            raise ValueError("layer_types must describe every decoder layer")
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError("Linear value heads must be a multiple of key heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Attention query heads must be a multiple of KV heads")
        if self.linear_value_head_dim > 256:
            raise ValueError("CPU recurrence supports value head dimensions up to 256")
        if not (0 < self.rotary_dim <= self.head_dim and self.rotary_dim % 2 == 0):
            raise ValueError("Unsupported rotary dimension")
        if not (self.rope_theta > 0 and self.rms_norm_eps > 0):
            raise ValueError("RoPE theta and RMSNorm epsilon must be positive")

    @property
    def rotary_dim(self):
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def key_dim(self):
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def value_dim(self):
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def conv_dim(self):
        return 2 * self.key_dim + self.value_dim

    @property
    def full_layers(self):
        return [i for i, kind in enumerate(self.layer_types) if kind == "full_attention"]

    @classmethod
    def preset(cls, name):
        if name.lower() == "9b":
            return cls()
        if name.lower() == "27b":
            return cls(hidden_size=5120, num_hidden_layers=64, intermediate_size=17408,
                       linear_num_value_heads=48, num_attention_heads=24)
        raise ValueError(f"Unknown Qwen3.5 preset: {name}")

    @classmethod
    def from_dict(cls, data):
        text = data.get("text_config", data)
        if text.get("model_type") not in (None, "qwen3_5_text", "qwen3_5"):
            raise ValueError("Only dense Qwen3.5 text decoders are supported")
        if text.get("num_experts", 0) or text.get("attention_bias", False):
            raise ValueError("MoE and attention-bias variants are not supported")
        if text.get("hidden_act", "silu") != "silu" or not text.get("attn_output_gate", True):
            raise ValueError("The CPU decoder requires SiLU and gated attention")
        required = [f.name for f in fields(cls) if f.type is int
                    and f.name != "max_position_embeddings"]
        missing = [key for key in required if key not in text]
        if missing:
            raise ValueError(f"Incomplete Qwen text config; missing {missing}")
        rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default":
            raise ValueError("Scaled RoPE/YaRN is not implemented in the CPU decoder")
        values = {f.name: text[f.name] for f in fields(cls) if f.name in text}
        for name in ("rope_theta", "partial_rotary_factor"):
            if name in rope:
                values[name] = rope[name]
        if "layer_types" not in values:
            interval = text.get("full_attention_interval", 4)
            if type(interval) is not int or interval <= 0:
                raise ValueError("full_attention_interval must be positive")
            values["layer_types"] = tuple(
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(text["num_hidden_layers"]))
        return cls(**values)

    @classmethod
    def resolve(cls, model_path, config=None, graph=None):
        if isinstance(config, cls):
            return config
        if isinstance(config, dict):
            return cls.from_dict(config)
        if config is not None and str(config).lower() in ("9b", "27b"):
            return cls.preset(str(config))
        path = Path(config) if config is not None else Path(model_path).parent / "config.json"
        if path.is_file():
            return cls.from_dict(json.loads(path.read_text()))
        if config is not None:
            raise FileNotFoundError(path)
        # Legacy 9B exports sometimes omit the HF config. Infer only known sizes;
        # the complete node/weight contract is checked before any native call.
        if graph is not None:
            heads = [n for n in graph.nodes if n.op_type == "MatMulNBits" and "lm_head" in n.name]
            if len(heads) == 1:
                for name in ("9b", "27b"):
                    candidate = cls.preset(name)
                    if (heads[0].attrs.get("K"), heads[0].attrs.get("N")) == (
                            candidate.hidden_size, candidate.vocab_size):
                        return candidate
        raise ValueError("Supply the HF config.json or an explicit 9b/27b config preset")

    def matmul_shapes(self):
        """Required graph node-name fragments -> (input, output) widths."""
        shapes = {}
        H, I = self.hidden_size, self.intermediate_size
        for i, kind in enumerate(self.layer_types):
            p = f"layers.{i}/"
            shapes.update({p + "mlp/gate_proj": (H, I), p + "mlp/up_proj": (H, I),
                           p + "mlp/down_proj": (I, H)})
            if kind == "linear_attention":
                shapes.update({p + "linear_attn/in_proj_qkv": (H, self.conv_dim),
                               p + "linear_attn/in_proj_z": (H, self.value_dim),
                               p + "linear_attn/in_proj_a": (H, self.linear_num_value_heads),
                               p + "linear_attn/in_proj_b": (H, self.linear_num_value_heads),
                               p + "linear_attn/out_proj": (self.value_dim, H)})
            else:
                Q = self.num_attention_heads * self.head_dim
                KV = self.num_key_value_heads * self.head_dim
                shapes.update({p + "attn/q_proj": (H, 2 * Q), p + "attn/k_proj": (H, KV),
                               p + "attn/v_proj": (H, KV), p + "attn/o_proj": (Q, H)})
        shapes["lm_head"] = (H, self.vocab_size)
        return shapes