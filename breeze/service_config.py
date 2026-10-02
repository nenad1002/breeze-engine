"""Validated, secret-safe settings for the optional single-process service."""
from dataclasses import dataclass, field
import ipaddress
import math
import os
from pathlib import Path
import platform
import re
import sys


def _valid_host(name):
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return isinstance(name, str) and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name))


@dataclass(frozen=True)
class ServiceSettings:
    model: Path | None = None
    demo: bool = False
    model_id: str = "breeze-local"
    host: str = "127.0.0.1"
    port: int = 8080
    threads: int = 8
    max_seq: int = 4096
    max_output_tokens: int = 512
    chunk_size: int = 128
    max_pending: int = 4
    request_timeout: float = 120.0
    max_body_bytes: int = 131072
    api_key: str | None = field(default=None, repr=False)
    allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "::1")

    def __post_init__(self):
        if self.demo == (self.model is not None):
            raise ValueError("Choose exactly one of a local model or demo mode")
        if self.model is not None:
            object.__setattr__(self, "model", Path(self.model).absolute())
        if self.demo:
            object.__setattr__(self, "model_id", "breeze-demo")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", self.model_id):
            raise ValueError("Model ID must use 1–100 letters, digits, periods, underscores or hyphens")
        for name in ("threads", "max_seq", "max_output_tokens", "chunk_size", "max_pending",
                     "max_body_bytes", "port"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.port > 65535 or self.max_output_tokens > 4096:
            raise ValueError("Port must be at most 65535; output limit at most 4096 tokens")
        if self.max_output_tokens > self.max_seq or self.chunk_size > self.max_seq:
            raise ValueError("Output and prefill chunk limits cannot exceed the context window")
        if self.max_pending > 64:
            raise ValueError("At most 64 admitted requests are allowed")
        if not math.isfinite(self.request_timeout) or not 0 < self.request_timeout <= 3600:
            raise ValueError("Request timeout must be finite and between 0 and 3600 seconds")
        if not self.allowed_hosts or any(not _valid_host(name) for name in self.allowed_hosts):
            raise ValueError("Specify exact allowed hostnames or IP addresses, without wildcards or ports")
        if self.api_key is not None and (len(self.api_key) < 24 or not self.api_key.isascii()
                                         or any(char.isspace() for char in self.api_key)):
            raise ValueError("BREEZE_API_KEY must contain at least 24 non-whitespace ASCII characters")
        try:
            loopback = ipaddress.ip_address(self.host).is_loopback
        except ValueError:
            loopback = self.host.lower() == "localhost"
        if not loopback and not self.api_key:
            raise ValueError("Non-loopback binding requires BREEZE_API_KEY")
        if os.environ.get("I4_QWEN_MAXL") is not None:
            raise ValueError("Unset I4_QWEN_MAXL; partial-layer runs cannot serve requests")


def preflight(settings):
    """Check local prerequisites without loading weights or importing the CPU bridge."""
    if settings.demo:
        return [{"check": "preview", "ok": True, "detail": "Scripted demo; no model or native library loaded"}]
    checks = []
    supported = sys.platform.startswith("linux") and platform.machine().lower() in {"x86_64", "amd64"}
    checks.append({"check": "platform", "ok": supported, "detail": "Linux x86-64 required"})
    try:
        flags = [set(line.split(":", 1)[1].split()) for line in
                 Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("flags")]
        instructions = bool(flags) and all(
            {"avx512f", "avx512vbmi"} <= row and bool({"avx512_vnni", "avx512vnni"} & row)
            for row in flags)
    except OSError:
        instructions = False
    checks.append({"check": "cpu", "ok": instructions, "detail": "AVX-512 VNNI and VBMI required"})
    library = Path(__file__).with_name("_breeze_cpu.so")
    checks.append({"check": "native_library", "ok": library.is_file(),
                   "detail": "Build the native library on the deployment CPU"})
    for name, path in (("decoder", settings.model), ("configuration", settings.model.parent / "config.json"),
                       ("tokenizer", settings.model.parent / "tokenizer.json"),
                       ("embeddings", settings.model.parent / "embeddings.npy")):
        checks.append({"check": name, "ok": path.is_file(), "detail": str(path)})
    return checks