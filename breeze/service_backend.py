"""Worker-owned backends. Native calls, including load and close, stay on one thread."""
from dataclasses import dataclass
import os
import re
import threading
import time

from .service_config import preflight
from .service_schemas import ServiceError


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    decode_steps: int = 0

    @property
    def usage(self):
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens}


def check_cancelled(cancelled, deadline):
    if cancelled.is_set():
        raise ServiceError("Request cancelled", 499, "cancelled", "request_cancelled")
    if time.monotonic() >= deadline:
        raise ServiceError("Request deadline exceeded", 504, "deadline_exceeded", "timeout_error")


def render_messages(messages):
    text = "".join(f"<|im_start|>{message.role}\n{message.content}<|im_end|>\n" for message in messages)
    return text + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


_NATIVE_OWNER = threading.Lock()


class NativeBackend:
    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.tokenizer = None
        self._owns_native = False

    def load(self):
        failures = [item["check"] for item in preflight(self.settings) if not item["ok"]]
        if failures:
            raise RuntimeError("Preflight failed: " + ", ".join(failures) + "; run serve_breeze.py --check")
        if not _NATIVE_OWNER.acquire(blocking=False):
            raise RuntimeError("Only one native service instance is allowed in a Python process")
        self._owns_native = True
        try:
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ[name] = str(self.settings.threads if name == "OMP_NUM_THREADS" else 1)
            os.environ.setdefault("OMP_PROC_BIND", "spread")
            os.environ.setdefault("OMP_PLACES", "cores")
            from tokenizers import Tokenizer
            from . import cpu_backend
            from .qwen35_config import QwenConfig
            from .qwen35_cpu import Qwen35CpuModel

            config = QwenConfig.resolve(self.settings.model)
            if self.settings.max_seq > config.max_position_embeddings:
                raise ValueError("Service context exceeds the model's native context window")
            self.tokenizer = Tokenizer.from_file(str(self.settings.model.parent / "tokenizer.json"))
            self.eos = {token for name in ("<|im_end|>", "<|endoftext|>")
                        if (token := self.tokenizer.token_to_id(name)) is not None}
            if not self.eos:
                raise ValueError("Tokenizer must define a recognized end-of-response token")
            cpu_backend.set_threads(self.settings.threads)
            self.model = Qwen35CpuModel(self.settings.model, config=config,
                                        max_seq=self.settings.max_seq, hi_prec=True, verbose=False)
        except BaseException:
            self.close()
            raise

    def prepare(self, request):
        ids = self.tokenizer.encode(render_messages(request.messages)).ids
        if not ids or len(ids) + request.max_tokens - 1 > self.settings.max_seq:
            raise ServiceError("Prompt plus requested output exceeds the configured context window. "
                               "Shorten the conversation or reduce max_tokens.", code="context_length_exceeded")
        return ids

    def generate(self, request, ids, emit, cancelled, deadline):
        import numpy as np

        start = time.monotonic()
        last = None
        for offset in range(0, len(ids), self.settings.chunk_size):
            check_cancelled(cancelled, deadline)
            # Every request starts at zero; no conversation state crosses requests.
            last = self.model.run(self.model.embed(ids[offset:offset + self.settings.chunk_size]), offset)[-1:].copy()
        prefill_seconds = time.monotonic() - start
        generated, text, past = [], "", len(ids)
        decode_seconds, decode_steps = 0.0, 0
        reason = "length"
        for step in range(request.max_tokens):
            check_cancelled(cancelled, deadline)
            if last is None or not np.isfinite(last).all():
                raise RuntimeError("Nonfinite output from the native decoder")
            token = int(last[-1].argmax())
            generated.append(token)
            decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
            if not decoded.endswith("\ufffd"):
                if not decoded.startswith(text):
                    raise RuntimeError("Tokenizer changed an already-emitted text prefix")
                if decoded[len(text):]:
                    emit(decoded[len(text):])
                    text = decoded
            if token in self.eos:
                reason = "stop"
                break
            if step + 1 == request.max_tokens:
                break
            check_cancelled(cancelled, deadline)
            embeddings = self.model.embed([token])
            tick = time.monotonic()
            last = self.model.run(embeddings, past_len=past)
            decode_seconds += time.monotonic() - tick
            decode_steps += 1
            past += 1
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        if not decoded.startswith(text):
            raise RuntimeError("Tokenizer changed an already-emitted text prefix")
        if decoded[len(text):]:
            emit(decoded[len(text):])
        return Completion(decoded, len(ids), len(generated), reason, prefill_seconds,
                          decode_seconds, decode_steps)

    def close(self):
        try:
            if self.model is not None:
                self.model.close()
                self.model = None
            self.tokenizer = None
        finally:
            if self._owns_native:
                self._owns_native = False
                _NATIVE_OWNER.release()


class DemoBackend:
    """Explicitly scripted preview, never presented as model inference."""
    def __init__(self, settings):
        self.settings = settings

    def load(self):
        pass

    def prepare(self, request):
        words = re.findall(r"\S+", render_messages(request.messages))
        if len(words) + request.max_tokens - 1 > self.settings.max_seq:
            raise ServiceError("Preview conversation exceeds the context budget", code="context_length_exceeded")
        return words

    def generate(self, request, words, emit, cancelled, deadline):
        text = ("This is a scripted preview, not a model-generated answer.\n\n"
                "Breeze brings a local chat API and a focused workspace to your CPU server. "
                "Connect a prepared Qwen3.5 checkpoint to draft replies, summarize working notes, "
                "and turn source material into actionable updates.\n\n"
                "This preview demonstrates streaming, conversation controls, and API integration. "
                "It does not evaluate or answer the submitted content.")
        pieces = re.findall(r"\S+\s*", text)
        selected = pieces[:request.max_tokens]
        for piece in selected:
            check_cancelled(cancelled, deadline)
            emit(piece)
        return Completion("".join(selected), len(words), len(selected),
                          "stop" if len(selected) == len(pieces) else "length")

    def close(self):
        pass