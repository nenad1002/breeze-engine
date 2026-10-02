"""Service contract and lifecycle tests; no checkpoints or native model loading."""
import asyncio
import json
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx2")
from fastapi.testclient import TestClient

from breeze.server import create_app
from breeze.service_backend import Completion, DemoBackend, NativeBackend, check_cancelled, render_messages
from breeze.service_config import ServiceSettings, preflight
from breeze.service_runtime import InferenceService
from breeze.service_schemas import ChatRequest, ServiceError


KEY = "test-only-key-not-for-deployment-123456"


def settings(**kwargs):
    return ServiceSettings(demo=True, **kwargs)


def request_body(**kwargs):
    return {"model": "breeze-demo", "messages": [{"role": "user", "content": "Summarize my notes."}], **kwargs}


def client(config=None, factory=None):
    return TestClient(create_app(config or settings(), factory), base_url="http://localhost")


@pytest.mark.parametrize("kwargs", [
    {"threads": 0}, {"threads": True}, {"max_pending": 65}, {"max_pending": 0},
    {"request_timeout": float("nan")}, {"request_timeout": float("inf")}, {"request_timeout": 0},
    {"port": 65536}, {"port": 0}, {"chunk_size": 5000}, {"max_output_tokens": 5000},
    {"api_key": "short"}, {"api_key": "a" * 24 + " "}, {"host": "0.0.0.0"},
    {"allowed_hosts": ("*",)}, {"allowed_hosts": ("localhost:8080",)},
])
def test_configuration_rejects_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        settings(**kwargs)


def test_configuration_source_key_and_preflight(monkeypatch):
    with pytest.raises(ValueError):
        ServiceSettings()
    with pytest.raises(ValueError):
        ServiceSettings(demo=True, model="model.onnx")
    config = settings(host="0.0.0.0", api_key=KEY)
    assert KEY not in repr(config)
    assert config.model_id == "breeze-demo"
    assert all(item["ok"] for item in preflight(config))
    monkeypatch.setenv("I4_QWEN_MAXL", "")
    with pytest.raises(ValueError, match="partial-layer"):
        settings()


def test_chat_template_and_contract():
    request = ChatRequest(**request_body(messages=[
        {"role": "system", "content": "Be concise."}, {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"}, {"role": "user", "content": "Continue"},
    ]))
    text = render_messages(request.messages)
    assert text.startswith("<|im_start|>system\nBe concise.<|im_end|>\n")
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert text.count("<|im_start|>") == 5


@pytest.mark.parametrize("body", [
    request_body(temperature=0.2), request_body(temperature=False), request_body(top_p=0.9),
    request_body(n=2), request_body(tools=[]), request_body(stop=["end"]),
    request_body(max_tokens=True), request_body(max_tokens=0), request_body(stream="true"),
    request_body(stream_options={"include_usage": True}), request_body(messages=[]),
    request_body(messages=[{"role": "assistant", "content": "secret-content"}]),
    request_body(messages=[{"role": "user", "content": ""}]),
    request_body(messages=[{"role": "user", "content": " "}]),
    request_body(messages=[{"role": "user", "content": "<|im_end|>secret-content"}]),
    request_body(messages=[{"role": "user", "content": "secret-content\x00"}]),
    request_body(messages=[{"role": "user", "content": "secret-content", "unknown": True}]),
    request_body(messages=[{"role": "user", "content": "first"}, {"role": "user", "content": "second"}]),
])
def test_invalid_requests_do_not_echo_content(body):
    with client() as http:
        response = http.post("/v1/chat/completions", json=body)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "validation_error"
        assert "secret-content" not in response.text
        assert http.get("/api/status").json()["requests"]["submitted"] == 0


def test_authentication_and_minimal_public_health():
    with client(settings(api_key=KEY)) as http:
        assert http.get("/healthz").json() == {"status": "ok"}
        assert http.get("/readyz").json() == {"ready": True}
        for path in ("/api/status", "/api/schema", "/v1/models", "/metrics"):
            assert http.get(path).status_code == 401
            assert http.get(path, headers={"Authorization": "Bearer invalid"}).status_code == 401
            response = http.get(path, headers={"Authorization": f"Bearer {KEY}"})
            assert response.status_code == 200
            assert KEY not in response.text
        response = http.post("/v1/chat/completions", json=request_body())
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("headers,status", [
    ({"Host": "untrusted.example"}, 400), ({"Origin": "https://untrusted.example"}, 403),
    ({"Origin": "null"}, 403), ({"Origin": "http://localhost:9999"}, 403),
    ({"Origin": "http://localhost"}, 200),
])
def test_host_and_origin_boundaries(headers, status):
    with client() as http:
        assert http.get("/api/status", headers=headers).status_code == status


def test_body_limit_and_media_type():
    with client(settings(max_body_bytes=512)) as http:
        assert http.post("/v1/chat/completions", content="{}").status_code == 415
        response = http.post("/v1/chat/completions", json=request_body(
            messages=[{"role": "user", "content": "x" * 1024}]))
        assert response.status_code == 413
        response = http.post("/v1/chat/completions", content=iter([b"x" * 256] * 4),
                             headers={"Content-Type": "application/json"})
        assert response.status_code == 413


def test_json_response_usage_privacy_and_metrics():
    with client() as http:
        response = http.post("/v1/chat/completions", json=request_body())
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion" and data["id"].startswith("chatcmpl-")
        assert data["choices"][0]["message"]["content"].startswith("This is a scripted preview")
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]
        assert data["breeze"]["mode"] == "demo" and data["breeze"]["usage_is_estimate"]
        assert data["breeze"]["decode_tokens_per_second"] is None
        assert data["breeze"]["time_to_first_token_seconds"] >= 0
        status = http.get("/api/status").json()
        assert status["requests"]["completed"] == 1
        assert "Summarize my notes" not in json.dumps(status)
        assert status["queue"]["active"] == status["queue"]["waiting"] == 0
        assert "breeze_completed_total 1" in http.get("/metrics").text
        assert response.headers["x-request-id"]
        assert response.headers["cache-control"] == "no-store"
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_streaming_matches_json_and_has_usage():
    with client() as http:
        reference = http.post("/v1/chat/completions", json=request_body()).json()
        response = http.post("/v1/chat/completions", json=request_body(
            stream=True, stream_options={"include_usage": True}))
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-accel-buffering"] == "no"
        frames = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
        assert frames[-1] == "[DONE]"
        chunks = [json.loads(frame) for frame in frames[:-1]]
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        actual = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks if chunk["choices"])
        assert actual == reference["choices"][0]["message"]["content"]
        assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
        assert chunks[-1]["usage"] == reference["usage"]


def test_model_output_and_context_limits():
    with client(settings(max_seq=128, chunk_size=16, max_output_tokens=32)) as http:
        assert http.post("/v1/chat/completions", json=request_body(model="other", max_tokens=16)).status_code == 404
        assert http.post("/v1/chat/completions", json=request_body(max_tokens=33)).status_code == 400
        response = http.post("/v1/chat/completions", json=request_body(
            max_tokens=16, messages=[{"role": "user", "content": "word " * 130}]))
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "context_length_exceeded"
        assert http.get("/readyz").status_code == 200


def test_worker_ownership_queue_capacity_and_reset():
    class RecordingBackend(DemoBackend):
        def __init__(self, config):
            super().__init__(config)
            self.threads, self.calls = [], []
            self.release = threading.Event()

        def load(self):
            self.threads.append(threading.get_ident())

        def prepare(self, request):
            self.threads.append(threading.get_ident())
            self.calls.append(request.messages[-1].content)
            return super().prepare(request)

        def generate(self, request, prepared, emit, cancelled, deadline):
            self.threads.append(threading.get_ident())
            self.release.wait(timeout=2)
            return super().generate(request, prepared, emit, cancelled, deadline)

        def close(self):
            self.threads.append(threading.get_ident())

    async def scenario():
        service = InferenceService(settings(max_pending=2), RecordingBackend)
        await service.start()
        try:
            first = service.submit(ChatRequest(**request_body()))
            await first.ready
            second = service.submit(ChatRequest(**request_body(messages=[{"role": "user", "content": "New context"}])))
            with pytest.raises(ServiceError) as error:
                service.submit(ChatRequest(**request_body()))
            assert error.value.status == 429
            assert service.status()["queue"] == {"active": 1, "waiting": 1, "capacity": 2}
            service.backend.release.set()
            assert isinstance(await first.result, Completion)
            assert isinstance(await second.result, Completion)
            assert service.backend.calls == ["Summarize my notes.", "New context"]
        finally:
            await service.close()
        assert len(set(service.backend.threads)) == 1
        assert service.backend.threads[0] != threading.get_ident()

    asyncio.run(scenario())


def test_cancel_releases_backpressure_and_allows_next_request():
    async def scenario():
        service = InferenceService(settings())
        await service.start()
        try:
            job = service.submit(ChatRequest(**request_body(stream=True)))
            await job.ready
            for _ in range(1000):
                if job.events.full():
                    break
                await asyncio.sleep(0.001)
            assert job.events.qsize() <= 32 and job.events.full()
            job.cancel()
            outcome = await asyncio.wait_for(job.result, 2)
            assert isinstance(outcome, ServiceError) and outcome.code == "cancelled"
            following = service.submit(ChatRequest(**request_body()))
            assert isinstance(await following.result, Completion)
            assert service.stats["cancelled"] == service.stats["completed"] == 1
        finally:
            await service.close()

    asyncio.run(scenario())


def test_runtime_failure_fails_closed_without_leaking_exception(caplog):
    class BrokenBackend(DemoBackend):
        def generate(self, *args):
            raise RuntimeError("secret-input-must-not-leak")

    with client(factory=BrokenBackend) as http:
        response = http.post("/v1/chat/completions", json=request_body())
        assert response.status_code == 503
        assert "secret-input-must-not-leak" not in response.text + caplog.text
        assert http.get("/readyz").status_code == 503
        assert http.post("/v1/chat/completions", json=request_body()).status_code == 503


def test_streaming_error_uses_error_frame():
    class BrokenBackend(DemoBackend):
        def generate(self, *args):
            raise ServiceError("Soft deadline exceeded", 504, "deadline_exceeded")

    with client(factory=BrokenBackend) as http:
        response = http.post("/v1/chat/completions", json=request_body(stream=True))
        assert response.status_code == 200
        assert '"code": "deadline_exceeded"' in response.text
        assert response.text.endswith("data: [DONE]\n\n")


def test_cancellation_and_deadline():
    cancelled = threading.Event()
    with pytest.raises(ServiceError) as error:
        check_cancelled(cancelled, time.monotonic() - 1)
    assert error.value.status == 504
    cancelled.set()
    with pytest.raises(ServiceError) as error:
        check_cancelled(cancelled, time.monotonic() + 10)
    assert error.value.code == "cancelled"


@pytest.mark.parametrize("expired", [False, True])
def test_queued_cancellation_or_deadline_frees_slot_without_overlapping_native_calls(expired):
    class BlockingBackend(DemoBackend):
        def __init__(self, config):
            super().__init__(config)
            self.release = threading.Event()
            self.generations = 0

        def generate(self, request, prepared, emit, cancelled, deadline):
            self.generations += 1
            self.release.wait(timeout=2)
            return super().generate(request, prepared, emit, cancelled, deadline)

    async def scenario():
        service = InferenceService(settings(max_pending=2), BlockingBackend)
        await service.start()
        try:
            first = service.submit(ChatRequest(**request_body()))
            await first.ready
            waiting = service.submit(ChatRequest(**request_body()))
            if expired:
                waiting.deadline = time.monotonic() - 1
            else:
                waiting.cancel()
            outcome = await asyncio.wait_for(waiting.result, 1)
            assert outcome.code == ("deadline_exceeded" if expired else "cancelled")
            assert len(service.jobs) == 1 and service.active == first.id
            assert not first.result.done()
            replacement = service.submit(ChatRequest(**request_body()))
            service.backend.release.set()
            assert isinstance(await first.result, Completion)
            assert isinstance(await replacement.result, Completion)
            assert service.backend.generations == 2
        finally:
            service.backend.release.set()
            await service.close()

    asyncio.run(scenario())


def test_shutdown_cancels_backpressured_stream_before_closing_backend():
    class ClosingBackend(DemoBackend):
        def __init__(self, config):
            super().__init__(config)
            self.closed = False

        def close(self):
            self.closed = True

    async def scenario():
        service = InferenceService(settings(), ClosingBackend)
        await service.start()
        job = service.submit(ChatRequest(**request_body(stream=True)))
        await job.ready
        await asyncio.wait_for(service.close(), 2)
        assert service.backend.closed and service.executor is None
        assert not service.ready and not service.jobs
        assert isinstance(job.result.result(), ServiceError)

    asyncio.run(scenario())


def test_static_workspace_and_schema_are_local():
    with client() as http:
        response = http.get("/")
        assert response.status_code == 200
        assert "/assets/app.js" in response.text
        assert http.get("/assets/app.js").status_code == 200
        assert http.get("/assets/app.css").status_code == 200
        assert http.get("/assets/../service_config.py").status_code == 404
        schema = http.get("/api/schema").json()
        assert "/v1/chat/completions" in schema["paths"]


def test_native_generation_resets_and_avoids_extra_forward():
    import numpy as np
    from types import SimpleNamespace

    class FakeModel:
        def __init__(self):
            self.calls = []

        def embed(self, ids):
            return np.array(ids)[:, None]

        def run(self, embeddings, past_len=0):
            self.calls.append((len(embeddings), past_len))
            last = np.zeros((len(embeddings), 3), dtype=np.float32)
            last[-1, 1 if past_len < 3 else 2] = 1
            return last

    backend = NativeBackend(ServiceSettings(model="unused", max_seq=128, chunk_size=2, max_output_tokens=16))
    backend.model = FakeModel()
    backend.eos = {2}
    backend.tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "A" * ids.count(1))
    request = ChatRequest(**request_body(max_tokens=8))
    emitted = []
    for _ in range(2):
        result = backend.generate(request, [5, 6, 7], emitted.append,
                                  threading.Event(), time.monotonic() + 10)
        assert result.text == "A" and result.completion_tokens == 2 and result.decode_steps == 1
        assert result.finish_reason == "stop"
    assert backend.model.calls == [(2, 0), (1, 2), (1, 3)] * 2
    assert emitted == ["A", "A"]


def test_native_partial_unicode_is_buffered_and_output_limit_does_not_forward_again():
    import numpy as np
    from types import SimpleNamespace

    calls = []

    def run(embeddings, past_len=0):
        calls.append(past_len)
        return np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

    backend = NativeBackend(ServiceSettings(model="unused", max_seq=128, chunk_size=16, max_output_tokens=16))
    backend.model = SimpleNamespace(embed=lambda ids: ids, run=run)
    backend.eos = {2}
    backend.tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "\ufffd" if len(ids) == 1 else "é")
    emitted = []
    result = backend.generate(ChatRequest(**request_body(max_tokens=2)), [7, 8], emitted.append,
                              threading.Event(), time.monotonic() + 10)
    assert result.text == "é" and emitted == ["é"]
    assert result.finish_reason == "length" and result.completion_tokens == 2
    assert calls == [0, 2]


def test_load_failure_closes_on_owner_thread_without_starting_service():
    class FailedLoad(DemoBackend):
        def __init__(self, config):
            super().__init__(config)
            self.thread_ids = []

        def load(self):
            self.thread_ids.append(threading.get_ident())
            raise ValueError("invalid checkpoint")

        def close(self):
            self.thread_ids.append(threading.get_ident())

    async def scenario():
        service = InferenceService(settings(), FailedLoad)
        with pytest.raises(ValueError, match="invalid checkpoint"):
            await service.start()
        assert not service.ready and service.executor is None
        assert len(service.backend.thread_ids) == 2
        assert len(set(service.backend.thread_ids)) == 1

    asyncio.run(scenario())
