"""Optional HTTP service. Importing the model library does not import this module."""
import asyncio
from contextlib import asynccontextmanager
import hmac
import json
from pathlib import Path
import time
from urllib.parse import urlsplit
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers

from .service_backend import Completion
from .service_runtime import InferenceService
from .service_schemas import ChatRequest, ServiceError


WEB = Path(__file__).with_name("web")
SECURITY_HEADERS = {
    "cache-control": "no-store",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "permissions-policy": "camera=(), microphone=(), geolocation=()",
    "content-security-policy": "default-src 'none'; script-src 'self'; style-src 'self'; "
                               "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
                               "base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
}


class RequestBoundary:
    """Check host/origin, bearer credentials and body size before application parsing."""
    def __init__(self, app, settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        async def secure_send(message):
            if message["type"] == "http.response.start":
                headers = [(key, value) for key, value in message.get("headers", [])
                           if key.decode("latin1").lower() not in SECURITY_HEADERS]
                headers += [(name.encode(), value.encode()) for name, value in SECURITY_HEADERS.items()]
                headers.append((b"x-request-id", request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        async def reject(error):
            headers = {"WWW-Authenticate": "Bearer"} if error.status == 401 else {}
            return await JSONResponse(error.payload(request_id), error.status, headers=headers)(scope, receive, secure_send)

        headers = Headers(scope=scope)
        try:
            host = urlsplit("//" + headers.get("host", ""))
            host_port = host.port  # validate the port as well as the hostname
            if host.username or host.password or host.path or host.query or host.fragment:
                raise ValueError
            if host.hostname is None or host.hostname.lower() not in {
                    name.lower() for name in self.settings.allowed_hosts}:
                raise ValueError
        except ValueError:
            return await reject(ServiceError("Host is not allowed", 400, "invalid_host"))
        protected = scope["path"].startswith(("/v1/", "/api/")) or scope["path"] == "/metrics"
        if protected:
            if self.settings.api_key:
                scheme, _, token = headers.get("authorization", "").partition(" ")
                if scheme.lower() != "bearer" or not hmac.compare_digest(
                        token.encode("utf-8"), self.settings.api_key.encode("ascii")):
                    return await reject(ServiceError("A valid bearer API key is required", 401,
                                                     "invalid_api_key", "authentication_error"))
            if headers.get("origin"):
                try:
                    origin = urlsplit(headers["origin"])
                    scheme = scope.get("scheme", "http")
                    if (origin.scheme not in {"http", "https"} or origin.username or origin.password
                            or origin.path not in {"", "/"} or origin.query or origin.fragment
                            or origin.scheme != scheme or origin.hostname != host.hostname
                            or (origin.port or (443 if origin.scheme == "https" else 80))
                            != (host_port or (443 if scheme == "https" else 80))):
                        raise ValueError
                except ValueError:
                    return await reject(ServiceError("Cross-origin API requests are not allowed", 403, "invalid_origin"))
        if scope["method"] == "POST" and scope["path"].rstrip("/") == "/v1/chat/completions":
            if headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
                return await reject(ServiceError("Use Content-Type: application/json", 415, "unsupported_media_type"))
            try:
                length = int(headers.get("content-length", "0"))
                if length < 0:
                    raise ValueError
            except ValueError:
                return await reject(ServiceError("Invalid Content-Length", 400, "invalid_content_length"))
            if length > self.settings.max_body_bytes:
                return await reject(ServiceError("Request body is too large", 413, "body_too_large"))
            body = bytearray()
            try:
                async with asyncio.timeout(10):
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            return
                        body.extend(message.get("body", b""))
                        if len(body) > self.settings.max_body_bytes:
                            return await reject(ServiceError("Request body is too large", 413, "body_too_large"))
                        if not message.get("more_body", False):
                            break
            except TimeoutError:
                return await reject(ServiceError("Request body took too long to receive", 408, "body_timeout"))
            replayed = False

            async def replay():
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            return await self.app(scope, replay, secure_send)
        return await self.app(scope, receive, secure_send)


async def wait_for_client(future, request, job):
    while not future.done():
        if await request.is_disconnected():
            job.cancel()
            raise ServiceError("Client disconnected", 499, "cancelled", "request_cancelled")
        try:
            await asyncio.wait_for(asyncio.shield(future), 0.2)
        except asyncio.TimeoutError:
            pass
    outcome = future.result()
    if isinstance(outcome, ServiceError):
        raise outcome
    return outcome


def create_app(settings, backend_factory=None):
    service = InferenceService(settings, backend_factory)

    @asynccontextmanager
    async def lifespan(app):
        await service.start()
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(title="Breeze local inference", version="0.2.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.service = service
    app.add_middleware(RequestBoundary, settings=settings)

    @app.exception_handler(ServiceError)
    async def service_error(request, error):
        headers = {"Retry-After": "1"} if error.status == 429 else None
        return JSONResponse(error.payload(request.state.request_id), error.status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        # Validation responses never echo submitted message content or credentials.
        problem = ServiceError("Invalid request. Use text messages, a final user turn, "
                               "and the documented greedy-only options.", code="validation_error")
        return JSONResponse(problem.payload(request.state.request_id), 400)

    @app.get("/healthz", include_in_schema=False)
    async def health():
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readiness():
        return JSONResponse({"ready": service.ready}, 200 if service.ready else 503)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": settings.model_id, "object": "model",
                                           "created": 0, "owned_by": "local"}]}

    @app.get("/api/status")
    async def status():
        return service.status()

    @app.get("/api/schema")
    async def schema():
        return app.openapi()

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return PlainTextResponse(service.metrics(), media_type="text/plain; version=0.0.4")

    @app.post("/v1/chat/completions")
    async def chat(body: ChatRequest, request: Request):
        job = service.submit(body, request.state.request_id)
        streaming = False
        created = int(time.time())
        base = {"id": job.id, "created": created, "model": settings.model_id}
        try:
            await wait_for_client(job.ready, request, job)
            if not body.stream:
                completion = await wait_for_client(job.result, request, job)
                return {**base, "object": "chat.completion", "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": completion.text},
                     "finish_reason": completion.finish_reason}], "usage": completion.usage,
                    "breeze": job.timings(completion, settings.demo)}

            def chunk(delta=None, finish=None, usage=None, timings=None):
                payload = {**base, "object": "chat.completion.chunk", "choices": [] if usage else [
                    {"index": 0, "delta": delta or {}, "finish_reason": finish}]}
                if usage:
                    payload["usage"] = usage
                if timings:
                    payload["breeze"] = timings
                return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

            async def events():
                try:
                    yield chunk({"role": "assistant", "content": ""})
                    while True:
                        try:
                            event = await asyncio.wait_for(service.next_event(job), timeout=10.0)
                        except asyncio.TimeoutError:
                            yield ": keep-alive\n\n"
                            continue
                        if isinstance(event, str):
                            yield chunk({"content": event})
                        elif isinstance(event, Completion):
                            yield chunk(finish=event.finish_reason, timings=job.timings(event, settings.demo))
                            if body.stream_options and body.stream_options.include_usage:
                                yield chunk(usage=event.usage)
                            yield "data: [DONE]\n\n"
                            return
                        else:
                            yield "data: " + json.dumps(event.payload(request.state.request_id)) + "\n\n"
                            yield "data: [DONE]\n\n"
                            return
                finally:
                    job.cancel()

            streaming = True
            return StreamingResponse(events(), media_type="text/event-stream",
                                     headers={"X-Accel-Buffering": "no"})
        finally:
            if not streaming:
                job.cancel()

    @app.get("/", include_in_schema=False)
    async def playground():
        return FileResponse(WEB / "index.html")

    app.mount("/assets", StaticFiles(directory=WEB, check_dir=False), name="assets")
    return app