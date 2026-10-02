"""Bounded admission and cooperative cancellation around one native worker."""
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
import logging
import threading
import time
import uuid

from .service_backend import Completion, DemoBackend, NativeBackend, check_cancelled
from .service_schemas import ServiceError


LOG = logging.getLogger("breeze.service")


@dataclass
class Job:
    id: str
    request: object = field(repr=False)
    created: float
    deadline: float
    ready: asyncio.Future = field(repr=False)
    result: asyncio.Future = field(repr=False)
    events: asyncio.Queue = field(repr=False)
    cancelled: threading.Event = field(default_factory=threading.Event, repr=False)
    started: float | None = None
    finished: float | None = None
    first_token_seconds: float | None = None

    def cancel(self):
        self.cancelled.set()

    def timings(self, completion, demo=False):
        return {
            "mode": "demo" if demo else "native",
            "usage_is_estimate": demo,
            "queue_seconds": max(0.0, (self.started or self.created) - self.created),
            "total_seconds": max(0.0, (self.finished or time.monotonic()) - self.created),
            "time_to_first_token_seconds": self.first_token_seconds,
            "prefill_seconds": None if demo else completion.prefill_seconds,
            "decode_seconds": None if demo else completion.decode_seconds,
            "decode_steps": None if demo else completion.decode_steps,
            "decode_tokens_per_second": (completion.decode_steps / completion.decode_seconds
                                         if not demo and completion.decode_seconds > 0 else None),
        }


class InferenceService:
    def __init__(self, settings, backend_factory=None):
        self.settings = settings
        self.backend = (backend_factory or (DemoBackend if settings.demo else NativeBackend))(settings)
        self.executor = None
        self.loop = None
        self._serial = asyncio.Lock()
        self._tasks = set()
        self.jobs = {}
        self.active = None
        self.ready = False
        self.closing = False
        self.created = time.monotonic()
        self.stats = dict(submitted=0, completed=0, failed=0, cancelled=0, rejected=0,
                          prompt_tokens=0, completion_tokens=0)

    async def start(self):
        self.loop = asyncio.get_running_loop()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="breeze-inference")
        try:
            await self.loop.run_in_executor(self.executor, self.backend.load)
        except BaseException:
            await self.loop.run_in_executor(self.executor, self.backend.close)
            await asyncio.to_thread(self.executor.shutdown, wait=True)
            self.executor = None
            raise
        self.ready = True

    async def close(self):
        self.closing, self.ready = True, False
        for job in list(self.jobs.values()):
            job.cancel()
        # A native call cannot be interrupted; do not free its state while it runs.
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
        if self.executor is not None:
            await self.loop.run_in_executor(self.executor, self.backend.close)
            await asyncio.to_thread(self.executor.shutdown, wait=True)
            self.executor = None

    def submit(self, request, request_id=None):
        if not self.ready or self.closing:
            raise ServiceError("Inference service is unavailable", 503, "not_ready", "service_error")
        if request.model != self.settings.model_id:
            raise ServiceError("Unknown model; use the ID returned by /v1/models", 404, "model_not_found")
        if request.max_tokens > self.settings.max_output_tokens:
            raise ServiceError(f"max_tokens exceeds the service limit of {self.settings.max_output_tokens}",
                               code="output_limit_exceeded")
        if len(self.jobs) >= self.settings.max_pending:
            self.stats["rejected"] += 1
            raise ServiceError("Inference queue is full; retry later", 429, "queue_full", "rate_limit_error")
        created = time.monotonic()
        job = Job("chatcmpl-" + (request_id or uuid.uuid4().hex), request, created,
                  created + self.settings.request_timeout, self.loop.create_future(),
                  self.loop.create_future(), asyncio.Queue(maxsize=32))
        self.jobs[job.id] = job
        self.stats["submitted"] += 1
        task = asyncio.create_task(self._run(job))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    def _emit(self, job, text):
        check_cancelled(job.cancelled, job.deadline)
        if job.first_token_seconds is None:
            job.first_token_seconds = time.monotonic() - job.created
        if not job.request.stream:
            return
        pending = asyncio.run_coroutine_threadsafe(job.events.put(text), self.loop)
        try:
            while True:
                try:
                    pending.result(timeout=0.1)
                    return
                except FutureTimeout:
                    check_cancelled(job.cancelled, job.deadline)
        except BaseException:
            pending.cancel()
            raise

    async def _acquire(self, job):
        waiting = asyncio.create_task(self._serial.acquire())
        acquired = False
        try:
            while not waiting.done():
                check_cancelled(job.cancelled, job.deadline)
                await asyncio.wait((waiting,), timeout=min(0.1, max(0.0, job.deadline - time.monotonic())))
            await waiting
            acquired = True
        finally:
            if not acquired:
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)
                # The lock can become available at the cancellation boundary.
                if not waiting.cancelled() and waiting.exception() is None and waiting.result():
                    self._serial.release()

    async def _run(self, job):
        acquired = False
        outcome = None
        try:
            await self._acquire(job)
            acquired = True
            check_cancelled(job.cancelled, job.deadline)
            if not self.ready:
                raise ServiceError("Inference service is unavailable", 503, "not_ready", "service_error")
            self.active = job.id
            job.started = time.monotonic()
            prepared = await self.loop.run_in_executor(self.executor, self.backend.prepare, job.request)
            check_cancelled(job.cancelled, job.deadline)
            job.ready.set_result(len(prepared))
            outcome = await self.loop.run_in_executor(
                self.executor, self.backend.generate, job.request, prepared,
                lambda text: self._emit(job, text), job.cancelled, job.deadline)
            check_cancelled(job.cancelled, job.deadline)
        except ServiceError as error:
            outcome = error
        except Exception as error:
            # Do not record prompt text, response text, secrets, or exception messages.
            LOG.error("Inference failed: request_id=%s exception_type=%s", job.id, type(error).__name__)
            self.ready = False
            outcome = ServiceError("Inference failed; the service needs to be restarted", 503,
                                   "inference_failed", "service_error")
        finally:
            if acquired:
                self.active = None
                self._serial.release()
            job.finished = time.monotonic()
            self.jobs.pop(job.id, None)
            if isinstance(outcome, Completion):
                self.stats["completed"] += 1
                self.stats["prompt_tokens"] += outcome.prompt_tokens
                self.stats["completion_tokens"] += outcome.completion_tokens
            else:
                outcome = outcome or ServiceError("Request cancelled", 499, "cancelled", "request_cancelled")
                self.stats["cancelled" if outcome.code == "cancelled" else "failed"] += 1
            # Futures carry outcomes rather than exceptions, including for disconnected clients.
            if not job.ready.done():
                job.ready.set_result(outcome)
            if not job.result.done():
                job.result.set_result(outcome)

    async def next_event(self, job):
        """Return queued text first, then the final outcome; no unbounded buffering."""
        if not job.events.empty():
            return job.events.get_nowait()
        if job.result.done():
            return job.result.result()
        pending = asyncio.create_task(job.events.get())
        try:
            done, _ = await asyncio.wait((pending, job.result), return_when=asyncio.FIRST_COMPLETED)
            if pending in done:
                return pending.result()
            if not job.events.empty():
                return job.events.get_nowait()
            return job.result.result()
        finally:
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

    def status(self):
        return {
            "product": "Breeze", "status": "ready" if self.ready else "unavailable",
            "mode": "demo" if self.settings.demo else "native",
            "model": self.settings.model_id, "threads": self.settings.threads,
            "max_context_tokens": self.settings.max_seq, "max_output_tokens": self.settings.max_output_tokens,
            "queue": {"active": int(self.active is not None), "waiting": len(self.jobs) - int(self.active is not None),
                      "capacity": self.settings.max_pending},
            "request_timeout_seconds": self.settings.request_timeout,
            "authentication_required": self.settings.api_key is not None,
            "uptime_seconds": time.monotonic() - self.created,
            "requests": self.stats.copy(),
            "capabilities": {"streaming": True, "multi_turn": True, "sampling": "greedy",
                             "tools": False, "images": False, "prompt_persistence": False},
        }

    def metrics(self):
        values = {
            "breeze_ready": ("gauge", int(self.ready)),
            "breeze_active_requests": ("gauge", int(self.active is not None)),
            "breeze_waiting_requests": ("gauge", len(self.jobs) - int(self.active is not None)),
            **{f"breeze_{name}_total": ("counter", value) for name, value in self.stats.items()},
        }
        return "".join(f"# TYPE {name} {kind}\n{name} {value}\n" for name, (kind, value) in values.items())