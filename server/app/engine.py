"""Continuous-batching scheduler running on a simulated GPU.

The scheduler follows the vLLM/SGLang design: a FIFO waiting queue, a running batch that is re-formed at *every*
decode step (so new requests join without waiting for the batch to drain), admission control against KV-cache
capacity, and prefix reuse through ``KVCacheManager``.

The "GPU" is a cost model (see ``Settings.sim_*``):
    step_time = decode_base + decode_per_seq * running_sequences + prefill_per_token * uncached_prompt_tokens
Decode is memory-bound, so ``decode_base`` dominates and adding sequences to a batch is nearly free: that is what
makes batching raise throughput. Prefix-cache hits shrink the ``prefill`` term.
"""
from __future__ import annotations

import asyncio
import re
import time
import uuid
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional, Union

from .config import RuntimeConfig, Settings
from .kvcache import KVCacheManager

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_VOCAB = (
    "the model serves requests with low latency using batched decoding on a shared gpu while cache "
    "blocks keep prefixes warm and queues stay short under load"
).split()


def tokenize(text: str) -> list[int]:
    return [zlib.crc32(t.encode()) % 50_000 for t in _TOKEN_RE.findall(text)]


def detokenize(tokens) -> str:
    return " ".join(_VOCAB[t % len(_VOCAB)] for t in tokens)


class QueueFull(Exception):
    """Admission queue is full; the request was shed."""


class RequestTooLarge(Exception):
    """Prompt + max_tokens can never fit in the KV cache."""


class RequestTimeout(Exception):
    """Request exceeded request_timeout_s."""


@dataclass
class GenResult:
    id: str
    text: str
    tokens: list[int]
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    queue_ms: Optional[float]
    ttft_ms: Optional[float]
    total_ms: float


@dataclass(eq=False)
class SeqRequest:
    id: str
    tenant: str
    prompt_tokens: list[int]
    max_tokens: int
    arrival: float
    future: asyncio.Future
    stream: Optional[asyncio.Queue]
    seed: int
    hashes: list[int] = field(default_factory=list)
    matched: list[int] = field(default_factory=list)
    held: list[int] = field(default_factory=list)
    private_blocks: int = 0
    cached_tokens: int = 0
    admitted_at: Optional[float] = None
    first_token_at: Optional[float] = None
    out: list[int] = field(default_factory=list)
    cancelled: bool = False


class SimEngine:
    kind = "sim"

    def __init__(
        self,
        settings: Settings,
        cfg: RuntimeConfig,
        name: str = "primary",
        on_step: Optional[Callable[[int, int, float], None]] = None,
    ) -> None:
        self.settings = settings
        self.cfg = cfg
        self.name = name
        self.on_step = on_step
        self.kv = KVCacheManager(settings.kv_cache_blocks, settings.kv_block_size)
        self.waiting: deque[SeqRequest] = deque()
        self.running: list[SeqRequest] = []
        self._wakeup = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._busy: deque[tuple[float, float]] = deque()  # (end_time, duration) of recent steps
        self.tokens_generated = 0
        self.requests_finished = 0
        self.steps = 0
        self.batch_sum = 0

    # -- lifecycle -------------------------------------------------------------------------------------
    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name=f"engine-{self.name}")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        for r in list(self.running) + list(self.waiting):
            if not r.future.done():
                r.future.cancel()
            if r.stream:
                r.stream.put_nowait(None)

    def apply_config(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg
        self._wakeup.set()

    # -- public API ------------------------------------------------------------------------------------
    def submit(self, tenant: str, prompt: str, max_tokens: int, stream: bool = False) -> SeqRequest:
        tokens = tokenize(prompt) or [0]
        if self.kv.blocks_for(len(tokens) + max_tokens) > self.kv.num_blocks:
            raise RequestTooLarge(f"{len(tokens)} prompt + {max_tokens} output tokens exceed KV cache capacity")
        if len(self.waiting) >= self.cfg.max_queue:
            raise QueueFull(f"queue full ({self.cfg.max_queue})")
        req = SeqRequest(
            id=uuid.uuid4().hex[:16],
            tenant=tenant,
            prompt_tokens=tokens,
            max_tokens=max_tokens,
            arrival=time.perf_counter(),
            future=asyncio.get_running_loop().create_future(),
            stream=asyncio.Queue() if stream else None,
            seed=zlib.crc32(repr(tokens).encode()),
        )
        self.waiting.append(req)
        self._wakeup.set()
        return req

    async def generate(self, tenant: str, prompt: str, max_tokens: int) -> GenResult:
        req = self.submit(tenant, prompt, max_tokens)
        try:
            return await asyncio.wait_for(req.future, timeout=self.cfg.request_timeout_s)
        except asyncio.TimeoutError:
            req.cancelled = True
            raise RequestTimeout(f"timed out after {self.cfg.request_timeout_s}s") from None
        except asyncio.CancelledError:
            req.cancelled = True
            raise

    async def stream_generate(
        self, tenant: str, prompt: str, max_tokens: int
    ) -> AsyncIterator[Union[str, GenResult]]:
        """Yields text pieces, then a final ``GenResult``."""
        req = self.submit(tenant, prompt, max_tokens, stream=True)
        try:
            while True:
                tok = await asyncio.wait_for(req.stream.get(), timeout=self.cfg.request_timeout_s)
                if tok is None:
                    break
                yield detokenize([tok])
            if req.future.cancelled():
                raise asyncio.CancelledError()
            yield req.future.result()
        except asyncio.TimeoutError:
            raise RequestTimeout(f"timed out after {self.cfg.request_timeout_s}s") from None
        finally:
            if not req.future.done():
                req.cancelled = True  # consumer went away or timed out: free the sequence

    def stats(self) -> dict:
        now = time.perf_counter()
        while self._busy and self._busy[0][0] < now - 1.0:
            self._busy.popleft()
        busy = sum(d for _, d in self._busy)
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "kv_usage": self.kv.usage(),
            "gpu_util": min(100.0, busy * 100.0),
            "prefix_hit_tokens": self.kv.hit_tokens,
            "prefix_lookup_tokens": self.kv.lookup_tokens,
            "tokens_generated": self.tokens_generated,
            "requests_finished": self.requests_finished,
            "steps": self.steps,
            "batch_sum": self.batch_sum,
        }

    # -- scheduler -------------------------------------------------------------------------------------
    @staticmethod
    def _dead(r: SeqRequest) -> bool:
        # The future is only resolved by _finish(); if it is done while the request is still queued/running,
        # the caller timed out or disconnected.
        return r.cancelled or r.future.done()

    def _reap(self) -> None:
        if any(self._dead(r) for r in self.running):
            alive = []
            for r in self.running:
                if self._dead(r):
                    self.kv.release(r.held, r.private_blocks)
                    r.held, r.private_blocks = [], 0
                    if not r.future.done():
                        r.future.cancel()
                    if r.stream:
                        r.stream.put_nowait(None)
                else:
                    alive.append(r)
            self.running = alive
        if any(self._dead(r) for r in self.waiting):
            self.waiting = deque(r for r in self.waiting if not self._dead(r))

    def _admit(self, now: float) -> list[SeqRequest]:
        cfg = self.cfg
        limit = cfg.max_batch_size if cfg.enable_batching else 1
        budget = cfg.max_batch_tokens
        bs = self.kv.block_size
        admitted: list[SeqRequest] = []
        while self.waiting and len(self.running) < limit:
            r = self.waiting[0]
            hashes = self.kv.block_hashes(r.prompt_tokens) if cfg.enable_prefix_cache else []
            matched = self.kv.match(hashes)
            if matched and len(matched) * bs >= len(r.prompt_tokens):
                matched = matched[:-1]  # a fully cached prompt still needs a forward pass for its last block
            cached = len(matched) * bs
            private = self.kv.blocks_for(len(r.prompt_tokens) + r.max_tokens) - len(matched)
            new_tokens = len(r.prompt_tokens) - cached
            if not self.kv.can_admit(private, matched):
                break
            if admitted and new_tokens > budget:
                break
            self.waiting.popleft()
            self.kv.admit(private, matched, len(r.prompt_tokens))
            r.hashes, r.matched, r.held = hashes, matched, list(matched)
            r.private_blocks, r.cached_tokens, r.admitted_at = private, cached, now
            budget -= new_tokens
            self.running.append(r)
            admitted.append(r)
        return admitted

    def _step_seconds(self, n_running: int, prefill_tokens: int) -> float:
        s = self.settings
        ms = s.sim_decode_base_ms + s.sim_decode_per_seq_ms * n_running + s.sim_prefill_ms_per_token * prefill_tokens
        return ms / 1000.0 / s.sim_speed

    def _next_token(self, r: SeqRequest) -> int:
        return zlib.crc32(f"{r.seed}:{len(r.out)}".encode()) % 50_000

    def _finish(self, r: SeqRequest, now: float) -> None:
        self.kv.release(r.held, r.private_blocks)
        r.held, r.private_blocks = [], 0
        ttft = (r.first_token_at - r.arrival) * 1000 if r.first_token_at else None
        queue = (r.admitted_at - r.arrival) * 1000 if r.admitted_at else None
        res = GenResult(
            id=r.id,
            text=detokenize(r.out),
            tokens=r.out,
            prompt_tokens=len(r.prompt_tokens),
            completion_tokens=len(r.out),
            cached_tokens=r.cached_tokens,
            queue_ms=queue,
            ttft_ms=ttft,
            total_ms=(now - r.arrival) * 1000,
        )
        if not r.future.done():
            r.future.set_result(res)
        if r.stream:
            r.stream.put_nowait(None)
        self.requests_finished += 1

    async def _loop(self) -> None:
        while True:
            if not self.waiting and not self.running:
                self._wakeup.clear()
                await self._wakeup.wait()
                if self.cfg.enable_batching and self.cfg.max_wait_ms > 0:
                    await asyncio.sleep(self.cfg.max_wait_ms / 1000.0)  # let a batch form
            self._reap()
            admitted = self._admit(time.perf_counter())
            if not self.running:
                if self.waiting:  # cannot happen (submit() rejects oversize), but never spin
                    r = self.waiting.popleft()
                    if not r.future.done():
                        r.future.set_exception(RequestTooLarge("cannot be scheduled"))
                await asyncio.sleep(0)
                continue

            prefill_tokens = sum(len(r.prompt_tokens) - r.cached_tokens for r in admitted)
            n = len(self.running)
            step_s = self._step_seconds(n, prefill_tokens)
            await asyncio.sleep(step_s)
            now = time.perf_counter()
            self._busy.append((now, step_s))
            self.steps += 1
            self.batch_sum += n
            if self.on_step:
                self.on_step(n, prefill_tokens, step_s)

            for r in admitted:  # prefill done: its full prompt blocks become shareable
                if not self._dead(r):
                    new = r.hashes[len(r.matched):]
                    r.held += self.kv.commit_prefill(new)
                    r.private_blocks -= len(new)

            done: list[SeqRequest] = []
            for r in self.running:
                if self._dead(r):
                    continue
                tok = self._next_token(r)
                r.out.append(tok)
                self.tokens_generated += 1
                if r.first_token_at is None:
                    r.first_token_at = now
                if r.stream:
                    r.stream.put_nowait(tok)
                if len(r.out) >= r.max_tokens:
                    done.append(r)
            if done:
                finished = {id(r) for r in done}
                self.running = [r for r in self.running if id(r) not in finished]
                for r in done:
                    self._finish(r, now)
            await asyncio.sleep(0)
