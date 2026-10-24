"""Connection-scoped transport backpressure policy stacks.

This module adds a *policy stack* on top of the existing transport flow
control machinery (``HIGH_WATER_LIMIT`` / :class:`FlowControl`):

* a stack is owned by exactly one TCP connection and lives for as long as the
  HTTP protocol (or the WebSocket connection an HTTP upgrade produced) lives;
* application code (typically middleware) can :func:`push_policy` a frame on
  the stack, tighten a knob temporarily, and :func:`pop_policy` it again - the
  :class:`PolicyScope` context manager pops even when the guarded block exits
  through an exception;
* protocol read paths consult the *effective* view of the stack merged with the
  process-wide :data:`~uvicorn.protocols.http.flow_control.HIGH_WATER_LIMIT`
  baseline;
* the stack is mounted through a :class:`contextvars.ContextVar`, so two
  concurrent connections may push different policies without ever sharing a
  frame;
* on HTTP/2 a frame may additionally be pushed *inside a single stream*; the
  shared transport never applies one stream's inner frame to a sibling stream.

The stack deliberately does **not** replace :class:`FlowControl`, the
WebSocket ``writable`` events, ASGI ``scope["state"]``, the ASGI context reset
strategy (``Config.reset_contextvars``), or the ``limit_concurrency`` 503
path. Deployments that never push a frame keep the historical behavior.
"""

from __future__ import annotations

import contextvars
import sys
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

from uvicorn.protocols.http.flow_control import HIGH_WATER_LIMIT

if TYPE_CHECKING:
    from asyncio import AbstractEventLoop

#: Resume reading as soon as the current request/response cycle (HTTP/1.x) or
#: the current HTTP/2 stream is drained, as HTTP/1.x always did.
CYCLE_DONE = "cycle_done"

#: Resume reading only once every other still-alive HTTP/2 stream on the same
#: connection is below its own effective inbound high-water mark. Mirrors the
#: historical ``resume_reading_if_idle`` behavior.
SIBLINGS_CLEAR = "siblings_clear"

ResumeGate: TypeAlias = Literal["cycle_done", "siblings_clear"]

_RESUME_GATES = (CYCLE_DONE, SIBLINGS_CLEAR)


@dataclass(frozen=True, slots=True)
class BackpressurePolicy:
    """One frame of a backpressure policy stack.

    Every knob is tri-state: ``None`` means "do not override whatever an outer
    frame (or the baseline) says", so frames pushed from generic middleware
    compose with frames pushed closer to the application.
    """

    inbound_high_water: int | None = None
    write_await: bool | None = None
    resume_gate: ResumeGate | None = None

    def __post_init__(self) -> None:
        if self.inbound_high_water is not None and (
            not isinstance(self.inbound_high_water, int)
            or isinstance(self.inbound_high_water, bool)
            or self.inbound_high_water <= 0
        ):
            raise ValueError("inbound_high_water must be a positive integer or None")
        if self.resume_gate is not None and self.resume_gate not in _RESUME_GATES:
            raise ValueError(f"resume_gate must be one of {_RESUME_GATES!r}, not {self.resume_gate!r}")


@dataclass(frozen=True, slots=True)
class EffectivePolicy:
    """The fully merged view protocols actually act on.

    ``resume_gate`` stays nullable on purpose: leaving it unset lets every
    protocol family apply its own historical default (HTTP/1.x resumes with
    the cycle, HTTP/2 waits for sibling streams).
    """

    inbound_high_water: int
    write_await: bool
    resume_gate: ResumeGate | None


#: Behavior of a connection that never had a frame pushed, identical to the
#: pre-stack machinery: module-level inbound limit, ASGI sends wait for the
#: write side, resume-gate falls back to each protocol family's habit.
BASELINE_POLICY = EffectivePolicy(
    inbound_high_water=HIGH_WATER_LIMIT,
    write_await=True,
    resume_gate=None,
)


def _merge(
    connection_frames: list[BackpressurePolicy],
    stream_frames: list[BackpressurePolicy] | None = None,
) -> EffectivePolicy:
    """Merge frames bottom-to-top; inner stream frames sit above connection frames."""

    inbound_high_water = HIGH_WATER_LIMIT
    write_await = True
    resume_gate: ResumeGate | None = None
    for frame in connection_frames:
        if frame.inbound_high_water is not None:
            inbound_high_water = frame.inbound_high_water
        if frame.write_await is not None:
            write_await = frame.write_await
        if frame.resume_gate is not None:
            resume_gate = frame.resume_gate
    for frame in stream_frames or ():
        if frame.inbound_high_water is not None:
            inbound_high_water = frame.inbound_high_water
        if frame.write_await is not None:
            write_await = frame.write_await
        if frame.resume_gate is not None:
            resume_gate = frame.resume_gate
    return EffectivePolicy(inbound_high_water, write_await, resume_gate)


class BackpressureStack:
    """The frame list owned by a single TCP connection.

    Protocol instances hold one of these. User code never manipulates the
    stack directly - it goes through the contextvar-bound public API - but
    protocols read :meth:`view` from their transport callbacks.
    """

    __slots__ = ("_frames",)

    def __init__(self) -> None:
        self._frames: list[BackpressurePolicy] = []

    def push(self, frame: BackpressurePolicy) -> None:
        self._frames.append(frame)

    def pop(self, frame: BackpressurePolicy) -> None:
        try:
            self._frames.remove(frame)
        except ValueError:
            raise RuntimeError("Cannot pop a backpressure policy frame that is not on this stack") from None

    def view(self, stream_frames: list[BackpressurePolicy] | None = None) -> EffectivePolicy:
        """Effective view: connection frames, optional inner stream frames, baseline."""

        return _merge(self._frames, stream_frames)

    def snapshot(self, stream_frames: list[BackpressurePolicy] | None = None) -> EffectivePolicy:
        """Immutable view captured at a fixed point (an HTTP -> WebSocket upgrade)."""

        return self.view(stream_frames)


@dataclass(slots=True)
class _Binding:
    """Where pushes from the current task land."""

    connection: BackpressureStack
    stream: list[BackpressurePolicy] | None = None


_current_binding: contextvars.ContextVar[_Binding] = contextvars.ContextVar("uvicorn_backpressure_binding")


@dataclass(slots=True)
class _PolicyToken:
    binding: _Binding
    frame: BackpressurePolicy
    target: list[BackpressurePolicy]
    stream: bool


def _current_binding_required() -> _Binding:
    binding = _current_binding.get(None)
    if binding is None:
        raise RuntimeError(
            "No connection backpressure stack is bound here. Push a backpressure policy from "
            "inside the ASGI callable of an HTTP request/stream; the lifespan scope, the "
            "limit_concurrency 503 path and connections before/after an HTTP upgrade have no stack."
        )
    return binding


def push_policy(policy: BackpressurePolicy, *, stream: bool = False) -> _PolicyToken:
    """Push a frame onto the current connection's (or stream's) stack.

    ``stream=True`` is only valid for an HTTP/2 stream task; it pushes an
    inner frame that is invisible to sibling streams. Pair every push with a
    :func:`pop_policy`, or prefer :class:`PolicyScope`.
    """

    binding = _current_binding_required()
    if stream:
        if binding.stream is None:
            raise RuntimeError("Stream-scoped backpressure policies require an HTTP/2 stream context")
        target = binding.stream
    else:
        target = binding.connection._frames
    target.append(policy)
    return _PolicyToken(binding=binding, frame=policy, target=target, stream=stream)


def pop_policy(token: _PolicyToken) -> None:
    """Pop the frame recorded by ``token``; safe on exception unwinding."""

    try:
        token.target.remove(token.frame)
    except ValueError:
        raise RuntimeError("Backpressure policy frame was already removed from its stack") from None


class PolicyScope:
    """Context manager that pushes on entry and pops on exit.

    Works as both a synchronous and an asynchronous context manager, so it
    nests naturally inside middleware and ASGI helper coroutines. Exiting
    through an exception still pops the frame - this is the entry point that
    replaces the fragile "set a global, restore in ``finally``" pattern.
    """

    __slots__ = ("_policy", "_stream", "_token")

    def __init__(self, policy: BackpressurePolicy, *, stream: bool = False) -> None:
        self._policy = policy
        self._stream = stream
        self._token: _PolicyToken | None = None

    def __enter__(self) -> PolicyScope:
        self._token = push_policy(self._policy, stream=self._stream)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        assert self._token is not None
        pop_policy(self._token)

    async def __aenter__(self) -> PolicyScope:
        return self.__enter__()

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        assert self._token is not None
        pop_policy(self._token)


def current_policy() -> EffectivePolicy:
    """Effective policy view visible at the call site; baseline when unbound."""

    binding = _current_binding.get(None)
    if binding is None:
        return BASELINE_POLICY
    return binding.connection.view(binding.stream)


async def _run_bound(coroutine: Coroutine[object, object, object], binding: _Binding) -> object:
    """Run ``coroutine`` with ``binding`` installed in its context."""

    token = _current_binding.set(binding)
    try:
        return await coroutine
    finally:
        _current_binding.reset(token)


def start_bound_task(
    loop: AbstractEventLoop,
    coroutine: Coroutine[object, object, object],
    stack: BackpressureStack,
    *,
    reset_contextvars: bool,
    stream_frames: list[BackpressurePolicy] | None = None,
) -> object:
    """Schedule the ASGI cycle task with the connection stack bound.

    Honors ``Config.reset_contextvars`` independently of the binding: with the
    opt-in reset the task runs in a fresh context that still carries only the
    backpressure binding; without it the task copies the ambient context and
    installs the binding on task entry (works on loops whose ``create_task``
    does not support the ``context`` argument).
    """

    binding = _Binding(connection=stack, stream=stream_frames)
    if reset_contextvars:
        # Opt-in workaround for contextvar leaking: start from an empty
        # context, but pre-bind the connection stack inside that context.
        ctx = contextvars.Context()
        ctx.run(_current_binding.set, binding)
        if sys.version_info >= (3, 11):  # pragma: py-lt-311
            return loop.create_task(coroutine, context=ctx)  # type: ignore[arg-type]
        return ctx.run(loop.create_task, coroutine)  # type: ignore[no-any-return]  # pragma: py-gte-311
    return loop.create_task(_run_bound(coroutine, binding))
