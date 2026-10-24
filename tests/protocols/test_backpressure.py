"""Acceptance tests for the connection-level backpressure policy stack.

The tests are split per *exit* the requirement names:

* the policy stack itself (merge, push/pop, exception cleanup, binding)
* HTTP/1 inbound high-water and outbound drain, once per implementation
* httptools pipelining (scheduling untouched, stack only layers on top)
* HTTP/2 sibling-stream resume gate and per-stream inner frames
* the limit_concurrency 503 path and lifespan (not subscribed)
* HTTP -> WebSocket upgrade snapshot, one check per WebSocket implementation
* concurrent-connection isolation
"""

from __future__ import annotations

import asyncio
import contextvars
import importlib.util
import pathlib
import warnings
from typing import Any

import pytest
from websockets.client import ClientProtocol
from websockets.uri import parse_uri

from tests.protocols.test_http import (
    SIMPLE_GET_REQUEST,
    UPGRADE_REQUEST,
    MockTransport,
    get_connected_protocol as get_h1_protocol,
)
from tests.protocols.test_http2 import (
    H2Client,
    get_connected_protocol as get_h2_protocol,
)
from tests.protocols.test_websocket import MockWebSocketConnection, MockWriteTransport
from uvicorn.config import Config
from uvicorn.lifespan.on import LifespanOn
from uvicorn.protocols import backpressure as bp
from uvicorn.protocols.backpressure import (
    BASELINE_POLICY,
    CYCLE_DONE,
    SIBLINGS_CLEAR,
    BackpressurePolicy,
    BackpressureStack,
    EffectivePolicy,
    PolicyScope,
    current_policy,
    push_policy,
)
from uvicorn.protocols.http.flow_control import HIGH_WATER_LIMIT
from uvicorn.protocols.http.h11_impl import H11Protocol
from uvicorn.server import ServerState

try:
    from uvicorn.protocols.http.httptools_impl import HttpToolsProtocol
except ModuleNotFoundError:  # pragma: no cover
    HttpToolsProtocol = None  # type: ignore[assignment,misc]

try:
    from uvicorn.protocols.http.zttp_impl import ZttpProtocol
except ModuleNotFoundError:  # pragma: no cover
    ZttpProtocol = None  # type: ignore[assignment,misc]

try:
    import zttp as _zttp

    has_h2 = hasattr(_zttp, "HTTP2")
except ModuleNotFoundError:  # pragma: no cover
    has_h2 = False

pytestmark = pytest.mark.anyio

H1_PROTOCOLS = [p for p in (H11Protocol, HttpToolsProtocol, ZttpProtocol) if p is not None]
H1_IDS = [p.__module__.rsplit(".", 1)[-1] for p in H1_PROTOCOLS]

skip_without_h2 = pytest.mark.skipif(not has_h2, reason="zttp with HTTP/2 support is not installed")
skip_without_httptools = pytest.mark.skipif(HttpToolsProtocol is None, reason="httptools is not installed")

POST_HEADERS = b"\r\n".join([b"POST / HTTP/1.1", b"Host: example.org", b"Content-Length: 9", b"", b""])
POST_BODY = b"123456789"

RESPONSE_START = {
    "type": "http.response.start",
    "status": 200,
    "headers": [(b"content-length", b"2")],
}
RESPONSE_BODY = {"type": "http.response.body", "body": b"ok", "more_body": False}


def bind(stack: BackpressureStack, stream: list[BackpressurePolicy] | None = None) -> contextvars.Token[Any]:
    return bp._current_binding.set(bp._Binding(connection=stack, stream=stream))


# ---------------------------------------------------------------------------
# Exit 1: the policy stack itself
# ---------------------------------------------------------------------------


def test_empty_stack_is_baseline_unchanged() -> None:
    assert BackpressureStack().view() == BASELINE_POLICY
    assert BASELINE_POLICY == EffectivePolicy(HIGH_WATER_LIMIT, True, None)


def test_connection_stack_push_pop_round_trip() -> None:
    stack = BackpressureStack()
    frame = BackpressurePolicy(inbound_high_water=1)
    stack.push(frame)
    assert stack.view().inbound_high_water == 1
    stack.pop(frame)
    assert stack.view() == BASELINE_POLICY
    # Popping a frame the connection never held is an error, not a silent
    # reset to some global default.
    with pytest.raises(RuntimeError):
        stack.pop(BackpressurePolicy(inbound_high_water=2))


def test_merged_view_prefers_top_frame_and_restores_on_pop() -> None:
    stack = BackpressureStack()
    token = bind(stack)
    try:
        with PolicyScope(BackpressurePolicy(inbound_high_water=1024, write_await=False)):
            view = current_policy()
            assert view.inbound_high_water == 1024
            assert view.write_await is False
            # Unset fields fall through to the outer layer / baseline.
            assert view.resume_gate is None
            with PolicyScope(BackpressurePolicy(inbound_high_water=256)):
                inner = current_policy()
                assert inner.inbound_high_water == 256
                # Inner frame did not mention write_await: outer value stands.
                assert inner.write_await is False
            assert current_policy().inbound_high_water == 1024
        # Back to baseline - never to a process-global singleton.
        assert current_policy() == BASELINE_POLICY
    finally:
        bp._current_binding.reset(token)


def test_stream_inner_frames_compose_above_connection_stack() -> None:
    stack = BackpressureStack()
    stream_frames: list[BackpressurePolicy] = []
    token = bind(stack, stream_frames)
    try:
        stack.push(BackpressurePolicy(inbound_high_water=4096, resume_gate=SIBLINGS_CLEAR))
        stream_frames.append(BackpressurePolicy(inbound_high_water=512))
        view = stack.view(stream_frames)
        assert view.inbound_high_water == 512
        assert view.resume_gate == SIBLINGS_CLEAR
        # A sibling stream with no inner frame sees only the connection frame.
        assert stack.view().inbound_high_water == 4096
    finally:
        bp._current_binding.reset(token)


def test_scope_pops_even_on_exception() -> None:
    stack = BackpressureStack()
    token = bind(stack)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            with PolicyScope(BackpressurePolicy(inbound_high_water=1)):
                raise RuntimeError("boom")
        assert stack.view() == BASELINE_POLICY

        with pytest.raises(RuntimeError, match="boom-async"):
            with PolicyScope(BackpressurePolicy(write_await=False)):  # noqa: SIM117
                raise RuntimeError("boom-async")
        assert stack.view() == BASELINE_POLICY
    finally:
        bp._current_binding.reset(token)


async def test_async_scope_pops_even_on_exception() -> None:
    stack = BackpressureStack()
    token = bind(stack)
    try:
        with pytest.raises(RuntimeError, match="cancelled"):
            async with PolicyScope(BackpressurePolicy(inbound_high_water=7)):
                assert current_policy().inbound_high_water == 7
                raise RuntimeError("cancelled")
        assert stack.view() == BASELINE_POLICY
    finally:
        bp._current_binding.reset(token)


def test_push_without_binding_raises_instead_of_touching_a_global() -> None:
    # No binding installed in this test context.
    with pytest.raises(RuntimeError, match="No connection backpressure stack"):
        push_policy(BackpressurePolicy(inbound_high_water=1))
    assert current_policy() == BASELINE_POLICY


def test_stream_push_outside_http2_stream_raises() -> None:
    stack = BackpressureStack()
    token = bind(stack, stream=None)
    try:
        with pytest.raises(RuntimeError, match="HTTP/2 stream context"):
            push_policy(BackpressurePolicy(inbound_high_water=1), stream=True)
    finally:
        bp._current_binding.reset(token)


def test_double_pop_raises() -> None:
    stack = BackpressureStack()
    token = bind(stack)
    try:
        frame_token = push_policy(BackpressurePolicy(inbound_high_water=1))
        bp.pop_policy(frame_token)
        with pytest.raises(RuntimeError):
            bp.pop_policy(frame_token)
    finally:
        bp._current_binding.reset(token)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"inbound_high_water": 0},
        {"inbound_high_water": -1},
        {"inbound_high_water": True},
        {"resume_gate": "not-a-gate"},
    ],
)
def test_policy_frame_validates_its_fields(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        BackpressurePolicy(**kwargs)


# ---------------------------------------------------------------------------
# Exit 2: HTTP/1 inbound high-water (h11, httptools, zttp read the merged view)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("protocol_cls", H1_PROTOCOLS, ids=H1_IDS)
async def test_h1_inbound_pause_reads_merged_watermark(protocol_cls: type[Any]) -> None:
    entered = asyncio.Event()

    async def app(scope: Any, receive: Any, send: Any) -> None:
        with PolicyScope(BackpressurePolicy(inbound_high_water=4)):
            entered.set()
            await receive()  # parked until POST_BODY arrives
        await send(RESPONSE_START)
        await send(RESPONSE_BODY)

    protocol = get_h1_protocol(app, protocol_cls)
    protocol.data_received(POST_HEADERS)
    # The app runs as its own real-loop task so the test can keep feeding the
    # transport while the app is parked inside receive() (MockLoop's run_one
    # would otherwise suspend the test itself).
    task = asyncio.ensure_future(protocol.loop.run_one())
    await entered.wait()
    await asyncio.sleep(0)
    assert not protocol.transport.read_paused

    protocol.data_received(POST_BODY)  # 9 buffered bytes vs watermark 4
    assert protocol.transport.read_paused  # merged watermark paused the read side

    await asyncio.wait_for(task, timeout=1)  # receive returns, frame pops, response completes
    assert not protocol.transport.read_paused
    assert b"200" in protocol.transport.buffer
    assert b"ok" in protocol.transport.buffer


@pytest.mark.parametrize("protocol_cls", H1_PROTOCOLS, ids=H1_IDS)
async def test_h1_baseline_watermark_still_applies(protocol_cls: type[Any]) -> None:
    """No frame pushed: the module HIGH_WATER_LIMIT keeps driving pause_reading."""
    entered = asyncio.Event()

    async def app(scope: Any, receive: Any, send: Any) -> None:
        entered.set()
        await receive()
        await send(RESPONSE_START)
        await send(RESPONSE_BODY)

    protocol = get_h1_protocol(app, protocol_cls)
    large_body = b"x" * (HIGH_WATER_LIMIT + 1)
    large_headers = b"\r\n".join(
        [
            b"POST / HTTP/1.1",
            b"Host: example.org",
            b"Content-Length: " + str(len(large_body)).encode(),
            b"",
            b"",
        ]
    )
    protocol.data_received(large_headers)
    task = asyncio.ensure_future(protocol.loop.run_one())
    await entered.wait()
    await asyncio.sleep(0)

    protocol.data_received(large_body)
    assert protocol.transport.read_paused
    await asyncio.wait_for(task, timeout=1)
    assert not protocol.transport.read_paused


# ---------------------------------------------------------------------------
# Exit 3: HTTP/1 outbound drain (write_await)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("protocol_cls", H1_PROTOCOLS, ids=H1_IDS)
async def test_h1_write_await_false_skips_drain(protocol_cls: type[Any]) -> None:
    async def app(scope: Any, receive: Any, send: Any) -> None:
        with PolicyScope(BackpressurePolicy(write_await=False)):
            await send(RESPONSE_START)
            await send(RESPONSE_BODY)

    protocol = get_h1_protocol(app, protocol_cls)
    protocol.data_received(SIMPLE_GET_REQUEST)
    protocol.flow.pause_writing()  # transport write side is full
    await protocol.loop.run_one()  # must not block on flow.drain()
    assert b"200" in protocol.transport.buffer
    assert b"ok" in protocol.transport.buffer


@pytest.mark.parametrize("protocol_cls", H1_PROTOCOLS, ids=H1_IDS)
async def test_h1_baseline_send_waits_for_drain(protocol_cls: type[Any]) -> None:
    sent = asyncio.Event()

    async def app(scope: Any, receive: Any, send: Any) -> None:
        await send(RESPONSE_START)
        sent.set()

    protocol = get_h1_protocol(app, protocol_cls)
    protocol.data_received(SIMPLE_GET_REQUEST)
    protocol.flow.pause_writing()
    task = asyncio.ensure_future(protocol.loop.run_one())
    for _ in range(10):
        await asyncio.sleep(0)
    assert not sent.is_set()
    assert b"200" not in protocol.transport.buffer

    protocol.flow.resume_writing()
    await asyncio.wait_for(task, timeout=1)
    assert sent.is_set()
    assert b"200" in protocol.transport.buffer


# ---------------------------------------------------------------------------
# Exit 4: httptools pipelining keeps its queue; the stack never masquerades
# the HTTP/2 siblings_clear gate as pipelining.
# ---------------------------------------------------------------------------


@skip_without_httptools
async def test_httptools_pipeline_scheduling_unchanged_under_policy() -> None:
    async def app(scope: Any, receive: Any, send: Any) -> None:
        # An HTTP/2-only gate and a tight watermark on an HTTP/1 connection
        # must not change pipeline enqueue/dispatch semantics.
        with PolicyScope(BackpressurePolicy(inbound_high_water=4, resume_gate=SIBLINGS_CLEAR)):
            pass
        await send(RESPONSE_START)
        await send(RESPONSE_BODY)

    protocol = get_h1_protocol(app, HttpToolsProtocol)
    request = b"GET / HTTP/1.1\r\nHost: example.org\r\n\r\n"
    protocol.data_received(request + request)

    assert len(protocol.pipeline) == 1
    assert protocol.transport.read_paused  # queued second request paused reads

    await protocol.loop.run_one()  # first response: queue pops, second cycle starts
    assert len(protocol.pipeline) == 0
    await protocol.loop.run_one()  # second response

    assert protocol.transport.buffer.count(b"200 OK") == 2
    assert not protocol.transport.read_paused


# ---------------------------------------------------------------------------
# Exit 5: HTTP/2 resume gate and per-stream inner frames
# ---------------------------------------------------------------------------


def _h2_path_app(entered: dict[str, asyncio.Event]) -> Any:
    async def app(scope: Any, receive: Any, send: Any) -> None:
        path = scope["path"]
        if path == "/a":
            with PolicyScope(
                BackpressurePolicy(inbound_high_water=10, resume_gate=SIBLINGS_CLEAR),
                stream=True,
            ):
                entered[path].set()
                await receive()
        elif path == "/b":
            with PolicyScope(BackpressurePolicy(inbound_high_water=20, write_await=False), stream=True):
                entered[path].set()
                await receive()
        elif path == "/c":
            with PolicyScope(BackpressurePolicy(resume_gate=CYCLE_DONE), stream=True):
                entered[path].set()
                await receive()
        else:
            entered[path].set()
            await receive()
        await send(RESPONSE_START)
        await send(RESPONSE_BODY)

    return app


async def _start_parked_streams(
    protocol: Any, client: H2Client, entered: dict[str, asyncio.Event], paths: list[str]
) -> tuple[list[Any], list[Any]]:
    """Send headers-only requests for ``paths`` and run each stream's app as
    its own real-loop task; return (streams, tasks) once every app is parked
    inside its first receive()."""

    streams = []
    for path in paths:
        streams.append(client.request(b"POST", path.encode(), end=False))
    protocol.data_received(client.data_to_send())
    tasks = [asyncio.ensure_future(protocol.loop.run_one()) for _ in paths]
    for path in paths:
        await entered[path].wait()
    for _ in range(10):
        await asyncio.sleep(0)
    return streams, tasks


def _entered(paths: list[str]) -> dict[str, asyncio.Event]:
    return {path: asyncio.Event() for path in paths}


@skip_without_h2
async def test_h2_inner_frames_are_per_stream_and_shared_transport_scans_siblings() -> None:
    entered = _entered(["/a", "/b"])
    protocol = get_h2_protocol(_h2_path_app(entered))
    client = H2Client()
    (stream_a, stream_b), tasks = await _start_parked_streams(protocol, client, entered, ["/a", "/b"])

    cycle_a = protocol.cycles[stream_a.stream_id]
    cycle_b = protocol.cycles[stream_b.stream_id]

    # Inner frames landed on the stream, not on the shared connection stack.
    assert len(cycle_a.stream_policies) == 1
    assert len(cycle_b.stream_policies) == 1
    assert protocol.backpressure.view().inbound_high_water == HIGH_WATER_LIMIT

    # Each stream reads its own merged watermark: no cross-stream leakage.
    assert cycle_a.effective_policy().inbound_high_water == 10
    assert cycle_b.effective_policy().inbound_high_water == 20

    protocol.flow.pause_reading()
    assert protocol.transport.read_paused

    # Stream A asks to resume while sibling B is over B's own watermark:
    # siblings_clear must keep the shared transport paused.
    cycle_b.body += b"x" * 50
    protocol.request_resume_reading(cycle_a)
    assert protocol.transport.read_paused

    # Even when A itself is drained, the over-watermark sibling still blocks.
    cycle_a.body += b"x" * 11
    protocol.request_resume_reading(cycle_a)
    assert protocol.transport.read_paused
    cycle_a.body = bytearray()
    protocol.request_resume_reading(cycle_a)
    assert protocol.transport.read_paused

    # A is at 15 bytes: over A's own 10-byte watermark but under B's 20.
    cycle_b.body = bytearray()
    cycle_a.body += b"x" * 15
    protocol.request_resume_reading(cycle_a)
    assert protocol.transport.read_paused  # A blocked by its own 10

    # B at 15 is clear under its own 20-byte watermark.
    cycle_a.body = bytearray()
    cycle_b.body += b"x" * 15
    protocol.request_resume_reading(cycle_a)
    assert not protocol.transport.read_paused  # sibling B is fine at 15 < 20

    # Let both apps finish cleanly.
    stream_a.end_message()
    stream_b.end_message()
    protocol.data_received(client.data_to_send())
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)


@skip_without_h2
async def test_h2_connection_frame_is_shared_but_inner_frame_wins() -> None:
    entered = _entered(["/a", "/b"])
    protocol = get_h2_protocol(_h2_path_app(entered))
    client = H2Client()
    (stream_a, stream_b), tasks = await _start_parked_streams(protocol, client, entered, ["/a", "/b"])
    cycle_a = protocol.cycles[stream_a.stream_id]
    cycle_b = protocol.cycles[stream_b.stream_id]

    frame = BackpressurePolicy(write_await=False)
    # Stream B already overrides write_await on its inner frame; A still shows
    # the baseline before the connection frame is pushed.
    assert cycle_a.effective_policy().write_await is True
    assert cycle_b.effective_policy().write_await is False
    protocol.backpressure.push(frame)
    try:
        assert cycle_a.effective_policy().write_await is False
        assert cycle_b.effective_policy().write_await is False
        # Inner inbound watermarks still take precedence over the connection.
        assert cycle_a.effective_policy().inbound_high_water == 10
        assert cycle_b.effective_policy().inbound_high_water == 20
    finally:
        protocol.backpressure.pop(frame)

    assert cycle_a.effective_policy().write_await is True
    assert cycle_b.effective_policy().write_await is False  # back to B's own inner view

    stream_a.end_message()
    stream_b.end_message()
    protocol.data_received(client.data_to_send())
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)


@skip_without_h2
async def test_h2_cycle_done_gate_resumes_despite_over_limit_sibling() -> None:
    entered = _entered(["/c", "/d"])
    protocol = get_h2_protocol(_h2_path_app(entered))
    client = H2Client()
    (stream_c, stream_d), tasks = await _start_parked_streams(protocol, client, entered, ["/c", "/d"])
    cycle_c = protocol.cycles[stream_c.stream_id]
    cycle_d = protocol.cycles[stream_d.stream_id]

    protocol.flow.pause_reading()
    cycle_d.body += b"x" * (HIGH_WATER_LIMIT + 1)
    # C explicitly chose the HTTP/1-style gate: its own drain is enough.
    protocol.request_resume_reading(cycle_c)
    assert not protocol.transport.read_paused

    # The same gate on the connection stack applies when a stream actually
    # closes: finishing C resumes reads even though over-limit sibling D is
    # still alive.
    protocol.flow.pause_reading()
    connection_gate = BackpressurePolicy(resume_gate=CYCLE_DONE)
    protocol.backpressure.push(connection_gate)
    try:
        stream_c.end_message()
        protocol.data_received(client.data_to_send())
        await asyncio.wait_for(tasks[0], timeout=1)
        assert not protocol.transport.read_paused
    finally:
        protocol.backpressure.pop(connection_gate)

    stream_d.end_message()
    protocol.data_received(client.data_to_send())
    await asyncio.wait_for(tasks[1], timeout=1)


@skip_without_h2
async def test_h2_baseline_gate_still_waits_for_siblings() -> None:
    """Never pushed: HTTP/2 keeps resume_reading_if_idle behavior exactly."""

    entered = _entered(["/e", "/f"])
    protocol = get_h2_protocol(_h2_path_app(entered))
    client = H2Client()
    (stream_e, stream_f), tasks = await _start_parked_streams(protocol, client, entered, ["/e", "/f"])
    cycle_e = protocol.cycles[stream_e.stream_id]
    cycle_f = protocol.cycles[stream_f.stream_id]

    protocol.flow.pause_reading()
    cycle_f.body += b"x" * (HIGH_WATER_LIMIT + 1)
    protocol.request_resume_reading(cycle_e)
    assert protocol.transport.read_paused

    cycle_f.body = bytearray()
    protocol.request_resume_reading(cycle_e)
    assert not protocol.transport.read_paused

    stream_e.end_message()
    stream_f.end_message()
    protocol.data_received(client.data_to_send())
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)


# ---------------------------------------------------------------------------
# Exit 6: limit_concurrency 503 does not subscribe; lifespan has no stack
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("protocol_cls", H1_PROTOCOLS, ids=H1_IDS)
async def test_h1_503_cycle_ignores_the_stack(protocol_cls: type[Any]) -> None:
    async def real_app(scope: Any, receive: Any, send: Any) -> None:  # pragma: no cover
        await send(RESPONSE_START)

    protocol = get_h1_protocol(real_app, protocol_cls, limit_concurrency=1)
    # Even with a frame present, the 503 cycle reads the baseline view.
    protocol.backpressure.push(BackpressurePolicy(write_await=False, inbound_high_water=1))
    post_request = b"\r\n".join([b"POST / HTTP/1.1", b"Host: example.org", b"Content-Length: 2", b"", b""]) + b"ok"
    protocol.data_received(post_request)

    cycle = protocol.cycle
    assert cycle is not None
    assert cycle.subscribed is False
    assert cycle.effective_policy() == BASELINE_POLICY
    await protocol.loop.run_one()
    assert b"503" in protocol.transport.buffer
    assert b"Service Unavailable" in protocol.transport.buffer


@skip_without_h2
async def test_h2_503_cycle_ignores_the_stack() -> None:
    async def real_app(scope: Any, receive: Any, send: Any) -> None:  # pragma: no cover
        await receive()

    protocol = get_h2_protocol(real_app, limit_concurrency=1)
    protocol.backpressure.push(BackpressurePolicy(write_await=False, inbound_high_water=1))
    client = H2Client()
    stream = client.request(b"GET", b"/")
    protocol.data_received(client.data_to_send())

    cycle = protocol.cycles[stream.stream_id]
    assert cycle.subscribed is False
    assert cycle.effective_policy() == BASELINE_POLICY
    await protocol.loop.run_one()
    assert b"503" in protocol.transport.buffer


def test_lifespan_scope_has_no_stack_binding() -> None:
    async def app(scope: Any, receive: Any, send: Any) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                with pytest.raises(RuntimeError, match="No connection backpressure stack"):
                    push_policy(BackpressurePolicy(inbound_high_water=1))
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def run() -> None:
        config = Config(app=app, lifespan="on")
        lifespan = LifespanOn(config)
        await lifespan.startup()
        await lifespan.shutdown()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(run())
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Exit 7: HTTP -> WebSocket upgrade snapshots the stack top; the three
# WebSocket implementations keep writable / direct pause_reading semantics.
# ---------------------------------------------------------------------------


class RecordingWSProtocol:
    """Stands in for a WebSocket implementation; records the upgrade snapshot."""

    captured: EffectivePolicy | None = None
    last_kwargs: dict[str, Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs
        type(self).captured = kwargs["backpressure"]

    def connection_made(self, transport: Any) -> None:
        self.transport = transport

    def data_received(self, data: bytes) -> None:
        self.replayed = data


@pytest.mark.parametrize("protocol_cls", H1_PROTOCOLS, ids=H1_IDS)
async def test_ws_upgrade_snapshots_stack_top(protocol_cls: type[Any]) -> None:
    async def app(scope: Any, receive: Any, send: Any) -> None:  # pragma: no cover
        pass

    protocol = get_h1_protocol(app, protocol_cls)
    protocol.ws_protocol_class = RecordingWSProtocol

    frame = BackpressurePolicy(write_await=False)
    protocol.backpressure.push(frame)
    try:
        protocol.data_received(UPGRADE_REQUEST)
    finally:
        protocol.backpressure.pop(frame)

    snapshot = RecordingWSProtocol.captured
    assert snapshot is not None
    assert snapshot.write_await is False  # followed the stack top at upgrade time
    assert snapshot.inbound_high_water == HIGH_WATER_LIMIT  # unset field merged to baseline
    # Frozen: popping the HTTP-side frame does not mutate the WebSocket view.
    assert snapshot.write_await is False


@pytest.mark.parametrize("protocol_cls", H1_PROTOCOLS, ids=H1_IDS)
async def test_ws_upgrade_without_frames_keeps_baseline(protocol_cls: type[Any]) -> None:
    async def app(scope: Any, receive: Any, send: Any) -> None:  # pragma: no cover
        pass

    protocol = get_h1_protocol(app, protocol_cls)
    protocol.ws_protocol_class = RecordingWSProtocol
    protocol.data_received(UPGRADE_REQUEST)
    assert RecordingWSProtocol.captured == BASELINE_POLICY


class PolicyMockWebSocketConnection(MockWebSocketConnection):
    """Like MockWebSocketConnection, but constructs the WS protocol with an
    upgrade-time backpressure snapshot."""

    def __init__(
        self,
        app: Any,
        ws_protocol_cls: Any,
        http_protocol_cls: Any,
        backpressure: EffectivePolicy | None,
    ) -> None:
        config = Config(app=app, ws=ws_protocol_cls, http=http_protocol_cls, lifespan="off")
        config.load()
        self._server_state = ServerState()
        self._protocol = ws_protocol_cls(
            config=config,
            server_state=self._server_state,
            app_state={},
            backpressure=backpressure,
        )
        self._protocol.close_timeout = 10.0
        self._transport = MockWriteTransport()
        self._client = ClientProtocol(parse_uri("ws://127.0.0.1/"))
        self._bytes_read = 0
        self._outbox = b""
        self._connection_lost = False


async def test_ws_sansio_and_wsproto_skip_writable_wait_when_snapshot_says_so(
    ws_protocol_cls: Any, http_protocol_cls: Any
) -> None:
    send_requested = asyncio.Event()
    send_completed = asyncio.Event()

    async def app(scope: Any, receive: Any, send: Any) -> None:
        await receive()  # websocket.connect
        await send({"type": "websocket.accept"})
        await send_requested.wait()
        await send({"type": "websocket.send", "text": "x"})
        send_completed.set()

    snapshot = EffectivePolicy(HIGH_WATER_LIMIT, False, None)
    connection = PolicyMockWebSocketConnection(app, ws_protocol_cls, http_protocol_cls, snapshot)
    assert connection._protocol.write_await is False
    async with connection:
        # accept happens as part of entering
        connection.pause_writing()
        send_requested.set()
        for _ in range(10):
            await asyncio.sleep(0)
        assert send_completed.is_set()  # send did not await the writable event
        assert connection.receive_text() == ["x"]
        connection.resume_writing()


async def test_ws_default_snapshot_keeps_writable_wait(ws_protocol_cls: Any, http_protocol_cls: Any) -> None:
    send_requested = asyncio.Event()
    send_completed = asyncio.Event()

    async def app(scope: Any, receive: Any, send: Any) -> None:
        await receive()
        await send({"type": "websocket.accept"})
        await send_requested.wait()
        await send({"type": "websocket.send", "text": "x"})
        send_completed.set()

    connection = PolicyMockWebSocketConnection(app, ws_protocol_cls, http_protocol_cls, None)
    assert connection._protocol.write_await is True
    async with connection:
        connection.pause_writing()
        send_requested.set()
        for _ in range(10):
            await asyncio.sleep(0)
        assert not send_completed.is_set()
        connection.resume_writing()
        await send_completed.wait()
        assert connection.receive_text() == ["x"]


WS_IMPL_PATHS = [
    "uvicorn/protocols/websockets/websockets_impl.py",
    "uvicorn/protocols/websockets/websockets_sansio_impl.py",
    "uvicorn/protocols/websockets/wsproto_impl.py",
]


@pytest.mark.parametrize("relative_path", WS_IMPL_PATHS)
def test_ws_implementations_never_use_http_flow_control(relative_path: str) -> None:
    source = (pathlib.Path(__file__).parents[2] / relative_path).read_text(encoding="utf-8")
    code_lines = [line for line in source.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    code = "\n".join(code_lines)
    # No import of the HTTP flow-control module and no FlowControl use: the
    # three WebSocket implementations only gate on their writable machinery.
    assert "uvicorn.protocols.http.flow_control" not in code
    assert "FlowControl" not in code


def test_legacy_websockets_drain_follows_snapshot() -> None:
    if importlib.util.find_spec("websockets") is None:  # pragma: no cover
        pytest.skip("websockets is not installed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module = __import__("uvicorn.protocols.websockets.websockets_impl", fromlist=["WebSocketProtocol"])

        config = Config(app=lambda scope, receive, send: None, lifespan="off")  # pragma: no branch

        async def run() -> None:
            default = module.WebSocketProtocol(config=config, server_state=ServerState(), app_state={})
            assert default.write_await is True

            nonwaiting = module.WebSocketProtocol(
                config=config,
                server_state=ServerState(),
                app_state={},
                backpressure=EffectivePolicy(HIGH_WATER_LIMIT, False, None),
            )
            assert nonwaiting.write_await is False
            # Returns immediately without a connected transport: the override
            # bypasses the library's writable waiter entirely.
            await nonwaiting.drain()

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Exit 8: concurrent connections isolate their stacks; reset_contextvars
# still binds the stack.
# ---------------------------------------------------------------------------


async def test_two_connections_push_different_policies_without_crossing() -> None:
    entered = {1: asyncio.Event(), 2: asyncio.Event()}
    release = asyncio.Event()

    def make_app(name: int, limit: int) -> Any:
        async def app(scope: Any, receive: Any, send: Any) -> None:
            with PolicyScope(BackpressurePolicy(inbound_high_water=limit)):
                entered[name].set()
                await release.wait()
                assert current_policy().inbound_high_water == limit
            await send(RESPONSE_START)
            await send(RESPONSE_BODY)

        return app

    protocols: list[Any] = []
    for name, limit in ((1, 111), (2, 222)):
        transport = MockTransport()
        config = Config(app=make_app(name, limit), lifespan="off")
        protocol = H11Protocol(
            config=config,
            server_state=ServerState(),
            app_state={},
            _loop=asyncio.get_running_loop(),
        )
        protocol.connection_made(transport)
        protocol.data_received(POST_HEADERS)
        protocols.append((protocol, transport))

    await entered[1].wait()
    await entered[2].wait()

    protocol_1, transport_1 = protocols[0]
    protocol_2, transport_2 = protocols[1]
    assert protocol_1.backpressure.view().inbound_high_water == 111
    assert protocol_2.backpressure.view().inbound_high_water == 222

    release.set()
    await asyncio.sleep(0.1)
    assert b"200" in transport_1.buffer
    assert b"200" in transport_2.buffer
    assert protocol_1.backpressure.view() == BASELINE_POLICY
    assert protocol_2.backpressure.view() == BASELINE_POLICY


async def test_reset_contextvars_still_binds_the_stack() -> None:
    entered = asyncio.Event()

    async def app(scope: Any, receive: Any, send: Any) -> None:
        assert current_policy().inbound_high_water == HIGH_WATER_LIMIT
        with PolicyScope(BackpressurePolicy(inbound_high_water=1234)):
            assert current_policy().inbound_high_water == 1234
            entered.set()
            await asyncio.sleep(0)
        assert current_policy().inbound_high_water == HIGH_WATER_LIMIT
        await send(RESPONSE_START)
        await send(RESPONSE_BODY)

    transport = MockTransport()
    config = Config(app=app, reset_contextvars=True, lifespan="off")
    protocol = H11Protocol(
        config=config,
        server_state=ServerState(),
        app_state={},
        _loop=asyncio.get_running_loop(),
    )
    protocol.connection_made(transport)
    protocol.data_received(SIMPLE_GET_REQUEST)

    await asyncio.wait_for(entered.wait(), timeout=1)
    await asyncio.sleep(0.05)
    assert b"200" in transport.buffer
