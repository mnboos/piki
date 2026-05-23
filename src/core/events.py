"""Thread-safe push to all connected WebSocket clients.

Producers (inference loop, splash logic, recording/replay) call
`publish(topic, payload)` after mutating shared state. The call is
non-blocking and safe from any thread: it schedules a
`channel_layer.group_send` onto daphne's asyncio loop, which fans out to
every consumer in the `"events"` group.

If no client has ever connected (loop not yet bound), publish is a no-op.
Reconnecting clients get current state via the consumer's snapshot, so
events dropped during the no-client window aren't a correctness problem.
"""

import asyncio
import logging
import threading
import time

from channels.layers import get_channel_layer

logger = logging.getLogger(__name__)

GROUP = "events"

_loop: asyncio.AbstractEventLoop | None = None
_channel_layer = None
_throttle_last: dict[str, float] = {}
_throttle_lock = threading.Lock()


def bind_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Capture daphne's running loop on first WS connect."""
    global _loop, _channel_layer
    if _loop is None:
        _loop = loop
        _channel_layer = get_channel_layer()
        logger.info("events.publish bound to loop=%r layer=%r", _loop, _channel_layer)


def publish(topic: str, payload: dict) -> None:
    """Broadcast an event envelope to all WebSocket clients (thread-safe)."""
    loop = _loop
    layer = _channel_layer
    if loop is None or layer is None:
        return

    envelope = {"topic": topic, "payload": payload, "ts": time.time()}
    try:
        asyncio.run_coroutine_threadsafe(
            layer.group_send(GROUP, {"type": "event.broadcast", "msg": envelope}),
            loop,
        )
    except RuntimeError:
        # Loop closed during shutdown — drop quietly.
        pass


def publish_throttled(topic: str, payload: dict, min_interval: float) -> None:
    """Publish at most once per `min_interval` seconds for a given topic."""
    now = time.monotonic()
    with _throttle_lock:
        last = _throttle_last.get(topic, 0.0)
        if now - last < min_interval:
            return
        _throttle_last[topic] = now
    publish(topic, payload)
