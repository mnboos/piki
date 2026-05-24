"""Single WebSocket consumer that fans out push events to the SPA."""

import asyncio
import logging
import time

from channels.generic.websocket import AsyncJsonWebsocketConsumer

from . import events
from .utils.event_payloads import snapshot

logger = logging.getLogger(__name__)


class EventsConsumer(AsyncJsonWebsocketConsumer):
    GROUP = events.GROUP

    async def connect(self) -> None:
        events.bind_loop(asyncio.get_running_loop())
        await self.channel_layer.group_add(self.GROUP, self.channel_name)
        await self.accept()
        snap = snapshot()
        ts = time.time()
        for topic, payload in snap.items():
            await self.send_json({"topic": topic, "payload": payload, "ts": ts})

    async def disconnect(self, code: int) -> None:
        await self.channel_layer.group_discard(self.GROUP, self.channel_name)

    async def receive(self, text_data=None, bytes_data=None, **kwargs) -> None:
        # Intercept raw `ping` heartbeats before the base class tries to JSON-decode
        # them (the SPA uses @vueuse useWebSocket which sends plain text, not JSON).
        if text_data == "ping":
            await self.send(text_data="pong")
            return
        await super().receive(text_data=text_data, bytes_data=bytes_data, **kwargs)

    async def receive_json(self, content, **kwargs) -> None:  # noqa: ARG002
        return

    async def event_broadcast(self, event: dict) -> None:
        await self.send_json(event["msg"])
