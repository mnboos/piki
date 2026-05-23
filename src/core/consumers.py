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

    async def receive_json(self, content, **kwargs) -> None:  # noqa: ARG002
        # Client may send `ping` heartbeats — silently accept.
        return

    async def event_broadcast(self, event: dict) -> None:
        await self.send_json(event["msg"])
