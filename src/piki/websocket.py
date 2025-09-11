import asyncio
import json
import struct
import time

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
from channels.consumer import SyncConsumer
from channels.generic.websocket import AsyncWebsocketConsumer

from core.utils import shared


class VideoStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def receive(self, text_data):
        data = json.loads(text_data)

        if data["type"] == "start_stream":
            await self.stream_raw_frames(
                width=data.get("width", 640), height=data.get("height", 480), fps=data.get("fps", 30)
            )

    async def stream_raw_frames(self, width, height, fps):
        """Stream raw RGB frames"""
        frame_size = width * height * 3  # RGB24

        async for frame in self.get_raw_frame():
            # Get raw frame data (from camera, FFmpeg, etc.)
            # raw_frame = await self.get_raw_frame(width, height)

            if frame is not None:
                # Send frame header + raw data
                header = struct.pack(
                    "<IHHQ",
                    0xDEADBEEF,  # Magic number
                    width,
                    height,
                    int(time.time() * 1000),  # timestamp
                )

                message = header + frame.tobytes()
                await self.send(bytes_data=message)

            await asyncio.sleep(1 / fps)

    async def get_raw_frame(self):
        while True:
            # This is an efficient way to wait for new frames without burning CPU
            # shared.output_buffer.wait_for_data()
            if shared.output_buffer.queue.empty():
                await asyncio.sleep(0.01)  # Non-blocking sleep
                continue

            latest_result = shared.output_buffer.popleft()
            if latest_result is None:
                # time.sleep(0.01)
                await asyncio.sleep(0.01)  # Non-blocking sleep
                continue

            # The 'frame' here is the low-resolution preview frame
            _worker_pid, _timestamp, frame, detected_objects = latest_result
            yield frame
