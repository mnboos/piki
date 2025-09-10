import asyncio
import json
import subprocess

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
from channels.consumer import SyncConsumer
from channels.generic.websocket import AsyncWebsocketConsumer


class VideoStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.pc = RTCPeerConnection()
        await self.accept()

    async def receive(self, text_data):
        print("receive in consumer!!!")
        params = json.loads(text_data)
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        @self.pc.on("track")
        async def on_track(track):
            # This is for receiving streams from the client, which we don't need in this case.
            pass

        player = MediaPlayer("/dev/video0", format="v4l2", options={"video_size": "640x480"})

        # Create a video track from the FFmpeg RTP stream
        # player = MediaPlayer("rtp://127.0.0.1:5004")
        video_sender = self.pc.addTrack(player.video)

        await self.pc.setRemoteDescription(offer)
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)

        await self.send(
            text_data=json.dumps(
                {
                    "sdp": self.pc.localDescription.sdp,
                    "type": self.pc.localDescription.type,
                }
            )
        )

    async def disconnect(self, close_code):
        await self.pc.close()
