import asyncio
import json
import os
import subprocess
import time

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

        # @self.pc.on("track")
        # async def on_track(track):
        #     # This is for receiving streams from the client, which we don't need in this case.
        #     pass

        # sdp_file_path = "/tmp/webrtc_piki.sdp"
        #
        # # Wait for the SDP file to be created by FFmpeg
        # max_wait_time = 10
        # start_time = time.time()
        # while not os.path.exists(sdp_file_path):
        #     await asyncio.sleep(0.1)
        #     # Check if the process died
        #     # if self.ffmpeg_process.poll() is not None:
        #     #     print("!!! FFmpeg process terminated prematurely. !!!")
        #     #     stderr_output = self.ffmpeg_process.stderr.read().decode("utf-8", errors="ignore")
        #     #     print("FFmpeg stderr:\n", stderr_output)
        #     #     return None
        #     if time.time() - start_time > max_wait_time:
        #         print("Error: FFmpeg timed out creating SDP file.")
        #         return None

        # player = MediaPlayer("/dev/video0", format="v4l2", options={"video_size": "640x480"})

        await asyncio.sleep(2)  # Wait 2 seconds. Adjust as needed.

        # Create a video track from the FFmpeg RTP stream
        player = MediaPlayer("udp://127.0.0.1:5004", timeout=30000)
        # player = MediaPlayer(sdp_file_path)
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
