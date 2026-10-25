"""Reconnectable DroidCam capture with a single slot for the newest JPEG."""

import socket
import struct
import threading
import time
from urllib.parse import urlsplit

import cv2
import numpy as np

RETRY_S = 2.0
FRESH_FRAME_S = 0.75
MAX_PACKET = 16 * 1024 * 1024
VIDEO_REQUEST = b"GET /v5/video/jpg/640x480/port/0/os/windows/obs/30/client/2.4.3/hdr/0/nonce/5912/"


def open_camera(source):
    if isinstance(source, str) and source.startswith("droidcam://"):
        url = urlsplit(source)
        if not url.hostname:
            raise ValueError("Set camera_index to droidcam://PHONE_IP:4747")
        return DroidCamCapture(url.hostname, url.port or 4747)
    return cv2.VideoCapture(source)


class DroidCamCapture:
    """One reader owns the phone socket and retries until release() is called.

    read() never returns an old frame from before a disconnect, and never queues
    frames behind inference. Opening an unavailable phone does not block the UI.
    """

    def __init__(self, host, port=4747, *, connect=socket.create_connection,
                 clock=time.monotonic, retry_s=RETRY_S):
        self.address = (host, port)
        self.connect, self.clock, self.retry_s = connect, clock, retry_s
        self.condition = threading.Condition()
        self.closed = threading.Event()
        self.socket = None
        self.jpeg = None
        self.sequence = self.consumed = 0
        self.received = 0.0
        self.error = "Waiting for DroidCam. Keep the phone app open."
        self.worker = threading.Thread(target=self._receive, name="droidcam", daemon=True)
        self.worker.start()

    def _exact(self, connection, length):
        data = bytearray()
        while len(data) < length:
            if self.closed.is_set():
                raise EOFError("Camera closed")
            part = connection.recv(length - len(data))
            if not part:
                raise EOFError("DroidCam disconnected")
            data.extend(part)
        return data

    def _receive(self):
        while not self.closed.is_set():
            connection = None
            try:
                connection = self.connect(self.address, timeout=2)
                connection.settimeout(1)
                with self.condition:
                    self.socket = connection
                connection.sendall(VIDEO_REQUEST)
                while not self.closed.is_set():
                    _, length = struct.unpack(">QI", self._exact(connection, 12))
                    if not 0 < length <= MAX_PACKET:
                        raise ValueError("DroidCam is busy or sent an invalid frame")
                    jpeg = self._exact(connection, length)
                    if not jpeg.startswith(b"\xff\xd8"):
                        continue  # codec configuration packets are not images
                    with self.condition:
                        self.jpeg = jpeg
                        self.sequence += 1
                        self.received = self.clock()
                        self.error = None
                        self.condition.notify_all()
            except (OSError, EOFError, ValueError, struct.error) as exc:
                with self.condition:
                    self.jpeg = None
                    self.error = (
                        "Reconnecting DroidCam. Keep the phone app open and the desktop "
                        "DroidCam client closed. " + str(exc)
                    )
                    self.condition.notify_all()
            finally:
                with self.condition:
                    self.socket = None
                if connection is not None:
                    connection.close()
            self.closed.wait(self.retry_s)

    def isOpened(self):
        with self.condition:
            return (not self.closed.is_set() and self.jpeg is not None
                    and self.clock() - self.received <= FRESH_FRAME_S)

    def read(self):
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence != self.consumed or self.closed.is_set()
                or self.jpeg is None, timeout=FRESH_FRAME_S
            )
            if (self.closed.is_set() or self.jpeg is None
                    or self.sequence == self.consumed
                    or self.clock() - self.received > FRESH_FRAME_S):
                return False, None
            jpeg = self.jpeg
            self.consumed = self.sequence
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame is not None, frame

    def set(self, *args):
        return True

    def release(self):
        self.closed.set()
        with self.condition:
            connection = self.socket
            self.jpeg = None
            self.condition.notify_all()
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        self.worker.join(timeout=3)
