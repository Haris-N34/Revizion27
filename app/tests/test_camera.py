import queue
import socket
import struct
import threading
import time

import cv2
import numpy as np

from app.camera import DroidCamCapture, FRESH_FRAME_S


def wait_for(predicate):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate(), "camera did not reach the expected state"


class Phone:
    def __init__(self, value=None):
        self.condition = threading.Condition()
        self.data = bytearray()
        self.closed = False
        if value is not None:
            self.feed(value)

    def feed(self, value):
        image = np.full((12, 16, 3), value, dtype=np.uint8)
        ok, jpeg = cv2.imencode(".jpg", image)
        assert ok
        self.raw(struct.pack(">QI", 0, jpeg.size) + jpeg.tobytes())

    def raw(self, data):
        with self.condition:
            self.data.extend(data)
            self.condition.notify_all()

    def recv(self, length):
        with self.condition:
            if not self.condition.wait_for(lambda: self.data or self.closed, timeout=1):
                raise socket.timeout()
            if self.closed:
                return b""
            part = self.data[:length]
            del self.data[:length]
            return part

    def settimeout(self, value):
        pass

    def sendall(self, request):
        assert b"/v5/video/jpg/" in request

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    shutdown = lambda self, how: self.close()


class Connections:
    def __init__(self):
        self.phones = queue.Queue()
        self.calls = 0

    def __call__(self, address, timeout):
        assert address == ("phone.test", 4747)
        self.calls += 1
        try:
            return self.phones.get_nowait()
        except queue.Empty:
            raise ConnectionRefusedError("phone offline")


def test_offline_start_and_three_disconnect_reconnect_cycles():
    connections = Connections()
    camera = DroidCamCapture("phone.test", connect=connections, retry_s=0.01)
    try:
        wait_for(lambda: connections.calls >= 2)
        assert camera.read() == (False, None)
        for value in (40, 100, 200):
            phone = Phone(value)
            connections.phones.put(phone)
            wait_for(camera.isOpened)
            ok, frame = camera.read()
            assert ok and abs(float(frame.mean()) - value) < 2
            phone.close()
            wait_for(lambda: not camera.isOpened())
            assert camera.read() == (False, None)
    finally:
        camera.release()
    assert not camera.worker.is_alive()


def test_only_latest_frame_is_returned_and_expired_frames_are_rejected():
    connections, phone = Connections(), Phone()
    for value in (20, 80, 160):
        phone.feed(value)
    connections.phones.put(phone)
    clock = [10.0]
    camera = DroidCamCapture("phone.test", connect=connections,
                             clock=lambda: clock[0], retry_s=0.01)
    try:
        wait_for(lambda: camera.sequence == 3)
        ok, frame = camera.read()
        assert ok and abs(float(frame.mean()) - 160) < 2
        clock[0] += FRESH_FRAME_S + 0.1
        assert not camera.isOpened()
        assert camera.read() == (False, None)
        phone.feed(220)
        wait_for(lambda: camera.sequence == 4)
        ok, frame = camera.read()
        assert ok and abs(float(frame.mean()) - 220) < 2
    finally:
        camera.release()


def test_busy_or_invalid_stream_retries_and_shutdown_interrupts_reader():
    connections, busy = Connections(), Phone()
    busy.raw(b"HTTP/1.1 503 Busy\r\n")
    connections.phones.put(busy)
    camera = DroidCamCapture("phone.test", connect=connections, retry_s=0.01)
    try:
        wait_for(lambda: busy.closed)
        assert camera.read() == (False, None)
        connections.phones.put(Phone(70))
        wait_for(camera.isOpened)
        assert camera.read()[0]
    finally:
        camera.release()
    assert not camera.worker.is_alive()


def test_release_stops_offline_retries():
    connections = Connections()
    camera = DroidCamCapture("phone.test", connect=connections, retry_s=0.01)
    wait_for(lambda: connections.calls > 0)
    camera.release()
    calls = connections.calls
    time.sleep(0.03)
    assert calls == connections.calls
    assert not camera.worker.is_alive()
