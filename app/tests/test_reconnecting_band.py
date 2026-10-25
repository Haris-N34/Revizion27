import threading
import time

from app.reconnecting_band import ReconnectingBand
from shared.contracts import Direction, GuidanceCommand


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


class USB:
    def __init__(self):
        self.unplugged = False
        self.commands = []
        self.closed = False

    def send(self, cmd):
        if self.unplugged:
            raise OSError("WriteFile failed")
        self.commands.append(cmd)

    def poll_button(self):
        if self.unplugged:
            raise OSError("ClearCommError failed")
        return True

    def close(self):
        self.closed = True
        if self.unplugged:
            raise OSError("device removed during close")


def test_usb_loss_does_not_escape_and_reconnect_restores_button_and_commands():
    devices = []

    def factory(config):
        device = USB()
        devices.append(device)
        return device

    band = ReconnectingBand({}, factory=factory, retry_s=0.01)
    cmd = GuidanceCommand(Direction.LEFT, 100, 50)
    try:
        wait_for(lambda: band.connected)
        for method in (lambda: band.poll_button(), lambda: band.send(cmd)):
            device = devices[-1]
            device.unplugged = True
            method()
            assert device.closed
            wait_for(lambda: band.connected and devices[-1] is not device)
            assert band.poll_button()
            # No stale movement is replayed automatically after reconnecting.
            assert devices[-1].commands == []
            band.send(cmd)
            assert devices[-1].commands == [cmd]
    finally:
        band.close()
    assert not band.connected


def test_initial_usb_absence_and_slow_reconnect_do_not_block_camera_calls():
    attempts = []
    release = threading.Event()

    def factory(config):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("COM5 missing")
        release.wait(3)
        return USB()

    band = ReconnectingBand({}, factory=factory, retry_s=0.01)
    try:
        wait_for(lambda: len(attempts) >= 2)
        assert not band.connected
        assert band.poll_button() is False
        band.send(GuidanceCommand(Direction.STOP, 0, 0))
        release.set()
        wait_for(lambda: band.connected)
    finally:
        release.set()
        band.close()


def test_closing_during_connect_closes_new_handle():
    release, entered = threading.Event(), threading.Event()
    device = USB()

    def factory(config):
        entered.set()
        release.wait(3)
        return device

    band = ReconnectingBand({}, factory=factory, retry_s=0.01)
    assert entered.wait(1)
    band.close()
    release.set()
    band.worker.join(1)
    assert device.closed and not band.connected and not band.worker.is_alive()
