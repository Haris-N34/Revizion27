"""USB serial transport for the shared wristband protocol."""
import time

from shared.contracts import Direction, GuidanceCommand


def _encode(cmd):
    direction = Direction(cmd.direction)
    if isinstance(cmd.intensity, bool) or not isinstance(cmd.intensity, int) or not 0 <= cmd.intensity <= 255:
        raise ValueError("Intensity must be an integer from 0 to 255")
    intensity = 0 if direction == Direction.STOP else cmd.intensity
    return f"{direction.value},{intensity}\n".encode("ascii")


class Band:
    def __init__(self, config):
        import serial
        port = config.get("serial_port")
        if not port:
            raise ValueError("Set serial_port in config.yaml")
        self._serial = serial.Serial(port, config.get("baud", 115200), timeout=0, write_timeout=1)
        self._closed = False
        self._last = None
        self._sent_at = float("-inf")
        self._rx = bytearray()
        self._button_pending = False
        try:
            time.sleep(2)
            self.send(GuidanceCommand(Direction.STOP, 0, 0))
        except BaseException:
            self._serial.close()
            raise

    def send(self, cmd):
        if self._closed:
            raise RuntimeError("Band is closed")
        payload = _encode(cmd)
        now = time.monotonic()
        elapsed = now - self._sent_at
        if elapsed < 1 / 30 or payload == self._last and elapsed < 0.2:
            return
        self._read_events()
        if self._serial.write(payload) != len(payload):
            raise OSError("Incomplete serial command")
        self._last, self._sent_at = payload, now

    def _read_events(self):
        """Drain replies, preserving button events and incomplete serial lines."""
        waiting = self._serial.in_waiting
        if waiting:
            self._rx.extend(self._serial.read(waiting))
        while b"\n" in self._rx:
            line, _, remaining = self._rx.partition(b"\n")
            self._rx = bytearray(remaining)
            if line.rstrip(b"\r") == b"BTN,1":
                self._button_pending = True
        if len(self._rx) > 128:
            self._rx.clear()

    def poll_button(self):
        """Consume a press once; also works while no guidance is being sent."""
        if self._closed:
            return False
        self._read_events()
        pressed, self._button_pending = self._button_pending, False
        return pressed

    def close(self):
        if not self._closed:
            try:
                self._serial.write(b"S,0\n")
                self._serial.flush()
            finally:
                self._serial.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class MockBand:
    def __init__(self, config=None):
        self._closed = False

    def send(self, cmd):
        if self._closed:
            raise RuntimeError("Band is closed")
        print(_encode(cmd).decode("ascii"), end="")

    def poll_button(self):
        return False

    def close(self):
        if not self._closed:
            self.send(GuidanceCommand(Direction.STOP, 0, 0))
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
