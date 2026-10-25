"""Keep USB failures and reconnect delays out of the live camera loop."""

import threading


class ReconnectingBand:
    def __init__(self, config, *, factory=None, retry_s=2):
        if factory is None:
            from hardware.host.band import Band
            factory = Band
        self.config, self.factory, self.retry_s = config, factory, retry_s
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.band = None
        self.error = "Connecting wristband"
        self.worker = threading.Thread(target=self._reconnect, daemon=True,
                                       name="wristband-reconnect")
        self.worker.start()

    @property
    def connected(self):
        with self.lock:
            return self.band is not None

    @staticmethod
    def _close(band):
        try:
            band.close()
        except (OSError, RuntimeError):
            pass  # An unplugged serial handle can also fail during close.

    def _reconnect(self):
        while not self.closed.is_set():
            if not self.connected:
                try:
                    band = self.factory(self.config)
                except (OSError, RuntimeError) as exc:
                    self.error = str(exc)
                else:
                    with self.lock:
                        if not self.closed.is_set():
                            self.band, self.error = band, None
                            band = None
                    if band is not None:
                        self._close(band)
            self.closed.wait(self.retry_s)

    def _call(self, method, *args):
        failed = None
        with self.lock:
            if self.band is None or self.closed.is_set():
                return False
            try:
                return getattr(self.band, method)(*args)
            except (OSError, RuntimeError) as exc:
                failed, self.band = self.band, None
                self.error = str(exc)
        self._close(failed)
        return False

    def send(self, cmd):
        self._call("send", cmd)

    def poll_button(self):
        return self._call("poll_button")

    def close(self):
        self.closed.set()
        with self.lock:
            band, self.band = self.band, None
        if band is not None:
            self._close(band)
        self.worker.join(timeout=0.1)
