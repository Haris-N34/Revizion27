from unittest.mock import Mock, patch
import pytest
from hardware.host.band import Band, MockBand, _encode
from shared.contracts import Direction, GuidanceCommand

def test_format():
    assert _encode(GuidanceCommand(Direction.UP_LEFT, 123, 20)) == b"UL,123\n"
    with pytest.raises(ValueError):
        _encode(GuidanceCommand(Direction.UP, 256, 0))

def test_resend_rate_and_close():
    serial = Mock(in_waiting=0)
    serial.write.side_effect = lambda value: len(value)
    band = Band.__new__(Band)
    band._serial, band._closed = serial, False
    band._last, band._sent_at = None, float("-inf")
    band._rx, band._button_pending = bytearray(), False
    cmd = GuidanceCommand(Direction.RIGHT, 180, 100)
    with patch("hardware.host.band.time.monotonic", side_effect=[0, 0.01, 0.1, 0.201]):
        for _ in range(4):
            band.send(cmd)
    assert serial.write.call_count == 2
    band.close()
    serial.write.assert_called_with(b"S,0\n")
    serial.close.assert_called_once()

def test_mock(capsys):
    with MockBand() as band:
        band.send(GuidanceCommand(Direction.LEFT, 80, 1))
    assert capsys.readouterr().out == "L,80\nS,0\n"


def test_button_event_survives_send_and_split_serial_lines():
    serial = Mock(in_waiting=12)
    serial.read.return_value = b"OK\r\nBTN,"
    serial.write.side_effect = len
    band = Band.__new__(Band)
    band._serial, band._closed = serial, False
    band._last, band._sent_at = None, float("-inf")
    band._rx, band._button_pending = bytearray(), False
    band.send(GuidanceCommand(Direction.RIGHT, 180, 100))
    assert band._rx == b"BTN,"
    serial.read.return_value = b"1\r\nERR\r\n"
    assert band.poll_button()
    serial.in_waiting = 0
    assert not band.poll_button()
    serial.in_waiting = 7
    serial.read.return_value = b"BTN,1\r\n"
    band._sent_at = float("-inf")
    band.send(GuidanceCommand(Direction.RIGHT, 180, 100))
    serial.in_waiting = 0
    assert band.poll_button()  # send must not discard the press with the ACKs
    assert not band.poll_button()


def test_mock_has_no_button_events():
    assert not MockBand().poll_button()
