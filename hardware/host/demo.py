"""Read shared-contract JSON lines and send computed commands to the band."""
import argparse
import json
import sys

from guidance.compute import compute
from hardware.host.band import Band, MockBand
from shared.config import load_config
from shared.contracts import Detection, HandPosition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock-band", action="store_true")
    parser.add_argument("--list-ports", action="store_true")
    args = parser.parse_args()
    if args.list_ports:
        from serial.tools.list_ports import comports
        for port in comports():
            print(port.device, port.description)
        return
    config = load_config()
    with (MockBand(config) if args.mock_band else Band(config)) as band:
        for line in sys.stdin:
            data = json.loads(line)
            det = Detection(**data["detection"])
            hand = HandPosition(**data["hand"]) if data.get("hand") else None
            if hand is not None and hand.frame_id != det.frame_id:
                raise ValueError("Detection and hand must belong to the same frame")
            cmd = compute(det, hand, data["frame_shape"], config)
            print(f"distance_px={cmd.distance_px:.1f} direction={cmd.direction.value}", file=sys.stderr)
            band.send(cmd)


if __name__ == "__main__":
    main()
