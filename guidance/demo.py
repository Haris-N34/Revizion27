"""python -m guidance.demo

No camera: the mouse is the fingertip. Left-click moves the can, right-click hides or
shows the hand, m toggles mirror, q / Esc quits. Shows raw compute() and the
Stabilizer output, and prints each change of the stabilized command.
"""

import cv2
import numpy as np

from guidance.compute import compute
from guidance.stabilize import Stabilizer
from shared.config import load_config
from shared.contracts import Detection, HandPosition

W, H = 960, 540
CAN_W, CAN_H = 80, 150
WINDOW = "octaby guidance"


def main() -> None:
    config = load_config()
    state = {"can": (W * 0.65, H * 0.45), "tip": (W * 0.2, H * 0.8), "hand": True}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            state["tip"] = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            state["can"] = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["hand"] = not state["hand"]

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    stabilizer = Stabilizer(config)
    last = None
    frame_id = 0
    while True:
        cx, cy = state["can"]
        det = Detection(
            "can", 0.9, cx - CAN_W / 2, cy - CAN_H / 2, cx + CAN_W / 2, cy + CAN_H / 2, 0
        )
        tx, ty = state["tip"]
        hand = HandPosition(tx, ty, tx, ty + 60, "Right", frame_id) if state["hand"] else None
        raw = compute(det, hand, (H, W, 3), config)
        cmd = stabilizer.update(raw)
        if cmd.direction != last:
            print(
                f"{cmd.direction.value},{cmd.intensity}  ({cmd.direction.name}, {cmd.distance_px:.0f}px)"
            )
            last = cmd.direction

        img = np.full((H, W, 3), 40, np.uint8)
        cv2.rectangle(img, (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2)), (0, 255, 0), 2)
        if hand:
            cv2.arrowedLine(img, (int(tx), int(ty)), (int(cx), int(cy)), (0, 255, 255), 2)
            cv2.circle(img, (int(tx), int(ty)), 9, (0, 0, 255), -1)
        lines = [
            f"raw:    {raw.direction.name} ({raw.direction.value},{raw.intensity})",
            f"stable: {cmd.direction.name} ({cmd.direction.value},{cmd.intensity})",
            f"distance {cmd.distance_px:.0f}px   mirror={bool(config.get('mirror'))}",
            "mouse: fingertip  L-click: move can  R-click: hide hand  m: mirror  q: quit",
        ]
        for i, text in enumerate(lines):
            cv2.putText(
                img, text, (10, 28 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
            )
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            config["mirror"] = not config.get("mirror")
        frame_id += 1
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
