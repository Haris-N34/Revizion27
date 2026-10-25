Install: `python -m pip install -r hardware/requirements.txt`.
List ports: `python -m hardware.host.demo --list-ports`.
Set serial_port and baud in the ignored root config.yaml.
Run offline: `python -m hardware.host.demo --mock-band < hardware/sample.jsonl`.
Upload `firmware/octaby_band/octaby_band.ino` using Arduino IDE and the Servo library.
Input JSON lines contain detection, hand (nullable), and frame_shape from the same frame.
Call Band.send every frame; unchanged commands resend after 200 ms, capped at 30 Hz.
Closing sends STOP; firmware must implement the 500 ms watchdog.
Pins: left 8, right 9, forward 10, talk button 12. Keep the button released on reset; its idle level is learned automatically. See wiring.md.
