# AGENTS.md: hardware/

Owner: **Azfar**. Branch: `feat/azfar`. Read the root `AGENTS.md` first.

## Scope

You own everything in `hardware/` and nothing else:

```
hardware/
  firmware/octaby_band/octaby_band.ino   # Arduino sketch
  host/band.py                           # Python: Band and MockBand
  host/demo.py                           # python -m hardware.host.demo
  tests/test_band.py
  wiring.md                              # pin map + wiring diagram/photo
  requirements.txt                       # pyserial
  README.md
```

## Hardware

- Arduino Uno, Nano or similar, connected to the laptop by USB.
- 4 coin vibration motors at the top, bottom, thumb side and pinky side of
  the wrist. Vibration motors, not servos.
- Each motor driven through a transistor (e.g. 2N2222 or a logic-level
  MOSFET) with a flyback diode. Never power a motor straight from a pin.
- Use PWM pins so intensity (0 to 255) maps to `analogWrite`.
- Record the pin map in `wiring.md` and as constants at the top of the sketch.

## Firmware behaviour

Implements the serial protocol in the root `AGENTS.md` exactly:

| Command | Motors |
| --- | --- |
| `U`, `D`, `L`, `R` | That one motor pulses (about 200 ms on, 100 ms off) at the given intensity |
| `UL`, `UR`, `DL`, `DR` | The two matching motors pulse together |
| `G` | All four: two long buzzes (grab now) |
| `X` | All four: three quick taps (hand lost), then off |
| `S` | All off |

- "Left" and "right" are from the user's point of view. The sketch has a
  `WRIST_IS_LEFT` constant that swaps thumb-side and pinky-side motors.
- Pulse timing runs on the Arduino with `millis()`, never `delay()`, so new
  commands are read immediately.
- 500 ms watchdog: no valid line received means all motors off.
- Reply `OK\n` or `ERR\n` to every line.

## Python host driver (`host/band.py`)

- `Band(config)`: opens `config["serial_port"]` at `config["baud"]` with
  pyserial; waits about 2 s after opening (Uno resets on connect).
- `send(cmd: GuidanceCommand)`: writes `f"{cmd.direction.value},{cmd.intensity}\n"`.
  Skip duplicates but resend the current command every 200 ms.
- `close()`: sends `S,0` then closes the port.
- `MockBand`: same methods, prints commands instead. The app and other
  modules use this when no Arduino is plugged in.
- Never hard-code a port name. Windows uses `COM3` style, macOS
  `/dev/tty.usbmodem...`. Add a `--list-ports` option to the demo that prints
  available ports.

## Tests

`tests/test_band.py` must run without an Arduino: test command formatting,
duplicate suppression and `MockBand` using a fake serial object.

## Don't

- Don't edit `shared/contracts.py`. If the protocol needs a new command, ask
  the team for a shared PR (root `AGENTS.md`, rule 3).
- Don't add Python dependencies other than `pyserial` without telling the
  team.
