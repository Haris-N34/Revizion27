# Servo wristband

User-specified positional servos replace the original coin motors.
Confirmed hardware: Arduino Uno with DMS-MG90 positional servos (4.8–6 V).
Manufacturer specifications: https://www.dfrobot.com.cn/goods-1174.html

| Guidance | Servo signal pin |
| --- | --- |
| Left (`L`) | 8 |
| Right (`R`) | 9 |
| Forward (`F`) | 10 |
| Talk button (input) | 12 |

Pin 12 uses `INPUT_PULLUP` and learns its released level after 250 ms of
startup, with at least 100 ms stable. Keep the button released during reset
and USB reconnection. A press is a change away from that learned level;
this supports a switch to GND or an existing externally pulled-down circuit.
A stable press for 30 ms sends `BTN,1\r\n` over the same USB serial connection.
Holding the button sends no repeats; release for 30 ms before pressing again.
The host preserves these events while discarding `OK`/`ERR` acknowledgements.
Press once, speak into the laptop microphone, then pause to finish the request.

Diagonals use their left or right component. Left/right are already mapped
by guidance; firmware does not mirror them again. `F` pulses only pin 10.
There is no reverse or vertical actuator; the host converts unavailable
directions to STOP. `G` gives two long pulses on all three; `X` three quick taps.
Repeated commands keep the watchdog alive without restarting these patterns.
`S`, zero intensity and 500 ms without a valid command detach all servos.
Detaching removes the signal; it does not switch off the external power supply.

Use an external regulated supply rated for your servo voltage and total stall
current. Connect servo grounds, supply ground and Arduino GND together; only
the signal wires go to pins 8, 9 and 10. Pin 11 is unused. USB powers the Arduino.
Reference: https://docs.arduino.cc/learn/electronics/servo-motors/

The sketch assumes ordinary positional servos, resting at 90 degrees, moving
by 3–12 degrees each way every 50 ms during a pulse. Test off-wrist with horns
free before fitting; adjust REST/MAX_SWING for the mechanism. Continuous
rotation servos need different neutral/stop calibration; do not use this sketch
with those until adapted. Mechanical servos produce taps, not coin-motor buzz.

Upload hardware/firmware/octaby_band/octaby_band.ino in Arduino IDE with the
Servo library, selecting the actual board and USB port. Set serial_port and
baud (115200) in the ignored config.yaml. Close Serial Monitor before running
the app. Run `python -m app.main` to use camera guidance with the real band.
