# Servo wristband

User-specified positional servos replace the original coin motors.
Confirmed hardware: Arduino Uno with DMS-MG90 positional servos (4.8–6 V).
Manufacturer specifications: https://www.dfrobot.com.cn/goods-1174.html

| Guidance | Servo signal pin |
| --- | --- |
| Right (`R`) | 8 |
| Top/up (`U`) | 11 |
| Left (`L`) | 10 |
| Back/down (`D`) | 9 |
| Talk button (input) | 12 |

The button is LOW when released and HIGH when pressed. Pin 12 uses `INPUT`,
with the existing external pull-down/button circuit (no internal pull-up).
A stable press for 30 ms sends `BTN,1\r\n` over the same USB serial connection.
Holding the button sends no repeats; release for 30 ms before pressing again.
The host preserves these events while discarding `OK`/`ERR` acknowledgements.
Press once, speak into the laptop microphone, then pause to finish the request.

Diagonals activate both corresponding servos. Left/right are already mapped
by guidance; firmware does not mirror them again. Depth `F` pulses all four
quickly, `B` uses long pulses. `G` gives two long pulses; `X` three quick taps.
Repeated commands keep the watchdog alive without restarting these patterns.
`S`, zero intensity and 500 ms without a valid command detach all servos.
Detaching removes the signal; it does not switch off the external power supply.

Use an external regulated supply rated for your servo voltage and total stall
current. Connect servo grounds, supply ground and Arduino GND together; only
the signal wires go to pins 8–11. USB powers the Arduino.
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
