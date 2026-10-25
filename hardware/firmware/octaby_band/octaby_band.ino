#include <Servo.h>
#include <string.h>

// Positional servos: signal pins, in user-facing direction order.
const byte PINS[] = {8, 9, 10, 11}; // right, top, left, back (down)
const byte REST[] = {90, 90, 90, 90};
const byte MAX_SWING = 12; // degrees either side of rest; calibrate off-wrist
const unsigned long WATCHDOG_MS = 500;
const byte BUTTON_PIN = 12; // externally pulled LOW; HIGH when pressed
const unsigned long BUTTON_DEBOUNCE_MS = 30;
bool buttonRaw = false, buttonPressed = false;
unsigned long buttonChanged = 0;
Servo servos[4];
char direction[3] = "S";
byte intensity = 0;
unsigned long lastValid = 0, started = 0;
char line[16];
byte used = 0;
bool overflowed = false;

byte maskFor(const char *d) {
  if (!strcmp(d, "R")) return 1;
  if (!strcmp(d, "U")) return 2;
  if (!strcmp(d, "L")) return 4;
  if (!strcmp(d, "D")) return 8;
  if (!strcmp(d, "UR")) return 3;
  if (!strcmp(d, "UL")) return 6;
  if (!strcmp(d, "DR")) return 9;
  if (!strcmp(d, "DL")) return 12;
  if (!strcmp(d, "G") || !strcmp(d, "X") ||
      !strcmp(d, "F") || !strcmp(d, "B")) return 15;
  return 0;
}

void stopServos() {
  // Remove control pulses, including on watchdog expiry.
  for (byte i = 0; i < 4; ++i) {
    servos[i].detach();
    digitalWrite(PINS[i], LOW);
  }
}

bool acceptLine(unsigned long now) {
  line[used] = '\0';
  char *comma = strchr(line, ',');
  if (!comma || comma == line || comma - line > 2) return false;
  *comma = '\0';
  if (!maskFor(line) && strcmp(line, "S")) return false;
  char *p = comma + 1;
  if (!*p) return false;
  unsigned int value = 0;
  for (; *p; ++p) {
    if (*p < '0' || *p > '9') return false;
    value = value * 10 + (*p - '0');
    if (value > 255) return false;
  }
  // Heartbeats must not restart one-shot grab/lost patterns.
  if (strcmp(direction, line)) started = now;
  strcpy(direction, line);
  intensity = value;
  lastValid = now;
  return true;
}

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT); // preserve the button's active-HIGH wiring
  for (byte i = 0; i < 4; ++i) pinMode(PINS[i], OUTPUT);
  stopServos();
}

void loop() {
  unsigned long now = millis();
  bool pressed = digitalRead(BUTTON_PIN) == HIGH;
  if (pressed != buttonRaw) {
    buttonRaw = pressed;
    buttonChanged = now;
  }
  if (now - buttonChanged >= BUTTON_DEBOUNCE_MS && pressed != buttonPressed) {
    buttonPressed = pressed;
    if (buttonPressed) Serial.println("BTN,1");
  }
  // Bound serial work so malformed/flooded input cannot starve the watchdog.
  for (byte n = 0; n < 64 && Serial.available(); ++n) {
    char c = Serial.read();
    if (c == '\n') {
      bool valid = !overflowed && acceptLine(now);
      Serial.println(valid ? "OK" : "ERR");
      used = 0;
      overflowed = false;
    } else if (used < sizeof(line) - 1) {
      line[used++] = c;
    } else {
      overflowed = true;
    }
  }
  if (now - lastValid >= WATCHDOG_MS) {
    strcpy(direction, "S");
    intensity = 0;
  }
  unsigned long elapsed = now - started;
  byte mask = maskFor(direction);
  bool active = elapsed % 300 < 200;
  if (!strcmp(direction, "G"))
    active = elapsed < 1000 && elapsed % 500 < 350;
  else if (!strcmp(direction, "X"))
    active = elapsed < 450 && elapsed % 150 < 75;
  else if (!strcmp(direction, "F")) {
    unsigned int period = 160 + intensity; // closer -> faster
    active = elapsed % period < period / 2;
  } else if (!strcmp(direction, "B"))
    active = elapsed % 900 < 650;
  if (!mask || !intensity ||
      ((!strcmp(direction, "G") && elapsed >= 1000) ||
       (!strcmp(direction, "X") && elapsed >= 450))) {
    stopServos();
    return;
  }
  byte swing = 3 + (unsigned int)intensity * (MAX_SWING - 3) / 255;
  int offset = active ? ((elapsed / 50) % 2 ? swing : -swing) : 0;
  for (byte i = 0; i < 4; ++i) {
    if (mask & (1 << i)) {
      // Set target before attaching to avoid an old target on reattachment.
      servos[i].write(REST[i] + offset);
      if (!servos[i].attached()) servos[i].attach(PINS[i]);
    } else {
      servos[i].detach();
      digitalWrite(PINS[i], LOW);
    }
  }
}
