#include <Servo.h>
#include <string.h>

// Positional servos: signal pins, in user-facing direction order.
const byte PINS[] = {8, 9, 10}; // left, right, forward
const byte SERVO_COUNT = sizeof(PINS) / sizeof(PINS[0]);
const byte REST[] = {90, 90, 90};
const byte MAX_SWING = 12; // degrees either side of rest; calibrate off-wrist
const unsigned long WATCHDOG_MS = 500;
const byte BUTTON_PIN = 12;
const unsigned long BUTTON_DEBOUNCE_MS = 30;
bool buttonRaw = false, buttonPressed = false;
bool buttonReady = false;
byte buttonIdle = HIGH;
unsigned long buttonChanged = 0;
Servo servos[SERVO_COUNT];
char direction[3] = "S";
byte intensity = 0;
unsigned long lastValid = 0, started = 0;
char line[16];
byte used = 0;
bool overflowed = false;

byte maskFor(const char *d) {
  if (!strcmp(d, "L") || !strcmp(d, "UL") || !strcmp(d, "DL")) return 1;
  if (!strcmp(d, "R") || !strcmp(d, "UR") || !strcmp(d, "DR")) return 2;
  if (!strcmp(d, "F")) return 4;
  if (!strcmp(d, "G") || !strcmp(d, "X")) return 7;
  return 0;
}

void stopServos() {
  // Remove control pulses, including on watchdog expiry.
  for (byte i = 0; i < SERVO_COUNT; ++i) {
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
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  // Learn the released level after startup; keep the button released on reset.
  buttonIdle = digitalRead(BUTTON_PIN);
  buttonChanged = millis();
  for (byte i = 0; i < SERVO_COUNT; ++i) pinMode(PINS[i], OUTPUT);
  stopServos();
}

void loop() {
  unsigned long now = millis();
  byte level = digitalRead(BUTTON_PIN);
  if (!buttonReady) {
    if (level != buttonIdle) {
      buttonIdle = level;
      buttonChanged = now;
    }
    if (now >= 250 && now - buttonChanged >= 100) buttonReady = true;
  }
  bool pressed = buttonReady && level != buttonIdle;
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
  }
  if (!mask || !intensity ||
      ((!strcmp(direction, "G") && elapsed >= 1000) ||
       (!strcmp(direction, "X") && elapsed >= 450))) {
    stopServos();
    return;
  }
  byte swing = 3 + (unsigned int)intensity * (MAX_SWING - 3) / 255;
  int offset = active ? ((elapsed / 50) % 2 ? swing : -swing) : 0;
  for (byte i = 0; i < SERVO_COUNT; ++i) {
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
