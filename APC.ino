#include <Servo.h>
#include <Braccio.h>
#include <Arduino.h>

Servo baseServo;
Servo shoulderServo;
Servo elbowServo;
Servo wristVerServo;
Servo wristRotServo;
Servo gripperServo;

int currentBase = 100;
int currentShoulder = 100;
int currentElbow = 100;
int currentWrist_ver = 100;
int currentWrist_rot = 100;
int currentGripper = 100;

String incomingMessage = "";  // Hier speichern wir die eingehende Nachricht

void setup() {
    baseServo.attach(11);
    shoulderServo.attach(10);
    elbowServo.attach(9);
    wristVerServo.attach(6);
    wristRotServo.attach(5);
    gripperServo.attach(3);
    
    baseServo.write(currentBase);
    shoulderServo.write(currentShoulder);
    elbowServo.write(currentElbow);
    wristVerServo.write(currentWrist_ver);
    wristRotServo.write(currentWrist_rot);
    gripperServo.write(currentGripper);

    // Pause for 2 seconds to let the servos reach their positions
    delay(2000);

    Serial.begin(9600);  // Initialisiert die serielle Verbindung mit einer Baudrate von 9600
}

void loop() {
    // Überprüft, ob Daten auf der seriellen Verbindung verfügbar sind
    while (Serial.available()) {
        char c = Serial.read();  // Ein Zeichen von der seriellen Verbindung lesen
        if (c == '\n') {  // Ende der Nachricht
            if (incomingMessage == "start") {
                executeBraccioCommand();
            }
            incomingMessage = "";  // Nachricht zurücksetzen
        } else {
            incomingMessage += c;  // Fügt das Zeichen zur Nachricht hinzu
        }
    }
}

void executeBraccioCommand() {
    // Hier den Code hinzufügen, den der Braccio ausführen soll, wenn er die "start"-Nachricht empfängt
    // Zum Beispiel:
    baseServo.write(45);  // Diese Zeilen sind nur Beispiele. Ersetzen Sie sie durch Ihre tatsächlichen Befehle.
    shoulderServo.write(90);
    // ... Weitere Befehle ...
}
