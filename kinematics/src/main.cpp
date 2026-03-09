#include <Arduino.h>
#include <AccelStepper.h>
#include <iostream>


#define motorInterfaceType 1
const int stepLeft = 2;
const int dirLeft = 5;
const int stepRight = 3;
const int dirRight = 6;
const int enablePin = 8;
int stepPerRev = 200;
int microStepping = 64;

AccelStepper leftStepper(motorInterfaceType, stepLeft, dirLeft);
AccelStepper rightStepper(motorInterfaceType, stepRight, dirRight);

void setup() {
    Serial.begin(115200);
    pinMode(enablePin, OUTPUT);
    digitalWrite(enablePin, LOW);

    leftStepper.setMaxSpeed(1000);
    leftStepper.setAcceleration(500);

    rightStepper.setMaxSpeed(1000);
    rightStepper.setAcceleration(500);
}

void loop() {
    if (Serial.available() > 0) {
        String commands = Serial.readStringUntil(">");

    }
}