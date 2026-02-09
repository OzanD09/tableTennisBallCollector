#include <Arduino.h>
#include <AccelStepper.h>

#define motorInterfaceType 1
const int stepLeft = 2;
const int dirLeft = 5;
const int stepRight = 3;
const int dirRight = 6;
const int enablePin = 8;
int stepPerRev = 200;
int microStepping = 16;

const double pi = 3.14159265358979323846;
float wheelDia = 1.5; //diameter of wheel in cm

AccelStepper leftStepper(motorInterfaceType, stepLeft, dirLeft);
AccelStepper rightStepper(motorInterfaceType, stepRight, dirRight);

float calcRotations(float distance) {
  float rotations = distance/(1.5*pi);
  return rotations;
}

int moveForward(float distanceCM) {
  float rotations = calcRotations(distanceCM);
  long targetSteps = (long)(rotations * stepPerRev * microStepping);

  leftStepper.move(targetSteps);
  rightStepper.move(targetSteps);

  while (leftStepper.distanceToGo() != 0 || rightStepper.distanceToGo() != 0){
    leftStepper.run();
    rightStepper.run();
  }
}


// put function declarations here:
int moveForward(int, int); //forwards movement in cm
float calcRotations(float);



void setup() {
  // put your setup code here, to run once:

}

void loop() {
  // put your main code here, to run repeatedly:
}


// put function definitions here:
