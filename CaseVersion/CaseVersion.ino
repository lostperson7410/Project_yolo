#include <cvzone.h>

SerialData serialData(1, 2); //(numOfValsRec,digitsPerValRec)
int valsRec[1]; // array of int with size numOfValsRec 

// Motor A

int dir1PinA = 2;

int dir2PinA = 3;

int speedPinA = 6; //   เพื่อให้ PWM สามารถควบคุมความเร็วมอเตอร์ 


// Motor B

int dir1PinB = 4;

int dir2PinB = 5;

int speedPinB = 7; // เพื่อให้ PWM สามารถควบคุมความเร็วมอเตอร์


void setup()
{
  pinMode(13, OUTPUT);
  
  serialData.begin();
  
  Serial.begin(9600);

  //กำหนด ขา เพื่อใช้ในการควบคุมการทำงานของ  Motor ผ่านทาง L298N

  pinMode(dir1PinA,OUTPUT);

  pinMode(dir2PinA,OUTPUT);

  pinMode(speedPinA,OUTPUT);

  pinMode(dir1PinB,OUTPUT);

  pinMode(dir2PinB,OUTPUT);

  pinMode(speedPinB,OUTPUT);
}

void loop()
{
  serialData.Get(valsRec);


  switch(valsRec[0]){
    case 1:
    Serial.print("NO");
    
    analogWrite(speedPinB, 10); //ตั้งค่าความเร็ว PWM ผ่านตัวแปร ค่าต่ำลง มอเตอร์จะหมุนช้าลง
    
    digitalWrite(dir1PinB, LOW);
    
    digitalWrite(dir2PinB, HIGH);
    delay(8000);
//    Waiting Car
    digitalWrite(dir1PinB, LOW);
    
    digitalWrite(dir2PinB, LOW);
    delay(3000);
//    Continuse
    digitalWrite(dir1PinB, HIGH);
    
    digitalWrite(dir2PinB, LOW);
    delay(8000); 
    
    digitalWrite(13, HIGH);
    delay(500);    
    digitalWrite(13, LOW);
    delay(500); 
    break;
    default:
    digitalWrite(dir1PinB, LOW);
    
    digitalWrite(dir2PinB, LOW);
    digitalWrite(13, HIGH);
    delay(100);    
    digitalWrite(13, LOW);
    delay(100); 
    break;
  }
//
//  if(valsRec[0] == 1){
//    Serial.print("NO");
//    
//    analogWrite(speedPinB, 10); //ตั้งค่าความเร็ว PWM ผ่านตัวแปร ค่าต่ำลง มอเตอร์จะหมุนช้าลง
//    
//    digitalWrite(dir1PinB, LOW);
//    
//    digitalWrite(dir2PinB, HIGH);
//    delay(1000);
//    digitalWrite(dir1PinB, HIGH);
//    
//    digitalWrite(dir2PinB, LOW);
//    delay(1000); 
//    
//    digitalWrite(13, HIGH);
//    delay(500);    
//    digitalWrite(13, LOW);
//    delay(500); 
//
//
//  }else if(valsRec[0] == 2){
//    Serial.print("Rewrase");
//    analogWrite(speedPinB, 100); //ตั้งค่าความเร็ว PWM ผ่านตัวแปร ค่าต่ำลง มอเตอร์จะหมุนช้าลง
//    
//    digitalWrite(dir1PinB, HIGH);
//    
//    digitalWrite(dir2PinB, LOW);
//
//  }else if(valsRec[0] == 0){
//    Serial.print("Stop");
//    
//    analogWrite(speedPinB, 0); //ตั้งค่าความเร็ว PWM ผ่านตัวแปร ค่าต่ำลง มอเตอร์จะหมุนช้าลง
//  
//    digitalWrite(dir1PinB, LOW);
//  
//    digitalWrite(dir2PinB, LOW);
//    digitalWrite(13, HIGH);
//    delay(100);
//    digitalWrite(13, LOW);
//    delay(100);
//  }
  
// Motor A

//  analogWrite(speedPinA, 255); //ตั้งค่าความเร็ว PWM ผ่านตัวแปร ค่าต่ำลง มอเตอร์จะหมุนช้าลง
//
//  digitalWrite(dir1PinA, LOW);
//
//  digitalWrite(dir2PinA, HIGH);
//

// Motor B

//  analogWrite(speedPinB, 255); //ตั้งค่าความเร็ว PWM ผ่านตัวแปร ค่าต่ำลง มอเตอร์จะหมุนช้าลง
//
//  digitalWrite(dir1PinB, LOW);
//
//  digitalWrite(dir2PinB, HIGH);
}
