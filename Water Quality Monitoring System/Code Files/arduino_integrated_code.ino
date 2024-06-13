
//Temperature sensor intilization
#include <DallasTemperature.h>
#define ONE_WIRE_BUS 4 // Pin for DS18B20 data line
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

//TDS Sensor Configuration/Dependencies
#include "CQRobotTDS.h"
float temp = 25.0; 
#define TDS_SENSOR A0
#define A_REF 5.0
CQRobotTDS tds(TDS_SENSOR,A_REF);

//Turbidity Sensor
#define TURBIDITY_SENSOR A2


//Software Serial 
#include <SoftwareSerial.h>
SoftwareSerial espSerial(2, 3); // RX - Pin 2, TX - Pin 3 pins on Arduino

//pH Sensor
#define pH_SENSOR A1
float calibration_value = 21.34; 
int sensorValue = 0; 
unsigned long int avgValue; 
float b;
int buf[10],t=0;

//Time out Trackers
unsigned long serial_timeout = 300;
unsigned long thingSpeak_timeout = 15000;

void read_ph_vtg()
{
 for(int i=0;i<10;i++) 
 { 
    buf[i]=analogRead(pH_SENSOR);
    delay(10);
 }

 //Sort the read values
  for(int i=0;i<9;i++)
 {
  for(int j=i+1;j<10;j++)
  {
   if(buf[i]>buf[j])
   {
    t=buf[i];
    buf[i]=buf[j];
    buf[j]=t;
   }
  }
 }

 avgValue = 0;
 for(int i=2;i<8;i++)
 {
  avgValue+=buf[i];
 }
  
}

float get_ph_value()
{
  float volt=(float)avgValue*5.0/1024/6; 
  float ph_act = -5.70 * volt + calibration_value;
  return ph_act;
}

void print_solution_turbidity(int turbidity)
{
   if (turbidity <= 20) {
      Serial.println(" its CLEAR ");
    }
    if ((turbidity > 20) && (turbidity <= 50)) {
      Serial.println(" its CLOUDY ");
    }
    if (turbidity > 50) {
      Serial.println(" its DIRTY ");
    }

}


void setup()
{
  //Initialize Hardware Serial Communication
  Serial.begin(115200);

  //Initialize Software Serial Communication
  espSerial.begin(115200);  
  
  // Initialize DS18B20 sensor
  sensors.begin();
}

void loop()
{

  float temperatureC;
  float phVal;
  int sensorValue;
  int turbidity;
  
  float tdsValue = tds.update(temperatureC);
  read_ph_vtg();
  

  //Update the local serial terminal every 1 second
  if (serial_timeout<millis())
  {
    sensors.requestTemperatures();
    temperatureC = sensors.getTempCByIndex(0);
    phVal = get_ph_value();
    sensorValue = analogRead(TURBIDITY_SENSOR);
    turbidity = map(sensorValue, 0, 750, 100, 0);
    
    
    Serial.print("TDS value: ");
    Serial.print(tdsValue, 0);
    Serial.println(" ppm");    
    Serial.print("Temperature: ");
    Serial.print(temperatureC);
    Serial.println("°C");
    Serial.print("pH Value:");
    Serial.println(phVal);
    Serial.print("Turbidity (Raw): ");
    Serial.print(turbidity);
    print_solution_turbidity(turbidity);
    Serial.println();
     
    serial_timeout = millis() + 300;
  }

  if(thingSpeak_timeout<millis())
  {
    Serial.println("Sending data to thingspeak!");
    // Send data to ESP32
    espSerial.print("DATA,");
    espSerial.print(temperatureC);
    espSerial.print(",");
    espSerial.print(tdsValue);
    espSerial.print(",");
    espSerial.print(phVal);
    espSerial.print(",");
    espSerial.println(turbidity);
    thingSpeak_timeout = millis() + 16000;
  }
}
