#include <WiFi.h>
#include <HTTPClient.h>
#include <ThingSpeak.h>

const char* ssid = "Sambhav's Pixel 7";
const char* password = "abcd1234";
const char* WRITE_API_KEY = "7BAX7WO3IMPOSVY3";
const unsigned long CHANNEL_ID = 2413723;
const int updateInterval = 15000; // Update interval in milliseconds (15 seconds)
WiFiClient client;  // Define a WiFi client object


// Function to send data to ThingSpeak
void sendData(float *data_to_send) {
  
  if (WiFi.status() == WL_CONNECTED) 
  {
    ThingSpeak.begin(client);  // Initialize ThingSpeak with the WiFi client
    ThingSpeak.setField(1, data_to_send[0]);  // Set field 1 with temperature
    ThingSpeak.setField(2, data_to_send[1]);     // Set field 2 with TDS Value
    ThingSpeak.setField(3, data_to_send[2]);     // Set field 3 with ph Value
    ThingSpeak.setField(4, data_to_send[3]);     // Set field 4 with Turbidity
    
    int response = ThingSpeak.writeFields(CHANNEL_ID, WRITE_API_KEY);
    if (response == 200) 
    {
      Serial.println("Data sent to ThingSpeak successfully!");
    } 
    else 
    {
      Serial.print("Error sending data to ThingSpeak. HTTP error code: ");
      Serial.println(response);
    }      
  }
  else 
  {
      Serial.println("WiFi not connected. Retrying...");
      delay(1000);
      WiFi.begin(ssid, password);
  }
}



void setup() {
  Serial.begin(115200);
  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to ");
  Serial.println(ssid);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");
}

void loop() 
{
    float sensors_data[4];
    // Read data from Arduino via Serial
    if (Serial.available() > 0) 
    {
      String data = Serial.readStringUntil('\n');
      if (data.startsWith("DATA,")) {
        // Extract sensor values from received data
        data.remove(0, 5); // Remove "DATA," from the beginning
        String sensorValues[4];
        int index = 0;
        while (data.length() > 0 && index < 4) {
          int commaIndex = data.indexOf(',');
          if (commaIndex != -1) {
            sensorValues[index] = data.substring(0, commaIndex);
            data.remove(0, commaIndex + 1); // Remove the parsed value and the comma
            index++;
          } else {
            sensorValues[index] = data; // Last value
            break;
          }
        }

        // Convert sensor values to floats
        sensors_data[0] = sensorValues[0].toFloat();
        sensors_data[1] = sensorValues[1].toFloat();
        sensors_data[2] = sensorValues[2].toFloat();
        sensors_data[3] = sensorValues[3].toFloat();
        Serial.print("Temperature:");
        Serial.println(sensors_data[0]);
        Serial.print("TDS:");
        Serial.println(sensors_data[1]);
        Serial.print("ph Value:");
        Serial.println(sensors_data[2]);
        Serial.print("Turbidity:");
        Serial.println(sensors_data[3]);
        
        sendData(sensors_data);

        
        
      }
   }
}
