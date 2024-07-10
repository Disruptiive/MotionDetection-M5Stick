#include <M5StickCPlus.h>
#include <ArduinoMqttClient.h>
#include "WiFi.h"
#include "time.h"
#include <ArduinoJson.h>

#define DELAY 0.1


// NTP server to request epoch time
const char* ntpServer = "europe.pool.ntp.org";

// Variable to save current epoch time
unsigned long Epoch_Time; 

int send_button = 0;
int curr_read = 0;

int send_button2 = 0;
int curr_read2 = 0;

float accX = 0;
float accY = 0;
float accZ = 0;

float gyroX = 0;
float gyroY = 0;
float gyroZ = 0;

float roll = 0;
float pitch = 0;
float yaw = 0;

float temp = 0;

char ssid[] = "HCN-07F980";    // your network SSID (name)
char pass[] = "77313111074341044391";  

const char broker[] = "192.168.100.100";
int        port     = 1883;
const char topic[]  = "test/imu";


static int cnt = 0;

unsigned long Get_Epoch_Time() {
  time_t now;
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) {
    //Serial.println("Failed to obtain time");
    return(0);
  }
  time(&now);
  return now;
}

String payload;
WiFiClient wifiClient;
MqttClient mqttClient(wifiClient);

/* After M5StickC is started or reset
  the program in the setUp () function will be run, and this part will only be run once.
  After M5StickCPlus is started or reset, the program in the setup() function will be executed, and this part will only be executed once. */
void setup(){
  M5.begin();
  Serial.begin(115200);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }
  Serial.print("Attempting to connect to WPA SSID: ");
  Serial.println(ssid);
  while (!WiFi.begin(ssid, pass)) {
    // failed, retry
    Serial.print(".");
    delay(5000);
  }
  Serial.print("WIFI CONNECTED");
  while (!mqttClient.connect(broker, port)) {
    Serial.print("MQTT connection failed! Error code = ");
    Serial.println(mqttClient.connectError());

    delay (5000);
  }

  // LCD display. LCd display
  M5.Imu.Init();

  M5.Lcd.setRotation(3);
  M5.Lcd.fillScreen(BLUE);
  M5.Lcd.setTextSize(1);
  M5.Lcd.setCursor(40, 0);
  M5.Lcd.println("MPU6886 TEST");

  configTime(0, 0, ntpServer);
  /*
  mqttClient.beginMessage(topic);
  payload = "start";
  mqttClient.print(payload);
  mqttClient.endMessage();
  */
  
}



/* After the program in setup() runs, it runs the program in loop()
The loop() function is an infinite loop in which the program runs repeatedly
After the program in the setup() function is executed, the program in the loop() function will be executed
The loop() function is an endless loop, in which the program will continue to run repeatedly */
void loop() {
    mqttClient.poll();
    static float measurements[60];
    M5.Imu.getGyroData(&gyroX, &gyroY, &gyroZ);
    M5.Imu.getAccelData(&accX, &accY, &accZ);

    accZ -= 0.11;
    gyroX +=3;
    gyroY += 16;
    gyroZ += 8;

    if (curr_read != M5.BtnA.read() && curr_read == 1){
      curr_read = M5.BtnA.read();
    }
    else if (curr_read != M5.BtnA.read()){
      send_button++;
      curr_read = M5.BtnA.read();
    }

    if (curr_read2 != M5.BtnB.read() && curr_read2 == 1){
      curr_read2 = M5.BtnB.read();
    }
    else if (curr_read2 != M5.BtnB.read()){
      send_button2++;
      curr_read2 = M5.BtnB.read();
    }

    measurements[cnt*6] = gyroX;
    measurements[cnt*6+1] = gyroY;
    measurements[cnt*6+2] = gyroZ;
    measurements[cnt*6+3] = accX;
    measurements[cnt*6+4] = accY;
    measurements[cnt*6+5] = accZ;
  
    cnt++; 

    if (cnt == 10){
      DynamicJsonDocument doc(3000);
      JsonArray jsonArray;
      jsonArray = doc.createNestedArray("data");
      for (int i =0;i<60;i++){
        jsonArray.add(measurements[i]);
        //Serial.println(measurements[i]);
      }
      Epoch_Time = Get_Epoch_Time();
      jsonArray.add((float) Epoch_Time);
      jsonArray.add((float) (send_button%2));
      jsonArray.add((float) (send_button2%2));
      jsonArray.add((int)  curr_read2);
      Serial.println(Epoch_Time);
      mqttClient.beginMessage(topic,(unsigned long)measureJson(doc));
      serializeJson(doc, mqttClient);
      mqttClient.endMessage();
      cnt = 0;
    }
    
    delay(10);
}
