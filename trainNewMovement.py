import time
import random
import json
import argparse
import os
import pandas as pd
from pathlib import Path
from dataset_features import createDataset
from create_model import model
import warnings
warnings.filterwarnings("ignore")

import paho.mqtt.client as mqtt

def json_serializer(obj):
    if isinstance(obj, bytes):
        return obj.decode('utf-8')

    return obj

class CustomData:
    def __init__(self, name, value):
        self.name = name
        self.value = value

class CustomEncoder(json.JSONEncoder):
    def default(self, o):
            return o.__dict__

class Data:
    def __init__(self,size,data_directory,model_directory,cnt,movement):
        self.size = size
        self.record = False

        self.movement = movement

        self.delete_1 = False
        self.delete_cnt = 0
        self.delete = False
        self.btn_b_val = 0

        self.btn_b_holding = False
        self.btn_b_cnt = 0
        self.proceed = False

        self.DatasetRecordAccel = []
        self.DatasetRecordGyro = []

        self.sampleN  = cnt
        self.DatafoldDir = data_directory
        self.ModelDir = model_directory
        self.modified = False




def writeFile(userdata):
    f_name = userdata.DatafoldDir + "/" + str(userdata.sampleN+1) + ".csv"
    with open(f_name,"w") as f:
        print("Motion Recorded")
        for i in range(len(userdata.DatasetRecordAccel)):
            data = str(userdata.DatasetRecordAccel[i])[1:-1]+", "+str(userdata.DatasetRecordGyro[i])[1:-1]+ "\n"
            f.write(data)
        userdata.sampleN += 1
        print("Movement " + userdata.movement + " sample size: "+ str(userdata.sampleN))
        userdata.modified = True

def remove(userdata):
    print("Removed motion: "+ userdata.DatafoldDir + "/" + str(userdata.sampleN) + ".csv")
    if userdata.sampleN > 0:
        file_n = userdata.DatafoldDir + "/" + str(userdata.sampleN) + ".csv"
        if os.path.exists(file_n):
            os.remove(file_n)
            userdata.sampleN -= 1
            userdata.modified = True
    print("Movement " + userdata.movement + " sample size: "+ str(userdata.sampleN))


def calculateRecord(userdata,b_a,data):
    if userdata.record == False and b_a == 1:
        userdata.record = True
        print("Now recording")
        userdata.DatasetRecordAccel = []
        userdata.DatasetRecordGyro = []
    elif userdata.record == True and b_a  == 1:
        for i in range(0,userdata.size,6):
            userdata.DatasetRecordGyro.append((data[i],data[i+1],data[i+2]))
            userdata.DatasetRecordAccel.append((data[i+3],data[i+4],data[i+5]))
    elif userdata.record == True and b_a == 0:
        writeFile(userdata)
        userdata.record = False
    else:
        userdata.record = False

def calculateDelete(userdata,b_b):
    if userdata.delete_1 == False and b_b != userdata.btn_b_val:
        userdata.delete_1 = True
        userdata.delete_cnt = 15
        userdata.btn_b_val = b_b

    elif userdata.delete_1 == True and b_b  != userdata.btn_b_val and userdata.delete_cnt > 0:
        userdata.delete = True
        userdata.delete_1 = False
        userdata.delete_cnt = 0
        userdata.btn_b_val = b_b
        remove(userdata)

    elif userdata.delete_1 == True and userdata.delete_cnt < 0:
        userdata.delete_1 = False
        userdata.delete_cnt = 0

    userdata.delete_cnt -= 1

def calculateProceed(userdata,b_b):
    if userdata.btn_b_holding == False and b_b == 1:
        userdata.btn_b_holding = True
        userdata.btn_b_cnt = 25
    elif userdata.btn_b_holding == True and b_b == 1 and userdata.btn_b_cnt > 0:
        userdata.btn_b_cnt -= 1
    elif userdata.btn_b_holding == True and b_b == 0 and userdata.btn_b_cnt > 0:
        userdata.btn_b_holding = False
        userdata.btn_b_cnt = 0
    elif userdata.btn_b_holding == True and userdata.btn_b_cnt <= 0:
        userdata.btn_b_holding = False
        userdata.btn_b_cnt = 0
        userdata.proceed  = True
        print("Finished recording motions")

    
    
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("test/imu")

def on_message(client, userdata, msg):
    if 'start' in str(msg.payload):
        return
    else:
        decoded_bytes = msg.payload.decode("utf-8")
        sensor_info = json.loads(decoded_bytes)
        #print(userdata.sampleN,userdata.DatafoldDir)
        if not userdata.proceed:
            button_a = sensor_info["data"][userdata.size+1]
            button_b = sensor_info["data"][userdata.size+2]
            button_b_curr = sensor_info["data"][userdata.size+3]
            calculateRecord(userdata,button_a,sensor_info["data"])
            calculateDelete(userdata,button_b)
            calculateProceed(userdata,button_b_curr)
        else:
            client.disconnect()

def on_publish(client, userdata, mid):
    print("Message published")

def on_subscribe(client, userdata, mid, granted_qos):
    print("Subscribed to topic")


parser = argparse.ArgumentParser(description="Train model to detect a new motion")
parser.add_argument("motionName", help="Folder name to store IMU data")
args = parser.parse_args()

data_directory = "../50hz/Data/"+args.motionName
model_directory = "../Models/"+args.motionName
path = Path(data_directory)
path.mkdir(parents=True, exist_ok=True)

# Create the directory
path2 = Path(model_directory)
path2.mkdir(parents=True, exist_ok=True)
cnt = 0
for x in path.iterdir():
    cnt+=1

client_id = f'publish-{random.randint(0, 1000)}'

user_data = Data(60,data_directory,model_directory,cnt,args.motionName)
client = mqtt.Client(client_id,userdata=user_data)

# Set the callback functions
client.on_connect = on_connect
client.on_message = on_message
client.on_publish = on_publish
client.on_subscribe = on_subscribe



client.connect("192.168.100.100", 1883)
client.loop_start()

# Keep the client running
try:
    while not user_data.proceed:
        time.sleep(0.01)

except KeyboardInterrupt:
    client.loop_stop()
    client.disconnect()


print("Processing Dataset")

if user_data.modified:
    df = createDataset()
    df.to_csv("df.csv")
else:
    df = pd.read_csv("df.csv",index_col=0)
print("Creating Model")


#Preprocess data before creating model
motions = [args.motionName]

model(motions,df,model_directory)

print("Done!")