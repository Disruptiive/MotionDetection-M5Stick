import time
from time import gmtime, strftime
import argparse
import random
import datetime
import json
import paho.mqtt.client as mqtt
import numpy as np
import joblib
import os
import pandas as pd
from pathlib import Path
from extract_features import extractFeatures2
from dtw import *
from create_onevsall_model import onevsall_model
from create_onevsall_model import multiclass_classifier
import warnings
warnings.filterwarnings("ignore")

def writeFile(roll,pitch,yaw,i,j,n):
    now = datetime.datetime.now().strftime("%d-%m-%Y-")
    now = now + str(i) + "-" + str(j) + "-" + str(n) + ".csv"
    with open(now,"w") as f:
        print("opened file",f.writable())
        for i in range(roll.size):
            data = str(roll[0,i]) + ", "+str(pitch[0,i]) + ", " + str(yaw[0,i]) + "\n"
            f.write(data)
        print("Done")

def getDF(title):

    file_loc = title
    samplerate = 1/50
    cutoff = 0.9
    df = pd.read_csv(file_loc,header = 0, names = ['accx','accy','accz','gyrox','gyroy','gyroz'])
    
    df.loc[0,'Roll'] = 0
    df.loc[0,'Pitch'] = 0
    df.loc[0,'Yaw'] = 0
    
    for i in range(len(df)):
        roll_acc = np.arctan2(df.loc[i,'accy'], df.loc[i,'accz'] + 0.05*df.loc[i,'accx'])
        pitch_acc = np.arctan2(-1*df.loc[i,'accx'], np.sqrt(np.square(df.loc[i,'accy']) + np.square(df.loc[i,'accz'])))
        roll_acc = np.degrees(roll_acc)
        pitch_acc = np.degrees(pitch_acc)
    
        if i != 0:
            roll_g =  df.loc[i-1,'Roll'] +  df.loc[i,'gyrox'] * samplerate
            pitch_g =  df.loc[i-1,'Pitch'] +  df.loc[i,'gyroy'] * samplerate
            yaw_g =  df.loc[i-1,'Yaw'] +  df.loc[i,'gyroz'] * samplerate
        else:
            roll_g =  df.loc[i,'gyrox'] * samplerate
            pitch_g = df.loc[i,'gyroy'] * samplerate
            yaw_g =   df.loc[i,'gyroz'] * samplerate
    
        df.loc[i,'Roll'] = roll_g * cutoff + roll_acc * (1-cutoff)
        df.loc[i,'Pitch'] = pitch_g * cutoff + pitch_acc * (1-cutoff)
        df.loc[i,'Yaw'] = yaw_g
    
    return df.loc[:,'Roll'],df.loc[:,'Pitch'],df.loc[:,'Yaw']

def calculateRPY(accx,accy,accz,gyrox,gyroy,gyroz,samplerate,cutoff,roll,pitch,yaw):
    prev_roll = 0
    prev_pitch = 0
    prev_yaw = 0


    for i in range(accx.size):
        roll_acc = np.arctan2(accy[0,i], accz[0,i] + 0.05*accx[0,i])
        pitch_acc = np.arctan2(-1*accx[0,i], np.sqrt(np.square(accy[0,i]) + np.square(accz[0,i])))
        roll_acc = np.degrees(roll_acc)
        pitch_acc = np.degrees(pitch_acc)
    
        if i > 0:
            roll_g =  prev_roll +  gyrox[0,i] * samplerate
            pitch_g =  prev_pitch +   gyroy[0,i] * samplerate
            yaw_g =  prev_yaw +   gyroz[0,i]* samplerate
        else:
            roll_g =  gyrox[0,i] * samplerate
            pitch_g = gyroy[0,i] * samplerate
            yaw_g =   gyroz[0,i] * samplerate
    
        prev_roll = roll_g * cutoff + roll_acc * (1-cutoff)
        prev_pitch = pitch_g * cutoff + pitch_acc * (1-cutoff)
        prev_yaw = yaw_g

        roll[0,i] = prev_roll
        pitch[0,i] = prev_pitch
        yaw[0,i] = prev_yaw

    return roll,pitch,yaw



class Data:
    def __init__(self,size,model,buffsize,tmp,scaler,motions):
        self.buffsize = buffsize
        self.template = tmp
        self.samplerate = 1/50
        self.cutoff = 0.9
        self.predict = True
        self.counter = 0
        self.timez = []
        self.model = model
        self.scaler = scaler
        self.accelx = np.zeros([1,size*50])
        self.accely = np.zeros([1,size*50])
        self.accelz = np.zeros([1,size*50])
        self.gyrox = np.zeros([1,size*50])
        self.gyroy = np.zeros([1,size*50])
        self.gyroz = np.zeros([1,size*50])
        self.roll = np.zeros([1,size*50])
        self.pitch = np.zeros([1,size*50])
        self.yaw = np.zeros([1,size*50])
        self.motions = motions
        self.x = []
        self.now = 0

def json_serializer(obj):
    if isinstance(obj, bytes):
        return obj.decode('utf-8')

    return obj

def dtwPipeline(userdata,i,rpy):
    if rpy == 0:
        alignment = dtw(userdata.template[i],userdata.roll,step_pattern=asymmetric,open_end=True,open_begin=True)
    elif rpy == 1:
        alignment = dtw(userdata.template[i],userdata.pitch,step_pattern=asymmetric,open_end=True,open_begin=True)
    else:
        alignment = dtw(userdata.template[i],userdata.yaw,step_pattern=asymmetric,open_end=True,open_begin=True)

    if alignment.index2[-1] > alignment.index2[0] + 5:
        features = extractFeatures2(userdata.accelx[0,alignment.index2[0]:alignment.index2[-1]],userdata.accely[0,alignment.index2[0]:alignment.index2[-1]],userdata.accelz[0,alignment.index2[0]:alignment.index2[-1]],userdata.gyrox[0,alignment.index2[0]:alignment.index2[-1]],userdata.gyroy[0,alignment.index2[0]:alignment.index2[-1]],userdata.gyroz[0,alignment.index2[0]:alignment.index2[-1]],userdata.roll[0,alignment.index2[0]:alignment.index2[-1]],userdata.pitch[0,alignment.index2[0]:alignment.index2[-1]],userdata.yaw[0,alignment.index2[0]:alignment.index2[-1]])
        features = np.nan_to_num(features)
        scaled_features = userdata.scaler.transform(features)
        userdata.x.append(scaled_features.reshape(78))

class CustomData:
    def __init__(self, name, value):
        self.name = name
        self.value = value

class CustomEncoder(json.JSONEncoder):
    def default(self, o):
            return o.__dict__

# Callback functions for MQTT events
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("test/imu")

def on_message(client, userdata, msg):
    if 'start' in str(msg.payload):
        return
    else:
        d = []
        decoded_bytes = msg.payload.decode("utf-8")
        sensor_info = json.loads(decoded_bytes)
        #print(time.time())
        #print(sensor_info["data"])
        #-------------
        userdata.accelx = np.roll(userdata.accelx, -10)
        userdata.accely = np.roll(userdata.accely, -10)
        userdata.accelz = np.roll(userdata.accelz, -10)
        userdata.gyrox = np.roll(userdata.gyrox, -10)
        userdata.gyroy = np.roll(userdata.gyroy, -10)
        userdata.gyroz = np.roll(userdata.gyroz, -10)

        for i in range(0,user_data.buffsize,6):
            userdata.gyrox[0,(-10+i//6)] = sensor_info["data"][i]
            userdata.gyroy[0,(-10+i//6)] = sensor_info["data"][i+1]
            userdata.gyroz[0,(-10+i//6)] = sensor_info["data"][i+2]
            userdata.accelx[0,(-10+i//6)] = sensor_info["data"][i+3]
            userdata.accely[0,(-10+i//6)] = sensor_info["data"][i+4]
            userdata.accelz[0,(-10+i//6)] = sensor_info["data"][i+5]
        if user_data.predict:
            t0 = time.time()
            userdata.roll,userdata.pitch,userdata.yaw = calculateRPY(userdata.accelx,userdata.accely,userdata.accelz,userdata.gyrox,userdata.gyroy,userdata.gyroz,userdata.samplerate,userdata.cutoff,userdata.roll,userdata.pitch,userdata.yaw)

            for i in range(len(motions)):
                    dtwPipeline(userdata,i*6+userdata.now*3,0)
                    dtwPipeline(userdata,i*6+userdata.now*3 + 1,1)
                    dtwPipeline(userdata,i*6+userdata.now*3,2)
            t1 = time.time()
            if userdata.x:
                d = userdata.model.predict(userdata.x)
            t2 = time.time()
            userdata.x = []
            userdata.now = 1 if userdata.now == 0 else 0
            #print("BEFORE: ", t1-t0)
            #print("AFTER :", t2-t1)
            #print(t2-t0)
            is_any_positive = [val-1  for val in d if val > 0]
            if is_any_positive:
                print(d)
                user_data.accelx[0,:] = 0
                user_data.accely[0,:] = 0
                user_data.accelz[0,:] = 0
                user_data.gyrox[0,:] = 0
                user_data.gyroy[0,:] = 0
                user_data.gyroz[0,:] = 0

                user_data.counter = 10
                user_data.predict = False
                #writeFile(userdata.roll,userdata.pitch,userdata.yaw,alignment.index2[0],alignment.index2[-1],2)
                curr_tm = strftime("%Y-%m-%d %H:%M:%S", gmtime())
                print("["+curr_tm+"]" + "Motion " + user_data.motions[is_any_positive[0]] + " detected")                   
            
        else:
            #print(str(user_data.counter) + " more ")
            user_data.counter -= 1
            if user_data.counter <=0:
                user_data.predict = True

def on_publish(client, userdata, mid):
    print("Message published")

def on_subscribe(client, userdata, mid, granted_qos):
    print("Subscribed to topic")


parser = argparse.ArgumentParser(description="Connects to MQTT and detects specific motion")
parser.add_argument("motionName", help="Motion to detect", nargs="*")
args = parser.parse_args()

motions = sorted(args.motionName)
delim = "_"
model_name = delim.join(motions)

data_directories = []
for motion in motions:
    data_directories.append("../50hz/Data/"+motion)
model_directory = "../Models/"+model_name

for data_dir in data_directories:
    if not os.path.exists(data_dir):
        error_msg = "No data found for "+ str(data_dir.split('/')[-1])+ ", add data for this motion first!"
        sys.exit(error_msg)
if not os.path.exists(model_directory):
    print("No model found, need to create it first")
    df = pd.read_csv("df.csv",index_col=0)
    #onevsall_model(motions,df,model_directory)
    multiclass_classifier(motions,df,model_directory)

filename_model = model_directory+ '/' + 'model.sav'
filename_scaler = model_directory+ '/' + 'scaler.sav'
loaded_model = joblib.load(filename_model)
loaded_scaler = joblib.load(filename_scaler)
print("Model loaded")
# Start the MQTT client
client_id = f'publish-{random.randint(0, 1000)}'

rpy = []
for data_dir in data_directories:
    path = Path(data_dir)
    cnt = 0
    for x in path.iterdir():
        cnt+=1
    values = list(range(1, cnt + 1))  # Create a list of all values in the range
    random_numbers = random.sample(values, 2)
    print(data_dir + '/' + str(random_numbers[0]) +".csv", data_dir + '/' + str(random_numbers[1]) +".csv")
    r,p,y = getDF(data_dir + '/' + str(random_numbers[0]) +".csv")
    r2,p2,y2 = getDF(data_dir + '/' + str(random_numbers[1]) +".csv")
    rpy.extend([r,p,y,r2,p2,y2])


user_data = Data(7,loaded_model,60,rpy,loaded_scaler,motions)
client = mqtt.Client(client_id,userdata=user_data)



# Set the callback functions
client.on_connect = on_connect
client.on_message = on_message
client.on_publish = on_publish
client.on_subscribe = on_subscribe


client.connect("192.168.100.102", 1883)
client.loop_start()

# Keep the client running
try:
    while True:
        time.sleep(0.01)

except KeyboardInterrupt:
    client.loop_stop()
    client.disconnect()


