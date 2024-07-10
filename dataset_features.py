import os
import pandas as pd
import numpy as np
import antropy as ant
from tsfresh.feature_extraction.feature_calculators import *
from scipy.signal import butter, lfilter

def createDataset(data_folder = "../50hz/Data", samplerate = 1/50, cutoff = 0.9):
    # Define filter parameters
    cutoff_frequency = 0.3  
    nyquist_frequency = 0.5 * (1/samplerate)
    normal_cutoff = cutoff_frequency / nyquist_frequency

    # Design a low-pass Butterworth filter
    b, a = butter(1, normal_cutoff, btype='low', analog=False)

    df_new = pd.DataFrame(columns=['motion_type','mean_accx','mean_accy','mean_accz','mean_gyrox','mean_gyroy','mean_gyroz','mean_roll','mean_pitch','mean_yaw','median_accx','median_accy','median_accz','median_gyrox','median_gyroy','median_gyroz','median_roll','median_pitch','median_yaw','variance_accx','variance_accy','variance_accz','variance_gyrox','variance_gyroy','variance_gyroz','variance_roll','variance_pitch','variance_yaw','rms_accx','rms_accy','rms_accz','rms_gyrox','rms_gyroy','rms_gyroz','rms_roll','rms_pitch','rms_yaw','zc_accx','zc_accy','zc_accz','zc_gyrox','zc_gyroy','zc_gyroz','zc_roll','zc_pitch','zc_yaw','sep_accx','sep_accy','sep_accz','roll_pitch_corr','roll_yaw_corr','pitch_yaw_corr','median_freq_accx','median_freq_accy','median_freq_accz','median_freq_gyrox','median_freq_gyroy','median_freq_gyroz','median_freq_roll','median_freq_pitch','median_freq_yaw','mean_freq_accx','mean_freq_accy','mean_freq_accz','mean_freq_gyrox','mean_freq_gyroy','mean_freq_gyroz','mean_freq_roll','mean_freq_pitch','mean_freq_yaw','spec_ent_accx','spec_ent_accy','spec_ent_accz','spec_ent_gyrox','spec_ent_gyroy','spec_ent_gyroz','spec_ent_roll','spec_ent_pitch','spec_ent_yaw'])

    root = os.getcwd()+"\\"+data_folder
    total = 0
    fs = 0
    for root, subdirs, files in os.walk(root):
        for f in files:
            if "csv" in f:
                fs += 1
                motion_type = root.split('\\')[-1]
                file_loc = root+"\\"+f

                df = pd.read_csv(file_loc,header = 0, names = ['accx','accy','accz','gyrox','gyroy','gyroz'])
                
                #calculate roll pitch yaw
                df.loc[0,'Roll'] = 0
                df.loc[0,'Pitch'] = 0
                df.loc[0,'Yaw'] = 0

                for i in range(len(df)):
                    total += 1
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
                
                #feauture extraction
                
                #mean
                mean_accx = mean(df[df.columns[0]].to_numpy())
                mean_accy = mean(df[df.columns[1]].to_numpy())
                mean_accz = mean(df[df.columns[2]].to_numpy())
                mean_gyrox = mean(df[df.columns[3]].to_numpy())
                mean_gyroy = mean(df[df.columns[4]].to_numpy())
                mean_gyroz = mean(df[df.columns[5]].to_numpy())
                mean_roll = mean(df[df.columns[6]].to_numpy())
                mean_pitch = mean(df[df.columns[7]].to_numpy())
                mean_yaw = mean(df[df.columns[8]].to_numpy())
                
                #median
                median_accx = median(df[df.columns[0]].to_numpy())
                median_accy = median(df[df.columns[1]].to_numpy())
                median_accz = median(df[df.columns[2]].to_numpy())
                median_gyrox = median(df[df.columns[3]].to_numpy())
                median_gyroy = median(df[df.columns[4]].to_numpy())
                median_gyroz = median(df[df.columns[5]].to_numpy())
                median_roll = median(df[df.columns[6]].to_numpy())
                median_pitch = median(df[df.columns[7]].to_numpy())
                median_yaw = median(df[df.columns[8]].to_numpy())
                
                #variance
                variance_accx = variance(df[df.columns[0]].to_numpy())
                variance_accy = variance(df[df.columns[1]].to_numpy())
                variance_accz = variance(df[df.columns[2]].to_numpy())
                variance_gyrox = variance(df[df.columns[3]].to_numpy())
                variance_gyroy = variance(df[df.columns[4]].to_numpy())
                variance_gyroz = variance(df[df.columns[5]].to_numpy())
                variance_roll = variance(df[df.columns[6]].to_numpy())
                variance_pitch = variance(df[df.columns[7]].to_numpy())
                variance_yaw = variance(df[df.columns[8]].to_numpy())
                
                #rms
                rms_accx = root_mean_square(df[df.columns[0]].to_numpy())
                rms_accy = root_mean_square(df[df.columns[1]].to_numpy())
                rms_accz = root_mean_square(df[df.columns[2]].to_numpy())
                rms_gyrox = root_mean_square(df[df.columns[3]].to_numpy())
                rms_gyroy = root_mean_square(df[df.columns[4]].to_numpy())
                rms_gyroz = root_mean_square(df[df.columns[5]].to_numpy())
                rms_roll = root_mean_square(df[df.columns[6]].to_numpy())
                rms_pitch = root_mean_square(df[df.columns[7]].to_numpy())
                rms_yaw = root_mean_square(df[df.columns[8]].to_numpy())
                
                #zero-crossings
                zc_accx = number_crossing_m(df[df.columns[0]].to_numpy(),0)
                zc_accy = number_crossing_m(df[df.columns[1]].to_numpy(),0)
                zc_accz = number_crossing_m(df[df.columns[2]].to_numpy(),0)
                zc_gyrox = number_crossing_m(df[df.columns[3]].to_numpy(),0)
                zc_gyroy = number_crossing_m(df[df.columns[4]].to_numpy(),0)
                zc_gyroz = number_crossing_m(df[df.columns[5]].to_numpy(),0)
                zc_roll = number_crossing_m(df[df.columns[6]].to_numpy(),0)
                zc_pitch = number_crossing_m(df[df.columns[7]].to_numpy(),0)
                zc_yaw = number_crossing_m(df[df.columns[8]].to_numpy(),0)
                
                ####Accelaration seperation
                
                # Apply the low-pass filter to separate gravity
                gravity_component_x = lfilter(b, a, df[df.columns[0]].to_numpy())
                gravity_component_y = lfilter(b, a, df[df.columns[1]].to_numpy())
                gravity_component_z = lfilter(b, a, df[df.columns[2]].to_numpy())

                # Subtract the gravity component to obtain the motion component
                motion_component_x = df[df.columns[0]].to_numpy() - gravity_component_x
                motion_component_y = df[df.columns[1]].to_numpy() - gravity_component_y
                motion_component_z = df[df.columns[2]].to_numpy() - gravity_component_z

                # You now have the separated gravity and motion components
                sep_accx = np.corrcoef(gravity_component_x,motion_component_x)[0][1]
                sep_accy = np.corrcoef(gravity_component_y,motion_component_y)[0][1]
                sep_accz = np.corrcoef(gravity_component_z,motion_component_z)[0][1]
                
                #Roll-Pitch-Yaw Correlations
                roll_pitch_corr = np.corrcoef(df[df.columns[6]].to_numpy(),df[df.columns[7]].to_numpy())[0][1]
                roll_yaw_corr = np.corrcoef(df[df.columns[6]].to_numpy(),df[df.columns[8]].to_numpy())[0][1]
                pitch_yaw_corr = np.corrcoef(df[df.columns[7]].to_numpy(),df[df.columns[8]].to_numpy())[0][1]
                
                #frequency domain
                freq_accx = np.fft.fft(df[df.columns[0]].to_numpy())
                freq_accy = np.fft.fft(df[df.columns[1]].to_numpy())
                freq_accz = np.fft.fft(df[df.columns[2]].to_numpy())
                freq_gyrox = np.fft.fft(df[df.columns[3]].to_numpy())
                freq_gyroy = np.fft.fft(df[df.columns[4]].to_numpy())
                freq_gyroz = np.fft.fft(df[df.columns[5]].to_numpy())
                freq_roll = np.fft.fft(df[df.columns[6]].to_numpy())
                freq_pitch = np.fft.fft(df[df.columns[7]].to_numpy())
                freq_yaw = np.fft.fft(df[df.columns[8]].to_numpy())
                
                #median frequency
                median_freq_accx = median(np.abs(freq_accx))
                median_freq_accy = median(np.abs(freq_accy))
                median_freq_accz = median(np.abs(freq_accz))
                median_freq_gyrox = median(np.abs(freq_gyrox))
                median_freq_gyroy = median(np.abs(freq_gyroy))
                median_freq_gyroz = median(np.abs(freq_gyroz))
                median_freq_roll = median(np.abs(freq_roll))
                median_freq_pitch = median(np.abs(freq_pitch))
                median_freq_yaw = median(np.abs(freq_yaw))
                
                #mean frequency
                mean_freq_accx = mean(np.abs(freq_accx))
                mean_freq_accy = mean(np.abs(freq_accy))
                mean_freq_accz = mean(np.abs(freq_accz))
                mean_freq_gyrox = mean(np.abs(freq_gyrox))
                mean_freq_gyroy = mean(np.abs(freq_gyroy))
                mean_freq_gyroz = mean(np.abs(freq_gyroz))
                mean_freq_roll = mean(np.abs(freq_roll))
                mean_freq_pitch = mean(np.abs(freq_pitch))
                mean_freq_yaw = mean(np.abs(freq_yaw))
                
                #spectral entropy
                spec_ent_accx = ant.spectral_entropy(df[df.columns[0]].to_numpy(), sf=10, method='welch')
                spec_ent_accy = ant.spectral_entropy(df[df.columns[1]].to_numpy(), sf=10, method='welch')
                spec_ent_accz = ant.spectral_entropy(df[df.columns[2]].to_numpy(), sf=10, method='welch')
                spec_ent_gyrox = ant.spectral_entropy(df[df.columns[3]].to_numpy(), sf=10, method='welch')
                spec_ent_gyroy = ant.spectral_entropy(df[df.columns[4]].to_numpy(), sf=10, method='welch')
                spec_ent_gyroz = ant.spectral_entropy(df[df.columns[5]].to_numpy(), sf=10, method='welch')
                spec_ent_roll = ant.spectral_entropy(df[df.columns[6]].to_numpy(), sf=10, method='welch')
                spec_ent_pitch = ant.spectral_entropy(df[df.columns[7]].to_numpy(), sf=10, method='welch')
                spec_ent_yaw = ant.spectral_entropy(df[df.columns[8]].to_numpy(), sf=10, method='welch')

                df_new.loc[len(df_new)] = [motion_type,mean_accx,mean_accy,mean_accz,mean_gyrox,mean_gyroy,mean_gyroz,mean_roll,mean_pitch,mean_yaw,median_accx,median_accy,median_accz,median_gyrox,median_gyroy,median_gyroz,median_roll,median_pitch,median_yaw,variance_accx,variance_accy,variance_accz,variance_gyrox,variance_gyroy,variance_gyroz,variance_roll,variance_pitch,variance_yaw,rms_accx,rms_accy,rms_accz,rms_gyrox,rms_gyroy,rms_gyroz,rms_roll,rms_pitch,rms_yaw,zc_accx,zc_accy,zc_accz,zc_gyrox,zc_gyroy,zc_gyroz,zc_roll,zc_pitch,zc_yaw,sep_accx,sep_accy,sep_accz,roll_pitch_corr,roll_yaw_corr,pitch_yaw_corr,median_freq_accx,median_freq_accy,median_freq_accz,median_freq_gyrox,median_freq_gyroy,median_freq_gyroz,median_freq_roll,median_freq_pitch,median_freq_yaw,mean_freq_accx,mean_freq_accy,mean_freq_accz,mean_freq_gyrox,mean_freq_gyroy,mean_freq_gyroz,mean_freq_roll,mean_freq_pitch,mean_freq_yaw,spec_ent_accx,spec_ent_accy,spec_ent_accz,spec_ent_gyrox,spec_ent_gyroy,spec_ent_gyroz,spec_ent_roll,spec_ent_pitch,spec_ent_yaw]
    return df_new

