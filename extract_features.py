from scipy.signal import butter, lfilter
import numpy as np
import antropy as ant
from tsfresh.feature_extraction.feature_calculators import *

def extractFeatures2(accelx,accely,accelz,gyrox,gyroy,gyroz,roll,pitch,yaw):

    ##For gravity/motion accelaration seperation
    sampling_rate = 50
    # Define filter parameters
    cutoff_frequency = 0.3  
    nyquist_frequency = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist_frequency

    # Design a low-pass Butterworth filter
    b, a = butter(1, normal_cutoff, btype='low', analog=False)

    np_new = np.empty([1,78])
            
    #feauture extraction
            
    #mean
    np_new[0,0] = mean(accelx)
    np_new[0,1] = mean(accely)
    np_new[0,2] = mean(accelz)
    np_new[0,3] = mean(gyrox)
    np_new[0,4] = mean(gyroy)
    np_new[0,5] = mean(gyroz)
    np_new[0,6] = mean(roll)
    np_new[0,7] = mean(pitch)
    np_new[0,8] = mean(yaw)
            
    #median
    np_new[0,9] = median(accelx)
    np_new[0,10] = median(accely)
    np_new[0,11] = median(accelz)
    np_new[0,12] = median(gyrox)
    np_new[0,13] = median(gyroy)
    np_new[0,14] = median(gyroz)
    np_new[0,15] = median(roll)
    np_new[0,16] = median(pitch)
    np_new[0,17] = median(yaw)
            
    #variance
    np_new[0,18] = variance(accelx)
    np_new[0,19] = variance(accely)
    np_new[0,20] = variance(accelz)
    np_new[0,21] = variance(gyrox)
    np_new[0,22] = variance(gyroy)
    np_new[0,23] = variance(gyroz)
    np_new[0,24] = variance(roll)
    np_new[0,25] = variance(pitch)
    np_new[0,26] = variance(yaw)
            
    #rms
    np_new[0,27] = root_mean_square(accelx)
    np_new[0,28] = root_mean_square(accely)
    np_new[0,29] = root_mean_square(accelz)
    np_new[0,30] = root_mean_square(gyrox)
    np_new[0,31] = root_mean_square(gyroy)
    np_new[0,32] = root_mean_square(gyroz)
    np_new[0,33] = root_mean_square(roll)
    np_new[0,34] = root_mean_square(pitch)
    np_new[0,35] = root_mean_square(yaw)
            
    #zero-crossings
    np_new[0,36] = number_crossing_m(accelx,0)
    np_new[0,37] = number_crossing_m(accely,0)
    np_new[0,38] = number_crossing_m(accelz,0)
    np_new[0,39] = number_crossing_m(gyrox,0)
    np_new[0,40] = number_crossing_m(gyroy,0)
    np_new[0,41] = number_crossing_m(gyroz,0)
    np_new[0,42] = number_crossing_m(roll,0)
    np_new[0,43] = number_crossing_m(pitch,0)
    np_new[0,44] = number_crossing_m(yaw,0)
            
    ####Accelaration seperation
            
    # Apply the low-pass filter to separate gravity
    gravity_component_x = lfilter(b, a, accelx)
    gravity_component_y = lfilter(b, a, accely)
    gravity_component_z = lfilter(b, a, accelz)

    # Subtract the gravity component to obtain the motion component
    motion_component_x = accelx - gravity_component_x
    motion_component_y = accely - gravity_component_y
    motion_component_z = accelz - gravity_component_z

    # You now have the separated gravity and motion components
    np_new[0,45] = np.corrcoef(gravity_component_x,motion_component_x)[0][1]
    np_new[0,46] = np.corrcoef(gravity_component_y,motion_component_y)[0][1]
    np_new[0,47] = np.corrcoef(gravity_component_z,motion_component_z)[0][1]
            
    #Roll-Pitch-Yaw Correlations
    np_new[0,48] = np.corrcoef(roll,pitch)[0][1]
    np_new[0,49] = np.corrcoef(roll,yaw)[0][1]
    np_new[0,50] = np.corrcoef(pitch,yaw)[0][1]
            
    #frequency domain
    freq_accx = np.fft.fft(accelx)
    freq_accy = np.fft.fft(accely)
    freq_accz = np.fft.fft(accelz)
    freq_gyrox = np.fft.fft(gyrox)
    freq_gyroy = np.fft.fft(gyroy)
    freq_gyroz = np.fft.fft(gyroz)
    freq_roll = np.fft.fft(roll)
    freq_pitch = np.fft.fft(pitch)
    freq_yaw = np.fft.fft(yaw)
            
    #median frequency
    np_new[0,51] = median(np.abs(freq_accx))
    np_new[0,52] = median(np.abs(freq_accy))
    np_new[0,53] = median(np.abs(freq_accz))
    np_new[0,54] = median(np.abs(freq_gyrox))
    np_new[0,55] = median(np.abs(freq_gyroy))
    np_new[0,56] = median(np.abs(freq_gyroz))
    np_new[0,57] = median(np.abs(freq_roll))
    np_new[0,58] = median(np.abs(freq_pitch))
    np_new[0,59] = median(np.abs(freq_yaw))
            
    #mean frequency
    np_new[0,60] = mean(np.abs(freq_accx))
    np_new[0,61] = mean(np.abs(freq_accy))
    np_new[0,62] = mean(np.abs(freq_accz))
    np_new[0,63] = mean(np.abs(freq_gyrox))
    np_new[0,64] = mean(np.abs(freq_gyroy))
    np_new[0,65] = mean(np.abs(freq_gyroz))
    np_new[0,66] = mean(np.abs(freq_roll))
    np_new[0,67] = mean(np.abs(freq_pitch))
    np_new[0,68] = mean(np.abs(freq_yaw))
            
    #spectral entropy
    np_new[0,69] = ant.spectral_entropy(accelx, sf=10, method='welch')
    np_new[0,70] = ant.spectral_entropy(accely, sf=10, method='welch')
    np_new[0,71] = ant.spectral_entropy(accelz, sf=10, method='welch')
    np_new[0,72] = ant.spectral_entropy(gyrox, sf=10, method='welch')
    np_new[0,73] = ant.spectral_entropy(gyroy, sf=10, method='welch')
    np_new[0,74] = ant.spectral_entropy(gyroz, sf=10, method='welch')
    np_new[0,75] = ant.spectral_entropy(roll, sf=10, method='welch')
    np_new[0,76] = ant.spectral_entropy(pitch, sf=10, method='welch')
    np_new[0,77] = ant.spectral_entropy(yaw, sf=10, method='welch')

    return np_new