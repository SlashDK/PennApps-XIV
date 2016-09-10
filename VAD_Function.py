#### Initializations ####

modulePath = '' ####### Set Module Path of Christian's Python Library
import sys
sys.path.append(modulePath)

#Imports dependencies
import csv
import os
import scipy.io.wavfile
import numpy as np
from scipy.fftpack import fft

import generalUtility
import dspUtil
import matplotlibUtil




def CustomVAD(Wav_File_Full_Path):


    #### Data Aquisition ####
    for f in wav_data:
        fs, wav = scipy.io.wavfile.read(Wav_File_Full_Path)

    #### Data Windowing ####
    #20ms Hamming Window Weights
    N = (fs/1000) * 20                    #Number of samples in 20ms
    Weight = np.hamming(N)                #Hamming window weights

    #20ms Windowing (10ms overlap) with Hamming Weights
    step = N/2                            #Number of values in a 10ms window
    end = len(wav)-(N/2)-1                #Starting point of the last window

    Windows = []
    for i in range(0,end,step):
        temp = wav[i:i+N]
        temp = np.multiply(temp,Weight)   #Applies Hamming weights to windows
        Windows.append(temp)

    Windows = np.array(Windows)
    Windows = Windows[1:len(Windows)]


    #### Data Pre-Processing #### 

    S = 1.00/float(fs)                                    #Sample Spacing
    xf = np.linspace(0.0, 1.0/(2.0*S), N/2)               #FFT List Index to Frequency Relationships (in Hz)

    ####Finding Mel-Scale Frequency Per Index of fft vector (Transforming ^ into Mel-Scale)####
    def MelTransform(FHz): #Transforms Frequency in Hz to Mel-Scale Frequency
        FMel = (1000/np.log(2))*(1+(FHz/1000))
        return FMel

    xMel = []
    for i in xf:
        temp = MelTransform(i)
        xMel.append(temp)

    ####Finding index of Frequency range between: 0kHz - 4kHz####
    ####print xMel                                           ####
    ####for i,val in enumerate(xMel):                        ####
    ####    if val > 4000:                                   ####
    ####        print i                                      ####
    ######## INDEX for 0-4kHz == [0:35] in fft vector############ 


    ####Log Spectral Energy of Each Window####

    #For quick evaluation*
    # j = [] #global min
    # k = [] #global max

    #Loop Preallocations:
    windows = []
    for i in range(0,len(Windows)):
        temp = Windows[i]
        temp = fft(temp)
        temp = temp[0:(len(temp)/2)]   #Restrict Range to Single Sided Spectrum
        temp = temp[0:35]              #Evaluates only the values in the 0-4kHz Mel-Scale frequency range (valaid vocal ranges)
        temp = np.absolute(temp)
        temp = np.square(temp)
        temp = np.log10(temp)
        windows.append(temp)

    #* Quick Evaluation (cont)       
    #     temp2 = min(windows[i])
    #     temp3 = max(windows[i])
    #     j.append(temp2)
    #     k.append(temp3)
    # print min(j)
    # print max(k)


    #Divide each Window into 5 smaller Sub-Bands
    def chunks(l, n):
        n = max(1, n)
        return [l[i:i + n] for i in range(0, len(l), n)]

    windows = [chunks(x,7) for x in windows]
    #^^^ This "7" will give you a lot of problems later, find a better way to define Size of sub-bands SOON.



    ####Speech Detection Algorithm Initialization####
    #Base Threshold (T)
    list = windows[0][0]
    for x in range(1,5):
        list = np.append(list, windows[0][x],0)

    T = np.mean(list)
    Ts = T                           #Speech Threshold
    Tp = 1.2*T                       #Pause Threshold

    #Identify first "Silent Segment"
    first_silent = []
    for i in range(1,len(windows)):
        val_present = len([x for x in windows[i][0] if x<T])
        if  val_present > 0:
            first_silent.append(i)
            break

    ####Calculate Speech to Noise Ratio (SNR)####

    #Mean of all prior speech sections
    Prior_list = np.array([])
    for i in range(0,first_silent[0]):
        temp_list = windows[i][0]
        for x in range(1,5):
            temp_list = np.append(list, windows[0][x],0)
        Prior_list = np.append(Prior_list,temp_list,0)
    Prior_mean = np.mean(Prior_list)

    #Mean of first "Silent Section"
    Silent_mean = np.mean(windows[first_silent[0]][:])

    #SNR calculation (lambd)
    lambd = Prior_mean / Silent_mean


    ####Iterative Thresholding Re-Classification Function####
    def Thresh(Mean,Tp_):
        if (Mean < Tp_):
            new_Ts = Mean
            new_Tp = 1.2 * new_Ts
            return new_Ts , new_Tp
        return None

    def GammaClassify(Frame_mean,Ts_,Tp_):
        if Frame_mean < Tp_:
            Gamma = 1.6             #Use "Non-Speech" gamma
        if Frame_mean > Ts_:
            Gamma = 1               #Use "Speech" gamma
        return Gamma

    #### Algorithm Computation ####


    ####Re-thresholding and Gamma Calculations####
    mean_arr = []
    for data in windows:
        sum_agg = 0
        for x in data:
            sum_agg+= sum(x) /  float(len(windows[0])*len((windows[0][0])))
        mean_arr.append(sum_agg)

    Gamma_list = []
    moving_ts = Ts
    moving_tp = Tp
    tp_list = []
    for i,data in enumerate(mean_arr):
        Gamma = GammaClassify(data,moving_ts,moving_tp)        #Gamma Classification for Tk Calculation
        Gamma_list.append(Gamma)
        thresh = Thresh(data,moving_tp) 
        if thresh != None:
            (moving_ts,moving_tp) = thresh                     #Re-Thresholding of Ts and Tp
        tp_list.append(moving_tp)

    #### Threshold Correction (Fc) from Spectral Flatness Function
    Fc_windows = []
    for i in range(0,len(Windows)):
        temp = Windows[i]
        temp = fft(temp)
        temp = temp[0:(len(temp)/2)]      #Restrict Range to Single Sided Spectrum
        temp = temp[0:35]                 #Evaluates only the values in the 0-4kHz Mel-Scale frequency range (valaid vocal ranges)
        temp = np.absolute(temp)
        temp = dspUtil.calculateSpectralFlatness(temp)   #Computes Fc from Spectral Flatness Function
        Fc_windows.append(temp)



    #### Tk Calculations ####   
    first_pause_window_index = None
    for i,gamma in enumerate(Gamma_list):
        if gamma==1.6:
            first_pause_window_index = i
            break


    def final_classifier(Tk,Tp):
        if Tk > Tp:
            return 1
        return 0

    last_pause_mean = np.mean(windows[first_pause_window_index][0])
    # Skip the first pause window
    first_pause_window_index+=1
    results = []
    for i in range(first_pause_window_index,len(Gamma_list)):      
            if Gamma_list[i] == 1.6:
                last_pause_mean = np.mean(windows[i][0])
            Tk = lambd*last_pause_mean + (Gamma_list[i])*(1 - Fc_windows[i])
            results.append(final_classifier(Tk,tp_list[i]))


    #### Time at Each Classified Section ####
    Time_Total = (len(wav)/fs) * 1000
    end_time = Time_Total
    start_time = Time_Total - (len(results)*10)

    Time_Vect = range(start_time, end_time,10)

    Results = zip(results, Time_Vect)
    # print Results
    
    CVAD = []
    for i in range (0,(Time_Vect[0]/10)):
        temp = 0
        CVAD.append(temp)
    CVAD.extend(results)

    #### Quick-View Results ####
    # print wav_data[Q]

    # print sum(results)
    # print len(results)
    # print results
    
    return CVAD