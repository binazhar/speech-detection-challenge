# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 01:43:18 2019

@author: Aziz-Dqube
"""

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wavfile
import numpy
audio_path = 'C:\\Users\\Aziz-Dqube\\Desktop\\TMW\\Task1\\'
sample_rate, signal = wavfile.read(audio_path+'audio1.wav')  # WAV file 

frame_width = 0.025 #25ms
shift = 0.01 #10ms
ceptral_coefficients = 13 # 0 to 12th order 
number_filter = 27 # 27 Mel-spaced triangular band pass filters


#independent Variables
# MFCC with zeroth cepstral coefficient replaced with the log of the total frame energy.
mfcc_features = mfcc(signal,sample_rate, frame_width, shift, ceptral_coefficients, number_filter, 512, 0,sample_rate/2,0.97,22,1,winfunc=numpy.hamming)
logfbank_features = logfbank(signal,sample_rate, frame_width, shift, ceptral_coefficients, number_filter)

#Dependent / Derived Variables
delta_features = delta (mfcc_features, 2)
delta_delta_features = delta (delta_features, 2)

print (type(mfcc_features))
print (type(logfbank_features))

print (len(mfcc_features))
print (len(logfbank_features))

#print(logfbank_features[1:3,:])
print(mfcc_features[1:3,:])

# Combine MFFC features and logFBank features
audio_features = numpy.concatenate((mfcc_features,logfbank_features), axis=1) 

#Write feature set in file
numpy.savetxt('audio1.csv',audio_features,delimiter=',' )


############################################ Second Iteration - Better Feature Extraction ################################
# OpenSmile Toolkit
