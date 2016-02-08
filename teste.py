'''
UFPE - DES
Functions regarding Automatic Speaker Verification with GMM-UBM approach.

Sample rate of recording functions: 8kHz
Number of ceps: 26
'''

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.io.wavfile as spwave
from ASV import *

import pyaudio
import wave

from numpy.fft import fft, ifft
# from rec5sec_V1 import rec5sec
from Talkbox import mfcc,periodogram
from sklearn.mixture import GMM

filename = '02_true_speaker.wav'


[fs, np_audio] = read_audiofile(filename,normalize=True)


# --------------------------------------------
# MFCC calculation
t_frame = 20*10**(-3) # Duration in seconds of each frame
nwin = t_frame*fs
# nwin is the number of samples per frame.
# Para t_frame=20ms e fs=16kHz, nwin=320
nfft = 512

ceps, mspec, spec = mfcc(np_audio, nwin, nfft, fs, nceps)
[nframes, ncolumns] = ceps.shape
# print ceps[0]
# print ceps[1]
# print ceps[2]
# print np.mean(ceps[:,0],axis=0) # M
# print np.mean(ceps[:,1],axis=0)
# print np.mean(ceps[:,2],axis=0)

# print np.mean(ceps,axis=0) # Mean along the vertical dimension

print mspec.shape
print np.mean(mspec,axis=0)

mean_along_frames = np.mean(mspec,axis=0) # Mean along the vertical dimension of the mel-spectrum

mean_along_frames_stack = mean_along_frames
for i in range(nframes-1):
	mean_along_frames_stack = np.vstack((mean_along_frames_stack, mean_along_frames))

print ceps.shape
print mean_along_frames_stack.shape

ceps = ceps - mean_along_frames_stack[:,0:nceps]

print ceps[0]
print ceps[130]