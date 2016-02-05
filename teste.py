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

import pyaudio
import wave

from numpy.fft import fft, ifft
# from rec5sec_V1 import rec5sec
from Talkbox import mfcc,periodogram
from sklearn.mixture import GMM

filename = 'S_05_24.wav'

# File reading --------------------------------------
def read_audiofile(filename,normalize=True):
	fs, np_audio = spwave.read(filename)
	if normalize:
		np_audio = np_audio/float(np.sum(np_audio))
	return fs, np_audio

[fs, np_audio] = read_audiofile(filename,normalize=True)
plt.plot(np_audio,linewidth=3)
plt.show()