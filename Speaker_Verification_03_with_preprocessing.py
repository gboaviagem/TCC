#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Neste teste de verificação um único locutor é tido como verdadeiro (M_01_24) e
dez amostras do mesmo locutor M_01_24 (agora com prefixo S) são testadas,
para ver se o sistema acerca as 10.
'''

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.io.wavfile as spwave

from numpy.fft import fft, ifft
from rec5sec_V1 import rec5sec
from Talkbox import mfcc,periodogram
from UBM_generation_with_preprocessing import get_UBM_all

from sklearn.mixture import GMM
from sklearn import preprocessing

os.system('clear')

# ======================================================================
# TESTING WITH ALL SPEAKERS AGAINST A SINGLE TRUE SPEAKER

true_speaker = 'M_01_24.wav'
	
'''
----------------------------------------------------
Training UBM for all TRAINING speakers (the ones with UBM in the name)
----------------------------------------------------
'''
cov_type = 'full'
UBM_all = get_UBM_all(cov_type,true_speaker)

'''
----------------------------------------------------
Setting up .txt files and lists
----------------------------------------------------
'''
files_in_folder = os.listdir(os.getcwd()) # List of files in current directory
audiofiles_S = [x for x in files_in_folder if ('.wav' in x) and ('S' in x) and ('UBM' not in x)]

f = open('Verification_Test_03_' + true_speaker.split('.')[0] + '.txt','w')
f.write('# Header: a single ASV test is carried out for each speaker S, with the speaker ' + true_speaker.split('.')[0] + ' being the True Speaker in all tests.\n# GENDER\tINDEX\tAGE\tSCORE\n')

for current_test_speaker in audiofiles_S:
	print '\nTEST 02 (all against one)'
	print 'For file ' + current_test_speaker + ':'

	[gender,index,aux] = current_test_speaker.split('_') # "aux" is just a disposable variable
	[age,aux] = aux.split('.')

	'''
	----------------------------------------------------
	Training true speaker
	----------------------------------------------------
	'''
	fs, audio_true = spwave.read(true_speaker)
	# ------------------
	# Energy normalization
	# ------------------
	audio_true = audio_true/float(np.sum(audio_true))

	t_frame = 20*10**(-3) # Duration in seconds of each frame
	nwin = t_frame*fs
	# nwin is the number of samples per frame.
	# Para t_frame=20ms e fs=16kHz, nwin=320
	nfft = 512
	nceps = 26

	ceps_true, mspec, spec = mfcc(audio_true, nwin, nfft, fs, nceps)
	
	# ------------------
	# Standardization
	# ------------------
	ceps_true = preprocessing.normalize(ceps_true, norm='l2')
# 	ceps_true = preprocessing.scale(ceps_true, axis=0, with_mean=True, with_std=False, copy=True)
	
	ngaussians = 10
	cov_type = 'full'
	
	gmm = GMM(n_components = ngaussians, covariance_type = cov_type)
	model_true = gmm.fit(ceps_true)
	print "model_true converged? ",model_true.converged_

	'''
	----------------------------------------------------
	Test-speech parameterization
	----------------------------------------------------
	'''
	fs, audio_test = spwave.read(current_test_speaker)
	# ------------------
	# Energy normalization
	# ------------------
	audio_test = audio_true/float(np.sum(audio_test))

	nwin = t_frame*fs
	# nwin is the number of samples per frame.
	# Para t_frame=20ms e fs=16kHz, nwin=320

	ceps_test, mspec, spec = mfcc(audio_test, nwin, nfft, fs, nceps)
	
	# ------------------
	# Standardization
	# ------------------
	ceps_test = preprocessing.normalize(ceps_test, norm='l2')
# 	ceps_test = preprocessing.scale(ceps_test, axis=0, with_mean=True, with_std=False, copy=True)


	'''
	----------------------------------------------------
	Scoring
	----------------------------------------------------
	'''
	log_prob_true = model_true.score(ceps_test)
	log_prob_UBM = UBM_all.score(ceps_test)

	print np.sum(log_prob_true)
	print np.sum(log_prob_UBM)
	score = np.sum(log_prob_true) - np.sum(log_prob_UBM)
	print "Difference between sum(log) probabilites (True VS. UBM_all): ", score

	f.write(gender + '\t' + index + '\t' + age + '\t' + str(score) + '\n')
f.close()