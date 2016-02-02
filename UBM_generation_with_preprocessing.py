'''
UBM generation with standardization of data: audio signals are energy-normalized and cepstra are scaled and normalized.
'''

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.io.wavfile as spwave

from numpy.fft import fft, ifft
from rec5sec_V1 import rec5sec
from Talkbox import mfcc,periodogram

from sklearn.mixture import GMM
from sklearn import preprocessing


from matplotlib.backends.backend_pdf import PdfPages

def gaussian(x, mu, sig):
    return (1./(np.sqrt(2.*np.pi)*sig))*np.exp(-np.power((x - mu)/sig, 2.)/2)

def get_ceps(str_gender,exclude_speaker):
	
	'''
	----------------------------------------------------
	Concatenating audio files
	----------------------------------------------------
	'''
	files_in_folder = os.listdir(os.getcwd()) # List of files in current directory
	audiofiles_male = [x for x in files_in_folder if ('.wav' in x) and ('M' in x) and (x != exclude_speaker) and ('UBM' in x)]
	audiofiles_female = [x for x in files_in_folder if ('.wav' in x) and ('F' in x) and (x != exclude_speaker) and ('UBM' in x)]

	np_audio_male = np.array([])
	np_audio_female = np.array([])

	for filename in audiofiles_male:
		fs, audio = spwave.read(filename)
		# ------------------
		# Energy normalization
		# ------------------
		audio = audio/float(np.sum(audio))
		np_audio_male = np.concatenate((np_audio_male, audio))

	for filename in audiofiles_female:
		fs, audio = spwave.read(filename)
		# ------------------
		# Energy normalization
		# ------------------
		audio = audio/float(np.sum(audio))
		np_audio_female = np.concatenate((np_audio_female, audio))
	
	np_audio_all = np.concatenate((np_audio_male,np_audio_female))

	'''
	----------------------------------------------------
	Computing MFCC
	----------------------------------------------------
	'''
	t_frame = 20*10**(-3) # Duration in seconds of each frame

	nwin = t_frame*fs
	# nwin is the number of samples per frame.
	# Para t_frame=20ms e fs=16kHz, nwin=320
	nfft = 512
	nceps = 26
	
	if str_gender == 'male':
		ceps, mspec, spec = mfcc(np_audio_male, nwin, nfft, fs, nceps)
	if str_gender == 'female':
		ceps, mspec, spec = mfcc(np_audio_female, nwin, nfft, fs, nceps)
	else:
		ceps, mspec, spec = mfcc(np_audio_all, nwin, nfft, fs, nceps)
	
	# Normalization of MFCC
	ceps = preprocessing.normalize(ceps, norm='l2')
# 	ceps = preprocessing.scale(ceps, axis=0, with_mean=True, with_std=False, copy=True)
	
	return ceps

def get_UBM_all(cov_type,exclude_speaker):
	# The variable exclude_speaker holds the file of the True Speaker
	# in the current test, and so this file is not considered in the
	# UBM construction.
	
	ceps_all = get_ceps('all',exclude_speaker)
	
	'''
	----------------------------------------------------
	Generating UBM-GMM
	----------------------------------------------------
	'''
	ngaussians = 10
	
	gmm = GMM(n_components=ngaussians, covariance_type = cov_type)
	model_all = gmm.fit(ceps_all)
	print "model_all converged? ",model_all.converged_
	
	return model_all

def get_UBM_female(cov_type,exclude_speaker):
	# The variable exclude_speaker holds the file of the True Speaker
	# in the current test, and so this file is not considered in the
	# UBM construction.
	
	ceps_female = get_ceps('female',exclude_speaker)
	
	'''
	----------------------------------------------------
	Generating UBM-GMM
	----------------------------------------------------
	'''
	ngaussians = 10
	
	gmm = GMM(n_components=ngaussians, covariance_type = cov_type)
	model_female = gmm.fit(ceps_female)
	print "model_female converged? ",model_female.converged_
	
	return model_female

def get_UBM_male(cov_type,exclude_speaker):
	# The variable exclude_speaker holds the file of the True Speaker
	# in the current test, and so this file is not considered in the
	# UBM construction.
	
	ceps_male = get_ceps('male',exclude_speaker)
	
	'''
	----------------------------------------------------
	Generating UBM-GMM
	----------------------------------------------------
	'''
	ngaussians = 10
	
	gmm = GMM(n_components=ngaussians, covariance_type = cov_type)
	model_male = gmm.fit(ceps_male)
	print "model_male converged? ",model_male.converged_
	
	return model_male
