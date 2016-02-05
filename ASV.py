'''
UFPE - DES
Functions regarding Automatic Speaker Verification with GMM-UBM approach.

In this version, all parameterization steps consist only in reading the MFCC from the .txt files.
'''

# CONSTANTS ------------------------------------
nceps = 25
RATE = 8000 # Sample rate


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


current_folder = os.getcwd()
text_dependency = current_folder.split('/')[-1] # Takes the name of the enclosing folder, if it is TD (text dependent) or TI (text independent)


# Recording -----------------------------------------
def rec5sec(output_file_name,save_ceps=True):
	print '\nSample rate: ', RATE
	CHANNELS = 1
	
	if sys.platform=='linux2':
		# For when using in Raspbian
		os.system("arecord -D plughw:0,0 -d 5 -f S16_LE -c " + str(CHANNELS) + " -r " + str(RATE) + " ./" + output_file_name)
		if save_ceps:
			audio2ceps(output_file_name) # Saves a .txt file with the MFCC of the file
	else:
		# For another platforms, including Mac OS.
		FORMAT = pyaudio.paInt16 # 16-bits integers
		CHUNK = 1024
		RECORD_SECONDS = 5
		WAVE_OUTPUT_FILENAME = output_file_name # Must contain ".wav" 
		input_device = 2 # Index of my microfone "C-Media USB Audio Device"
		
		# ATTENTION: the input_device variable depends on the PC one runs this code!
		
		audio = pyaudio.PyAudio()
 
		# start Recording
		stream = audio.open(format=FORMAT, channels=CHANNELS,
						rate=RATE, input=True, input_device_index = input_device,
						frames_per_buffer=CHUNK)
		print "recording..."
		frames = []
 
		for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
			data = stream.read(CHUNK)
			frames.append(data)
		print "finished recording"
 
 
		# stop Recording
		stream.stop_stream()
		stream.close()
		audio.terminate()
 
		waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
		waveFile.setnchannels(CHANNELS)
		waveFile.setsampwidth(audio.get_sample_size(FORMAT))
		waveFile.setframerate(RATE)
		waveFile.writeframes(b''.join(frames))
		waveFile.close()
		
		if save_ceps:
			audio2ceps(output_file_name) # Saves a .txt file with the MFCC of the file


def record_true_speaker():
	'''
	Records 3 elocutions of 5 seconds and assigns them to the true speaker.
	'''
	
	for i in range(3):
		print '\n'
		print "True Speaker, recording #" + str(i+1) + ".\n"
	
		outputfile = '0' + str(i+1) + '_true_speaker.wav' # Choosing name of output file to be created
		rec5sec(outputfile)
		print outputfile + "\n"


def record_test_speaker():
	'''
	Function to be used in the operating phase, to record the speaker claiming to be the True Speaker.
	'''
	print '\n'
	print "Recording the test speaker.\n"

	outputfile = '00_test_speaker.wav' # Choosing name of output file to be created
	rec5sec(outputfile)
	print outputfile + "\n"


def record_for_threshold_calculation():
	'''
	This function records 3 elocutions from the True Speaker, tests them against
	the true speaker model and sets the lowest score as the threshold.
	'''
	
	for i in range(3):
		print '\n'
		print "True Speaker, audio for threshold, recording #" + str(i+1) + ".\n"
	
		outputfile = '0' + str(i+1) + '_threshold_audio.wav' # Choosing name of output file to be created
		rec5sec(outputfile)
		print outputfile + "\n"



# File reading --------------------------------------
def read_audiofile(filename,normalize=True):
	fs, np_audio = spwave.read(filename)
	if normalize:
		np_audio = np_audio/float(np.sum(np_audio))
	return fs, np_audio


def audio2ceps(filename,flag_normalize=True):
	'''
	Reads an audio file and creates a text file with its MFCC.
	
	Problem to fix: in the Raspberry, the function np.savetxt somehow doesn't accept the keyword 'header'.
	'''
	
	file = filename.split('.')[0] # Separates file name and extension
	
	# --------------------------------------------
	# File reading into numpy array	
	[fs, np_audio] = read_audiofile(filename,normalize = flag_normalize)
	
	# --------------------------------------------
	# MFCC calculation
	t_frame = 20*10**(-3) # Duration in seconds of each frame
	nwin = t_frame*fs
	# nwin is the number of samples per frame.
	# Para t_frame=20ms e fs=16kHz, nwin=320
	nfft = 512
	
	ceps, mspec, spec = mfcc(np_audio, nwin, nfft, fs, nceps)
	[nframes, ncolumns] = ceps.shape
	
	# --------------------------------------------
	# Text file creation
	str_header = 'MFCC from file ' + filename + '.\nInfo:\n\tSample rate: ' + str(RATE) + '\n\tNumber of MFCC per frame: ' + str(nceps) + '\n\tNumber of frames (samples): ' + str(nframes) + '\n\n'
	if sys.platform=='linux2': # in Raspbian
		np.savetxt(file + '_ceps.txt',ceps)
	else:
		np.savetxt(file + '_ceps.txt',ceps,header=str_header)


def convert_all_audiofile2ceps(normalize=True):
	'''
	Converts all recorded audio files from .wav to .txt, saving their MFCC.
	'''
	files_in_folder = os.listdir(os.getcwd()) # List of files in current directory
	audiofiles = [x for x in files_in_folder if ('.wav' in x)]
	
	for file in audiofiles:
		audio2ceps(file,flag_normalize = normalize)
	

def get_np_audiofiles(UBM=False, TS=False, normalize=True, exclude_speaker=None):
	'''
	Returns all the audio files for male and female elocutions in Numpy-array form.
	
	- UBM: if True, indicates that the audio files to be read are the ones separated exclusively for construction of the UBM.
	- TS: if True, only the files from the True Speaker are read.
	- normalize: if True, the audio files are energy-normalized.
	'''
	files_in_folder = os.listdir(os.getcwd()) # List of files in current directory
	
	# -----------------------------
	# Listing only the audio files
	# -----------------------------
	# Files just from the UBM ----------------------------
	if UBM:
		audiofiles_male = [x for x in files_in_folder if ('.wav' in x) and ('M' in x) and ('UBM' in x) and ('_test' not in x) and ('_true' not in x)]
		audiofiles_female = [x for x in files_in_folder if ('.wav' in x) and ('F' in x) and ('UBM' in x) and ('_test' not in x) and ('_true' not in x)]
		if exclude_speaker in audiofiles_male:
			audiofiles_male.remove(exclude_speaker)
		if exclude_speaker in audiofiles_female:
			audiofiles_female.remove(exclude_speaker)
		
		print 'Number of files for UBM: ', len(audiofiles_male) + len(audiofiles_female)
	
	# Files just from the true speaker ----------------------------
	if TS:
		audiofiles = [x for x in files_in_folder if ('.wav' in x) and ('_true' in x) and ('_test' not in x)]
		if exclude_speaker in audiofiles:
			audiofiles.remove(exclude_speaker)
		
		print 'Number of files for True Speaker GMM: ', len(audiofiles)
	
	# Files neither from the true speaker nor the UBM ----------------------------
	if ((not UBM) and (not TS)):
		audiofiles_male = [x for x in files_in_folder if ('.wav' in x) and ('M' in x) and ('UBM' not in x) and ('_true' not in x) and ('_test' not in x)]
		audiofiles_female = [x for x in files_in_folder if ('.wav' in x) and ('F' in x) and ('UBM' not in x) and ('_true' not in x) and ('_test' not in x)]
		if exclude_speaker in audiofiles_male:
			audiofiles_male.remove(exclude_speaker)
		if exclude_speaker in audiofiles_female:
			audiofiles_female.remove(exclude_speaker)

	
	if TS:
		np_audio_true = np.array([])
		for filename in audiofiles:
			[fs, np_audio] = read_audiofile(filename,normalize)
			np_audio_true = np.concatenate((np_audio_true, np_audio))
		return fs, np_audio_true
	else:
		np_audio_male = np.array([])
		np_audio_female = np.array([])

		# -----------------------------
		# Reading audio files into numpy arrays
		# -----------------------------
		for filename in audiofiles_male:
			[fs, np_audio] = read_audiofile(filename,normalize)
			np_audio_male = np.concatenate((np_audio_male, np_audio))

		for filename in audiofiles_female:
			[fs, np_audio] = read_audiofile(filename,normalize)
			np_audio_female = np.concatenate((np_audio_female, np_audio))
	
		np_audio_all = np.concatenate((np_audio_male,np_audio_female))	
		return fs, np_audio_male, np_audio_female, np_audio_all


# Parameterization ----------------------------------
def get_ceps_UBM():
	files_in_folder = os.listdir(os.getcwd()) # List of files in current directory
	
	# Files just from the UBM ----------------------------
	cepsfiles = [x for x in files_in_folder if ('UBM_ceps.txt' in x)]
	
	ceps = np.loadtxt(cepsfiles[0])
	
	for i in range(len(cepsfiles) - 1):
		current_ceps = np.loadtxt(cepsfiles[i + 1])
		ceps = np.vstack((ceps,current_ceps))
	
	return ceps


def get_ceps_true_speaker(speaker_samples_ceps='combined'):
	files_in_folder = os.listdir(os.getcwd()) # List of files in current directory
	true_speaker_files = [x for x in files_in_folder if ('true_speaker_ceps.txt' in x)]

	if speaker_samples_ceps=='combined':
		ceps = np.loadtxt(true_speaker_files[0])
	
		for i in range(len(true_speaker_files) - 1):
			current_ceps = np.loadtxt(true_speaker_files[i + 1])
			ceps = np.vstack((ceps,current_ceps))
			
		return ceps
	else:
		
		ceps_01 = np.loadtxt(true_speaker_files[0])
		ceps_02 = np.loadtxt(true_speaker_files[1])
		ceps_03 = np.loadtxt(true_speaker_files[2])
		
		return ceps_01, ceps_02, ceps_03
		

def get_ceps_test_speaker():
	
	files_in_folder = os.listdir(os.getcwd()) # List of files in current directory
	
	# Files just from the UBM ----------------------------
	cepsfiles = [x for x in files_in_folder if ('test_speaker_ceps.txt' in x)]
	
	ceps = np.loadtxt(cepsfiles[0])
	return ceps


def get_ceps_threshold_files():
	files_in_folder = os.listdir(os.getcwd()) # List of files in current directory
	threshold_files = [x for x in files_in_folder if ('threshold_audio_ceps.txt' in x)]
	
	ceps_01 = np.loadtxt(threshold_files[0])
	ceps_02 = np.loadtxt(threshold_files[1])
	ceps_03 = np.loadtxt(threshold_files[2])
	
	return ceps_01, ceps_02, ceps_03


# Universal Background Model ------------------------
def get_UBM_all(ngaussians = 10, cov_type='full', exclude_speaker=None):

	ceps_all = get_ceps_UBM()	
	gmm = GMM(n_components=ngaussians, covariance_type = cov_type)
	model_all = gmm.fit(ceps_all)
	print "model_all converged? ",model_all.converged_
	
	return model_all


def get_UBM_female(cov_type,exclude_speaker):
	
	ceps_female = get_ceps_UBM()
	ngaussians = 10	
	gmm = GMM(n_components=ngaussians, covariance_type = cov_type)
	model_female = gmm.fit(ceps_female)
	print "model_female converged? ",model_female.converged_
	
	return model_female


def get_UBM_male(cov_type,exclude_speaker):
		
	ceps_male = get_ceps_UBM()
	ngaussians = 10	
	gmm = GMM(n_components=ngaussians, covariance_type = cov_type)
	model_male = gmm.fit(ceps_male)
	print "model_male converged? ",model_male.converged_
	
	return model_male


# GMM generation ------------------------------------
def get_GMM_true_speaker(ngaussians=10, cov_type='full', speaker_samples='combined'):
	'''
	Returns the model(s) for the True Speaker.
	- speaker_samples: if this string equals 'combined', all three elocutions of the True Speaker are combined to generate a single GMM. Otherwise, three models are output.
	'''
	if speaker_samples=='combined':
		ceps = get_ceps_true_speaker(speaker_samples_ceps = speaker_samples)
		gmm = GMM(n_components=ngaussians, covariance_type = cov_type)
		model_true = gmm.fit(ceps)
		print "model_true converged? ",model_true.converged_
		return model_true
	else: # in this case speaker_samples='separated'
		[ceps_01, ceps_02, ceps_03] = get_ceps_true_speaker(speaker_samples_ceps = speaker_samples)

		gmm = GMM(n_components=ngaussians, covariance_type = cov_type)
		model_true01 = gmm.fit(ceps_01)
		model_true02 = gmm.fit(ceps_02)
		model_true03 = gmm.fit(ceps_03)
		print "all models converged? ",(model_true01.converged_ and model_true02.converged_ and model_true03.converged_)
		return model_true01, model_true02, model_true03


# Evaluation and scoring ----------------------------
def get_score_threshold(speaker_samples_threshold='combined'):
	
	[ceps_01, ceps_02, ceps_03] = get_ceps_threshold_files()
	
	if speaker_samples_threshold=='combined':
		model_true = get_GMM_true_speaker(speaker_samples = speaker_samples_threshold)
		
		scores = np.zeros(3)
		scores[0] = np.sum(model_true.score(ceps_01))
		scores[1] = np.sum(model_true.score(ceps_02))
		scores[2] = np.sum(model_true.score(ceps_03))
 
# 		print scores[0]
# 		print scores[1]
# 		print scores[2]
	
# 		return np.min(scores)
		scores_sorted = np.sort(scores)
		threshold = (scores_sorted[0] + scores_sorted[1] + scores_sorted[2])/3.0 # weighted average with weight=2 to the lowest score
		return threshold
		
	else: # In this case speaker_samples_threshold=='separated'
		scores = np.zeros(3)
		[model_true_01, model_true_02, model_true_03] = get_GMM_true_speaker(speaker_samples = speaker_samples_threshold)

		list_ceps = [ceps_01, ceps_02, ceps_03]
		
		for i in range(3):
			scores_GMM_true = np.zeros(3)
			scores_GMM_true[0] = np.sum(model_true_01.score(list_ceps[i]))
			scores_GMM_true[1] = np.sum(model_true_02.score(list_ceps[i]))
			scores_GMM_true[2] = np.sum(model_true_03.score(list_ceps[i]))
			
			scores[i] = np.max(scores_GMM_true)
		
# 		return np.min(scores)
		scores_sorted = np.sort(scores)
		print scores_sorted[0]
		print scores_sorted[1]
		print scores_sorted[2]
		threshold = (scores_sorted[0] + scores_sorted[1] + scores_sorted[2])/3.0 # weighted average with weight=2 to the lowest score
		return threshold


def evaluate_all_vs_true():
	'''
	# ======================================================================
	# TESTING WITH ALL SPEAKERS AGAINST A SINGLE TRUE SPEAKER
	'''
	files_in_folder = os.listdir(os.getcwd()) # List of files in current directory
	audiofiles_male = [x for x in files_in_folder if ('.wav' in x) and ('M' in x) and ('UBM' not in x) and ('_true' not in x) and ('_test' not in x)]
	audiofiles_female = [x for x in files_in_folder if ('.wav' in x) and ('F' in x) and ('UBM' not in x) and ('_true' not in x) and ('_test' not in x)]
	
	print 'len(audiofiles_male): ', len(audiofiles_male)
	print 'len(audiofiles_female): ', len(audiofiles_female)
	
	true_speakers_list = audiofiles_male + audiofiles_female + ['01_true_speaker.wav','02_true_speaker.wav','03_true_speaker.wav'] # Concatenation of both lists

	for true_speaker in true_speakers_list:
	
		'''
		----------------------------------------------------
		Training UBM for all training speakers (male & female) except the True one
		----------------------------------------------------
		'''
		cov_type = 'full'
		UBM_all = get_UBM_all(cov_type='full',exclude_speaker = true_speaker)
	
		'''
		----------------------------------------------------
		Setting up .txt files and lists
		----------------------------------------------------
		'''
		f = open('Verification_Test_02_' + true_speaker.split('.')[0] + '.txt','w')
		f.write('# Header: a single ASV test is carried out for each speaker S, with the speaker ' + true_speaker.split('.')[0] + ' being the True Speaker in all tests.\n# GENDER\tINDEX\tAGE\tSCORE\n')

		for gender_file_list in [audiofiles_male, audiofiles_female]:
			for current_test_speaker in gender_file_list:
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

				t_frame = 20*10**(-3) # Duration in seconds of each frame
				nwin = t_frame*fs
				# nwin is the number of samples per frame.
				# Para t_frame=20ms e fs=16kHz, nwin=320
				nfft = 512

				ceps_true, mspec, spec = mfcc(audio_true, nwin, nfft, fs, nceps)

				ngaussians = 10
				cov_type = 'full'
			
				gmm = GMM(n_components = ngaussians, covariance_type = cov_type)
				model_true = gmm.fit(ceps_true)
				print "model_true converged? ",model_true.converged_
		
				'''
				----------------------------------------------------
				Training test speaker
				----------------------------------------------------
				'''
				fs, audio_test = spwave.read(current_test_speaker)

				nwin = t_frame*fs
				# nwin is the number of samples per frame.
				# Para t_frame=20ms e fs=16kHz, nwin=320

				ceps_test, mspec, spec = mfcc(audio_test, nwin, nfft, fs, nceps)


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


def evaluate_all_vs_single_true(flag_speaker_samples='combined'):
	
	# ----------------------------------------
	# Model of True Speaker
	if flag_speaker_samples=='combined':
		model_true = get_GMM_true_speaker(speaker_samples=flag_speaker_samples)
	else: # Neste caso flag_speaker_samples=='separated'
		[model_true_01, model_true_02, model_true_03] = get_GMM_true_speaker(speaker_samples=flag_speaker_samples)
	
	# ----------------------------------------
	# Setting up text file
	
	files_in_folder = os.listdir(os.getcwd()) # List of files in current directory
	text_output_files = [x for x in files_in_folder if ('Verification_Test_04_true' in x) and (flag_speaker_samples in x)]

	number = len(text_output_files) + 1
	f = open('Verification_Test_04_true_' + flag_speaker_samples + '_0' + str(number) + '.txt','w')
	f.write('# ' + text_dependency + ' ASV\n# Header: a single ASV test is carried out for each test speaker, with the true speaker fixed, but having three elocutions.\n# GENDER\tINDEX\tAGE\tSCORE\tDECISION\n')

	n_accepted = 0
	n_false_acceptance = 0
	n_rejected = 0
	n_false_rejection = 0


	'''
	----------------------------------------------------
	Training UBM for all UBM-speakers except the True one
	----------------------------------------------------
	'''
	cov_type = 'full'
	UBM_all = get_UBM_all(cov_type='full')

	'''
	----------------------------------------------------
	Getting threshold
	----------------------------------------------------
	'''
	threshold = get_score_threshold(speaker_samples_threshold = flag_speaker_samples)


	files_in_folder = os.listdir(os.getcwd()) # List of files in current directory
	audiofiles_male = [x for x in files_in_folder if ('.wav' in x) and ('M' in x) and ('UBM' not in x) and ('_test' not in x) and ('_true' not in x)]
	audiofiles_female = [x for x in files_in_folder if ('.wav' in x) and ('F' in x) and ('UBM' not in x) and ('_test' not in x) and ('_true' not in x)]
	audiofiles_S = [x for x in files_in_folder if ('.wav' in x) and ('S' in x) and ('UBM' not in x) and ('_test' not in x) and ('_true' not in x)]

	# ------------------------------------------------------------------------
	# Running through the lists of utterances from non-True speakers
	for gender_file_list in [audiofiles_male, audiofiles_female]:
		for current_test_speaker in gender_file_list:
			print '\nTEST 04'
			print 'For file ' + current_test_speaker + ':'

			[gender,index,aux] = current_test_speaker.split('_') # "aux" is just a disposable variable
			[age,aux] = aux.split('.')

			'''
			----------------------------------------------------
			Training test speaker
			----------------------------------------------------
			'''
			[fs, audio_test] = read_audiofile(current_test_speaker,normalize=True)
	
			t_frame = 20*10**(-3) # Duration in seconds of each frame
			nwin = t_frame*fs # nwin is the number of samples per frame. Para t_frame=20ms e fs=16kHz, nwin=320
			nfft = 512

			ceps_test, mspec, spec = mfcc(audio_test, nwin, nfft, fs, nceps)

			'''
			----------------------------------------------------
			Scoring
			----------------------------------------------------
			'''
			if flag_speaker_samples=='combined':
				
				score_UBM = UBM_all.score(ceps_test)
				score_true = model_true.score(ceps_test)

				score = np.sum(score_true - score_UBM)
				print '\nScore: ', score
			else: # Neste caso flag_speaker_samples=='separated'
				
				score_UBM = UBM_all.score(ceps_test)
				score01 = model_true_01.score(ceps_test)
				score02 = model_true_02.score(ceps_test)
				score03 = model_true_03.score(ceps_test)

				score_true = np.max(np.array([score01, score02, score03]))

				score = np.sum(score_true - score_UBM)
				print '\nScore: ', score
	
			print "Difference between sum(log) probabilites (True VS. UBM_all): ", score
	
			# ----------------------------------
			# DECISION
			# ----------------------------------
			if score >= threshold:
				decision = 'ACCEPT'
				n_accepted += 1
				n_false_acceptance += 1
			else:
				decision = 'Reject'
				n_rejected += 1
		
			f.write(gender + '\t' + index + '\t' + age + '\t' + str(score) + '\t' + decision + '\n')
	
	# ------------------------------------------------------------------------
	# Running through the utterances of true speaker (prefix = S)
	for current_test_speaker in audiofiles_S:
		print '\nTEST 04'
		print 'For file ' + current_test_speaker + ':'

		[gender,index,aux] = current_test_speaker.split('_') # "aux" is just a disposable variable
		[age,aux] = aux.split('.')

		'''
		----------------------------------------------------
		Training test speaker
		----------------------------------------------------
		'''
		[fs, audio_test] = read_audiofile(current_test_speaker,normalize=True)

		t_frame = 20*10**(-3) # Duration in seconds of each frame
		nwin = t_frame*fs # nwin is the number of samples per frame. Para t_frame=20ms e fs=16kHz, nwin=320
		nfft = 512

		ceps_test, mspec, spec = mfcc(audio_test, nwin, nfft, fs, nceps)

		'''
		----------------------------------------------------
		Scoring
		----------------------------------------------------
		'''
		if flag_speaker_samples=='combined':
			
			score_UBM = UBM_all.score(ceps_test)
			score_true = model_true.score(ceps_test)

			score = np.sum(score_true - score_UBM)
			print '\nScore: ', score
		else: # Neste caso flag_speaker_samples=='separated'
			
			score_UBM = UBM_all.score(ceps_test)
			score01 = model_true_01.score(ceps_test)
			score02 = model_true_02.score(ceps_test)
			score03 = model_true_03.score(ceps_test)

			score_true = np.max(np.array([score01, score02, score03]))

			score = np.sum(score_true - score_UBM)
			print '\nScore: ', score

		print "Difference between sum(log) probabilites (True VS. UBM_all): ", score

		# ----------------------------------
		# DECISION
		# ----------------------------------
		if score >= threshold:
			decision = 'ACCEPT'
			n_accepted += 1
		else:
			decision = 'Reject'
			n_rejected += 1
			n_false_rejection += 1
	
		f.write(gender + '\t' + index + '\t' + age + '\t' + str(score) + '\t' + decision + '\n')
	
	
	f.write('# Number of accepted: ' + str(n_accepted) + '\n# Number of rejected: ' + str(n_rejected) + '\n# Number of false-acceptance: ' + str(n_false_acceptance) + '\n# Number of false-rejection: ' + str(n_false_rejection) + '\n# Threshold: ' + str(threshold))
	f.close()

