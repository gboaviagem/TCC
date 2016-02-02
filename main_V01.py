'''
UFPE - DES - 31/01/2016
ASV routine.
'''

from ASV import *
import os

os.system('clear')

keep_in_loop = 1
print '\nSetting up...'
covariance_matrix = 'full'
model_UBM = get_UBM_all(cov_type = covariance_matrix)

while keep_in_loop:
	option = raw_input('\nWhat to do?\n1- Exit\n2- Train true speaker\n3- Listen to test speaker\n4- Evaluate\n5- Recordings for threshold calculation\nOption: ')
	if option=='2':
		record_true_speaker()
	
	if option=='3':
		option_TS_GMM = raw_input('\nShould the true speaker generate 1 or 3 models? ')
		print '\nRecording Test Speaker elocution.'
		record_test_speaker()
		ceps_test = get_ceps_test_speaker()
		
		if option_TS_GMM=='1':
			model_true = get_GMM_true_speaker(speaker_samples='combined')
		
			score_UBM = model_UBM.score(ceps_test)
			score_true = model_true.score(ceps_test)
		
			score = np.sum(score_true - score_UBM)
			print '\nScore: ', score
		else:
			[model_true_01, model_true_02, model_true_03] = get_GMM_true_speaker(speaker_samples='separated')
		
			score_UBM = model_UBM.score(ceps_test)
			score01 = model_true_01.score(ceps_test)
			score02 = model_true_02.score(ceps_test)
			score03 = model_true_03.score(ceps_test)
			
			score_true = np.max(np.array([score01, score02, score03]))
		
			score = np.sum(score_true - score_UBM)
			print '\nScore: ', score
	
	if option=='1':
		keep_in_loop=0
	
	if option=='4':
		option_evaluate = raw_input('Speaker samples combined (1), separated (2) or both(3)? ')
		if option_evaluate=='1':
			evaluate_all_vs_single_true(speaker_samples='combined')
		if option_evaluate=='2':
			evaluate_all_vs_single_true(speaker_samples='separated')
		if option_evaluate=='3':
			evaluate_all_vs_single_true(speaker_samples='separated')
			evaluate_all_vs_single_true(speaker_samples='combined')
	
	if option=='5':
		record_for_threshold_calculation()