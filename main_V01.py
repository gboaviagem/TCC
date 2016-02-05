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

# ------------------------------------------------------
# Building GMM 
model_UBM = get_UBM_all(cov_type = covariance_matrix)
model_true = get_GMM_true_speaker(speaker_samples='combined')
threshold = get_score_threshold(speaker_samples_threshold = 'combined')



# ------------------------------------------------------
# Main loop
while keep_in_loop:
	option = raw_input('\nWhat to do?\n1- Exit\n2- Train true speaker\n3- Listen to test speaker\n4- Evaluate\n5- Recordings for threshold calculation\nOption: ')
	if option=='2':
		record_true_speaker()
	
	if option=='3':
		print '\nRecording Test Speaker elocution.'
		record_test_speaker()
		ceps_test = get_ceps_test_speaker()

		score_UBM = model_UBM.score(ceps_test)
		score_true = model_true.score(ceps_test)
	
		score = np.sum(score_true - score_UBM)
		print '\nScore: ', score
		print '\nThreshold: ', threshold

		
		# ----------------------------------
		# DECISION
		# ----------------------------------
		if score >= threshold:
			decision = 'ACCEPTED'
		else:
			decision = 'REJECTED'
		print '\n', decision

	
	if option=='1':
		keep_in_loop=0
	
	if option=='4':
		option_evaluate = raw_input('Speaker samples combined (1), separated (2) or both(3)? ')
		if option_evaluate=='1':
			evaluate_all_vs_single_true(flag_speaker_samples='combined')
		if option_evaluate=='2':
			evaluate_all_vs_single_true(flag_speaker_samples='separated')
		if option_evaluate=='3':
			evaluate_all_vs_single_true(flag_speaker_samples='separated')
			evaluate_all_vs_single_true(flag_speaker_samples='combined')
	
	if option=='5':
		record_for_threshold_calculation()