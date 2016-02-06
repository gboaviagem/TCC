'''
UFPE - DES - 06/02/2016
ASV routine, embedded in RPi.
'''

from ASV import *
from RPIO import *
import time
import os

def menu():
	'''
	The user gets to choose what action to perform by using two push buttons. PB_1 selects the options, while PB_2 rolls over the menu.
	Options:
		0- Verify speaker (test phase, outputs ACCEPT or REJECT, or synonyms)
		1- Train speaker (records 3 utterances from the true speaker GMM, then 3 utterances for threshold calculation)
		2- Exit
	Variables:
	opt - holds the current option
	'''
	opt = 0
	select = 0
	menu_length = 3 # How many options there are in the menu
	
	while select==0:
		if opt==0:
			lcd_string("# Opcoes (1/3)",LCD_LINE_1)
			lcd_string("1- Verificacao",LCD_LINE_2)
		if opt==1:
			lcd_string("# Opcoes (2/3)",LCD_LINE_1)
			lcd_string("2- Treino",LCD_LINE_2)
		if opt==2:
			lcd_string("# Opcoes (3/3)",LCD_LINE_1)
			lcd_string("3- Sair",LCD_LINE_2)	
			
		while (not PB_is_pressed()):
			# Polling the button, waiting for the user's choice.
			pass
		
		start_time = time.time()
		while PB_is_pressed():
			# Waiting and measuring for how long the user is pressing the button
			pass
		how_long_pressed = time.time() - start_time
		
		if how_long_pressed > 2.0: # Button pressed for more than 2 seconds
			select = 1
		else:
			opt = (opt+1) % menu_length	# opt is incremented mod 3.
	return opt
	

# ----------------------------
# Setup
pins_setup()
lcd_init()

lcd_string("Inicializando,",LCD_LINE_1)
lcd_string("aguarde. (1/3)",LCD_LINE_2)

# ------------------------------------------------------
# Building GMM
covariance_matrix = 'full'
model_UBM = get_UBM_all(cov_type = covariance_matrix)

lcd_string("Inicializando,",LCD_LINE_1)
lcd_string("aguarde. (2/3)",LCD_LINE_2)
model_true = get_GMM_true_speaker(speaker_samples='combined')

lcd_string("Inicializando,",LCD_LINE_1)
lcd_string("aguarde. (3/3)",LCD_LINE_2)
threshold = get_score_threshold(speaker_samples_threshold = 'combined')


# ------------------------------------------------------
# Main loop
keep_in_loop = 1

while keep_in_loop:
	option = menu()
	
	if option==0:
		lcd_string("Escutando...",LCD_LINE_1)
		lcd_string("(fale por 5s)",LCD_LINE_2)

		record_test_speaker()

		lcd_string("Processando,",LCD_LINE_1)
		lcd_string("aguarde.",LCD_LINE_2)

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
			lcd_string("Confirmado,",LCD_LINE_1)
			lcd_string("bem-vindo.",LCD_LINE_2)
		else:
			lcd_string("Por favor,",LCD_LINE_1)
			lcd_string("repita o teste.",LCD_LINE_2)
	
	if option==1:
		# True speaker recordings for GMM
		lcd_string("Gravacao (1/6):",LCD_LINE_1)
		lcd_string("Escutando...",LCD_LINE_2)
		record_true_speaker(in_embedded_system=True,index_rec=0)

		lcd_string("Gravacao (2/6):",LCD_LINE_1)
		lcd_string("Escutando...",LCD_LINE_2)
		record_true_speaker(in_embedded_system=True,index_rec=1)

		lcd_string("Gravacao (3/6):",LCD_LINE_1)
		lcd_string("Escutando...",LCD_LINE_2)
		record_true_speaker(in_embedded_system=True,index_rec=2)
		
		lcd_string("Processando,",LCD_LINE_1)
		lcd_string("aguarde.",LCD_LINE_2)	
		model_true = get_GMM_true_speaker(speaker_samples='combined') # True Speaker model updated
		
		# Recordings for threshold calculations
		lcd_string("Gravacao (4/6):",LCD_LINE_1)
		lcd_string("Escutando...",LCD_LINE_2)
		record_for_threshold_calculation(in_embedded_system=True,index_rec=0)

		lcd_string("Gravacao (5/6):",LCD_LINE_1)
		lcd_string("Escutando...",LCD_LINE_2)
		record_for_threshold_calculation(in_embedded_system=True,index_rec=1)

		lcd_string("Gravacao (6/6):",LCD_LINE_1)
		lcd_string("Escutando...",LCD_LINE_2)
		record_for_threshold_calculation(in_embedded_system=True,index_rec=2)

		lcd_string("Processando,",LCD_LINE_1)
		lcd_string("aguarde.",LCD_LINE_2)
		threshold = get_score_threshold(speaker_samples_threshold = 'combined') # Threshold updated
	
	if option==2:
		lcd_string("Desligando o",LCD_LINE_1)
		lcd_string("sistema.",LCD_LINE_2)
		keep_in_loop = 0

os.system("shutdown -h now")
