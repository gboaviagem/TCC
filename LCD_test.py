from RPIO import *
import time
import os

pins_setup()
lcd_init()

keep_in_loop = 1

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
		
		if how_long_pressed > 1.0: # Button pressed for more than 2 seconds
			select = 1
		else:
			opt = (opt+1) % menu_length	# opt is incremented mod 3.
	return opt

while keep_in_loop:
	option = menu()

	if option==0:
		lcd_string("Opcao 1",LCD_LINE_1)
		lcd_string("escolhida.",LCD_LINE_2)
		time.sleep(3)
	
	if option==1:		
		lcd_string("Opcao 2",LCD_LINE_1)
		lcd_string("escolhida.",LCD_LINE_2)
		time.sleep(3)
	
	if option==2:
		lcd_string("Opcao 3.",LCD_LINE_1)
		lcd_string("Tchau.",LCD_LINE_2)
		keep_in_loop=0
