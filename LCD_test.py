from RPIO import *
import time
import os

pins_setup()
lcd_init()

keep_in_loop = 1

while keep_in_loop:
	if (not PB1_is_pressed()) and (not PB2_is_pressed()):
		lcd_string("Aperte algum",LCD_LINE_1)
		lcd_string("botao",LCD_LINE_2)

	if PB1_is_pressed() and (not PB2_is_pressed()):
		lcd_string("Botao 01",LCD_LINE_1)
		lcd_string("funcionando",LCD_LINE_2)
		time.sleep(3) # 3 second delay

	if PB2_is_pressed() and (not PB1_is_pressed()):
		lcd_string("Botao 02",LCD_LINE_1)
		lcd_string("funcionando tbm",LCD_LINE_2)
		time.sleep(3) # 3 second delay
	
	if PB1_is_pressed() and PB2_is_pressed():
		lcd_string("Desligando",LCD_LINE_1)
		lcd_string("Tchau!",LCD_LINE_2)
		time.sleep(3) # 3 second delay
		keep_in_loop = 0
		os.system("shutdown -h now")
