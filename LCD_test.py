from RIPO import *
import time


lcd_pins_setup()
lcd_init()

# Push buttons constants (BCM GPIO)
PB_1 = 16
PB_2 = 12

GPIO.setup(PB_1, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(PB_2, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# PB1 = Button(pin=PB_1, pull_up=True, bounce_time=None)
# PB2 = Button(pin=PB_2, pull_up=True, bounce_time=None)

PB1_is_pressed():
	return GPIO.input(PB_1)

PB2_is_pressed():
	return GPIO.input(PB_2)

keep_in_loop = 1

while keep_in_loop:
	if (not PB1_is_pressed()) and (not PB2_is_pressed()):
		lcd_string("Aperte algum",LCD_LINE_1)
		lcd_string("botao",LCD_LINE_1)

	if PB1_is_pressed() and (not PB2_is_pressed()):
		lcd_string("Botao 01",LCD_LINE_1)
		lcd_string("funcionando",LCD_LINE_1)
		time.sleep(3) # 3 second delay

	if PB2_is_pressed() and (not PB1_is_pressed()):
		lcd_string("Botao 02",LCD_LINE_1)
		lcd_string("funcionando tbm",LCD_LINE_1)
		time.sleep(3) # 3 second delay
	
	if PB1_is_pressed() and PB2_is_pressed():
		lcd_string("Desligando",LCD_LINE_1)
		lcd_string("Tchau!",LCD_LINE_1)
		time.sleep(3) # 3 second delay
		keep_in_loop = 0
		os.system("shutdown -h now")