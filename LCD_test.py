from RIPO import *

lcd_pins_setup()
lcd_init()

# Push buttons constants (BCM GPIO)
PB_1 = 16
PB_2 = 12

PB1 = Button(pin=PB_1, pull_up=True, bounce_time=None)
PB2 = Button(pin=PB_2, pull_up=True, bounce_time=None)

if PB1.is_pressed and (not PB2.is_pressed):
	lcd_string("Botao 01",LCD_LINE_1)
	lcd_string("funcionando",LCD_LINE_1)

if PB2.is_pressed and (not PB1.is_pressed):
	lcd_string("Botao 02",LCD_LINE_1)
	lcd_string("funcionando tbm",LCD_LINE_1)
	
if PB1.is_pressed and PB2.is_pressed:
	lcd_string("Desligando",LCD_LINE_1)
	lcd_string("Tchau!",LCD_LINE_1)