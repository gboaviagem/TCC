#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para gravação de várias elocuções, para ser utilizado
em campo, na aquisição de amostras de voz de vários locutores
em série.
V1 - 05/01/2015
"""

import os
from ASV import *

n_recordings = 1

print '\n'
print "Say your NAME during recording.\n"
gender = raw_input("\nMale (M), female (F) or true speaker (S): ").upper()
if gender == 'S':
	n_recordings = int(raw_input('How many recordings? '))
	

for i in range(n_recordings):
	files_in_folder = os.listdir(os.getcwd())
	audiofiles_male = [x for x in files_in_folder if ('.wav' in x) and ('M' in x)]
	audiofiles_female = [x for x in files_in_folder if ('.wav' in x) and ('F' in x)]
	audiofiles_true_speaker = [x for x in files_in_folder if ('.wav' in x) and ('S' in x)]

	numberM = str(len(audiofiles_male) + 1)
	if len(audiofiles_male) < 9:
		numberM = '0' + numberM

	numberF = str(len(audiofiles_female) + 1)
	if len(audiofiles_female) < 9:
		numberF = '0' + numberF

	numberS = str(len(audiofiles_true_speaker) + 1)
	if len(audiofiles_true_speaker) < 9:
		numberS = '0' + numberS
	
	number = numberM
	if gender == 'F':
		number = numberF
	if gender == 'S':
		number = numberS

	if gender != 'S':
		age = raw_input("Age: ")
	else:
		age = '24'
	outputfile = gender + '_' + number + '_' + age + '.wav' # Choosing name of output file to be created
	rec5sec(outputfile)

	print outputfile + "\n"