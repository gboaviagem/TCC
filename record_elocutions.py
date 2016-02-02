#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para gravação de várias elocuções do mesmo locutor, chamado S.
V1 - 26/01/2015
"""

import os
from rec5sec_V1 import rec5sec

files_in_folder = os.listdir(os.getcwd())
audiofiles_S = [x for x in files_in_folder if ('.wav' in x) and ('S' in x)]

number = str(len(audiofiles_S) + 1)
if len(audiofiles_S) < 9:
	number = '0' + number

print '\n'
print "Say your NAME during recording.\n"
# age = raw_input("Age: ")
age = '24'

outputfile = 'S_' + number + '_' + age + '.wav' # Choosing name of output file to be created
rec5sec(outputfile)

print outputfile + "\n"