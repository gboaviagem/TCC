#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.io.wavfile as spwave

from numpy.fft import fft, ifft
from rec5sec_V1 import rec5sec
from Talkbox import mfcc,periodogram
from sklearn.mixture import GMM
from UBM_generation import get_UBM_all

from matplotlib.backends.backend_pdf import PdfPages

os.system('clear')

files_in_folder = os.listdir(os.getcwd()) # List of files in current directory
ver_files_m = [x for x in files_in_folder if ('Verification_Test_02_' in x) and ('M' in x)]
ver_files_f = [x for x in files_in_folder if ('Verification_Test_02_' in x) and ('F' in x)]
ver_files = ver_files_m + ver_files_f

[index,age,score] = np.loadtxt(ver_files[0],usecols=[1,2,3],unpack=True)
matrix_scores = score.T

for i in range(len(ver_files) - 1):
	[index,age,score] = np.loadtxt(ver_files[i+1],usecols=[1,2,3],unpack=True)
	matrix_scores = np.vstack((score.T,matrix_scores))

min_value = - np.min(np.min(matrix_scores))
aux = matrix_scores + min_value
aux = aux/(np.max(np.max(aux)))
print matrix_scores
print aux

plt.figure(num=None, figsize=(5, 4), dpi=80, facecolor='w', edgecolor='k')
pp = PdfPages('matrix_ASV_02.pdf')
plt.imshow(aux)
plt.colorbar()
plt.title('Scores de testes de ASV',fontsize=14)
plt.xlabel('Locutores de teste',fontsize=14)
plt.ylabel('Locutores de treino',fontsize=14)
# plt.show()
plt.tight_layout()
pp.savefig()
pp.close()

plt.figure(num=None, figsize=(5, 4), dpi=80, facecolor='w', edgecolor='k')
pp = PdfPages('matrix_ASV_02_positive.pdf')
ind = np.where(matrix_scores<0)
matrix_scores[ind] = 0
plt.imshow(matrix_scores)
plt.colorbar()
plt.title('Scores positivos de testes de ASV',fontsize=14)
plt.xlabel('Locutores de teste',fontsize=14)
plt.ylabel('Locutores de treino',fontsize=14)
# plt.show()
plt.tight_layout()
pp.savefig()
pp.close()