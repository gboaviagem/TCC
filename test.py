import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import scipy.io.wavfile as spwave
from rec5sec_V1 import rec5sec

recordedfilename = 'TesteAudio.wav' # Choosing name of output file to be created
rate, data = spwave.read(recordedfilename)

nchannels = np.size(data.shape) # Number of channels
nsamples = data.size # Number of samples

t = np.arange(len(data))/float(rate) # Duration in seconds

plt.figure(1)
plt.subplot(211)
plt.plot(t,data)
plt.title('A4 (440 Hz)')
plt.xlabel('Time [s]')

# FFT
data_fft = np.abs(fft(data))
k = np.arange(nsamples)
f = k*rate/nsamples


plt.subplot(212)
plt.plot(f[0:nsamples/2],2*data_fft[0:nsamples/2])
plt.title('Espectro unilateral (Mag)')
plt.xlabel('Frequencia [Hz]')
plt.xscale('log')

plt.tight_layout()

plt.show()