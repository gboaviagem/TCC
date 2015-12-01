import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as spwave
from rec5sec_V1 import rec5sec

# Recording 5 seconds of audio
recordedfilename = 'TesteAudio.wav' # Choosing name of output file to be created
rec5sec(recordedfilename) # Recording hand-made function
rate, data = spwave.read(recordedfilename)

nchannels = np.size(data.shape) # Number of channels
nsamples = data.size # Number of samples

if nchannels == 2:
	[data_left,data_right] = np.hsplit(data,2) # Left and right channels

t = np.arange(len(data))/float(rate) # Duration in seconds

# Plotting audio signals
if nchannels == 2:
	plt.figure(1)
	plt.subplot(211)
	plt.plot(t,data_left) # Left channel
	plt.title('Left Channel')
	plt.xlabel('Time [s]')

	plt.subplot(212)
	plt.plot(t,data_right) # Right channel
	plt.title('Right Channel')
	plt.xlabel('Time [s]')

	plt.tight_layout()
	plt.show()
else:
	plt.plot(t,data)
	plt.title('Audio signal')
	plt.xlabel('Time [s]')

	plt.tight_layout()
	plt.show()