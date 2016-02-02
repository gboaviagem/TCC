import os

def rec5sec_RPi(output_file_name):
	os.system("arecord -D plughw:0,0 -d 5 -f S16_LE -c 1 -r 16000 ./S16bitmono16kHz" + output_file_name)
