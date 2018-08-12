#=== TODO ===
# -support greyscale with varying bit depth

import warnings, sys, argparse

import scipy
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.io import wavfile

from functools import reduce

# fix these bad bois before scipy updates
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 


def readImage(fn):
	"""Reads an image to an array. 
	Returns 2d float[]

	Keyword arguments:
	domain		-- the signals time domain, float[]
	freq 		-- frequency of the sinusoid, float
	amp 		-- amplitude of the sinusoid, float
	phase 		-- phase of the sinusoid, float
	"""
	return scipy.ndimage.imread(fn, flatten = True)


def sinusoid(domain, freq, amp=1.0, phase=0.0):
	"""Creates a sinusoidal signal.
	Returns float[]

	Keyword arguments:
	domain		-- the signals time domain, float[]
	freq 		-- frequency of the sinusoid, float
	amp 		-- amplitude of the sinusoid, float
	phase 		-- phase of the sinusoid, float
	"""
	return amp * np.sin(domain * 2.0 * np.pi * freq + phase)


def writeWav(fn, signal, fs):
	"""Writes a signal to a wav file with 32 bit encoding.

	Keyword arguments:
	fn     		-- the filename of the file to write to, string
	signal 		-- the audio signal, float[]
	fs 			-- the sample rate of the signal, int
	"""
	scaled = np.int32(signal/np.max(np.abs(signal)) * 2147483647)
	wavfile.write(fn, fs, scaled)


def readWav(fn):
	"""Reads a wav file to an array.
	Returns float[]

	Keyword arguments:
	fn     		-- the filename of the file to read, string
	"""
	return scipy.io.wavfile.read(fn)


def butterHighpass(cutoff, fs):
	"""Create a highpass filter
	Returns scipy filter object

	Keyword arguments:
	cutoff     	-- the cutoff freq, int
	fs     		-- the sample rate, int
	"""
	nyq = fs / 2
	cutoff_normalised = cutoff / nyq
	b, a = signal.butter(5, cutoff_normalised, btype='high', analog=False)
	return b, a


def highpassFilter(sig, cutoff, fs):
	"""Applies a highpass filter to a signal
	Returns a float[]

	Keyword arguments:
	sig     	-- the signal to filter, float[]
	cutoff     	-- the cutoff freq, int
	fs     		-- the sample rate, int
	"""
	b, a = butterHighpass(cutoff, fs)
	return signal.filtfilt(b, a, sig)


def computeSinusoidCache(domain, freq_max):
	""" Precomputes a load of sinusoids into a cache
	Returns 3d array. Dim 1 is the frequency, Dim 2 is the amplitude. Dim 3 is the sinunsoid signal.

	Keyword arguments:
	domain		-- the signals time domain, float[]
	freq_max	-- the largest frequency in the cache
	"""
	N = len(domain)
	cache = np.empty([freq_max, 2, N])					#empty cache object, contains precomputed sine waves for all frequencies [0hz, 22050hz] and all possible amplitudes [0, 1]

	for f in range(1, freq_max + 1):
		cache[freq_max - f][0] = np.zeros(N)			# 0 amplitude is just zeroes
		cache[freq_max - f][1] = sinusoid(domain, f)  	# 1 ampliude is standard sine wave

	return cache

def encode(image_fn, fs, freq_min, STFT_window):
	""" Encodes a monochrome image into an audio signal
	Returns a float[]

	Keyword arguments:
	image_fn	-- the signals time domain, float[]
	fs     		-- the encodings sample rate, int
	freq_min  	-- the minimum frequency to encode, int
	STFT_window -- the size of the sample window, int
	"""
	freq_max = fs // 2

	image = readImage(image_fn)
	height, width = image.shape[0], image.shape[1]

	bitmap = (image/255).astype(int).transpose()		# transpose since we want to work in columns not rows, think 1 column = 1 window of audio sample

	N = np.floor((freq_max - freq_min) / height)		# calculate how many multiples of the image fit in our freq range
	bitmap = np.repeat(bitmap, N, axis=1)          		# scale image accordingly to fill specified freq range

	print('computing sinusoid series cache...')
	window_domain = np.arange(0.0, STFT_window) / fs   	#compute the domain for the sine wave cache
	STFT_cache = computeSinusoidCache(window_domain, freq_max)

	print('applying inverse short time fourier transform (this will take a while)...')
	sig = []
	for row in bitmap: 
		component_frequencies = [STFT_cache[i][row[i]] for i in range(len(row))]	# look up each sinusoid in the cache
		window = reduce(lambda x, y: x + y, component_frequencies)					# sum all the components to compute the STFT of the window
		sig = np.concatenate((sig, window))

	return sig


def main():
	# Preamble
	parser = argparse.ArgumentParser(description='Hides a monochrome .bmp into the spectrogram of a .wav file.')

	parser.add_argument('image_file', type=str, help='a monochrome .bmp')
	parser.add_argument('-fmin', type=int, metavar='hz', nargs='?', default=25000, help='the minimum frequency to appear in the spectrogram')
	parser.add_argument('-window', type=int, metavar='samples', nargs='?', default=1024, help='the STFT window size')
	parser.add_argument('-fs', type=int, metavar='hz', nargs='?', default=96000, help='the sample rate of the produced .wav')

	options = vars(parser.parse_args())

	image_fn = options['image_file']
	freq_min = options['fmin']
	STFT_window = options['window']
	fs = options['fs']
	result_fn = image_fn.split('.')[0] + '.wav'

	# Program
	print('encoding', image_fn, '@ sample rate', fs, 'hz\n')
	encoded = encode(image_fn, fs, freq_min, STFT_window)

	print('filtering low frequency artifacts')
	filtered = highpassFilter(encoded, freq_min, fs)

	print('writing audio to ' + result_fn + '...')
	writeWav(result_fn, filtered, fs)
	print('\ndone!')

if __name__ == '__main__':
	main()
