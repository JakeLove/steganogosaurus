import sys, argparse

import scipy
import numpy as np

from scipy.io import wavfile
from scipy import signal

def writeWav(fn, signal, sample_rate):
	"""Writes a signal to a wav file with 32 bit encoding.

	Keyword arguments:
	fn     		-- the filename of the file to write to, string
	signal 		-- the audio signal, float[]
	sample_rate -- the sample rate of the signal, int
	"""
	scaled = np.int32(signal/np.max(np.abs(signal)) * 2147483647)
	wavfile.write(fn, sample_rate, scaled)


def readWav(fn):
	"""Reads a wav file to an array.
	Returns float[]

	Keyword arguments:
	fn     		-- the filename of the file to read, string
	"""
	return scipy.io.wavfile.read(fn)


def merge(sig_a, sig_b, offset):
	"""Merges 2 signals by adding them together.
	Returns float[]

	Keyword arguments:
	sig_a     	-- a signal, float[]
	sig_b     	-- another signal, float[]
	offset		-- the number of samples to offset the smaller signal by. Yes this is programmed stupidly. Sorry.
	"""
	if sig_a.size >= sig_b.size:	# determine which signal if smaller so we can pad it to be the same size as the other
		largest = sig_a
		smallest = sig_b
	else:
		largest = sig_b
		smallest = sig_a

	diff = largest.size - smallest.size

	left_pad = offset		   		# number of zeros to left pad the smaller file
	right_pad = diff - offset		# number of zeros to right pad the smaller file
	smallest = np.concatenate([np.zeros(left_pad), smallest, np.zeros(right_pad)])
	
	return largest + smallest


def main():
	# Preamble
	parser = argparse.ArgumentParser(description='Additively merges 2 .wav files together.')

	parser.add_argument('file_a', type=str, help='a .wav file')
	parser.add_argument('file_b', type=str, help='another .wav file')
	parser.add_argument('-o', type=str, metavar='file_name', nargs='?', default='merged.wav', help='name of the outfile')
	parser.add_argument('-offset', type=int, metavar='seconds', nargs='?', default=0, help='the offset of file_b when merging with file_a')

	options = vars(parser.parse_args())

	wav_a = options['file_a']
	wav_b = options['file_b']
	outfile = options['o']
	offset_seconds = options['offset']
	# Program
	fs,  sig_a  = readWav(wav_a)
	fs2, sig_b  = readWav(wav_b)

	if fs != fs2:
		raise Exception('\nSample rates of files do not match')

	offset = offset_seconds * fs

	writeWav(outfile, merge(sig_a, sig_b, offset), fs)

	print('done!')



if __name__ == '__main__':
	main()
