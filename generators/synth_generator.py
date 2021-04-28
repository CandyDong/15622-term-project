from synthplayer.oscillators import *
import numpy as np
import random
import math
import argparse
import os
from scipy.io.wavfile import write
import csv
import pprint

pp = pprint.PrettyPrinter(indent=4, width=1)

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--length", type=float, dest="length", default=1.0, help="length of each sample in seconds")
	parser.add_argument("--num_classes", type=int, dest="num_classes", default=16)
	parser.add_argument("--size", type=int, dest="size", default=1500)
	parser.add_argument("--out_dir", type=str, dest="out_dir", default="../data")
	parser.add_argument("--wav_dir", type=str, dest="wav_dir", default="../data/wav_files")
	parser.add_argument("--sample_rate", type=int, dest="sample_rate", default=16384)
	args = parser.parse_args()

	# print(f"RUN: {vars(args)}")
	pp.pprint(vars(args))
	return args

''' 
	generate `num` values in [min, max] 
'''
def generate_param(num, min, max):
	ext = float(max - min)
	return [i * ext / (num - 1) + min for i in range(num)]

'''
	generates a set of frequencies as per paper
	paper: f = 2^(n/12)/ 440Hz with n in 0..15, 
	corresponding to A4-C6
'''
def generate_freq(num):
	return [math.pow(2, i / 12) * 440 for i in range(num)]

def sample(param):
	index = random.choice(range(len(param)))
	return param[index], index

def generate_one_hot_encoding(index, num_levels):
	encoding = np.zeros(num_levels).astype(float)
	encoding[index] = 1.0
	return encoding

def sample_param(parameters, size):
	# sample parameters
	param_set = []
	labels = []
	for i in range(size):
		encoding = []
		new_set = {}
		for k, param in sorted(parameters.items()):
			value, index = sample(param)
			# print("param: {}, index: {}, value: {}, encoding: {}".format(str(k), index, value, generate_one_hot_encoding(index, len(param))))
			new_set.update({k: value})
			encoding.append(generate_one_hot_encoding(index, len(param)))

		labels.append(np.hstack(encoding))
		param_set.append(new_set)
	return param_set, labels

def generate_dataset(args, param_set, labels):
	dataset = []

	for i, param in enumerate(param_set):
		d = {"filename": os.path.join(args.wav_dir, "{:05d}.wav".format(i))}
		d.update(param)
		d.update({"label": labels[i]})
		dataset.append(d)
		if i%100 == 0: 
			print("generating example {} ".format(i) + "."*30)
			pp.pprint(d)

		synth_gen = generate_synth(param, args.sample_rate)
		audio = generate_sound(synth_gen, param, args.length, args.sample_rate)
		write(d["filename"], args.sample_rate, audio)

	return dataset

'''
	synth defined in the paper
'''
def generate_synth(p, sample_rate):
	# currently the oscillator only consists of a frequency modulated sine wave
	m = Sine(frequency=p["M"], amplitude=p["D"], samplerate=sample_rate)
	y_osc = Sine(frequency=p["C"], amplitude=p["A"], fm_lfo=m, samplerate=sample_rate)

	y_env = EnvelopeFilter(y_osc, attack=p["attack"], 
								  decay=p["decay"], 
								  sustain=p["sustain"], 
								  sustain_level=p["sustain_level"],
								  release=p["release"])

	# TODO: low pass filter + resonance + gate
	synth = y_env
	return synth.blocks()


def generate_sound(gen, p, length, sample_rate):
	num_samples = int(sample_rate*length)
	data = []
	while len(data) < num_samples:
		data.extend(next(gen))
	return np.array(data)

def write_meta(meta_file, parameters, dataset):
	#write the meta file
	with open(meta_file, 'w') as f:
		writer = csv.DictWriter(f, fieldnames=["id", "filename", "label"]+list(sorted(parameters.keys())))
		writer.writeheader()
		for i, d in enumerate(dataset):
			d["id"] = i
			writer.writerow(d)


def main():
	args = get_args()
	os.makedirs(args.out_dir, exist_ok=True)
	os.makedirs(args.wav_dir, exist_ok=True)

	parameters = {
				  # FM parameters
				  "C": generate_freq(args.num_classes),
				  "M": generate_param(args.num_classes, 1, 30), 
				  "A": generate_param(args.num_classes, 0.001, 1.0),
				  "D": generate_param(args.num_classes, 0, 1.5),

				  # envelope function
				  "attack": generate_param(args.num_classes, 0, 1.0),
				  "decay": generate_param(args.num_classes, 0, 1.0),
				  "sustain": generate_param(args.num_classes, 0, 1.0),
				  "sustain_level": generate_param(args.num_classes, 0, 1.0),
				  "release": generate_param(args.num_classes, 0, 1.0)

				  # low-pass filter with resonance
				  # "f_cut": ,
				  # "q": # resonance
				  }
	# print(parameters)
	# print(sorted(parameters.keys()))
	
	param_set, labels = sample_param(parameters, args.size)
	dataset = generate_dataset(args, param_set, labels)

	# stores parameters <-> filename mapping
	meta_file = os.path.join(args.out_dir, "meta.csv")
	write_meta(meta_file, parameters, dataset)

if __name__ == '__main__':
	main()

