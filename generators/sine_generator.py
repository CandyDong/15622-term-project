import random
import math
import argparse
import os
from scipy.io.wavfile import write
import csv
import numpy as np

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--length", type=float, dest="length", default=1.0, help="length of each sample in seconds")
	parser.add_argument("--size", type=int, dest="size", default=150)
	parser.add_argument("--out_dir", type=str, dest="out_dir", default="../data")
	parser.add_argument("--wav_dir", type=str, dest="wav_dir", default="../data/wav_files")
	parser.add_argument("--sample_rate", type=int, dest="sample_rate", default=44100)
	args = parser.parse_args()
	print(f"RUN: {vars(args)}")
	return args

def generate_sine(sample_rate, length, params):
	num_samples = int(sample_rate*length)
	data = np.zeros(num_samples)
	for i in range(num_samples):
		t = float(i) / sample_rate
		v = ((params["a1"] * math.sin(t * params["f1"] * math.pi)) + (params["a2"] * math.sin(t * params["f2"] * math.pi))) * 0.5

		# normalization
		peak = np.max(np.absolute(v))
		if peak > 0:
		   v = v / peak

		data[i] = v

	return data 

def sample_param(param):
	index = random.choice(range(len(param)))
	return param[index]

def main():
	args = get_args()
	os.makedirs(args.out_dir, exist_ok=True)
	os.makedirs(args.wav_dir, exist_ok=True)

	# stores parameters <-> filename mapping
	meta_file = os.path.join(args.out_dir, "meta.csv")

	parameters = {"f1": [100, 200, 400],
				  "a1": [0.5, 0.7, 1.0],
				  "f2": [800, 1200, 1600],
				  "a2": [0.5, 0.7, 1.0]}

	# sample parameters
	param_set = []
	for i in range(args.size):
		new_set = {k: sample_param(v) for k, v in parameters.items()}
		# print(new_set)
		param_set.append(new_set)

	dataset = []
	for i, param in enumerate(param_set):
		d = {"filename": os.path.join(args.wav_dir, "{:05d}.wav".format(i))}
		d.update(param)
		dataset.append(d)
		print("generating example: {}".format(d))

		audio = generate_sine(args.sample_rate, args.length, param)
		write(d["filename"], args.sample_rate, audio)

	#write the meta file
	with open(meta_file, 'w') as f:
			writer = csv.DictWriter(f, fieldnames=["id", "filename", "a1", "f1", "a2", "f2"])
			writer.writeheader()
			for i, d in enumerate(dataset):
				d["id"] = i
				writer.writerow(d)

if __name__ == '__main__':
	main()
