import random
import math
import argparse
import os

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--length", type=float, dest="length", default=1.0, help="length of each sample in seconds")
	parser.add_argument("--size", type=int, dest="size", default=150)
	parser.add_argument("--out_dir", type=str, dest="out_dir", default="../data")
	parser.add_argument("--wave_dir", type=str, dest="wave_dir", default="../data_waves")
	parser.add_argument("--sample_rate", type=int, dest="sample_rate", default=44100)
	args = parser.parse_args()
	print(f"RUN: {vars(args)}")
	return args

def generate_sine(sample_rate, length, a, f):
	num_samples = int(sample_rate*length)
	data = np.zeros(num_samples)
	for i in range(num_samples):
		t = float(i) / sample_rate
		data[i] = a * math.sin(t * f * math.pi)
	return data 

def sample_param(param):
	index = random.choice(range(len(param)))
	return param[index]


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.wave_dir, exist_ok=True)

    parameters = {"f1": [100, 200, 400],
                  "a1": [0.5, 0.7, 1.0],
                  "f2": [800, 1200, 1600],
                  "a2": [0.5, 0.7, 1.0]}

    # sample parameters
    param_set = []
    for i in range(args.size):
    	new_set = {k: sample_param(v) for k, v in parameters.items()}
    	print(new_set)
    	if i > 10:
    		break


    # generate_sine(args.sample_rate, args.length, args.)

if __name__ == '__main__':
	main()
