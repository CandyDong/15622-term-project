import argparse
import os
import pprint
from scipy.io.wavfile import write
from synth_generator import *

pp = pprint.PrettyPrinter(indent=4, width=1)

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--sample_rate", type=int, dest="sample_rate", default=16384)
	parser.add_argument("--out_dir", type=str, dest="out_dir", default="../data/")
	parser.add_argument("--C", type=float, dest="C", default=783.991)
	parser.add_argument("--M", type=float, dest="M", default=12.6)
	parser.add_argument("--A", type=float, dest="A", default=1.0)
	parser.add_argument("--D", type=float, dest="D", default=0.5)
	parser.add_argument("--attack", type=float, dest="attack", default=0.2667)
	parser.add_argument("--decay", type=float, dest="decay", default=0.6)
	parser.add_argument("--sustain", type=float, dest="sustain", default=0.2667)
	parser.add_argument("--sustain_level", type=float, dest="sustain_level", default=0.0)
	parser.add_argument("--release", type=float, dest="release", default=0.8)
	args = parser.parse_args()

	# print(f"RUN: {vars(args)}")
	pp.pprint(vars(args))
	return args

def generate_sample(sample_rate, out_dir, params):
	synth_gen = generate_synth(params, sample_rate)
	audio = generate_sound(synth_gen, params, 1.0, sample_rate)
	write(os.path.join(out_dir, "sample.wav"), sample_rate, audio)

def main():
	args = get_args()

	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	params = {
			  "C": args.C,
			  "M": args.M, 
			  "A": args.A,
			  "D": args.D,
			  "attack": args.attack,
			  "decay": args.decay,
			  "sustain": args.sustain,
			  "sustain_level": args.sustain_level,
			  "release": args.release
			 }

	generate_sample(args.sample_rate, args.out_dir, params)

if __name__ == '__main__':
	main()