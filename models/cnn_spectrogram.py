import argparse
import os
import pprint
import csv

pp = pprint.PrettyPrinter(indent=4, width=1)

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--length", type=float, dest="length", default=1.0, help="length of each sample in seconds")
	parser.add_argument("--num_classes", type=int, dest="num_classes", default=16)
	parser.add_argument("--size", type=int, dest="size", default=150)
	args = parser.parse_args()

	# print(f"RUN: {vars(args)}")
	pp.pprint(vars(args))
	return args

SAMPLE_RATE = 16384
OUT_DIR = "../data"
WAV_DIR = "../data/wav_files"
META_FILE_PATH = os.path.join(OUT_DIR, "meta.csv")
PARAMETERS = sorted(["C", "M", "A", "D", "attack", "decay", "sustain", "sustain_level", "release"])


def read_meta():
	print("read_meta")
	dataset = []
	with open(META_FILE_PATH, 'r') as f:
		reader = csv.DictReader(f)
		for row in reader:
			dataset.append(row)
	return dataset


def main():
	dataset = read_meta()
	


if __name__ == '__main__':
	main()