import argparse
import os, sys
import pprint
from synth_generator import *
from scipy.io.wavfile import write
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from cnn import top_k_mean_accuracy

pp = pprint.PrettyPrinter(indent=4, width=1)

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--sample_path", type=str, dest="sample_path", default="../data/sample.wav")
	parser.add_argument("--out_path", type=str, dest="out_path", default="../data/reconstructed.wav")
	parser.add_argument("--model_path", type=str, dest="model_path", default="../saved_models/E2E_best.h5")
	args = parser.parse_args()

	# print(f"RUN: {vars(args)}")
	pp.pprint(vars(args))
	return args

# PARAMETERS = sorted(["C", "M", "A", "D", "attack", "decay", "sustain", "sustain_level", "release"])
NUM_CLASSES = 16
PARAM_DICT = {
			  # FM parameters
			  "C": generate_freq(NUM_CLASSES),
			  "M": generate_param(NUM_CLASSES, 1, 30), 
			  "A": generate_param(NUM_CLASSES, 0.001, 1.0),
			  "D": generate_param(NUM_CLASSES, 0, 1.5),

			  # envelope function
			  "attack": generate_param(NUM_CLASSES, 0, 1.0),
			  "decay": generate_param(NUM_CLASSES, 0, 1.0),
			  "sustain": generate_param(NUM_CLASSES, 0, 1.0),
			  "sustain_level": generate_param(NUM_CLASSES, 0, 1.0),
			  "release": generate_param(NUM_CLASSES, 0, 1.0)

			  # low-pass filter with resonance
			  # "f_cut": ,
			  # "q": # resonance
			}
PARAMETERS = sorted(list(PARAM_DICT.keys()))

def evaluate_sample(model, sample_path, out_path):
	sound = read(sample_path)
	sample_rate, data = sound
	print("sample rate: {}, sound_data: {}".format(sample_rate, data))

	X = [data]
	Xd = np.expand_dims(np.vstack(X), axis=2)
	# print("Xd.shape: {}".format(Xd.shape))

	y_pred = model.predict(Xd)[0]

	result = []
	start = 0
	for i, p in enumerate(PARAMETERS):
		vals = PARAM_DICT[p]
		num_levels = len(vals)
		# print("{}: num_levels={}, vals={}".format(p, num_levels, vals))

		s_pred = y_pred[start:start+num_levels]
		pred_ind = np.argmax(s_pred)
		
		result.append((p, pred_ind, PARAM_DICT[p][pred_ind]))

		start += num_levels

	print("result: {}".format(result))

	params = {p: v for p, _, v in result}
	synth_gen = generate_synth(params, sample_rate)
	audio = generate_sound(synth_gen, params, 1.0, sample_rate)
	write(out_path, sample_rate, audio)

	# result: [('A', 15, 1.0), ('C', 9, 739.9888454232688), ('D', 3, 0.3), 
	# ('M', 6, 12.6), ('attack', 4, 0.26666666666666666), ('decay', 2, 0.13333333333333333), 
	# ('release', 3, 0.2), ('sustain', 13, 0.8666666666666667), ('sustain_level', 8, 0.5333333333333333)]

def main():
	args = get_args()

	model = keras.models.load_model(args.model_path, custom_objects={"top_k_mean_accuracy": top_k_mean_accuracy},)
	evaluate_sample(model, args.sample_path, args.out_path)

if __name__ == '__main__':
	main()