import argparse
import os
import pprint
import csv
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import CSVLogger
import ast
import numpy as np
from data_generator import DataGenerator
import math
from synthplayer.oscillators import *
from scipy.io.wavfile import write, read
from kapre.time_frequency import STFT, Magnitude, MagnitudeToDecibel
from synth_generator import *

BEST = False # whether the skip training and load the best model

NUM_EPOCH = 100
BATCH_SIZE = 64

SAMPLE_RATE = 16384
OUT_DIR = "../data"
MODEL_DIR = "../saved_models/"
WAV_DIR = "../data/wav_files"
SAMPLE_WAV_DIR = "../data/sample_wav_files/original"
RECONSTRUCT_WAV_DIR = "../data/reconstructed_wav_files"
SAMPLE_RECONSTRUCT_WAV_DIR = "../data/sample_wav_files/reconstructed"
# MODEL_NAME = "CONV6XL" 
MODEL_NAME = "E2E" 

META_FILE_PATH = os.path.join(OUT_DIR, "meta.csv")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "{}.h5".format(MODEL_NAME))
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "{}_best.h5".format(MODEL_NAME))
CHECKPOINT_MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "{}_checkpoint.h5".format(MODEL_NAME))
HISTORY_SAVE_PATH = os.path.join(MODEL_DIR, "{}_history.csv".format(MODEL_NAME))
PREDICTION_FILE_PATH = os.path.join(MODEL_DIR, "{}_prediction.csv".format(MODEL_NAME))
SUMMARY_FILE_PATH = os.path.join(MODEL_DIR, "{}_summary.txt".format(MODEL_NAME))


if not os.path.exists(OUT_DIR):
	os.makedirs(OUT_DIR)
if not os.path.exists(MODEL_DIR):
	os.makedirs(MODEL_DIR)
if not os.path.exists(RECONSTRUCT_WAV_DIR):
	os.makedirs(RECONSTRUCT_WAV_DIR)
if not os.path.exists(SAMPLE_WAV_DIR):
	os.makedirs(SAMPLE_WAV_DIR)
if not os.path.exists(SAMPLE_RECONSTRUCT_WAV_DIR):
	os.makedirs(SAMPLE_RECONSTRUCT_WAV_DIR)

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

def top_k_mean_accuracy(y_true, y_pred, k=5):
	"""
	@ paper
	The top-k mean accuracy is obtained by computing the top-k
	accuracy for each test example and then taking the mean across
	all examples. In the same manner as done in the MPR analysis,
	we compute the top-k mean accuracy per synthesizer
	parameter for ???? = 1, ... ,5.
	"""
	# TODO: per parameter?
	original_shape = tf.shape(y_true)
	y_true = tf.reshape(y_true, (-1, tf.shape(y_true)[-1]))
	y_pred = tf.reshape(y_pred, (-1, tf.shape(y_pred)[-1]))
	top_k = K.in_top_k(y_pred, tf.cast(tf.argmax(y_true, axis=-1), "int32"), k)
	correct_pred = tf.reshape(top_k, original_shape[:-1])
	return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def get_e2e_model(label_size):
	model = keras.Sequential()
	model.add(keras.Input(shape=(SAMPLE_RATE, 1)))
	# model.add(keras.layers.Conv1D(filters=96,  kernel_size=64, strides=4, activation='relu', padding="same"))
	# model.add(keras.layers.Conv1D(filters=96,  kernel_size=32, strides=4, activation='relu', padding="same"))
	# model.add(keras.layers.Conv1D(filters=128, kernel_size=16, strides=4, activation='relu', padding="same"))
	# model.add(keras.layers.Conv1D(filters=257, kernel_size=8,  strides=4, activation='relu', padding="same"))

	# model.add(keras.layers.Reshape((64, 257, 1), input_shape=(64, 257)))

	# model.add(keras.layers.Conv2D(filters=32,  kernel_size=(3, 3), strides=(2, 2), activation='relu', padding="same"))
	# model.add(keras.layers.Conv2D(filters=71,  kernel_size=(3, 3), strides=(2, 2), activation='relu', padding="same"))
	# model.add(keras.layers.Conv2D(filters=128,  kernel_size=(3, 4), strides=(2, 3), activation='relu', padding="same"))
	# model.add(keras.layers.Conv2D(filters=128,  kernel_size=(3, 3), strides=(2, 2), activation='relu', padding="same"))
	# model.add(keras.layers.Conv2D(filters=128,  kernel_size=(3, 3), strides=(2, 2), activation='relu', padding="same"))
	# model.add(keras.layers.Conv2D(filters=128,  kernel_size=(3, 3), strides=(1, 2), activation='relu', padding="same"))

	model.add(keras.layers.Conv1D(filters=96,  kernel_size=64, strides=4))
	model.add(keras.layers.Conv1D(filters=96,  kernel_size=32, strides=4))
	model.add(keras.layers.Conv1D(filters=128, kernel_size=16, strides=4))
	model.add(keras.layers.Conv1D(filters=257, kernel_size=8,  strides=4))

	model.add(keras.layers.Reshape((61, 257, 1), input_shape=(61, 257)))

	model.add(keras.layers.Conv2D(filters=32,  kernel_size=(3, 3), strides=(2, 2)))
	model.add(keras.layers.Conv2D(filters=71,  kernel_size=(3, 3), strides=(2, 2)))
	model.add(keras.layers.Conv2D(filters=128,  kernel_size=(3, 4), strides=(2, 3)))
	model.add(keras.layers.Conv2D(filters=128,  kernel_size=(3, 3), strides=(2, 2)))

	model.add(keras.layers.Conv2D(filters=128,  kernel_size=(1, 1), strides=(2, 2)))
	model.add(keras.layers.Conv2D(filters=128,  kernel_size=(1, 1), strides=(1, 2)))

	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(512))

	model.add(keras.layers.Dense(label_size, activation="sigmoid"))
	model.compile(
		optimizer='adam', 
		loss=keras.losses.BinaryCrossentropy(), 
		metrics=[
			# @paper: 1) Mean Percentile Rank?
			# mean_percentile_rank,
			# @paper: 2) Top-k mean accuracy based evaluation
			top_k_mean_accuracy,
			# @paper: 3) Mean Absolute Error based evaluation
			keras.metrics.MeanAbsoluteError(),
		],)

	print(model.summary())
	return model

def get_conv6xl_model(label_size):
	model = keras.Sequential()
	model.add(keras.Input(shape=(SAMPLE_RATE, 1)))

	model.add(STFT(
		n_fft=512,
		hop_length=256,
		input_shape=(SAMPLE_RATE, 1),
		input_data_format='channels_last'
	))
	model.add(Magnitude())
	model.add(MagnitudeToDecibel())
	model.add(keras.layers.Conv2D(filters=64,  kernel_size=(3, 3), strides=(2, 2)))
	model.add(keras.layers.Conv2D(filters=128,  kernel_size=(3, 3), strides=(2, 2)))
	model.add(keras.layers.Conv2D(filters=128,  kernel_size=(3, 4), strides=(2, 3)))
	model.add(keras.layers.Conv2D(filters=128,  kernel_size=(3, 3), strides=(2, 2)))
	model.add(keras.layers.Conv2D(filters=256,  kernel_size=(1, 1), strides=(2, 2)))
	model.add(keras.layers.Conv2D(filters=256,  kernel_size=(1, 1), strides=(1, 2)))

	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(512))

	model.add(keras.layers.Dense(label_size, activation="sigmoid"))
	model.compile(
		optimizer='adam', 
		loss=keras.losses.BinaryCrossentropy(), 
		metrics=[
			# @paper: 1) Mean Percentile Rank?
			# mean_percentile_rank,
			# @paper: 2) Top-k mean accuracy based evaluation
			top_k_mean_accuracy,
			# @paper: 3) Mean Absolute Error based evaluation
			keras.metrics.MeanAbsoluteError(),
			'accuracy'
		],)

	print(model.summary())
	return model


def read_meta():
	dataset = []
	with open(META_FILE_PATH, 'r') as f:
		reader = csv.DictReader(f)
		for row in reader:
			dataset.append(row)
	return dataset

def reconstruct(pred_inds, meta):
	with open(PREDICTION_FILE_PATH, 'w') as f:
		writer = csv.DictWriter(f, fieldnames=["id", "original_filename", "reconstruct_filename"]+PARAMETERS)
		writer.writeheader()
		for i in range(pred_inds.shape[0]):
			pred = {}
			pred["id"] = meta[i]["id"]
			pred["original_filename"] = meta[i]["filename"]
			pred["reconstruct_filename"] = os.path.join(RECONSTRUCT_WAV_DIR, "{:05d}.wav".format(int(pred["id"])))
			for j, param in enumerate(PARAMETERS):
				pred[param] = PARAM_DICT[param][pred_inds[i][j]]
			reconstruct_sound(pred)
			writer.writerow(pred)

def reconstruct_sound(param):
	synth_gen = generate_synth(param, SAMPLE_RATE)
	audio = generate_sound(synth_gen, param, 1.0, SAMPLE_RATE)
	write(param["reconstruct_filename"], SAMPLE_RATE, audio)
	print("{} reconstructed".format(param["reconstruct_filename"]))

def print_summary(example, pred, truth, save=False):
	PRECISION = 1
	width = 8

	names = "Parameter: "
	act_s = "Actual:    "
	pred_s = "Predicted: "
	pred_i = "Pred. Indx:"
	act_i = "Act. Index:"
	diff_i = "Index Diff:"
	for i, p in enumerate(PARAMETERS):
		names += p.rjust(width)[:width]
		act_s += f"{PARAM_DICT[p][truth[i]]:>8.2f}"
		pred_s += f"{PARAM_DICT[p][pred[i]]:>8.2f}"
		act_i += f"{truth[i]:>8}"
		pred_i += f"{pred[i]:>8}"
		diff_i += f"{pred[i]-truth[i]:>8}"

	exact = 0.0
	close = 0.0
	for i, p in enumerate(PARAMETERS):
		if pred[i] == truth[i]:
			exact = exact + 1.0
		if abs(pred[i] - truth[i]) <= PRECISION:
			close = close + 1.0
	exact_ratio = exact / len(PARAMETERS)
	close_ratio = close / len(PARAMETERS)
	print("Example {}: ".format(example))
	print(names)
	print(act_s)
	print(pred_s)
	print(act_i)
	print(pred_i)
	print(diff_i)
	print("exact_ratio: {}\nclose_ratio:{}".format(exact_ratio, close_ratio))
	print("-" * 30)

	with open(SUMMARY_FILE_PATH, 'a') as f:
		f.write("{}\n{}\n{}\n{}\n{}\n{}\n{}\nexact_ratio: {}\nclose_ratio: {}\n{}\n".format(
			"Example {}: ".format(example),
			names,
			act_s, pred_s, act_i, pred_i, diff_i, exact_ratio, close_ratio, "-" * 30))
	return exact_ratio, close_ratio

def evaluate(model, X, y, meta):
	num_examples = X.shape[0]
	print("Number of examples evaluated: {}".format(num_examples))

	y_pred = model.predict(X)
	print("prediction: {}, shape: {}".format(y_pred, y_pred.shape))
	
	# from one hot to indices
	y_pred_inds, y_inds = [], []
	start = 0
	for i, p in enumerate(PARAMETERS):
		vals = PARAM_DICT[p]
		num_levels = len(vals)
		# print("{}: num_levels={}, vals={}".format(p, num_levels, vals))

		s_pred, s = y_pred[:, start:start+num_levels], y[:, start:start+num_levels]
		pred_inds, inds = np.argmax(s_pred, axis=1), np.argmax(s, axis=1)
		
		y_pred_inds.append(np.expand_dims(pred_inds, axis=1))
		y_inds.append(np.expand_dims(inds, axis=1))

		start += num_levels

	y_pred_inds = np.hstack(y_pred_inds)
	y_inds = np.hstack(y_inds)
	print("y_pred_inds shape: {}, y_inds.shape: {}".format(y_pred_inds.shape, y_inds.shape))

	for i in range(y_inds.shape[0]):
		print_summary(meta[i]["id"], y_pred_inds[i], y_inds[i], save=True)

	reconstruct(y_pred_inds, meta)

def main():
	# delete previous history file
	if os.path.exists(HISTORY_SAVE_PATH):
		os.remove(HISTORY_SAVE_PATH)
	if os.path.exists(SUMMARY_FILE_PATH):
		os.remove(SUMMARY_FILE_PATH)
	for f in os.listdir(RECONSTRUCT_WAV_DIR):
		os.remove(os.path.join(RECONSTRUCT_WAV_DIR, f))

	gpu_avail = tf.test.is_gpu_available()  # True/False
	cuda_gpu_avail = tf.test.is_gpu_available(cuda_only=True)  # True/False

	print("+" * 30)
	print(
		f"Running model: {MODEL_NAME} for {NUM_EPOCH} epochs"
	)
	print(f"Saving model at {MODEL_SAVE_PATH}")
	print(f"GPU: {gpu_avail}, with CUDA: {cuda_gpu_avail}")
	print("+" * 30)


	dataset = read_meta()
	label_size = len(np.fromstring(dataset[0]['label'][1:-1], dtype=float, sep=' '))
	print("+" * 30)
	print("dataset loaded.")
	print("label size = {}".format(label_size))
	print("+" * 30)

	params = {"dataset": dataset, 
				"batch_size": BATCH_SIZE, 
				"shuffle": True,
				"n_samps": SAMPLE_RATE}
	training_generator = DataGenerator(first=0.8, **params)
	validation_generator = DataGenerator(last=0.2, **params)

	callbacks = []
	best_callback = keras.callbacks.ModelCheckpoint(
		filepath=BEST_MODEL_SAVE_PATH,
		save_weights_only=False,
		save_best_only=True,
		verbose=1,
	)
	checkpoint_callback = keras.callbacks.ModelCheckpoint(
		filepath=CHECKPOINT_MODEL_SAVE_PATH,
		save_weights_only=False,
		save_best_only=False,
		verbose=1,
	)
	callbacks.append(best_callback)
	callbacks.append(checkpoint_callback)
	callbacks.append(CSVLogger(HISTORY_SAVE_PATH, append=True))

	if BEST:
		print("Loading BEST model; Training skipped!!!")
		print(
			f"Evaluate best model: {BEST_MODEL_SAVE_PATH}"
		)
		model = keras.models.load_model(
			BEST_MODEL_SAVE_PATH,
			custom_objects={"top_k_mean_accuracy": top_k_mean_accuracy},
		)
	else:
		if MODEL_NAME is "E2E":
			model = get_e2e_model(label_size)
		elif MODEL_NAME is "CONV6XL":
			# not yet implemented
			model = get_conv6xl_model(label_size)
		history = model.fit(
			x=training_generator,
			validation_data=validation_generator,
			epochs=NUM_EPOCH,
			callbacks=callbacks,
			initial_epoch=0,
			verbose=1,
		)

		# Save model
		model.save(MODEL_SAVE_PATH)

	validation_generator.on_epoch_end()
	X_valid, y_valid = validation_generator.__getitem__(0)
	meta_valid = validation_generator.get_meta(0)

	evaluate(model, X_valid, y_valid, meta_valid)


if __name__ == '__main__':
	main()