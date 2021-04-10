import argparse
import os
import pprint
import csv
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import ast
import numpy as np
from data_generator import DataGenerator
import math

SAMPLE_RATE = 16384
OUT_DIR = "../data"
MODEL_DIR = "../saved_models/"
WAV_DIR = "../data/wav_files"
MODEL_NAME = "E2E" 

NUM_EPOCH = 100
BATCH_SIZE = 64

META_FILE_PATH = os.path.join(OUT_DIR, "meta.csv")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "{}.h5".format(MODEL_NAME))
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "{}_best.h5".format(MODEL_NAME))
CHECKPOINT_MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "{}_checkpoint.h5".format(MODEL_NAME))
PREDICTION_FILE_PATH = os.path.join(OUT_DIR, "prediction.csv")

if not os.path.exists(OUT_DIR):
	os.makedirs(OUT_DIR)
if not os.path.exists(MODEL_DIR):
	os.makedirs(MODEL_DIR)

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
	parameter for 𝑘 = 1, ... ,5.
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
	model.add(keras.layers.Conv1D(filters=96,  kernel_size=64, strides=4, activation='relu', padding="same"))
	model.add(keras.layers.Conv1D(filters=96,  kernel_size=32, strides=4, activation='relu', padding="same"))
	model.add(keras.layers.Conv1D(filters=128, kernel_size=16, strides=4, activation='relu', padding="same"))
	model.add(keras.layers.Conv1D(filters=257, kernel_size=8,  strides=4, activation='relu', padding="same"))

	model.add(keras.layers.Reshape((64, 257, 1), input_shape=(64, 257)))

	model.add(keras.layers.Conv2D(filters=32,  kernel_size=(3, 3), strides=(2, 2), activation='relu', padding="same"))
	model.add(keras.layers.Conv2D(filters=71,  kernel_size=(3, 3), strides=(2, 2), activation='relu', padding="same"))
	model.add(keras.layers.Conv2D(filters=128,  kernel_size=(3, 4), strides=(2, 3), activation='relu', padding="same"))
	model.add(keras.layers.Conv2D(filters=128,  kernel_size=(3, 3), strides=(2, 2), activation='relu', padding="same"))
	model.add(keras.layers.Conv2D(filters=128,  kernel_size=(3, 3), strides=(2, 2), activation='relu', padding="same"))
	model.add(keras.layers.Conv2D(filters=128,  kernel_size=(3, 3), strides=(1, 2), activation='relu', padding="same"))

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

def get_c4_model():
	model = keras.Sequential()
	model.add(keras.Input(shape=(1,16384)))
	model.add(keras.layers.Conv2D(filters=32,  kernel_size=(3, 4), strides=(2, 3), activation='relu', padding="same"))
	model.add(keras.layers.Conv2D(filters=65,  kernel_size=(3, 4), strides=(2, 3), activation='relu', padding="same"))
	model.add(keras.layers.Conv2D(filters=105,  kernel_size=(3, 4), strides=(2, 3), activation='relu', padding="same"))
	model.add(keras.layers.Conv2D(filters=128,  kernel_size=(4, 5), strides=(3, 4), activation='relu', padding="same"))

	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(512))

	model.add(keras.layers.Dense(368, activation="sigmoid"))
	model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy())

	print("+"*30)
	print(model.summary())
	print("+"*30)
	return model


def read_meta():
	dataset = []
	with open(META_FILE_PATH, 'r') as f:
		reader = csv.DictReader(f)
		for row in reader:
			dataset.append(row)
	return dataset


'''
	decode the predicted one hot vector
'''
def decode(key, x):
	ind = np.array(x).argmax()
	return PARAM_DICT[key][ind]


def reconstruct(prediction, meta):
	print("prediction: {}".format(prediction))
	with open(PREDICTION_FILE_PATH, 'w') as f:
		writer = csv.DictWriter(f, fieldnames=["id", "original_filename"]+PARAMETERS)
		writer.writeheader()
		for i in range(prediction.shape[0]):
			pred = {}
			pred["id"] = meta[i]["id"]
			pred["original_filename"] = meta[i]["filename"]
			for j, param in enumerate(PARAMETERS):
				vector = prediction[i][j:j+NUM_CLASSES]
				value = decode(param, vector)
				pred[param] = value
			writer.writerow(pred)


def main():
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

	if MODEL_NAME is "E2E":
		model = get_e2e_model(label_size)
	elif MODEL_NAME is "C4":
		# not yet implemented
		model = get_c4_model()


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

	params = {"dataset": dataset, 
				"batch_size": 64, 
				"shuffle": True,
				"n_samps": SAMPLE_RATE}
	training_generator = DataGenerator(first=0.8, **params)
	validation_generator = DataGenerator(last=0.2, **params)
	model.fit(
		x=training_generator,
		validation_data=validation_generator,
		epochs=NUM_EPOCH,
		callbacks=callbacks,
		initial_epoch=0,
		verbose=0,  # https://github.com/tensorflow/tensorflow/issues/38064
	)

	# Save model
	model.save(MODEL_SAVE_PATH)

	validation_generator.on_epoch_end()
	X_valid, y_valid = validation_generator.__getitem__(0)
	meta_valid = validation_generator.get_meta(0)
	print("X_valid: {}".format(X_valid))

	prediction = model.predict(X_valid)
	# print("prediction: {}, shape: {}".format(prediction, prediction.shape))
	reconstruct(prediction, meta_valid)



if __name__ == '__main__':
	main()