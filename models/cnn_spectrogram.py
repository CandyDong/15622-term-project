import argparse
import os
import pprint
import csv
import pandas as pd
import tensorflow as tf
from tensorflow import keras

pp = pprint.PrettyPrinter(indent=4, width=1)

SAMPLE_RATE = 16384
OUT_DIR = "../data"
MODEL_DIR = "../model/"
WAV_DIR = "../data/wav_files"
MODEL_NAME = "E2E" 

NUM_EPOCH = 100
BATCH_SIZE = 64
N_DFT = 512
N_HOP = 256

META_FILE_PATH = os.path.join(OUT_DIR, "meta.csv")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "{}.h5".format(MODEL_NAME))

if not os.path.exists(OUT_DIR):
	os.makedirs(OUT_DIR)
if not os.path.exists(MODEL_DIR):
	os.makedirs(MODEL_DIR)

PARAMETERS = sorted(["C", "M", "A", "D", "attack", "decay", "sustain", "sustain_level", "release"])

def get_e2e_model():
	model = keras.Sequential()
	model.add(keras.Input(shape=(1,16384)))
	model.add(keras.layers.Conv1D(filters=96,  kernel_size=64, strides=4, activation='relu', padding="same", data_format='channels_first'))
	model.add(keras.layers.Conv1D(filters=96,  kernel_size=32, strides=4, activation='relu', padding="same", data_format='channels_first'))
	model.add(keras.layers.Conv1D(filters=128, kernel_size=16, strides=4, activation='relu', padding="same", data_format='channels_first'))
	model.add(keras.layers.Conv1D(filters=257, kernel_size=8,  strides=4, activation='relu', padding="same", data_format='channels_first'))

	model.add(keras.layers.Reshape((61, 257, 1), input_shape=(61, 257)))

	model.add(keras.layers.Conv2D(filters=32,  kernel_size=(3, 3), strides=(2, 2), activation='relu', padding="same", data_format='channels_first'))
	model.add(keras.layers.Conv2D(filters=71,  kernel_size=(3, 3), strides=(2, 2), activation='relu', padding="same", data_format='channels_first'))
	model.add(keras.layers.Conv2D(filters=128,  kernel_size=(3, 4), strides=(2, 3), activation='relu', padding="same", data_format='channels_first'))
	model.add(keras.layers.Conv2D(filters=128,  kernel_size=(3, 3), strides=(2, 2), activation='relu', padding="same", data_format='channels_first'))
	model.add(keras.layers.Conv2D(filters=128,  kernel_size=(3, 3), strides=(2, 2), activation='relu', padding="same", data_format='channels_first'))
	model.add(keras.layers.Conv2D(filters=128,  kernel_size=(3, 3), strides=(1, 2), activation='relu', padding="same", data_format='channels_first'))

	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(512))

	model.add(keras.layers.Dense(9, activation="softmax", name="predictions"))
	model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy())

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
	print("read_meta")
	dataset = []
	with open(META_FILE_PATH, 'r') as f:
		reader = csv.DictReader(f)
		for row in reader:
			dataset.append(row)
	return dataset


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
	print("dataset loaded......")

	if MODEL_NAME is "E2E":
		model = get_e2e_model()
	elif MODEL_NAME is "C4":
		# not yet implemented
		model = get_c4_model()



if __name__ == '__main__':
	main()