import numpy as np
from tensorflow import keras

# inputs = keras.Input(shape=(1,16384), name="raw_audio")

# y = keras.layers.Conv1D(filters=96,  kernel_size=64, strides=4, activation='relu', padding="same", data_format='channels_first')(inputs)
# y = keras.layers.Conv1D(filters=96,  kernel_size=32, strides=4, activation='relu', padding="same", data_format='channels_first')(y)
# y = keras.layers.Conv1D(filters=128, kernel_size=16, strides=4, activation='relu', padding="same", data_format='channels_first')(y)
# y = keras.layers.Conv1D(filters=257, kernel_size=8,  strides=4, activation='relu', padding="same", data_format='channels_first')(y)
# print(y.shape) # (None, 257, 61)

# y = keras.layers.Reshape((61, 257, 1), input_shape=(61, 257))(y)
# print(y.shape) # (None, 61, 257, 1)

# y = keras.layers.Conv2D(filters=32,  kernel_size=(3, 3), strides=(2, 2), activation='relu', padding="same", data_format='channels_first')(y)
# y = keras.layers.Conv2D(filters=71,  kernel_size=(3, 3), strides=(2, 2), activation='relu', padding="same", data_format='channels_first')(y)
# y = keras.layers.Conv2D(filters=128,  kernel_size=(3, 4), strides=(2, 3), activation='relu', padding="same", data_format='channels_first')(y)
# y = keras.layers.Conv2D(filters=128,  kernel_size=(3, 3), strides=(2, 2), activation='relu', padding="same", data_format='channels_first')(y)
# y = keras.layers.Conv2D(filters=128,  kernel_size=(3, 3), strides=(2, 2), activation='relu', padding="same", data_format='channels_first')(y)
# y = keras.layers.Conv2D(filters=128,  kernel_size=(3, 3), strides=(1, 2), activation='relu', padding="same", data_format='channels_first')(y)
# print(y.shape)

# y = keras.layers.Flatten()(y)

# y = keras.layers.Dense(512)(y)

# n_outputs = 9
# outputs = keras.layers.Dense(n_outputs, activation="sigmoid", name="predictions")(y)


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


# model.fit(x, y, batch_size=32, epochs=100, verbose=2)





