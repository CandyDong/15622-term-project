import h5py
import numpy as np
from scipy.io import wavfile
from tensorflow import keras


class DataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        dataset=None,
        batch_size=32,
        n_samps=16384,
        shuffle=True,
        last: float = 0.0,
        first: float = 0.0,
        channels_last=False,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.label_size = len(np.fromstring(dataset[0]['label'][1:-1], dtype=float, sep=' '))
        self.n_channels = 1
        self.n_samps = n_samps
        # For the E2E model, need to return channels last?
        if channels_last:
            self.expand_axis = 2
        else:
            self.expand_axis = 1

        # set up list of IDs from data files
        self.list_IDs = range(self.dataset_size)

        print(f"Number of examples in dataset: {len(self.list_IDs)}")
        slice: int = 0
        if last > 0.0:
            slice = int(self.dataset_size * (1 - last))
            self.list_IDs = self.list_IDs[slice:]
            print(f"Taking Last {len(self.list_IDs)} points")
        elif first > 0.0:
            slice = int(self.dataset_size * first)
            self.list_IDs = self.list_IDs[:slice]
            print(f"Taking First {len(self.list_IDs)} points")

        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(indexes)

        # print("Returning data! Got X: {}, y: {}".format(X.shape,y.shape))
        return X, y

    def get_meta(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        meta = []
        for i in indexes:
            meta.append(self.dataset[i])
        return meta

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def read_file(self, index):
        filename = self.dataset[index]["filename"]
        fs, data = wavfile.read(filename)
        return data

    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, *dim, n_channels)
        "Generates data containing batch_size samples"
        # Initialization
        # X = np.empty((self.batch_size, *self.dim))
        # y = np.empty((self.batch_size), dtype=int)

        # Generate data
        X = []
        y = []
        for i in list_IDs_temp:
            # Read labels
            y.append(np.fromstring(self.dataset[i]["label"][1:-1], dtype=float, sep=' '))
            # Load soundfile data
            data = self.read_file(i)
            if data.shape[0] > self.n_samps:
                print(
                    "Warning - too many samples: {} > {}".format(
                        data.shape[0], self.n_samps
                    )
                )
            X.append(data[: self.n_samps])
        Xd = np.expand_dims(np.vstack(X), axis=2)
        yd = np.vstack(y)

        # print("X: {}, y: {} \n X shape: {}, Y shape: {}".format(Xd, yd, Xd.shape, yd.shape))
        return Xd, yd
