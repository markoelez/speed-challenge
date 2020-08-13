#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation, ELU
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from image_util import ImageUtil


class Model:

    def __init__(self, dsize=(100, 100)):

        self.DSIZE = dsize

        self.WEIGHTS_FN = "weights.h5"
        self.TENSORBOARD = "tensorboard"

        self.model = self.build()


    def build(self):

        print("Creating model...")

        input_shape = (self.DSIZE[0], self.DSIZE[1], 2)

        model = Sequential()
        model.add(Conv2D(4, kernel_size=(5, 5),
                        padding='same',
                        activation='relu',
                        data_format = "channels_last",
                        input_shape=input_shape))
        model.add(MaxPool2D())

        model.add(Conv2D(8, kernel_size=(5, 5),
                        padding='same',
                        activation='relu',
                        data_format = "channels_last"))
        model.add(MaxPool2D())

        model.add(Conv2D(32, kernel_size=(5, 5),
                        padding='same',
                        activation='relu',
                        data_format = "channels_last"))
        model.add(MaxPool2D())

        model.add(Conv2D(64, kernel_size=(3, 3),
                        padding='same',
                        activation='relu',
                        data_format = "channels_last"))
        model.add(MaxPool2D())

        model.add(Dropout(0.3))
        model.add(Conv2D(128, kernel_size=(3, 3),
                        padding='same',
                        activation='relu',
                        data_format = "channels_last"))
        model.add(MaxPool2D())

        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1))

        print(model.summary())

        model.compile(optimizer='adam', loss='mse')

        return model

    def train(self, x_train, y_train, epochs, batch_size, val_split):

        tensorBoard = TensorBoard(log_dir=self.TENSORBOARD, histogram_freq=0,
                                  write_graph=True,
                                  write_images=True)

        callbacks = [tensorBoard]

        x_train = x_train[..., [0, 2]]
        x_train = x_train / 127.5

        self.model.fit(x_train, y_train,
                       verbose=1,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=val_split,
                       callbacks=callbacks)
        
        print("Finished training. Saving weights to {}".format(self.WEIGHTS_FN))

        self.model.save_weights(self.WEIGHTS_FN)

    def load_weights(self):
        try:
            print("Loading weights")
            self.model.load_weights(self.WEIGHTS_FN)
            return True
        except ValueError:
            print("Unable to load weights")
            return False
        except IOError:
            print("Unable to load weights")
            return False

    def test(self, x_test, y_test):
        x_test = x_test[:,:,:,[0,2]]

        ret = self.load_weights()
        if ret:
            print("Testing")
            print(self.model.evaluate(x_test, y_test))
        else:
            print("Unable to begin testing")

