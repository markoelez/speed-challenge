#!/usr/bin/env python3

import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Activation, ELU, Lambda
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from image_util import ImageUtil


class Model:

    def __init__(self, dsize=(100, 100)):

        self.DSIZE = dsize

        self.WEIGHTS_FN = "weights.h5"
        self.TENSORBOARD = "tensorboard"

        self.model = self.build()


    def build(self, lr=1e-4):
        """Build CNN model

        Based on Nvidia's DAVE-2 architecture with ELU used for activation instead of RELU
        as in CommaAI model and some additional minor adjustments.
        """

        input_shape = (self.DSIZE[0], self.DSIZE[1], 2)

        model = Sequential()

        # Input normalization
        model.add(Lambda(lambda x: x / 127.5, input_shape=input_shape, name='lambda_norm'))

        # 5x5 Convolutional layers with stride of 2x2
        model.add(Conv2D(24, kernel_size=(5, 5),
                        strides=(2, 2),
                        kernel_initializer='he_normal',
                        name='conv1'))
        model.add(ELU(name='elu1')) 
        model.add(Conv2D(36, kernel_size=(5, 5),
                        strides=(2, 2),
                        kernel_initializer='he_normal',
                        name='conv2'))
        model.add(ELU(name='elu2')) 
        model.add(Conv2D(48, kernel_size=(5, 5),
                        strides=(2, 2),
                        kernel_initializer='he_normal',
                        name='conv3'))
        model.add(ELU(name='elu3')) 

        # 3x3 Convolutional layers with stride of 1x1
        model.add(Dropout(0.5))
        model.add(Conv2D(64, kernel_size=(3, 3),
                        strides=(1, 1),
                        kernel_initializer='he_normal',
                        name='conv4'))
        model.add(ELU(name='elu4')) 
        model.add(Conv2D(64, kernel_size=(3, 3),
                        strides=(1, 1),
                        kernel_initializer='he_normal',
                        name='conv5'))
        model.add(ELU(name='elu5')) 

        # Flatten before passing to fully connected layers
        model.add(Flatten())

        # Three fully connected layers
        model.add(Dropout(0.5, name='do1'))
        model.add(Dense(100, name='fc1', kernel_initializer='he_normal'))
        model.add(Dropout(0.5, name='do2'))
        model.add(ELU(name='elu6')) 
        model.add(Dense(50, name='fc2', kernel_initializer='he_normal'))
        model.add(Dropout(0.5, name='do3'))
        model.add(ELU(name='elu7')) 
        model.add(Dense(10, name='fc3', kernel_initializer='he_normal'))
        model.add(ELU(name='elu8')) 

        # Output
        model.add(Dense(1, kernel_initializer='he_normal', name='output'))

        print(model.summary())

        adam = Adam(lr=lr)

        model.compile(optimizer=adam, loss='mse')

        return model

    def train(self, x_train, y_train, epochs, batch_size, val_split):

        tensorBoard = TensorBoard(log_dir=self.TENSORBOARD, histogram_freq=0,
                                  write_graph=True,
                                  write_images=True)

        callbacks = [tensorBoard]

        x_train = x_train[..., [0, 2]]

        history = self.model.fit(x_train, y_train,
                       verbose=1,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=val_split,
                       callbacks=callbacks)

        pickle.dump(history.history, open('history.p', 'wb'))
        
        print("Finished training. Saving weights to {}".format(self.WEIGHTS_FN))

        self.model.save_weights(self.WEIGHTS_FN)

    def predict(self, x_test):
        x_test = x_test[..., [0, 2]]
        pred = self.model.predict(x_test)
        return np.array(pred)

    def load_weights(self, path):
        try:
            print("Loading weights")
            self.model.load_weights(path)
            return True
        except ValueError:
            print("Unable to load weights")
            return False
        except IOError:
            print("Unable to load weights")
            return False

    def test(self, x_test, y_test):
        x_test = x_test[..., [0 ,2]]
        return self.model.evaluate(x_test, y_test)

