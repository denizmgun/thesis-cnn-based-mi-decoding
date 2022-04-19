from .model import KerasModel
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import BatchNormalization, Dropout, Conv2D, MaxPooling2D, LayerNormalization

import kapre
from kapre import STFT, Magnitude
#from kapre.utils import Normalization2D


class CNN_STFT(KerasModel):


    def create_model(self, input_shape, dropout=0.5, print_summary=False, debug=True):

        # basis of the CNN_STFT is a Sequential network
        model = Sequential()

        
        # Gotta create a spectrogramm and Normalization2D layer with the new kape api.
        
        # spectrogram creation using STFT
        model.add(STFT(n_fft = 128, hop_length = 16, name = 'static_stft', input_shape=input_shape))
        model.add(Magnitude())
        model.add(LayerNormalization(axis=1))
        #model.add(Normalization2D(str_axis = 'freq'))
        
        if not(debug):
            # Conv Block 1
            model.add(Conv2D(filters = 24, kernel_size = (12, 12),
                             strides = (1, 1), name = 'conv1',
                             padding = 'same'))
            model.add(BatchNormalization(axis = 1))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2), padding = 'valid',
                                   data_format = 'channels_last'))

            # Conv Block 2
            model.add(Conv2D(filters = 48, kernel_size = (8, 8),
                             name = 'conv2', padding = 'same'))
            model.add(BatchNormalization(axis = 1))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid',
                                   data_format = 'channels_last'))

            # Conv Block 3
            model.add(Conv2D(filters = 96, kernel_size = (4, 4),
                             name = 'conv3', padding = 'same'))
            model.add(BatchNormalization(axis = 1))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2),
                                   padding = 'valid',
                                   data_format = 'channels_last'))
            model.add(Dropout(dropout))

        # classifier
        model.add(Flatten())
        model.add(Dense(2))  # two classes only
        model.add(Activation('softmax'))

        if print_summary:
            print(model.summary())

        # compile the model
        model.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam',
                      metrics = ['accuracy'])

        # assign model and return
        self.model = model
        return model
