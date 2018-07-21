#from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import GRU
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.utils import class_weight
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import CSVLogger
from methods import *
#misc
import numpy as np
import os
import keras.backend.tensorflow_backend as tfb
import tensorflow as tf

POS_WEIGHT = 10  # multiplier for positive targets, needs to be tuned


#if False:
if True:
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    session = tf.Session(config=config)

use_previous_model = 0
timesteps = 40
no_features = 60
batch = 750
dropout_rate = 0.2
epochs = 5
input_cols = timesteps * no_features

var_size = 0

if var_size:
    hidden_layers = int(input("\nNumber of Hidden Layers (Minimum 1): "))
    neurons = []
    if hidden_layers == 0:
        hidden_layers = 1;
    for i in range(0, hidden_layers):
        neurons.append(int(input("Number of Neurons in Hidden Layer " + str(i + 1) + ": ")))
else:
    #FIXED SIZE
    hidden_layers = 4
    neurons = [input_cols*2, input_cols*3, input_cols*2, int(input_cols*1.5)]
    neurons = [2000, 1400, 800, 400]
    #neurons = [150, 200,250, 100]
    neurons = [15, 20,25, 10]

if use_previous_model:
    model = load_model("mymodel.h5")
    print("Model loaded ")
else:

    optimizer = 'adam'
    lossfun = 'binary_crossentropy'
    lossfun = weighted_binary_crossentropy

    print("Builing model...")
    model = Sequential()
    if hidden_layers == 1:
        model.add(GRU(neurons[0], input_shape=(timesteps, no_features), stateful=False, reset_after=False))
    else:
        model.add(
            GRU(neurons[0], input_shape=(timesteps, no_features), stateful=False, activation='relu', return_sequences=True))
        model.add(Dropout(dropout_rate))
        for i in range(1, hidden_layers):
            if i == (hidden_layers - 1):
                model.add(GRU(neurons[i], stateful=False, activation='relu'))
            else:
                model.add(GRU(neurons[i], stateful=False, return_sequences=True))
            model.add(Dropout(dropout_rate))

        model.add(Dense(no_features, activation='sigmoid'))
        model.compile(loss=lossfun, metrics=["accuracy"], optimizer=optimizer)
        model.summary()

filepath = "BestGRUWeights.h5"  # Best weights for sampling will be saved here.
filepath2 = "BestGRUWeights2.h5"  # Best weights for sampling will be saved here.
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint2 = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

#values = np.array([]).reshape(0, no_features)

i = 0
dataset_path = "/home/dsabat/datasety/"

print("Scanning files...")

path, dirs, files = next(os.walk(dataset_path))
file_count = len(files)

class_weight = {0: 1.,
                1: 5.}

csv_logger = CSVLogger('log.csv', append=True, separator=';')

for f in range(0,file_count): ################ FOR every file in dataset
    print("Processing " + str(f) + "out of " + str(file_count))
    values = np.load(dataset_path+str(f)+".npy").astype('float16')
    if values.size == 0:
        continue
    print("Loaded file...")

    # frame as supervised learning
    reframed = series_to_supervised(values, timesteps, 1)
    print("Reframed file...")

    # split into train and test sets
    values = reframed.values
    trainPortion = int(0.8 * len(values))

    train = values[:trainPortion, :]
    test = values[trainPortion:, :]
    print(train.shape)
    # split into input and outputs
    train_X, train_y = train[:, :input_cols], train[:, -no_features:]
    test_X, test_y = test[:, :input_cols], test[:, -no_features:]
    #class_weight = class_weight.compute_class_weight('balanced',np.unique(train_y),train_y)
    #print(type(class_weight))
    train_X = train_X.reshape((train_X.shape[0], timesteps, no_features))
    test_X = test_X.reshape((test_X.shape[0], timesteps, no_features))
    print("Splitted data...")

    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch, validation_data=(test_X, test_y),
                        verbose=1, shuffle=True, callbacks=[checkpoint,checkpoint2, csv_logger])
    model.save('mymodel.h5')

