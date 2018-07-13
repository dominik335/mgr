#from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import GRU
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense

from methods import *
#misc
import numpy as np
import os

if False:
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    session = tf.Session(config=config)

timesteps = 40
no_features = 60
batch = 30
dropout_rate = 0.2
epochs = 30
input_cols = timesteps * no_features

if False:
    hidden_layers = int(input("\nNumber of Hidden Layers (Minimum 1): "))
    neurons = []
    if hidden_layers == 0:
        hidden_layers = 1;
    for i in range(0, hidden_layers):
        neurons.append(int(input("Number of Neurons in Hidden Layer " + str(i + 1) + ": ")))
else:
    #FIXED SIZE
    hidden_layers = 4
    neurons = [input_cols*2, input_cols*3, input_cols*2, input_cols*1.5]

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

    model.add(Dense(no_features, activation='tanh'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.summary()

filepath = "BestGRUWeights.h5"  # Best weights for sampling will be saved here.
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#values = np.array([]).reshape(0, no_features)

i = 0
dataset_path = "/home/dsabat/datasety/"

print("Scanning files...")

path, dirs, files = next(os.walk(dataset_path))
file_count = len(files)

for f in range(0,file_count): ################ FOR every file in dataset
    print("Processing " + str(f) + "out of " + str(file_count))

    values = np.load(dataset_path+str(f)+".npy")
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

    train_X = train_X.reshape((train_X.shape[0], timesteps, no_features))
    test_X = test_X.reshape((test_X.shape[0], timesteps, no_features))
    print("Splitted data...")

    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch, validation_data=(test_X, test_y),
                        verbose=1, shuffle=True, callbacks=[checkpoint])
    # plot history
    f = open('history_of_training', 'a')
    f.write(history)
    f.close()
model.save('mymodel.h5')
