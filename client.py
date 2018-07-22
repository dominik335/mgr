
from __future__ import print_function
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

#misc
import numpy as np

import os

from pandas import DataFrame
from pandas import concat
import tensorflow as tf

from methods import *

# client
steps = 6  # of simulation
timesteps = 40
no_features = 60
input_cols = timesteps * no_features

dir_model = "/home/dominik/Pulpit/MAGISTERKA/pobrane wagi/7/"
midi_path = '/home/dominik/Pulpit/MAGISTERKA/testoweMidiInput/4.midi'

weight_path = dir_model + 'mymodel.h5'
#weight_path = dir_model + 'BestGRUWeights.h5'

model = load_model(weight_path, custom_objects={ 'weighted_binary_crossentropy': weighted_binary_crossentropy})
values = convert(midi_path)
inputdata = values
pred_input = values

model.summary()

#while(pred_input.shape[0]<timesteps):
pred_input=np.vstack((pred_input,pred_input))
pred_input=np.vstack((pred_input,pred_input))
pred_input=np.vstack((pred_input,pred_input))

predictions = []
predictions = np.array(predictions).astype('float16')
predictions = predictions.reshape(predictions.shape[0], no_features)
np.set_printoptions(threshold=np.nan)

for i in range(1, steps):
    reframed = series_to_supervised(pred_input, timesteps, 1)
    # print(reframed)
    values = reframed.values
    inppred = values[-timesteps:, :]
    y = model.predict(inppred.reshape(inppred.shape[0] + 1, timesteps, no_features))
    #y = np.where(y > pred_tres, 1, 0)
    #print(y)
    predictions = np.concatenate((predictions, y.reshape(y.shape[0], no_features)), axis=0)
    pred_input = np.concatenate(((pred_input.reshape(pred_input.shape[0], no_features)), y.reshape(y.shape[0], no_features)), axis=0)
out=predictions

if len(predictions) > 0:
    print(  np.array2string(np.where(predictions>=0.5 ,1,0), max_line_width=np.inf))
    #out = np.vstack((inputdata,predictions))
    convert_back(out, midi_path)
else:
    print("generated nothing :/")