
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
import pretty_midi
import tensorflow as tf

#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#set_session(tf.Session(config=config))

from methods import *

pred_tres = 0.2

# client app

steps = 3  # of simulation
timesteps = 30
no_features = 60
input_cols = timesteps * no_features

dir_model = "/home/dominik/Pulpit/MAGISTERKA/pobrane wagi/5/"
midi_path = '/home/dominik/Pulpit/MAGISTERKA/testoweMidiInput/4.midi'

weight_path = dir_model + 'mymodel.h5'
weight_path = dir_model + 'BestGRUWeights.h5'



def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def convert(path):
    timeres = 20
    midi_data = pretty_midi.PrettyMIDI(path)
    roll = midi_data.get_piano_roll(fs=timeres)
    roll= np.where(roll>0 ,1,0)
    while(np.all(roll[:,0] == 0)): #drop "0" columns
        roll = np.delete(roll,0,1)
    return np.transpose(roll[40:100]) #pitch LtR, time UtD

def convert_back(roll,path):
    midi_out_path = path.split('.')[0] + "-enriched.midi"
    timeres = 20
    roll = np.transpose(roll)
    roll= np.where(roll>0 ,127,0)
    leading_zeros=np.zeros([40,roll.shape[1]])
    roll = np.vstack((leading_zeros, roll))
    bck = piano_roll_to_pretty_midi(roll, fs = timeres)
    bck.write(midi_out_path)

model = load_model(weight_path)
values = convert(midi_path)
inputdata = values
pred_input = values

model.summary()

#while(pred_input.shape[0]<timesteps):
pred_input=np.vstack((pred_input,pred_input))

predictions = []
predictions = np.array(predictions).astype('float32')
predictions = predictions.reshape(predictions.shape[0], no_features)
np.set_printoptions(threshold=np.nan)

for i in range(1, steps):
    reframed = series_to_supervised(pred_input, timesteps, 1)
    # print(reframed)
    values = reframed.values
    inppred = values[-timesteps:, :]
    y = model.predict(inppred.reshape(inppred.shape[0] + 1, timesteps, no_features))
    #y = np.where(y > pred_tres, 1, 0)
    print(y)
    predictions = np.concatenate((predictions, y.reshape(y.shape[0], no_features)), axis=0)
    pred_input = np.concatenate(((pred_input.reshape(pred_input.shape[0], no_features)), y.reshape(y.shape[0], no_features)), axis=0)
out=predictions

if len(predictions) > 0:
    # print(out)
    #out = np.vstack((inputdata,predictions))
    convert_back(out, midi_path)
else:
    print("generated nothing :/")