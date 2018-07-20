from __future__ import print_function
import numpy as np
import os

from methods import *

no_features = 60
values = np.array([]).reshape(0, no_features).astype('bool')
i = 0
j = 0
indir = '/home/dsabat/blog/classical/'
indir = '/home/dsabat/danekom/clementi/'
indir = '/home/dominik/Pulpit/MAGISTERKA/midi/midiPopover/'

for root, dirs, filenames in os.walk(indir):
    for f in filenames:
        if i % 40 == 0:
            print(i)
        i = 1 + i
        values = np.vstack((values, convert(indir + f)))

        if (values.shape[0]>=200000):
            np.save("/home/dsabat/datasety/" + str(j), values)
            j=j+1
            values=None
            values = np.array([]).reshape(0, no_features)

np.save("/home/dsabat/datasety/" + str(j), values)
j=j+1
