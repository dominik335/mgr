from __future__ import print_function
import numpy as np
import os

from methods import *

no_features = 60
values = np.array([]).reshape(0, no_features)

i = 0
indir = '/home/dsabat/blog/classical/'
for root, dirs, filenames in os.walk(indir):
    for f in filenames:
        if i % 40 == 0:
            print(i)
        i = 1 + i
        if (i==100):
            break
        values = np.vstack((values, convert(indir + f)))

        if values.size == 0:
            continue

np.save("dataset_b_cla.npy",values)
