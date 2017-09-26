import pandas as pds
import numpy as np
import os
import random

PATH = os.path.abspath('../data')

file = pds.read_csv(filepath_or_buffer=PATH + os.path.sep + 'temp_anal.csv', delimiter='\t', header=None).as_matrix()

dataset = file[:, 1:]

X_median = np.median(dataset, axis=0)
X_std = np.std(dataset, axis=0, dtype=np.float32)

# Normalization distribution
normal_good = np.random.normal(X_median[0], 3.0, 1000)
normal_bad = np.random.normal(X_median[0], 5.0, 1000)

normal_gmax = np.max(normal_good, axis=0)
normal_gmin = np.min(normal_good, axis=0)

normal_bmax = np.max(normal_bad, axis=0)
normal_bmin = np.min(normal_bad, axis=0)

# print(normal_gmin, normal_gmax, normal_bmin, normal_bmax)

lbls = []

for i in range(len(dataset)):
    if normal_gmin <= dataset[i, 0] <= normal_gmax:
        lbls.append(random.uniform(0, .25))
    elif (normal_bmin <= dataset[i, 0] < normal_gmin) or \
            (normal_gmax < dataset[i, 0] <= normal_bmax):
        lbls.append(random.uniform(.26, .4))
    else:
        lbls.append(random.uniform(.6, 1.0))

# print(lbls)

lbls = np.reshape(lbls, (-1, 1))

# print(file.shape)
# print(lbls.shape)

new_file = np.concatenate((file, lbls), axis=1)

np.savetxt(PATH + os.path.sep + 'temp_with_label2.csv', fmt='%s', X=new_file, delimiter='\t')
