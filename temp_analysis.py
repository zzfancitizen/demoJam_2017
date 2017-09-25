import numpy as np
import pandas as pds
import os

path = os.path.abspath('./data')

anal = pds.read_csv(filepath_or_buffer=path + os.path.sep + 'temp_anal.csv', sep='\t').as_matrix()

choose = np.random.choice(len(anal), int(len(anal) * 0.005))

lbls = []

for i in range(len(anal)):
    if i in choose:
        lbls.append(0)
    else:
        lbls.append(1)

lbls = np.reshape(lbls, (len(lbls), 1))

anal = np.concatenate((anal, lbls), axis=1)

np.savetxt(path + os.path.sep + 'temp_with_label.csv', X=anal, fmt='%s', delimiter='\t')
