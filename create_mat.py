import data_io
import numpy as np
from scipy.io import savemat, loadmat

info = data_io.read_train_info()
codes = np.zeros(info.values.shape)
lookup = {'Numerical':1,
        'Categorical':2,
        'Binary':3}
for i, t in enumerate(info.values):
    a,b = t
    codes[i,:] = [lookup[a], lookup[b]]

savemat('matlab/info.mat'.format(i),
        {'codes': codes})
exit()

train = data_io.read_train_pairs()
for i, t in enumerate(train.values):
    A, B = t
    savemat('matlab/{0:04}.mat'.format(i),
            {'A': A, 'B': B})

