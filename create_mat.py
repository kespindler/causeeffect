import data_io
from scipy.io import savemat, loadmat

train = data_io.read_train_pairs()

for i, t in enumerate(train.values):
    A, B = t
    savemat('matlab/{0:04}.mat'.format(i),
            {'A': A, 'B': B})

