import data_io
import numpy as np
from scipy.io import savemat, loadmat

DATA = 'valid'

def discretize(d):
    nbins = np.ceil(3.5*np.std(d) / len(d)**1.3);
    edges = np.linspace(np.amin(d),np.amax(d),nbins+1);
    return np.digitize(d, edges);

def singleH(d, dtype):
    if dtype == 'Numerical':
        d = discretize(d)
    freq = np.bincount(d.astype('int32')).astype('float64') / len(d)
    freqnz = freq[np.nonzero(freq)]
    hpos = np.dot(np.log(freqnz), freqnz)
    return -hpos

def jointH(a, b, atype, btype):
    if atype == 'Numerical':
        a = discretize(a)
    if btype == 'Numerical':
        b = discretize(b)
    
    proba = np.bincount(a.astype('int32')).astype('float64') / len(a)
    probb = np.bincount(b.astype('int32')).astype('float64') / len(b)

    proba_nz = proba[np.nonzero(proba)]
    probb_nz = probb[np.nonzero(probb)]

    jointp = np.outer(proba_nz, probb_nz)
    hpos = np.sum(np.log(jointp) * jointp)
    return -hpos

if __name__ == '__main__':

    print 'Reading in {} data...'.format(DATA)

    if DATA == 'train':
        info = data_io.read_train_info() 
        train = data_io.read_train_pairs()
    elif DATA == 'valid':
        info = data_io.read_valid_info() 
        train = data_io.read_valid_pairs()
    else:
        raise ValueError

    print 'Saving coded info matrix...'
    codes = np.zeros(info.values.shape)
    lookup = {'Numerical':1,
            'Categorical':2,
            'Binary':3}
    for i, t in enumerate(info.values):
        a,b = t
        codes[i,:] = [lookup[a], lookup[b]]

    savemat('matlab/{}info.mat'.format(DATA),
            {'codes': codes}, oned_as='column')

    print 'Saving value matrices...'
    for i, t in enumerate(train.values):
        A, B = t
        savemat('matlab/{0}{1:04}.mat'.format(DATA, i),
                {'A': A, 'B': B})

    n_samples = len(info.values)
    conditional_entropy = np.zeros((n_samples, 2))

    for i in range(n_samples):
        print 'Processing entropy %d...' % (i, )
        a, b = train.values[i]
        atype, btype = info.values[i]
        h_joint = jointH(a, b, atype, btype)
        h_a = singleH(a, atype)
        h_b = singleH(b, btype)
        h_a_cond_b = h_joint - h_a
        h_b_cond_a = h_joint - h_b
        conditional_entropy[i,:] = [h_a_cond_b, h_b_cond_a]

    savemat('matlab/{}entropy.mat'.format(DATA),
            {'entropy': conditional_entropy}, oned_as='column')
