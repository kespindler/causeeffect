import numpy as np
import data_io
from scipy.io import savemat, loadmat
from sklearn.base import BaseEstimator
from scipy.special import psi
from scipy.stats.stats import pearsonr
from scipy.integrate import quad, dblquad
from scipy.stats import gaussian_kde
import collections

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

print 'Reading in data...'
info = data_io.read_train_info()
train = data_io.read_train_pairs()
n_samples = len(info.values)
conditional_entropy = np.zeros((n_samples, 2))

for i in range(n_samples):
    print 'Processing %d...' % (i, )
    a, b = train.values[i]
    atype, btype = info.values[i]
    h_joint = jointH(a, b, atype, btype)
    h_a = singleH(a, atype)
    h_b = singleH(b, btype)
    h_a_cond_b = h_joint - h_a
    h_b_cond_a = h_joint - h_b
    conditional_entropy[i,:] = [h_a_cond_b, h_b_cond_a]

savemat('matlab/entropy.mat'.format(i),
        {'entropy': conditional_entropy}, oned_as='column')