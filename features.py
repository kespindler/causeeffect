import numpy as np
from sklearn.base import BaseEstimator
from numpy import log
from scipy.special import psi
from scipy.stats.stats import pearsonr
from scipy.integrate import quad, dblquad
from scipy.stats import gaussian_kde
import collections
from scipy.io import loadmat
from create_mat import DATA

inf = float('inf')
finf = lambda x: inf
nfinf = lambda x: -inf

entropy = loadmat('matlab/{}entropy.mat'.format(DATA))['entropy']
decisions = loadmat('matlab/{}decisions.mat'.format(DATA))['decisions'].flatten()

class FeatureMapper:
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        for feature_name, column_names, extractor in self.features:
            extractor.fit(X[column_names], y)

    def transform(self, X):
        extracted = []
        for feature_name, column_names, extractor in self.features:
            fea = extractor.transform(X[column_names])
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

    def fit_transform(self, X, y=None):
        extracted = []
        for feature_name, column_names, extractor in self.features:
            fea = extractor.fit_transform(X[column_names], y)
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

def identity(x):
    #type of x is <type 'numpy.ndarray'>
    return x

def measureAbsSum(list1):
    """takes a list of values of the y-values associated with one x value, returns their absolute distance
    from the mean --should account for number of samples?? """
    mean = np.mean(list1)
    
    return sum([abs(1.0*x-mean) for x in list1])

def injectivity(x,y):
    """given a measure, returns an metric of the 'injectivity' of x -> y"""
    assert (len(x) == len(y))
    dx = collections.defaultdict(list)
    for index in xrange(len(x)):
        dx[x[index]].append(y[index])
    
    totalscore = 0.0
    for k, v in dx.iteritems():
        if len(v) > 1: #each x value has more then 1 y-value
            totalscore += measureAbsSum(v)
    
    return totalscore    
        
def count_unique(x):
    return len(set(x))

def percentage_unique(x):
    return 1.0 * count_unique(x)/len(x)

def inverse_conditional_info(i):
    return entropy[i, 1]

def conditional_info(i):
    return entropy[i, 0]

def anm_decision(i):
    return decisions[i]

def normalized_entropy(x):
    x = (x - np.mean(x)) / np.std(x)
    x = np.sort(x)
    hx = 0.0;
    for i in range(len(x)-1):
        delta = x[i+1] - x[i];
        if delta != 0:
            hx += np.log(np.abs(delta));
    hx = hx / (len(x) - 1) + psi(len(x)) - psi(1);
    return hx

def entropy_difference(x, y):
    return normalized_entropy(x) - normalized_entropy(y)

def correlation(x, y):
    return pearsonr(x, y)[0]

def correlation_magnitude(x, y):
    return abs(correlation(x, y))

class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(x) for x in X], ndmin=2).T

class MultiColumnTransform(BaseEstimator):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(*x[1]) for x in X.iterrows()], ndmin=2).T
