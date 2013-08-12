import numpy as np
import collections
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


if __name__ == '__main__':
    a = np.array([1,2,1,1,5])
    b = np.array([1,2,3,4,5])
    print injectivity(a,b)