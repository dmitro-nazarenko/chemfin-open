import numpy as np

def mean_std(data, div = True):
    a = data.copy()
    mean = np.median(a, axis = 0, keepdims = True)
    a -= mean
    std  = np.std(a, axis = 0, keepdims = True)
    if div:
        a /= std
    return a, mean, std

def size2cd(data, c = 0., d = 1.):
    amin = np.min(data, axis = 0, keepdims = True)
    amax = np.max(data, axis = 0, keepdims = True)
    a = data - amin
    a /= amax / d
    return a
    
def p2norm(data):
    nrm = np.linalg.norm(data, axis = 0)
    nrm = np.reshape(nrm, (1, -1))
    return nrm
