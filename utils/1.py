import numpy as np
import pyopenms as ms
import os
import matplotlib.pyplot as plt
from itertools import islice

datadir = './data/'
fls = os.listdir(datadir)


file = ms.MzXMLFile()
exp = ms.MSExperiment()
file.load(datadir+fls[0], exp)

alim = 100.
blim = 900.
step = 0.01

def findOverlap(a, b, rtol = 1e-05, atol = 1e-08):
    ovr_a = []
    ovr_b = []
    start_b = 0
    for i, ai in enumerate(a):
        for j, bj in islice(enumerate(b), start_b, None):
            if np.isclose(ai, bj, rtol=rtol, atol=atol, equal_nan=False):
                ovr_a.append(i)
                ovr_b.append(j)
            elif bj > ai: # (more than tolerance)
                break # all the rest will be farther away
            else: # bj < ai (more than tolerance)
                start_b += 1 # ignore further tests of this item
    return (ovr_a, ovr_b)

def sp2list(exp, alim, blim, step):

    template = np.arange(alim, blim, step).reshape((1,-1))
    tmp = np.zeros((1, template.size))

    template = np.vstack((template, tmp))

    f = exp.getSpectra()
    sp = np.zeros((len(f), template.shape[1]))
    for i in xrange(2):#len(f)):
        tmp =  f[i].get_peaks()
        tmp = np.vstack((tmp[0].round(int(np.log10(1/step))), tmp[1]))
        ind, ind2 = findOverlap(template[0], tmp[0])
        sp[i, ind] = tmp[1]
    return sp

spec = sp2list(exp, alim, blim, step)

#a = np.array(spec)


# How to heal the files:
# row 4: <parentFile fileName="file://C:\LabSolutions\Data\Project1\2015\Aug\19_08_2015\27_1k10_10ul#1.lcd" fileType="RAWData" fileSha1="be3d0b00e80df8a021294dcea8e4a84e5c289f70" />
# (example)



# Working!
# exp.getTIC() -> get_peaks()
# exp.getInstrument() -> getVendor
#                     -> getSoftware (Name, Version)
# HPLC


# IMPORTANT:

## F = exp.getSpectra() - list of spectra as special objects
## F.get_peaks() - returns two np.arrays (mz, I)

