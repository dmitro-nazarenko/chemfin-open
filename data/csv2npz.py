import os, sys

sys.path.insert(0, "../utils")
from csvload import lcsv

import numpy as np

def load_labels(fnm):
    with open(fnm, 'rw') as f:
        buf = f.readlines()
    lb = []
    for x in buf:
        x = x.replace('\n', '')
        if x == '':
            lb.append( None )
        else:
            sep = x.split(',')
            if len(sep) > 1:
                x = [int(y) - 1 for y in sep]
            else:
                x = int(x) - 1
            lb.append(x)
    del buf
    return lb

outputs = ['ex1', 'ex1a', 'ex2', 'ex3', 'ex4', 'ex5', 'ex6']

# load data

fnms = ['neg.csv', 'pos.csv']


for dirnm in outputs:
    read = []
    path = "./" + dirnm + "/"
    for x in fnms:
        read.append( lcsv(path + x, delim = ',', quote = '"', blank_str = 'Blank', qc_str = None, sample_str = 'Sample') )

    neg = read[0]
    pos = read[1]
    lab = load_labels(path + 'labels.dat')

    np.savez_compressed(path+dirnm, negmz = neg[0], posmz = pos[0],
                                    negrt = neg[1], posrt = pos[1],
                                    negbl = neg[2], posbl = pos[2],
                                    negsp = neg[4], possp = pos[4],
                                    label = lab)

exit()

# construct appropriate input
mza = 100.
mzb = 1000.
mz_step = 1e-0 # 0.01

rta = 0.
rtb = 20.
rt_step = 5./60. # 5 sec as minutes

# evaluate negative peaks:
# polarity_ind = 0 # 0 = neg, 1 = pos
sample_ind = -1
mz_ind = 0
rt_ind = 1
bl_ind = 2
ulist = []

for part_batch in read:
    neg_bl = np.array(part_batch[0][    bl_ind])
    neg_s =  np.array(part_batch[0][sample_ind])
    neg_mz = np.array(part_batch[0][    mz_ind])
    neg_rt = np.array(part_batch[0][    rt_ind])

    pos_s =  np.array(part_batch[1][sample_ind])
    pos_mz = np.array(part_batch[1][    mz_ind])
    pos_rt = np.array(part_batch[1][    rt_ind])
    pos_bl = np.array(part_batch[1][    bl_ind])

    blind1  = [0]*5 + [1]*5 + [2]*5 + [3]*5 + [4]*6
    blind2  = [5]*5 + [6]*5 + [7]*5 + [8]*5 + [9]*6
    tmp = [11]*5 + [12]*5
    blind1  = blind1 + tmp
    blind2  = blind2 + tmp

    blind = np.empty(len(blind1) + len(blind2))
    blind[0::2] = np.array(blind1)
    blind[1::2] = np.array(blind2)

    for k in xrange(neg_s.shape[1]): # must be equal to pos_s.shape[1] // number of samples
        # u is 3D array: (m/z, RT, polarity) = (m/z, RT, 2), but polarity is fixednow
        u = np.zeros( ( (mzb - mza) / mz_step, (rtb - rta) / rt_step, 2))

        for i in xrange(neg_s.shape[0]):
            if neg_s[i, k] != 0:
                mzind = round((neg_mz[i] - mza)/ mz_step) - 1
                rtind = round((neg_rt[i] - rta)/ rt_step) - 1
                #if u[mzind, rtind, 0] != 0:
                u[mzind, rtind, 0] = max(u[mzind, rtind, 0], neg_s[i, k]) # to deal with crossing values
                # getting into account the blank:
                #if neg_bl[i, blind[k]] != 0:
                #    u[mzind, rtind, 0] = 0.
        for i in xrange(pos_s.shape[0]):
            if pos_s[i, k] != 0:
                mzind = round((pos_mz[i] - mza)/ mz_step) - 1
                rtind = round((pos_rt[i] - rta)/ rt_step) - 1
                #if u[mzind, rtind, 0] != 0:
                u[mzind, rtind, 1] = max(u[mzind, rtind, 1], pos_s[i, k]) # to deal with crossing values
                # getting into account the blank:
                #if pos_bl[i, blind[k]] != 0:
                #    u[mzind, rtind, 0] = 0.
        ulist.append(u)

        

u = np.array(ulist)

def gen_lbl(fnm= 'labels'):
    num_plants = 36
    num_trials = 2
    arge = np.arange(num_plants)
    lbl1 = np.zeros(num_plants*num_trials)
    lbl1[0::2] += arge
    lbl1[1::2] += arge
    num_trials = 8
    lbl2 = np.zeros((num_plants, num_trials))
    lbl2[[0, 17, 34], :4] += np.ones((3,1))
    lbl2[[2, 13, 20],4: ] += np.ones((3,1))
    num_trials = 6
    lbl3 = np.zeros(num_plants*num_trials)
    for i in xrange(num_trials):
        lbl3[i::num_trials] += arge
    np.savez_compressed(fnm, l1 = lbl1, l2 = lbl2, l3 = lbl3)










