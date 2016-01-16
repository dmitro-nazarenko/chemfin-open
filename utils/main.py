import os, sys

sys.path.insert(0, "utils")
from csvload import lcsv, loadpar

import numpy as np

# load data
rootdir = './data/peak_tables/'

dirnms = [x[0] for x in os.walk(rootdir)]
dirnms = filter(lambda x: 'batch' in x, dirnms)
dirnms.sort()

read = []
for curdir in dirnms:
    fnms = os.listdir(curdir)
    fnms.sort()
    fnms = filter(lambda x: x.endswith(".csv"), fnms)
    fnms = filter(lambda x: x != 'sample_batch.csv', fnms)    


    part_read = []
    for x in fnms:
        part_read.append( lcsv(curdir + '/'+ x) )
        print x
    read.append(part_read)


# construct appropriate input

exp_file = './data/sample_batch.csv'
#meaningful_interval = 
#1 26
#1 26


#blank, flav, trial = loadpar(exp_file)
#exit()



mza = 100.
mzb = 1000.
mz_step = 1e-0 # 0.01

rta = 0.
rtb = 20.
rt_step = 1./12 # 5 sec as minutes



# evaluate negative peaks:
#polarity_ind = 0 # 0 = neg, 1 = pos
sample_ind = -1
mz_ind = 0
rt_ind = 1
bl_ind = 2
ulist = []

def cut_ind(x, indices):
    rv = x.copy()
    h = np.arange(len(indices))
    indices = np.array(indices)
    indices -= h
    for i in indices:
        tmp = rv[:, i+1:]
        rv = np.hstack( (rv[:, :i], tmp) )
    return rv

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










