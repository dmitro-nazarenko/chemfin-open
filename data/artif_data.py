import os, sys
import numpy as np
import scipy.stats as stats
from sklearn.cross_validation import StratifiedShuffleSplit

from t2m import t2m_full

def art_gen(a, b, al, bl, ae, be, svnm, coef1,
            amp=100, # artificial data size = data size * amp
            m = 30, #  magnitude of peak area change is +/- m %
            r1 = 15,
            r2 = 30): # i.e 15-30% of peak's amplitude will be changed

    r1 /= 100.
    r2 /= 100.
    m /= 100.

    coef2 = 1. - coef1
    tamp = int(coef1 * amp)
    vamp = int(coef2 * amp)

    sa = a.shape
    sb = b.shape

    a = np.reshape(a, (len(a), -1))
    b = np.reshape(b, (len(b), -1))
    sf = np.array(sa[1:]).prod()
    assert sf == np.array(sb[1:]).prod(), "Incompatible feature spaces"

    S = np.arange(sf)#S = np.arange (1, len(A) + 1 ) # HERE WAS A DRASTIC ERROR!! I've saved original as comment (list for choosing index of peak)

    #Art = np.zeros(((len(A[:])*amp, len(A[1][:])))) # array for storing artificial data
    artae = [x + '_artificial' for x in ae]
    artbe = [x + '_artificial' for x in be]
    artal = al.copy()
    artbl = bl.copy()
    arta  =  a.copy()
    artb  =  b.copy()

    for i in xrange(max(tamp-1, vamp-1)):
        if i < tamp-1:
            artal = np.concatenate ((artal, al))
            artae = np.concatenate ((artae, ae))
            arta = np.concatenate((arta, a))
        if i < vamp-1:
            artbl = np.concatenate ((artbl, bl))
            artbe = np.concatenate ((artbe, be))
            artb = np.concatenate((artb, b))

    for j in xrange(max(vamp-1, tamp-1)):
        for i in xrange(max(sa[0], sb[0])):
            if i + (j*len(a)) < sa[0]*(tamp - 1):
                ka = 0
                lim = np.count_nonzero(a[i, :]) * np.random.uniform(r1, r2) # 15-30% nonzero peaks
                lim = np.trunc(lim)
                while ka < lim:
                    l = np.random.choice(S)
                    if a[i, l] != 0:
                        arta[i + (j*len(a)), l] = a[i, l] * (1. - np.random.uniform(-m, m)) # i.e magnitude of change is +/- 30 %
                        ka += 1
            if i + (j*len(b)) < sb[0]*(vamp - 1):
                kb = 0
                lim = np.count_nonzero(b[i, :]) * np.random.uniform(r1, r2) # 15-30% nonzero peaks
                lim = np.trunc(lim)
                while kb < lim:
                    l = np.random.choice(S)
                    if b[i, l] != 0:
                        artb[i + (j*len(b)), l] = b[i, l] * (1. - np.random.uniform(-m, m)) # i.e magnitude of change is +/- 30 %
                        kb += 1

    tind  = np.arange(sa[0] * (tamp-1))
    otind = np.arange(sa[0] * (tamp-1), sa[0] * tamp)

    vind = np.arange(sa[0]*tamp, sa[0]*tamp + sb[0]*vamp)
    ovind = np.arange(sa[0]*tamp + sb[0]*(vamp - 1), sa[0]*tamp + sb[0]*vamp)

    art = np.vstack((arta, artb))
    arte = np.concatenate((artae, artbe))
    artl = np.concatenate((artal, artbl))

    #print art.shape, sa[0]*tamp, sb[0]*vamp
    art = art.reshape(sa[0]*tamp + sb[0]*vamp, sa[1], sa[2])
    np.savez_compressed(svnm, data = art, label = artl, expnm = arte, tind = tind, vind = vind, otind = otind, ovind = ovind)
    
    mat_art, mean, std, p2n = t2m_full(art)
    ind = np.concatenate((tind, otind))
    np.savez_compressed('mat_ao_a', data = mat_art[ind], mean = mean, std = std, p2n = p2n, label = artl[ind], type = 'mean_std')
    ind = np.concatenate((vind, ovind))
    np.savez_compressed('mat_ao_b', data = mat_art[ind], mean = mean, std = std, p2n = p2n, label = artl[ind], type = 'mean_std')

    np.savez_compressed('mat_o_a', data = mat_art[otind], mean = mean, std = std, p2n = p2n, label = artl[otind], type = 'mean_std')
    np.savez_compressed('mat_o_b', data = mat_art[ovind], mean = mean, std = std, p2n = p2n, label = artl[ovind], type = 'mean_std')
    

if __name__ == '__main__':
    fnm = 'data_raw.npz'

    F = np.load(fnm)
    data = F["data"]
    label = F["label"]
    expnm = F["expnm"]

    slo = StratifiedShuffleSplit(label, n_iter=1, test_size=0.5, train_size=0.5, random_state=None)
    for train_ind, valid_ind in slo:
        pass

    svnm = 'data_art'
    #svnm5 = 'data_orig_full'

    #full_ind = train_ind.tolist() + valid_ind.tolist()

    #np.savez_compressed(svnm3, data = data[train_ind], label = label[train_ind], expnm = expnm[train_ind])
    #np.savez_compressed(svnm4, data = data[valid_ind], label = label[valid_ind], expnm = expnm[valid_ind])
    #np.savez_compressed(svnm5, data = data[full_ind], label = label[full_ind], expnm = expnm[full_ind])

    coef_train = 0.7
    #coef_valid = 1. - coef_train

    prodinc = 100

    a = data[train_ind]
    b = data[valid_ind]
    al = label[train_ind]
    bl = label[valid_ind]
    ae = expnm[train_ind]
    be = expnm[valid_ind]

    art_gen(a, b, al, bl, ae, be, svnm, coef_train, prodinc)









