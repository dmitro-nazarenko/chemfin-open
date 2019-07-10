import numpy as np
from scipy.sparse import csr
from normalize import *

def save_sparse_csr(filename,array):
    np.savez(filename, data = array.data, indices = array.indices,
             indptr = array.indptr, shape = array.shape )

def t2m_full(data, type = 'mean_std', saveindfnm = None):
    sa = data.shape
    data = np.reshape(data, (sa[0], -1), order = 'F')
    a = data.copy()
    a, mean, std = mean_std(a, div = False)
    ind = np.where(mean != 0)[1]
    if saveindfnm is not None:
        si = []
        for i in xrange(ind.size):
            tind = ind[i]
            pol_ind = tind / (sa[1]*sa[2])
            tind = tind % (sa[1]*sa[2])
            rtc_ind = tind / sa[1]
            mz_ind  = tind % sa[1]
            si.append([mz_ind, rtc_ind, pol_ind])
        np.savez(saveindfnm, si)
    a = data
    #ind = ind.astype('i').flatten()
    a = a[:, ind]
    # normalize
    p2n = p2norm(a)
    a /= p2n
    if type == '01':
        amax = np.max(a, axis = 1, keepdims=True)
        a /= amax
        return a, None, None, None
    elif type == 'mean_std':
        a, mean, std = mean_std(a)
        return a, mean, std, p2n


if __name__ == '__main__':
    #fnm = 'data_art.npz'
    #fnm = 'data_raw.npz'#'data_blcleaned.npz'#
    #fnm = 'data_art_train.npz'
    fnms = ['data_art+orig_valid.npz',
            'data_art+orig_train.npz',
            'data_orig_valid.npz',
            'data_orig_train.npz']
    for fnm in fnms:
        f = np.load(fnm)
        data = f['data']
        label = f['label']
        a, mean, std, p2n = t2m_full(data)#, 'ind')
        np.savez_compressed(fnm.replace('data', 'mat'), data = a, mean = mean, std = std, p2n = p2n, label = label, type = 'mean_std')



#asp = csr.csr_matrix(a)
#save_sparse_csr('blcleaned_csr', asp)


