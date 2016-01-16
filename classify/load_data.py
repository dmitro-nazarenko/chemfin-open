import numpy as np

def load_full(datanm, num):
    fl = np.load(datanm)

    x = fl['data']
    sx = x.shape
    #x = x.reshape((sx[0], -1))

    y = fl['label']
    y = y.astype('int32')
    numcl = y.max() + 1

    ind = np.arange(sx[0])
    p    = np.empty((0))

    for i in xrange(numcl):
        ind_y = np.where(y == i)[0]
        ind_y = np.random.permutation(ind_y)
        p    = np.concatenate( (p, ind_y[:num]))
    p = p.astype('i')

    return x[p,:], y[p]

def load_mat(datanm, num_train, num_valid):
    fl = np.load(datanm)

    x = fl['data']
    sx = x.shape
    #x = x.reshape((sx[0], -1))

    y = fl['label']
    y = y.astype('int32')
    numcl = y.max() + 1

    ind = np.arange(sx[0])
    p    = np.empty((0))
    pval = np.empty((0))

    for i in xrange(numcl):
        ind_y = np.where(y == i)[0]
        ind_y = np.random.permutation(ind_y)
        p    = np.concatenate( (p   , ind_y[         :num_train          ]))
        pval = np.concatenate( (pval, ind_y[num_train:num_train+num_valid]))

    p = p.astype('i')
    pval = pval.astype('i')
    train_data = [x[p   ,:], y[p   ]]
    valid_data = [x[pval,:], y[pval]]
    return train_data, valid_data
