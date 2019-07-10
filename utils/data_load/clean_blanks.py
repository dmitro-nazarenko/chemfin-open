import numpy as np

BLANK_THRESHOLD = 0.
dirnms = ['ex1', 'ex2', 'ex3', 'ex4', 'ex5', 'ex6']
bidx_fnm = 'blank_index.dat'

for dirnm in dirnms:
    f = open("./" + dirnm + '/' + bidx_fnm, 'rw')
    buf = f.readlines()
    f.close()

    bidx = []
    for x in buf:
        x = x.replace('\n', '')
        if x == '':
            bidx.append( None )
        else:
            bidx.append( int(x) )
    del buf

    f = np.load("./" + dirnm + '/' + dirnm + '.npz')

    negblmask = f['negbl']
    negmz = f['negmz']
    negrt = f['negrt']

    posblmask = f['posbl']
    posmz = f['posmz']
    posrt = f['posrt']

    label = f['label'].tolist()

    negblmask[abs(negblmask) >  BLANK_THRESHOLD] = -1.
    negblmask[abs(negblmask) <= BLANK_THRESHOLD] =  0.
    negblmask += 1

    posblmask[abs(posblmask) >  BLANK_THRESHOLD] = -1.
    posblmask[abs(posblmask) <= BLANK_THRESHOLD] =  0.
    posblmask += 1

    [_ , Nsamp] = f['negsp'].shape # must be same as posp axis 1

    negsp = []
    possp = []
    newlab = []
    abn = []
    for i in xrange(Nsamp):
        if bidx[i] is None:
            continue
        if bidx[i] < 0:
            abn.append(i)
        negsp.append( f['negsp'][:, i] * negblmask[:, abs(bidx[i])] ) 
        possp.append( f['possp'][:, i] * posblmask[:, abs(bidx[i])] )
        newlab.append(label[i])

    negsp = np.array(negsp)
    possp = np.array(possp)
    newlab = np.array(newlab, dtype=object)

    np.savez_compressed("./" + dirnm + '/' + dirnm + "_blcleaned", negmz = negmz, posmz = posmz,
                                     negrt = negrt, posrt = posrt,
                                     negblmask = negblmask,
                                     posblmask = posblmask,
                                     negsp = negsp.T, possp = possp.T,
                                     label = newlab, abnorm = abn)

