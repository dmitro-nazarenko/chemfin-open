import os, sys
import numpy as np

def tens_old(drnms, labels,
               mza = 100.,
               mzb = 1000.,
               mz_step = 1e-0, # 0.01
               rta = 0.,
               rtb = 20.,
               rt_step = 5./60., # 5 sec as minutes
               blcleaned = True):
    # evaluate negative peaks:
    # polarity_ind = 0 # 0 = neg, 1 = pos
    sample_ind = -1
    mz_ind = 0
    rt_ind = 1
    bl_ind = 2
    ulist = []
    llist = []
    elist = []

    for drnm in drnms:
        if blcleaned:
            fnm = drnm + '_blcleaned.npz'
        else:
            fnm = drnm + '.npz'
        f = np.load("./" + drnm + '/' + fnm)

        negsp = f['negsp']
        negmz = f['negmz']
        negrt = f['negrt']

        possp = f['possp']
        posmz = f['posmz']
        posrt = f['posrt']

        label = f['label']

        for k in xrange(negsp.shape[1]): # must be equal to pos_s.shape[1] // number of samples

            if label[k] not in labels:
                continue
            llist.append(label[k])
            elist.append(drnm)

            # u is 3D array: (m/z, RT, polarity) = (m/z, RT, 2), but polarity is fixednow
            u = np.zeros( ( (mzb - mza) / mz_step, (rtb - rta) / rt_step, 2))

            for i in xrange(negsp.shape[0]):
                if negsp[i, k] != 0:
                    mzind = round((negmz[i] - mza)/ mz_step) - 1
                    rtind = round((negrt[i] - rta)/ rt_step) - 1
                    #if u[mzind, rtind, 0] != 0:
                    u[mzind, rtind, 0] = max(u[mzind, rtind, 0], negsp[i, k]) # to deal with crossing values
            for i in xrange(possp.shape[0]):
                if possp[i, k] != 0:
                    mzind = round((posmz[i] - mza)/ mz_step) - 1
                    rtind = round((posrt[i] - rta)/ rt_step) - 1
                    #if u[mzind, rtind, 0] != 0:
                    u[mzind, rtind, 1] = max(u[mzind, rtind, 1], possp[i, k]) # to deal with crossing values
            ulist.append(u)
    u = np.array(ulist)
    return u, llist, elist

def categorizeRT(rt):
    minRT = 1.
    maxRT = 17.
    step = 3.
    if (rt > maxRT) or (rt <= minRT):
        return None
    scale = np.arange(minRT, maxRT, 3)
    scale[-1] = maxRT
    for i in xrange(len(scale) - 1):
        if rt <= scale[i+1]:
            return i


def tens(drnms, labels,
               mza = 100.,
               mzb = 1000.,
               mz_step = 1e-0, # 0.01
               blcleaned = True,
               catRT = False):


    ulist = []
    llist = []
    elist = []


    for drnm in drnms:
        if blcleaned:
            fnm = drnm + '_blcleaned.npz'
        else:
            fnm = drnm + '.npz'
        f = np.load("./" + drnm + '/' + fnm)

        negsp = f['negsp']
        negmz = f['negmz']
        negrt = f['negrt']

        possp = f['possp']
        posmz = f['posmz']
        posrt = f['posrt']

        label = f['label']

        for k in xrange(negsp.shape[1]): # must be equal to pos_s.shape[1] // number of samples

            if label[k] not in labels:
                continue
            llist.append(label[k])
            elist.append(drnm)

            # u is 3D array: (m/z, RT, polarity) = (m/z, RT, 2), but polarity is fixednow
            if catRT:
                numsRT = 5
            else:
                rta = 2.
                rtb = 18.
                rts = 60./60.
                numsRT = int((rtb - rta) / rts)
            
            u = np.zeros( ( (mzb - mza) / mz_step, numsRT, 2))

            for i in xrange(negsp.shape[0]):
                if negsp[i, k] != 0:
                    if (negmz[i] < mza) or (negmz[i] >= mzb):
                        continue
                    mzind = np.trunc((negmz[i] - mza)/ mz_step)
                    if catRT:
                        rtind = categorizeRT(negrt[i])
                    else:
                        if (negrt[i] >= rtb) or (negrt[i] < rta):
                            rtind = None
                        else:
                            rtind = np.trunc((negrt[i] - rta)/ rts)
                            rtind = int(rtind)
                    mzind = int(mzind)
                    if rtind is None:
                        continue
                    #if u[mzind, rtind, 0] != 0:
                    u[mzind, rtind, 0] = max(u[mzind, rtind, 0], negsp[i, k]) # to deal with crossing values
            for i in xrange(possp.shape[0]):
                if possp[i, k] != 0:
                    if (posmz[i] < mza) or (posmz[i] >= mzb):
                        continue
                    mzind = np.trunc((posmz[i] - mza)/ mz_step)
                    if catRT:
                        rtind = categorizeRT(posrt[i])
                    else:
                        if (posrt[i] >= rtb) or (posrt[i] < rta):
                            rtind = None
                        else:
                            rtind = np.trunc((posrt[i] - rta)/ rts)
                            rtind = int(rtind)
                    mzind = int(mzind)
                    if rtind is None:
                        continue
                    #if u[mzind, rtind, 0] != 0:
                    u[mzind, rtind, 1] = max(u[mzind, rtind, 1], possp[i, k]) # to deal with crossing values
            ulist.append(u)
    u = np.array(ulist)
    return u, llist, elist

def tens_nort(drnms, labels,
               mza = 100.,
               mzb = 1000.,
               mz_step = 1e-0, # 0.01
               blcleaned = True):

    ulist = []
    llist = []
    elist = []

    for drnm in drnms:
        if blcleaned:
            fnm = drnm + '_blcleaned.npz'
        else:
            fnm = drnm + '.npz'
        f = np.load("./" + drnm + '/' + fnm)

        negsp = f['negsp']
        negmz = f['negmz']

        possp = f['possp']
        posmz = f['posmz']

        label = f['label']

        for k in xrange(negsp.shape[1]): # must be equal to pos_s.shape[1] // number of samples

            if label[k] not in labels:
                continue
            llist.append(label[k])
            elist.append(drnm)

            # u is 2D array: (m/z, polarity) = (m/z, 2), but polarity is fixednow
            
            u = np.zeros( ( (mzb - mza) / mz_step, 2))

            for i in xrange(negsp.shape[0]):
                if negsp[i, k] != 0:
                    if (negmz[i] < mza) or (negmz[i] >= mzb):
                        continue
                    mzind = np.trunc((negmz[i] - mza)/ mz_step)
                    mzind = int(mzind)
                    #if u[mzind, rtind, 0] != 0:
                    u[mzind, 0] = max(u[mzind, 0], negsp[i, k]) # to deal with crossing values
            for i in xrange(possp.shape[0]):
                if possp[i, k] != 0:
                    if (posmz[i] < mza) or (posmz[i] >= mzb):
                        continue
                    mzind = np.trunc((posmz[i] - mza)/ mz_step)
                    mzind = int(mzind)
                    #if u[mzind, rtind, 0] != 0:
                    u[mzind, 1] = max(u[mzind, 1], possp[i, k]) # to deal with crossing values
            ulist.append(u)
    u = np.array(ulist)
    return u, llist, elist


if __name__ == '__main__':
    drnms = ['ex1', 'ex1a', 'ex2', 'ex3', 'ex4', 'ex5', 'ex6']
    labels = range(36)
    fnm = 'data_raw' #'data_blcleaned' #'data_raw'
    blclean = False

    t, l, e = tens_nort(drnms, labels,
               mza = 100.,
               mzb = 900.,
               mz_step = 1e-0, # 0.01
               blcleaned = blclean)
    np.savez_compressed(fnm, data = t, label = l, expnm = e)
