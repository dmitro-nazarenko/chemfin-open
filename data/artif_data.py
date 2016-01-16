import os, sys
import numpy as np
import scipy.stats as stats

def art_gen(fnm,
            amp=100, # artificial data size = data size * amp
            m = 30, #  magnitude of peak area change is +/- m %
            r1 = 15,
            r2 = 30,
            saveorig = False): # i.e 15-30% of peak's amplitude will be changed

    F = np.load(fnm)
    A = F["data"]
    C = F["label"]
    E = F["expnm"]

    r1 /= 100.
    r2 /= 100.
    m /= 100.

    sA = A.shape
    A = np.reshape(A, (len(A), -1)) # i.e 834*1600
    sf = sA[1]*sA[2]

    S = np.arange(sf)#S = np.arange (1, len(A) + 1 ) # HERE WAS A DRASTIC ERROR!! I've saved original as comment (list for choosing index of peak)

    #Art = np.zeros(((len(A[:])*amp, len(A[1][:])))) # array for storing artificial data
    Ea = [x + '_artificial' for x in E]
    Ca = C.copy()
    Art = A.copy()

    if not saveorig:
        amp -= 1

    for i in xrange(amp):
        Ca = np.concatenate ((Ca, C))
        Ea = np.concatenate ((Ea, E))
        Art = np.concatenate((Art,A))

    if not saveorig:
        amp += 1


    for j in xrange(amp):
        for i in xrange(sA[0]):
            k = 0
            lim = np.count_nonzero(A[i, :]) * np.random.uniform(r1, r2) # 15-30% nonzero peaks
            lim = np.trunc(lim)
            while k < lim:
                l = np.random.choice(S)
                if A[i, l] != 0:
                    Art[i + (j*len(A)), l] = A[i, l] * (1. - np.random.uniform(-m, m)) # i.e magnitude of change is +/- 30 %
                    k += 1

    Art = Art.reshape(Art.shape[0] + saveorig*sA[0], 800, 2)
    np.savez_compressed('data_art', data = Art, label = Ca, expnm = Ea)
    

if __name__ == '__main__':
    art_gen('data_raw.npz', 100)








