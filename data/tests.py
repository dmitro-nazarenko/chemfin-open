import numpy as np
from scipy.sparse import csr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def scat3D(x, y, z, nms = None, svfnm = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    if nms is not None:
        ax.set_xlabel(nms[0])
        ax.set_ylabel(nms[1])
        ax.set_zlabel(nms[2])
    if svfnm is None:
        plt.show()
    else:
        plt.savefig(svfnm)
    plt.clf()

def scat2D(x, y, nms = None, svfnm = None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, c='r', marker='o')
    if nms is not None:
        ax.set_xlabel(nms[0])
        ax.set_ylabel(nms[1])
    if svfnm is None:
        plt.show()
    else:
        plt.savefig(svfnm)
    plt.clf()

def hist(x, num_bins = 20, svfnm = None):
    mu = np.median(x)
    sigma = np.std(x)
    # the histogram of the data
    n, bins, patches = plt.hist(x, num_bins, normed=True, facecolor='green', alpha=0.5)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    if svfnm is None:
        plt.show()
    else:
        plt.savefig(svfnm)
    plt.clf()


#fnm = 'raw_csr.npz'
sfnm = 'data_raw.npz'

fnm = 'a119.npz'

f = np.load(fnm)#load_sparse_csr(fnm)
a = f['data']
mean = f['mean']
std = f['std']

# load labels
f = np.load(sfnm)
label = f['label']
del f



u, s, vt = np.linalg.svd(a)
b = np.dot(a, vt.T)

x = b[:, 2].flatten()
y = b[:, 3].flatten()
z = b[:, 4].flatten()

scat3D(x, y, z)
for k in xrange(36):
    ind = np.where(label == k)[0]
    c = a[ind, :]
    u, s, vt = np.linalg.svd(c)
    b = np.dot(c, vt.T)

    x = b[:, 0].flatten()
    y = b[:, 1].flatten()
    z = b[:, 2].flatten()

    scat2D(x, y, svfnm = str(k+1) + '_scat.png')
    scat3D(x, y, z, svfnm = str(k+1) + '_scat3D.png')

sa = a.shape
for k in xrange(sa[1]):
    b = a[:, k] * std[0, k] + mean[0, k]
    ind = np.where(b > 1e-2)[0]
    hist(b, svfnm = 'hist' + str(k) + '.png')
    hist(b[ind], svfnm = 'hist_wo_zero_' + str(k) + '.png')







if False:
    # barchat of sample numbers per class
    numclust = 36
    s = []
    for i in xrange(numclust):
        s.append(label[label == i].size)

    plt.bar(np.arange(numclust)+1, s)
    plt.xticks(np.arange(numclust)+1)
    plt.show() 

