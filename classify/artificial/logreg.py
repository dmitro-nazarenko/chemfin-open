import numpy as np
import sys
sys.path.append("../")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, recall_score, precision_score, accuracy_score, f1_score
from load_data import load_full, load_mat
from sklearn.cross_validation import StratifiedShuffleSplit

def kget(labels, num_classes, k):
    ind = []
    for i in xrange(num_classes):
        tmp = np.where(labels == i)[0]
        ind.append( np.random.permutation(tmp) )
        ind[-1] = ind[-1][:k]
    ind = np.array(ind)
    ind = ind.T
    ind = ind.flatten()
    print labels[ind]
    return ind


def check_vb(dirnm, datanm_train, datanm_valid, C, num_classes):
    spct = 10*70
    tdata, tlabels = load_full(dirnm+datanm_train, spct)
    #print tdata.shape, tlabels.shape

    spct = 10*30
    vdata, vlabels = load_full(dirnm+datanm_valid, spct)

    h = np.arange(0, 310, 10)
    h[0] +=1
    # artif
    ans = np.zeros((h.size, 2))

    tind = kget(tlabels, num_classes, h[-1])
    vind = kget(vlabels, num_classes, h[-1])

    for l in xrange(0, h.size):

        clf = LogisticRegression(C  =C,     penalty='l2', multi_class = 'ovr',
                                 tol=0.001, n_jobs = -1, verbose = 0, solver = 'newton-cg')
        clf.fit(tdata[tind[:h[l]*num_classes]], tlabels[tind[:h[l]*num_classes]])

        out_train = clf.predict_proba(tdata[tind[:h[l]*num_classes]])
        out_valid = clf.predict_proba(vdata[vind[:h[l]*num_classes]])

        ans[l, 0] += log_loss(tlabels[tind[:h[l]*num_classes]], out_train)
        ans[l, 1] += log_loss(vlabels[vind[:h[l]*num_classes]], out_valid)

    np.savez("logreg_bv", ans= ans, C = C, num_classes = num_classes)
    return ans

def check_lambda(dirnm, datanm_train, datanm_valid, datanm_orig_train, datanm_orig_valid, samples_per_class, Cs, num_classes):
    spct = 10*70
    tdata, tlabels = load_full(dirnm+datanm_train, spct)
    print tdata.shape, tlabels.shape

    spct = 10
    otdata, otlabels = load_full(dirnm+datanm_orig_train, spct)

    spct = 10*30
    vdata, vlabels = load_full(dirnm+datanm_valid, spct)

    spct = 10
    ovdata, ovlabels = load_full(dirnm+datanm_orig_valid, spct)

    # artif
    ans = np.zeros((len(Cs), 4))

    for i, C in enumerate(Cs):
        clf = LogisticRegression(C  =C,     penalty='l2', multi_class = 'ovr',
                                 tol=0.001, n_jobs = -1, verbose = 0, solver = 'newton-cg')
        clf.fit(tdata, tlabels)

        out_train = clf.predict_proba(tdata)
        out_valid = clf.predict_proba(vdata)
        out_train_real = clf.predict_proba(otdata)
        out_valid_real = clf.predict_proba(ovdata)

        ans[i, 0] += log_loss(tlabels, out_train)
        ans[i, 1] += log_loss(vlabels, out_valid)
        ans[i, 2] += log_loss(otlabels, out_train_real)
        ans[i, 3] += log_loss(ovlabels, out_valid_real)

    np.savez("logreg_lambda", ans= ans, Cs = Cs, num_classes = num_classes, samples_per_class = samples_per_class)
    return ans

def main_func(dirnm, datanm_train, datanm_valid, datanm_orig_train, datanm_orig_valid, Cs, num_classes):
    recall = np.zeros((len(Cs), num_classes+1, 4))
    precision = np.zeros((len(Cs), num_classes+1, 4))
    f1 = np.zeros((len(Cs), num_classes+1, 4))
    accuracy = np.zeros((len(Cs), 4))
    logloss = np.zeros((len(Cs), 4))
    
    spct = 10*70
    tdata, tlabels = load_full(dirnm+datanm_train, spct)
    print tdata.shape, tlabels.shape

    spct = 10
    otdata, otlabels = load_full(dirnm+datanm_orig_train, spct)

    spct = 10*30
    vdata, vlabels = load_full(dirnm+datanm_valid, spct)

    spct = 10
    ovdata, ovlabels = load_full(dirnm+datanm_orig_valid, spct)

    for i, C in enumerate(Cs):

        clf = LogisticRegression(C  =C,     penalty='l2', multi_class = 'ovr',
                                 tol=0.001, n_jobs = -1, verbose = 0)#, solver = 'newton-cg')
        clf.fit(tdata, tlabels)

        out_train = clf.predict_proba(tdata)
        out_valid = clf.predict_proba(vdata)
        out_train_real = clf.predict_proba(otdata)
        out_valid_real = clf.predict_proba(ovdata)

        logloss[i, 0] += log_loss(tlabels, out_train)
        logloss[i, 1] += log_loss(vlabels, out_valid)
        logloss[i, 2] += log_loss(otlabels, out_train_real)
        logloss[i, 3] += log_loss(ovlabels, out_valid_real)

        out_train = clf.predict(tdata)
        out_valid = clf.predict(vdata)
        out_train_real = clf.predict(otdata)
        out_valid_real = clf.predict(ovdata)

        accuracy[i, 0] += accuracy_score(tlabels, out_train)
        accuracy[i, 1] += accuracy_score(vlabels, out_valid)
        accuracy[i, 2] += accuracy_score(otlabels, out_train_real)
        accuracy[i, 3] += accuracy_score(ovlabels, out_valid_real)


        precision[i, :-1, 0] += precision_score(tlabels, out_train, average = None)
        precision[i, -1, 0] += precision_score(tlabels, out_train, average = 'macro')

        precision[i, :-1, 1] += precision_score(vlabels, out_valid, average = None)
        precision[i, -1, 1] += precision_score(vlabels, out_valid, average = 'macro')

        precision[i, :-1, 2] += precision_score(otlabels, out_train_real, average = None)
        precision[i, -1, 2] += precision_score(otlabels, out_train_real, average = 'macro')

        precision[i, :-1, 3] += precision_score(ovlabels, out_valid_real, average = None)
        precision[i, -1, 3] += precision_score(ovlabels, out_valid_real, average = 'macro')


        recall[i, :-1, 0] += recall_score(tlabels, out_train, average = None)
        recall[i, -1, 0] += recall_score(tlabels, out_train, average = 'macro')

        recall[i, :-1, 1] += recall_score(vlabels, out_valid, average = None)
        recall[i, -1, 1] += recall_score(vlabels, out_valid, average = 'macro')

        recall[i, :-1, 2] += recall_score(otlabels, out_train_real, average = None)
        recall[i, -1, 2] += recall_score(otlabels, out_train_real, average = 'macro')

        recall[i, :-1, 3] += recall_score(ovlabels, out_valid_real, average = None)
        recall[i, -1, 3] += recall_score(ovlabels, out_valid_real, average = 'macro')


        f1[i, :-1, 0] += f1_score(tlabels, out_train, average = None)
        f1[i, -1, 0] += f1_score(tlabels, out_train, average = 'macro')

        f1[i, :-1, 1] += f1_score(vlabels, out_valid, average = None)
        f1[i, -1, 1] += f1_score(vlabels, out_valid, average = 'macro')

        f1[i, :-1, 2] += f1_score(otlabels, out_train_real, average = None)
        f1[i, -1, 2] += f1_score(otlabels, out_train_real, average = 'macro')

        f1[i, :-1, 3] += f1_score(ovlabels, out_valid_real, average = None)
        f1[i, -1, 3] += f1_score(ovlabels, out_valid_real, average = 'macro')

    np.savez("logreg_final", accuracy = accuracy, recall = recall, f1 = f1,
                             precision = precision, logloss = logloss, C = C,
                             num_classes = num_classes)
    return [accuracy, recall, f1, precision, logloss]

if __name__ == '__main__':

    dirnm = '../../data/'

    datanm_train = 'mat_ao_a.npz'
    datanm_valid = 'mat_ao_b.npz'
    datanm_orig_train = 'mat_o_a.npz'
    datanm_orig_valid = 'mat_o_b.npz'


    #Cs = [100., 50., 10., 5., 1., 0.75, 0.5, 0.25, 0.01]
    Cs = [100., 10., 1., 0.1, 0.01, 0.0075, 0.005, 0.0025, 0.001]#, 0.0001]#, 0.01]#, 0.001, 0.0001, 0.00001, 0.000001] #=> 0.80
    Cb = 0.0075
    #a = check_lambda(dirnm, datanm_train, datanm_valid, datanm_orig_train, datanm_orig_valid, samples_per_class = 20, Cs = Cs, num_classes = 36)
    #a = check_vb(dirnm, datanm_train, datanm_valid, C = Cb, num_classes = 36)
    l = main_func(dirnm, datanm_train, datanm_valid, datanm_orig_train, datanm_orig_valid, Cs, num_classes = 36)






















