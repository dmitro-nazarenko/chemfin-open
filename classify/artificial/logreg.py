import numpy as np
import sys
sys.path.append("../")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, recall_score, precision_score, accuracy_score, f1_score
from load_data import load_full, load_mat
from sklearn.cross_validation import StratifiedShuffleSplit

def check_vb(datanm_train, datanm_valid, samples_per_class, spct, Cs, num_classes, num_iter = 100):
    tdata, tlabels = load_full(datanm_train, spct)
    # artif
    slo = StratifiedShuffleSplit(tlabels, n_iter=num_iter, test_size=0.5, train_size=0.5, random_state=None)
    ans = np.zeros((len(Cs), spct/200, 2))
    idx = 0
    for train_index, test_index in slo:
        train_data = [tdata[train_index, :], tlabels[train_index]]
        valid_data = [tdata[test_index , :], tlabels[test_index ]]

        #train_real, valid_real = load_mat(datanm_valid, 0, samples_per_class)

        for l in xrange(0, samples_per_class/2, 100):
            ind_train = []
            ind_valid = []
            for k in xrange(num_classes):
                ind_train = ind_train + np.where(train_data[1] == k)[0].tolist()[:l+1]
                ind_valid = ind_valid + np.where(valid_data[1] == k)[0].tolist()[:l+1]

            ctrain_data = [ train_data[0][ind_train], train_data[1][ind_train] ]
            cvalid_data = [ valid_data[0][ind_valid], valid_data[1][ind_valid] ]

            for i, C in enumerate(Cs):
                clf = LogisticRegression(C  =C   , penalty='l2', multi_class = 'ovr',
                                         tol=0.001, n_jobs = -1 , verbose = 0)#, solver = 'newton-cg')
                clf.fit(ctrain_data[0], ctrain_data[1])

                out_train = clf.predict_proba(ctrain_data[0])
                out_valid = clf.predict_proba(cvalid_data[0])

                ans[i, l, 0] += log_loss(ctrain_data[1], out_train)
                ans[i, l, 1] += log_loss(cvalid_data[1], out_valid)

    ans /= num_iter

    np.savez("logreg_bv", ans= ans, Cs = Cs, num_iter = num_iter, num_classes = num_classes, samples_per_class = samples_per_class)
    return ans

def check_lambda(datanm_train, datanm_valid, samples_per_class, spct, Cs, num_classes, num_iter = 100):
    tdata, tlabels = load_full(datanm_train, spct)
    # artif
    slo = StratifiedShuffleSplit(tlabels, n_iter=num_iter, test_size=0.3, train_size=0.7, random_state=None)
    ans = np.zeros((len(Cs), 3))
    idx = 0
    for train_index, test_index in slo:
        train_data = [tdata[train_index, :], tlabels[train_index]]
        valid_data = [tdata[test_index , :], tlabels[test_index ]]

        train_real, valid_real = load_mat(datanm_valid, 0, samples_per_class)

        for i, C in enumerate(Cs):
            clf = LogisticRegression(C  =C,     penalty='l2', multi_class = 'ovr',
                                     tol=0.001, n_jobs = -1, verbose = 0)#, solver = 'newton-cg')
            clf.fit(train_data[0], train_data[1])

            out_train = clf.predict_proba(train_data[0])
            out_valid = clf.predict_proba(valid_data[0])
            out_valid_real = clf.predict_proba(valid_real[0])

            ans[i, 0] += log_loss(train_data[1], out_train)
            ans[i, 1] += log_loss(valid_data[1], out_valid)
            ans[i, 2] += log_loss(valid_real[1], out_valid_real)
        idx += 1

    ans[:, :] /= num_iter

    np.savez("logreg_lambda", ans= ans, Cs = Cs, num_iter = num_iter, num_classes = num_classes, samples_per_class = samples_per_class)
    return ans

def main_func(datanm_train, datanm_valid, samples_per_class, spct, C, num_classes, num_iter = 100):
    recall = np.zeros((num_classes+1, 3))
    precision = np.zeros((num_classes+1, 3))
    f1 = np.zeros((num_classes+1, 3))
    accuracy = np.zeros((3))
    logloss = np.zeros((3))
    
    tdata, tlabels = load_full(datanm_train, spct)
    # artif
    slo = StratifiedShuffleSplit(tlabels, n_iter=num_iter, test_size=0.3, train_size=0.7, random_state=None)

    for train_index, test_index in slo:
        train_data = [tdata[train_index, :], tlabels[train_index]]
        valid_data = [tdata[test_index , :], tlabels[test_index ]]

        train_real, valid_real = load_mat(datanm_valid, 0, samples_per_class)

        clf = LogisticRegression(C  =C,     penalty='l2', multi_class = 'ovr',
                                 tol=0.001, n_jobs = -1, verbose = 0)#, solver = 'newton-cg')
        clf.fit(train_data[0], train_data[1])

        out_train = clf.predict_proba(train_data[0])
        out_valid = clf.predict_proba(valid_data[0])
        out_valid_real = clf.predict_proba(valid_real[0])

        logloss[0] += log_loss(train_data[1], out_train)
        logloss[1] += log_loss(valid_data[1], out_valid)
        logloss[2] += log_loss(valid_real[1], out_valid_real)

        out_train = clf.predict(train_data[0])
        out_valid = clf.predict(valid_data[0])
        out_valid_real = clf.predict(valid_real[0])

        accuracy[0] += accuracy_score(train_data[1], out_train)
        accuracy[1] += accuracy_score(valid_data[1], out_valid)
        accuracy[2] += accuracy_score(valid_real[1], out_valid_real)


        precision[:-1, 0] += precision_score(train_data[1], out_train, average = None)
        precision[-1, 0] += precision_score(train_data[1], out_train, average = 'macro')

        precision[:-1, 1] += precision_score(valid_data[1], out_valid, average = None)
        precision[-1, 1] += precision_score(valid_data[1], out_valid, average = 'macro')

        precision[:-1, 2] += precision_score(valid_real[1], out_valid_real, average = None)
        precision[-1, 2] += precision_score(valid_real[1], out_valid_real, average = 'macro')


        recall[:-1, 0] += recall_score(train_data[1], out_train, average = None)
        recall[-1, 0] += recall_score(train_data[1], out_train, average = 'macro')

        recall[:-1, 1] += recall_score(valid_data[1], out_valid, average = None)
        recall[-1, 1] += recall_score(valid_data[1], out_valid, average = 'macro')

        recall[:-1, 2] += recall_score(valid_real[1], out_valid_real, average = None)
        recall[-1, 2] += recall_score(valid_real[1], out_valid_real, average = 'macro')


        f1[:-1, 0] += f1_score(train_data[1], out_train, average = None)
        f1[-1, 0] += f1_score(train_data[1], out_train, average = 'macro')

        f1[:-1, 1] += f1_score(valid_data[1], out_valid, average = None)
        f1[-1, 1] += f1_score(valid_data[1], out_valid, average = 'macro')

        f1[:-1, 2] += f1_score(valid_real[1], out_valid_real, average = None)
        f1[-1, 2] += f1_score(valid_real[1], out_valid_real, average = 'macro')

    f1 /= num_iter
    recall  /= num_iter
    precision  /= num_iter
    logloss  /= num_iter
    accuracy  /= num_iter

    np.savez("logreg_final", accuracy = accuracy, recall = recall, f1 = f1,
                             precision = precision, logloss = logloss, C = C,
                             num_iter = num_iter, num_classes = num_classes,
                             samples_per_class = samples_per_class)
    return [accuracy, recall, f1, precision, logloss]

if __name__ == '__main__':
    datanm_learn = '../../data/a.npz'
    datanm_valid = '../../data/a.npz'
    #Cs = [100., 50., 10., 5., 1., 0.75, 0.5, 0.25, 0.01]
    Cs = [100., 50., 10., 5., 1., 0.80, 0.75, 0.6, 0.65, 0.5, 0.25, 0.01]#, 0.001, 0.0001, 0.00001, 0.000001] #=> 0.80
    Cb = 0.6
    #a = check_lambda(datanm_learn, datanm_valid, samples_per_class = 20, spct = 2000, Cs = Cs, num_classes = 36, num_iter = 100)
    #a = check_vb(datanm_learn, datanm_valid, samples_per_class = 20, spct = 2000, Cs = [Cb], num_classes = 36, num_iter = 100)
    l = main_func(datanm_learn, datanm_valid, samples_per_class = 20, spct = 2000, C = Cb, num_classes = 36, num_iter = 100)






















