from sklearn import svm
import numpy as np
#from sklearn.metrics import log_loss
from sklearn.metrics import log_loss, recall_score, precision_score, accuracy_score, f1_score, hinge_loss
from load_data import load_full
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import log_loss


def check_vb(datanm, samples_per_class, Cs, num_classes, gamma, num_iter = 100, kernel = 'linear', strat = 'ovr'):
    data, labels = load_full(datanm, samples_per_class)
    slo = StratifiedShuffleSplit(labels, n_iter=num_iter, test_size=0.5, train_size=0.5, random_state=None)
    ans = np.zeros((len(Cs), len(gamma), samples_per_class/2, 4))
    for train_index, test_index in slo:
        train_data = [data[train_index, :], labels[train_index]]
        valid_data = [data[test_index , :], labels[test_index ]]

        for l in xrange(samples_per_class/2):
            ind_train = []
            ind_valid = []
            for k in xrange(num_classes):
                ind_train = ind_train + np.where(train_data[1] == k)[0].tolist()[:l+1]
                ind_valid = ind_valid + np.where(valid_data[1] == k)[0].tolist()[:l+1]

            ctrain_data = [ train_data[0][ind_train], train_data[1][ind_train] ]
            cvalid_data = [ valid_data[0][ind_valid], valid_data[1][ind_valid] ]

            for i, C in enumerate(Cs):
                for j, g in enumerate(gamma):
                    clf = svm.SVC(C=C, kernel=kernel, degree=3, gamma=g, coef0=0.0, shrinking=True,
                                  probability=False, tol=0.001,  cache_size=10000, class_weight=None,
                                  verbose=False, max_iter=-1, decision_function_shape=strat, random_state=None)
                    clf.fit(ctrain_data[0], ctrain_data[1])

                    #out_train = clf.predict_proba(ctrain_data[0])
                    #out_valid = clf.predict_proba(cvalid_data[0])

                    #ans[i, l, 0] += log_loss(ctrain_data[1], out_train)
                    #ans[i, l, 1] += log_loss(cvalid_data[1], out_valid)
                    
                    out_train = clf.decision_function(train_data[0])
                    out_valid = clf.decision_function(valid_data[0])

                    ans[i, j, l, 2] += hinge_loss(train_data[1], out_train, range(num_classes))
                    ans[i, j, l, 3] += hinge_loss(valid_data[1], out_valid, range(num_classes))

    ans /= num_iter

    np.savez("svm_bv_" + kernel + '_' + strat, ans= ans, Cs = Cs, num_iter = num_iter, num_classes = num_classes, samples_per_class = samples_per_class)
    return ans

def check_lambda(datanm, samples_per_class, Cs, num_classes, gamma, num_iter = 100, kernel = 'linear', strat = 'ovr'):
    data, labels = load_full(datanm, samples_per_class)
    slo = StratifiedShuffleSplit(labels, n_iter=num_iter, test_size=0.3, train_size=0.7, random_state=None)
    ans = np.zeros((len(Cs), len(gamma), 4))
    for train_index, test_index in slo:
        train_data = [data[train_index, :], labels[train_index]]
        valid_data = [data[test_index , :], labels[test_index ]]

        for j, g in enumerate(gamma):
            for i, C in enumerate(Cs):
                clf = svm.SVC(C=C, kernel=kernel, degree=3, gamma=g, coef0=0.0, shrinking=True,
                                  probability=False, tol=0.001,  cache_size=10000, class_weight=None,
                                  verbose=False, max_iter=-1, decision_function_shape=strat, random_state=None)
                clf.fit(train_data[0], train_data[1])

                out_train = clf.decision_function(train_data[0])
                out_valid = clf.decision_function(valid_data[0])

                ans[i, j, 2] += hinge_loss(train_data[1], out_train, range(num_classes))
                ans[i, j, 3] += hinge_loss(valid_data[1], out_valid, range(num_classes))

                #ans[i, j, 0] += log_loss(train_data[1], clf.predict_proba(train_data[0]))
                #ans[i, j, 1] += log_loss(valid_data[1], clf.predict_proba(valid_data[0]))

    ans[:, :, :] /= num_iter

    np.savez("svm_lambda_" + kernel + '_' + strat, ans= ans, Cs = Cs, num_iter = num_iter, num_classes = num_classes, samples_per_class = samples_per_class)
    return ans

def main_func(datanm, samples_per_class, C, num_classes, gamma, num_iter = 100, kernel = 'linear', strat = 'ovr'):
    data, labels = load_full(datanm, samples_per_class)
    slo = StratifiedShuffleSplit(labels, n_iter=num_iter, test_size=0.3, train_size=0.7, random_state=None)
    recall = np.zeros((num_classes+1, 2))
    precision = np.zeros((num_classes+1, 2))
    f1 = np.zeros((num_classes+1, 2))
    accuracy = np.zeros((2))
    logloss = np.zeros((2))
    hingeloss = np.zeros((2))

    
    for train_index, test_index in slo:
        train_data = [data[train_index, :], labels[train_index]]
        valid_data = [data[test_index , :], labels[test_index ]]

        clf = svm.SVC(C=C, kernel=kernel, degree=3, gamma=gamma, coef0=0.0, shrinking=True,
                      probability=False, tol=0.001,  cache_size=10000, class_weight=None,
                      verbose=False, max_iter=-1, decision_function_shape=strat, random_state=None)
        clf.fit(train_data[0], train_data[1])

        #out_train = clf.predict_proba(train_data[0])
        #out_valid = clf.predict_proba(valid_data[0])

        #logloss[0] += log_loss(train_data[1], out_train)
        #logloss[1] += log_loss(valid_data[1], out_valid)

        out_train = clf.decision_function(train_data[0])
        out_valid = clf.decision_function(valid_data[0])

        hingeloss[0] += hinge_loss(train_data[1], out_train)
        hingeloss[1] += hinge_loss(valid_data[1], out_valid)

        out_train = clf.predict(train_data[0])
        out_valid = clf.predict(valid_data[0])

        accuracy[0] += accuracy_score(train_data[1], out_train)
        accuracy[1] += accuracy_score(valid_data[1], out_valid)

        precision[:-1, 0] += precision_score(train_data[1], out_train, average = None)
        precision[-1, 0] += precision_score(train_data[1], out_train, average = 'macro')
        precision[:-1, 1] += precision_score(valid_data[1], out_valid, average = None)
        precision[-1, 1] += precision_score(valid_data[1], out_valid, average = 'macro')

        recall[:-1, 0] += recall_score(train_data[1], out_train, average = None)
        recall[-1, 0] += recall_score(train_data[1], out_train, average = 'macro')
        recall[:-1, 1] += recall_score(valid_data[1], out_valid, average = None)
        recall[-1, 1] += recall_score(valid_data[1], out_valid, average = 'macro')

        f1[:-1, 0] += f1_score(train_data[1], out_train, average = None)
        f1[-1, 0] += f1_score(train_data[1], out_train, average = 'macro')
        f1[:-1, 1] += f1_score(valid_data[1], out_valid, average = None)
        f1[-1, 1] += f1_score(valid_data[1], out_valid, average = 'macro')

    f1 /= num_iter
    recall  /= num_iter
    precision  /= num_iter
    logloss  /= num_iter
    accuracy  /= num_iter

    np.savez("svm_final_" + kernel + '_' + strat, accuracy = accuracy, recall = recall, f1 = f1,
                             precision = precision, logloss = logloss, C = C,
                             num_iter = num_iter, num_classes = num_classes,
                             samples_per_class = samples_per_class,
                             hingeloss = hingeloss)
    return [accuracy, recall, f1, precision, logloss, hingeloss]

if __name__ == '__main__':
    datanm = '../data/a.npz'
    #kernel = 'rbf'
    kernel = 'linear'
    strat = 'ovr'
    #strat = 'ovo'

    gamma = [0.01, 0.1, 1., 10., 100.]

    #Cs = [100., 50., 10., 5., 1., 0.75, 0.5, 0.25, 0.01, 0.075, 0.05, 0.025, 0.001, 0.00075, 0.0005, 0.00025, 0.0001]
    #if (kernel == 'rbf') and (strat == 'ovo'):
    #    Cs = [1., 0.25, 0.0075, 0.001, 0.00025, 0.00005, 0.0000075]
        #Cb = 
    if (kernel == 'rbf') and (strat == 'ovr'):
        Cs = [100000., 10000., 1000., 100., 10., 1.]
        gamma = [0.0001, 0.001, 0.01, 0.1, 1.]
        bgamma = 0.0001
        Cb = 100.
    if (kernel == 'linear') and (strat == 'ovr'):
        Cs = [10., 1., 0.01, 0.001, 0.0001, 0.00001]
        Cb = 0.01
        bgamma = 'auto'
        gamma = [bgamma]
    #if (kernel == 'linear') and (strat == 'ovo'):
    #    Cs = [1., 0.001, 0.0000075, 0.00000075, 0.000000075]
    #Cs = [1./0.35]
    #a = check_lambda(datanm, gamma = gamma, samples_per_class = 20, Cs = Cs, num_classes = 36, num_iter = 100, kernel = kernel, strat = strat)
    a = check_vb(datanm, gamma = gamma, samples_per_class = 20, Cs = [Cb], num_classes = 36, num_iter = 100, kernel = kernel)

    #Cs = (1./np.arange(0.01, 2.01, 0.1)).tolist()
    #l = main_func(datanm, samples_per_class = 20, C = Cb, num_classes = 36, gamma = bgamma, num_iter = 100, kernel = kernel, strat = strat)



