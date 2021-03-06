import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, recall_score, precision_score, accuracy_score, f1_score, brier_score_loss
from load_data import load_full
from sklearn.cross_validation import StratifiedShuffleSplit

def brier(ytrue, yprob, num_classes):
    rv = 0.
    for i in xrange(num_classes):
        ind = np.where(ytrue == i)[0]
        tmp = np.zeros(ytrue.size)
        tmp[ind] += 1
        rv += brier_score_loss(ytrue, yprob[:, i])
    rv /= num_classes
    return rv

def check_vb(datanm, samples_per_class, depv, nest, num_classes, criterion, num_iter = 100):
    data, labels = load_full(datanm, samples_per_class)
    slo = StratifiedShuffleSplit(labels, n_iter=num_iter, test_size=0.5, train_size=0.5, random_state=None)
    ans = np.zeros((samples_per_class/2, 4))
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

            clf = RandomForestClassifier(n_estimators=nest, criterion=criterion, max_depth=depv,
                                    min_samples_split=2, min_samples_leaf=1,
                                    min_weight_fraction_leaf=0.0, max_features='auto',
                                    max_leaf_nodes=None, bootstrap=True, oob_score=False,
                                    n_jobs=8, random_state=None, verbose=0, warm_start=False,
                                    class_weight=None)
            clf.fit(ctrain_data[0], ctrain_data[1])

            out_train = clf.predict_proba(ctrain_data[0])
            out_valid = clf.predict_proba(cvalid_data[0])

            ans[l, 0] += log_loss(ctrain_data[1], out_train)
            ans[l, 1] += log_loss(cvalid_data[1], out_valid)

            ans[l, 2] += brier(ctrain_data[1], out_train, num_classes)
            ans[l, 3] += brier(cvalid_data[1], out_valid, num_classes)

    ans /= num_iter

    np.savez("rand_forest_bv_" + criterion, ans= ans, depv = depv, nest=nest, num_iter = num_iter, num_classes = num_classes, samples_per_class = samples_per_class)
    return ans

def check_lambda(datanm, samples_per_class,depv, num_classes, criterion, num_iter = 100):
    data, labels = load_full(datanm, samples_per_class)
    slo = StratifiedShuffleSplit(labels, n_iter=num_iter, test_size=0.3, train_size=0.7, random_state=None)
    ans = np.zeros((len(depv), 4))
    for train_index, test_index in slo:
        train_data = [data[train_index, :], labels[train_index]]
        valid_data = [data[test_index , :], labels[test_index ]]

        for i, d in enumerate(depv):
            clf = DecisionTreeClassifier(criterion=criterion, splitter='best',
                                         max_depth=d, min_samples_split=2,
                                         min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                         max_features=None, random_state=None,
                                         max_leaf_nodes=None, class_weight=None, presort=False)
            clf.fit(train_data[0], train_data[1])

            out_train = clf.predict_proba(train_data[0])
            out_valid = clf.predict_proba(valid_data[0])

            ans[i, 0] += log_loss(train_data[1], out_train)
            ans[i, 1] += log_loss(valid_data[1], out_valid)
            ans[i, 2] += brier(train_data[1], out_train, num_classes)
            ans[i, 3] += brier(valid_data[1], out_valid, num_classes)

    ans[:, :] /= num_iter

    np.savez("rand_forest_lambda_" + criterion, ans= ans, mdep = mdep, num_iter = num_iter, num_classes = num_classes, samples_per_class = samples_per_class)
    return ans

def main_func(datanm, samples_per_class, depv, nest, num_classes, criterion, num_iter = 100):
    data, labels = load_full(datanm, samples_per_class)
    slo = StratifiedShuffleSplit(labels, n_iter=num_iter, test_size=0.3, train_size=0.7, random_state=None)
    recall = np.zeros((len(mdep), len(nest), num_classes+1, 2))
    precision = np.zeros((len(mdep), len(nest), num_classes+1, 2))
    f1 = np.zeros((len(mdep), len(nest), num_classes+1, 2))
    accuracy = np.zeros((len(mdep), len(nest), 2))
    logloss = np.zeros((len(mdep), len(nest), 2))
    brierloss = np.zeros((len(mdep), len(nest), 2))
    
    for train_index, test_index in slo:
        train_data = [data[train_index, :], labels[train_index]]
        valid_data = [data[test_index , :], labels[test_index ]]

        for i, d in enumerate(mdep):
            for j, n in enumerate(nest):
                clf = RandomForestClassifier(n_estimators=n, criterion=criterion, max_depth=d,
                                        min_samples_split=2, min_samples_leaf=1,
                                        min_weight_fraction_leaf=0.0, max_features='auto',
                                        max_leaf_nodes=None, bootstrap=True, oob_score=False,
                                        n_jobs=8, random_state=None, verbose=0, warm_start=False,
                                        class_weight=None)
                clf.fit(train_data[0], train_data[1])

                out_train = clf.predict_proba(train_data[0])
                out_valid = clf.predict_proba(valid_data[0])

                logloss[i, j, 0] += log_loss(train_data[1], out_train)
                logloss[i, j, 1] += log_loss(valid_data[1], out_valid)

                brierloss[i, j, 0] += brier(train_data[1], out_train, num_classes)
                brierloss[i, j, 1] += brier(valid_data[1], out_valid, num_classes)


                out_train = clf.predict(train_data[0])
                out_valid = clf.predict(valid_data[0])

                accuracy[i, j, 0] += accuracy_score(train_data[1], out_train)
                accuracy[i, j, 1] += accuracy_score(valid_data[1], out_valid)

                precision[i, j, :-1, 0] += precision_score(train_data[1], out_train, average = None)
                precision[i, j, -1, 0] += precision_score(train_data[1], out_train, average = 'macro')
                precision[i, j, :-1, 1] += precision_score(valid_data[1], out_valid, average = None)
                precision[i, j, -1, 1] += precision_score(valid_data[1], out_valid, average = 'macro')

                recall[i, j, :-1, 0] += recall_score(train_data[1], out_train, average = None)
                recall[i, j, -1, 0] += recall_score(train_data[1], out_train, average = 'macro')
                recall[i, j, :-1, 1] += recall_score(valid_data[1], out_valid, average = None)
                recall[i, j, -1, 1] += recall_score(valid_data[1], out_valid, average = 'macro')

                f1[i, j, :-1, 0] += f1_score(train_data[1], out_train, average = None)
                f1[i, j, -1, 0] += f1_score(train_data[1], out_train, average = 'macro')
                f1[i, j, :-1, 1] += f1_score(valid_data[1], out_valid, average = None)
                f1[i, j, -1, 1] += f1_score(valid_data[1], out_valid, average = 'macro')

    f1 /= num_iter
    recall  /= num_iter
    precision  /= num_iter
    logloss  /= num_iter
    accuracy  /= num_iter

    np.savez("rand_forest_final", accuracy = accuracy, recall = recall, f1 = f1,
                             precision = precision, logloss = logloss, depv = depv,
                             num_iter = num_iter, num_classes = num_classes,
                             samples_per_class = samples_per_class, brierloss = brierloss)
    return [accuracy, recall, f1, precision, logloss, brierloss]

if __name__ == '__main__':
    datanm = '../data/a.npz'
    #Cs = [100., 50., 10., 5., 1., 0.75, 0.5, 0.25, 0.01]

    #crit = 'entropy'
    crit = 'gini'
    if crit == 'gini':
        mdep = range(1, 101)#[1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40]#, 35, 40, 45, 50]
        nest = range(1, 101)#[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


        bdep = 18
        bnest = 45
        #dep = 35
    if crit == 'entropy':
            # bad
        mdep = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


    #Cs = (1./np.arange(0.01, 2.01, 0.1)).tolist()
    #a = check_lambda(datanm, samples_per_class = 20, depv = mdep, num_classes = 36, num_iter = 100, criterion = crit)
    a = check_vb(datanm, samples_per_class = 20, depv = bdep, nest = bnest, num_classes = 36, num_iter = 100, criterion = crit)
    #l = main_func(datanm, samples_per_class = 20, depv = mdep, nest = nest, num_classes = 36, criterion = crit, num_iter = 100)



















