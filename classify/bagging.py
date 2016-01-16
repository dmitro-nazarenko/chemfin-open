import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
from load_data import load_full
from sklearn.cross_validation import StratifiedShuffleSplit

def check_vb(datanm, samples_per_class, Cs, num_classes, num_iter = 100):
    data, labels = load_full(datanm, samples_per_class)
    slo = StratifiedShuffleSplit(labels, n_iter=num_iter, test_size=0.5, train_size=0.5, random_state=None)
    ans = np.zeros((len(Cs), samples_per_class/2, 2))
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
                clf = LogisticRegression(C  =C   , penalty='l2', multi_class = 'ovr',
                                         tol=0.001, n_jobs = -1 , verbose = 0)#, solver = 'newton-cg')
                clf.fit(ctrain_data[0], ctrain_data[1])

                out_train = clf.predict_proba(ctrain_data[0])
                out_valid = clf.predict_proba(cvalid_data[0])

                ans[i, l, 0] += log_loss(ctrain_data[1], out_train)
                ans[i, l, 1] += log_loss(cvalid_data[1], out_valid)

    ans /= num_iter

    np.savez("rand_forest_bv", ans= ans, Cs = Cs, num_iter = num_iter, num_classes = num_classes, samples_per_class = samples_per_class)
    return ans

def check_lambda(datanm, samples_per_class, Cs, num_classes, mdep, num_iter = 100):
    data, labels = load_full(datanm, samples_per_class)
    slo = StratifiedShuffleSplit(labels, n_iter=num_iter, test_size=0.3, train_size=0.7, random_state=None)
    ans = np.zeros((len(Cs), len(mdep), 4))
    for train_index, test_index in slo:
        train_data = [data[train_index, :], labels[train_index]]
        valid_data = [data[test_index , :], labels[test_index ]]

        for j, d in enumerate(mdep):
            tr = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=d, min_samples_split=2, min_samples_leaf=1,      min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, class_weight=None, presort=False)

            for i, C in enumerate(Cs):

                

                clf = BaggingClassifier(base_estimator=tr, n_estimators=C,
                                         max_samples=1.0, max_features=1.0, bootstrap=True,
                                         bootstrap_features=True, oob_score=False, 
                                         warm_start=False, n_jobs=8, random_state=None, verbose=0)
                clf.fit(train_data[0], train_data[1])

                out_train = clf.predict_proba(train_data[0])
                out_valid = clf.predict_proba(valid_data[0])

                ans[i, j, 0] += log_loss(train_data[1], out_train)
                ans[i, j, 1] += log_loss(valid_data[1], out_valid)
                ans[i, j, 2] += clf.score(train_data[0], train_data[1])
                ans[i, j, 3] += clf.score(valid_data[0], valid_data[1])


    ans[:, :, :] /= num_iter

    np.savez("bagging_lambda", ans= ans, Cs = Cs, num_iter = num_iter, num_classes = num_classes, samples_per_class = samples_per_class)
    return ans

if __name__ == '__main__':
    datanm = '../data/a.npz'
    #Cs = [100., 50., 10., 5., 1., 0.75, 0.5, 0.25, 0.01]
    Cs = [10, 50, 100, 150, 200, 250, 300]
    mdep = [1, 5, 10, 15, 20, 25, 30, 35, 40]
    #check_vb(datanm, samples_per_class = 18, Cs = Cs, num_classes = 36, num_iter = 100)
    #Cs = (1./np.arange(0.01, 2.01, 0.1)).tolist()
    a = check_lambda(datanm, samples_per_class = 19, Cs = Cs, mdep = mdep, num_classes = 36, num_iter = 1000)





















