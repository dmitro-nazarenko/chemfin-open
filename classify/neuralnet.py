import theanets
import numpy as np
from load_data import load_full
from sklearn.cross_validation import StratifiedShuffleSplit

def main_test(datanm, sp1, sp2, lamval, samples_per_class,num_classes, num_iter = 100):
    data, labels = load_full(datanm, samples_per_class)
    slo = StratifiedShuffleSplit(labels, n_iter=num_iter, test_size=0.3, train_size=0.7, random_state=None)

    feat_num = data.shape[1]
    layer_out1 = range(num_classes, feat_num + 1, 60)

    recall = np.zeros((len(sp1), len(sp2), len(lamval), num_classes+1, 2))
    precision = np.zeros((len(sp1), len(sp2), len(lamval), num_classes+1, 2))
    f1 = np.zeros((len(sp1), len(sp2), len(lamval), num_classes+1, 2))
    accuracy = np.zeros((len(sp1), len(sp2), len(lamval), 2))
    logloss = np.zeros((len(sp1), len(sp2), len(lamval), 2))

    for train_index, test_index in slo:
        train_data = [data[train_index, :], labels[train_index]]
        valid_data = [data[test_index , :], labels[test_index ]]

        size = train_data[0].shape[1]

        for i, s1 in enumerate(sp1):
            for j, s2 in enumerate(sp2):
                for k, num_out in enumerate(layer_out1):
                    exp = theanets.Experiment(
                        theanets.Classifier,
                        layers = (size, dict(size = num_out    , sparsity = sp1, activation = 'sigmoid'),
                                        dict(size = num_classes, sparsity = sp2, activation = 'sigmoid')),
                        weighted=False)

                    for l, curlamval in enumerate(lamval):
                        exp.train(train_data, valid_data, hidden_l1 = curlamval, algorithm='nag')#, #learning_rate=0.01,
    #                                      min_improvement=0.01):#, train_batches=50):
                        logloss[i, j, k, l, 0] += train['loss']
                        logloss[i, j, k, l, 1] += valid['loss']

                        out_train = exp.predict(train_data[0])
                        out_valid = exp.predict(valid_data[0])

                        accuracy[i, j, k, l, 0] += accuracy_score(train_data[1], out_train)
                        accuracy[i, j, k, l, 1] += accuracy_score(valid_data[1], out_valid)

                        precision[i, j, k, l, :-1, 0] += precision_score(train_data[1], out_train, average = None)
                        precision[i, j, k, l, -1, 0] += precision_score(train_data[1], out_train, average = 'macro')
                        precision[i, j, k, l, :-1, 1] += precision_score(valid_data[1], out_valid, average = None)
                        precision[i, j, k, l, -1, 1] += precision_score(valid_data[1], out_valid, average = 'macro')

                        recall[i, j, k, l, :-1, 0] += recall_score(train_data[1], out_train, average = None)
                        recall[i, j, k, l, -1, 0] += recall_score(train_data[1], out_train, average = 'macro')
                        recall[i, j, k, l, :-1, 1] += recall_score(valid_data[1], out_valid, average = None)
                        recall[i, j, k, l, -1, 1] += recall_score(valid_data[1], out_valid, average = 'macro')

                        f1[i, j, k, l, :-1, 0] += f1_score(train_data[1], out_train, average = None)
                        f1[i, j, k, l, -1, 0] += f1_score(train_data[1], out_train, average = 'macro')
                        f1[i, j, k, l, :-1, 1] += f1_score(valid_data[1], out_valid, average = None)
                        f1[i, j, k, l, -1, 1] += f1_score(valid_data[1], out_valid, average = 'macro')

    f1 /= num_iter
    recall  /= num_iter
    precision  /= num_iter
    logloss  /= num_iter
    accuracy  /= num_iter

    np.savez("mlp_final", accuracy = accuracy, recall = recall, f1 = f1,
                             precision = precision, logloss = logloss, sp1 = sp1,
                             sp2 = sp2, layer1 = layer_out1, lamb = lamval,
                             num_iter = num_iter, num_classes = num_classes,
                             samples_per_class = samples_per_class)
    return [accuracy, recall, precision, f1, logloss]



if __name__ == '__main__':
    sp1 = np.arange(0., 95., 1.).tolist()
    sp2 = np.arange(0., 95., 1.).tolist()
    lamval = [0., 0.0001, 0.001, 0.01, 0.1, 1., 10., 100., 1000.]
    num_classes = 36
    spc = 20 # samples per class
    datanm = '../data/a.npz'
    main_test(datanm, sp1, sp2, lamval, spc, num_classes, num_iter = 100)


# laplace is our f[0], but gamma distribution was not included!
# it is a strange behaviour but if I put the concatenated realizations, it won't work appropriately
# that is the reason we put it one by one
