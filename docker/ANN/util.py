import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC

def load_CSV_file(file_name):
    tmp = np.loadtxt(file_name, dtype=np.str, delimiter=",")
    data = tmp[1:, 1:].astype(np.float)
    return data


def load_dataset(dataset_root, dataset, nDim, w, h):
    # load data
    data = np.zeros((nDim, w, h))
    for idx in range(nDim):
        file_name = dataset_root + dataset + f'\\{idx + 1}.csv'
        data[idx] = load_CSV_file(file_name)

    # load label
    label = load_CSV_file(dataset_root + dataset + "\\label.csv")

    print("Dataset Loaded. ")
    print("training data: ", data.shape)
    print("test data: ", data.shape)

    return data, label.reshape(1, w, h)


def remove_backgrounds(data, label):
    bg_idx = label.reshape(-1) != 0
    data_new = data[:, bg_idx]
    label_new = label[0, bg_idx]

    label_new = (label_new + 1) / 2  # convert label from {+1, -1} to {1, 0}
    return data_new.transpose(), label_new


def delete_one_dimension(data, iDim):
    data_new = np.zeros((data.shape[0], data.shape[1] - 1))
    count = 0
    for j in range(data.shape[0]):
        if j != iDim:
            data_new[:, count] = data[:, j]
            count += 1


def get_training_functions(Method, label):
    if Method == 'AdaBoost':
        clf = AdaBoostClassifier()
        trainFunc = clf.fit
        testFunc = clf.score
        outputFunc = clf.decision_function
        weight = label*100+1
    elif Method == 'DecisionTree':
        clf = DecisionTreeClassifier(max_depth=5)
        trainFunc = clf.fit
        testFunc = clf.score
        outputFunc = clf.predict_proba
        weight = label*100+1
    elif Method == 'LogisticRegression':
        clf = LogisticRegression()
        trainFunc = clf.fit
        testFunc = clf.score
        outputFunc = clf.decision_function
        weight = label*100+1
    elif Method == 'NaiveBayes':
        clf = GaussianNB()
        trainFunc = clf.fit
        testFunc = clf.score
        outputFunc = clf.predict_proba
        weight = label*999+1
    elif Method == 'RandomForest':
        clf = RandomForestRegressor(max_depth=3)
        trainFunc = clf.fit
        testFunc = clf.score
        outputFunc = clf.predict
        weight = label*100+1
    elif Method == 'SVM':
        clf = SVC(C=0.001, max_iter=200000, kernel='linear')
        # clf = SVC(C=100, max_iter=100)
        trainFunc = clf.fit
        testFunc = clf.score
        outputFunc = clf.decision_function
        weight = label*999+1
    else:
        print("Error Method" + Method + "!")
        return None, None, None
    return trainFunc, testFunc, outputFunc, weight
