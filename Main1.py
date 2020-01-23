import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import algorithms.ClassificationAlgorithms as cAlgo

# get DataSet
data = pd.read_csv("datasets/irisDataset.csv")
x = data.iloc[:, 0:4]
y = data.iloc[:, 4]

# split data to train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=63)

# train data
y_predict_knn = cAlgo.k_nearest_neighbor(x_train, y_train, x_test)
y_predict_svm = cAlgo.support_vector_machine(x_train, y_train, x_test)
y_predict_mlp = cAlgo.multilayer_perceptron(x_train, y_train, x_test, max_iter=500, activation='identity')
y_predict_nb = cAlgo.naive_bayes(x_train, y_train, x_test)
y_predict_dt = cAlgo.decision_tree(x_train, y_train, x_test)


def testResults(y_test, y_pred):
    result = {"true": 0, "false": 0}
    lst1 = y_test.tolist()
    lst2 = y_pred.tolist()
    for i in range(len(lst1)):
        if lst1[i] == lst2[i]:
            result["true"] = result["true"] + 1
        else:
            result["false"] = result["false"] + 1

    print((result["true"] / len(lst1)) * 100)


testResults(y_test, y_predict_knn)
testResults(y_test, y_predict_svm)
testResults(y_test, y_predict_mlp)
testResults(y_test, y_predict_nb)
testResults(y_test, y_predict_dt)
