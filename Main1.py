import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import algorithms.ClassificationAlgorithms as cAlgo
from sklearn.model_selection import KFold

# get DataSet
data = pd.read_csv("datasets/irisDataset.csv")
x = data.iloc[:, 0:4]
y = data.iloc[:, 4]

print("10-fold cross validation:")
print("-------------------------")

# define 10 fold cross validation
kf = KFold(n_splits=10)

# initial classifiers value

classifiers_value = {
    "knn": 0,
    "svm": 0,
    "mlp": 0,
    "nb": 0,
    "dt": 0,
    "nn": 0
}

for train_index, test_index in kf.split(x):
    # split data to train and test
    x_train, x_test, y_train, y_test = np.array(x)[train_index], np.array(x)[test_index], \
                                       np.array(y)[train_index], np.array(y)[test_index]

    # train data
    y_predict_knn = cAlgo.k_nearest_neighbor(x_train, y_train, x_test)
    y_predict_svm = cAlgo.support_vector_machine(x_train, y_train, x_test)
    y_predict_mlp = cAlgo.multilayer_perceptron(x_train, y_train, x_test, max_iter=500, activation='identity')
    y_predict_nb = cAlgo.naive_bayes(x_train, y_train, x_test)
    y_predict_dt = cAlgo.decision_tree(x_train, y_train, x_test)
    y_predict_nn = cAlgo.neural_network(x_train, y_train, x_test, max_iter=1000, activation='identity')


    def testResults(y_test, y_pred):
        result = {"true": 0, "false": 0}
        lst1 = y_test.tolist()
        lst2 = y_pred.tolist()
        for i in range(len(lst1)):
            if lst1[i] == lst2[i]:
                result["true"] = result["true"] + 1
            else:
                result["false"] = result["false"] + 1

        return result["true"] / len(lst1)


    classifiers_value["knn"] = classifiers_value["knn"] + testResults(y_test, y_predict_knn)
    classifiers_value["svm"] = classifiers_value["svm"] + testResults(y_test, y_predict_svm)
    classifiers_value["mlp"] = classifiers_value["mlp"] + testResults(y_test, y_predict_mlp)
    classifiers_value["nb"] = classifiers_value["nb"] + testResults(y_test, y_predict_nb)
    classifiers_value["dt"] = classifiers_value["dt"] + testResults(y_test, y_predict_dt)
    classifiers_value["nn"] = classifiers_value["nn"] + testResults(y_test, y_predict_nn)

print("k nearest neighbor:", "%.2f" % (classifiers_value["knn"] * 10))
print("support vector machine:", "%.2f" % (classifiers_value["svm"] * 10))
print("multilayer perceptron:", "%.2f" % (classifiers_value["mlp"] * 10))
print("naive bayes:", "%.2f" % (classifiers_value["nb"] * 10))
print("decision tree:", "%.2f" % (classifiers_value["dt"] * 10))
print("neural network:", "%.2f" % (classifiers_value["nn"] * 10))
print("------------------------------------------------")
print()
print("Leave one out cross validation:")
print("-------------------------------")