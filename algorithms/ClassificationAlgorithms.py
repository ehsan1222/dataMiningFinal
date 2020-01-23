from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier


def k_nearest_neighbor(x_train, y_train, x_test, n_neighbors=5):
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='auto')
    neigh.fit(x_train, y_train)
    return neigh.predict(x_test)


def support_vector_machine(x_train, y_train, x_test):
    sVM = svm.SVC(kernel='linear')
    sVM.fit(x_train, y_train)
    return sVM.predict(x_test)


def multilayer_perceptron(x_train, y_train, x_test, hidden_layer_sizes=(100,), activation='relu', max_iter=200):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter, random_state=0)
    mlp.fit(x_train, y_train)
    return mlp.predict(x_test)