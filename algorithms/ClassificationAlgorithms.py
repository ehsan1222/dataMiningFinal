from sklearn.neighbors import KNeighborsClassifier


def k_nearest_neighbor(x_train, y_train, x_test, n_neighbors=5):
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='auto')
    neigh.fit(x_train, y_train)
    return neigh.predict(x_test)

