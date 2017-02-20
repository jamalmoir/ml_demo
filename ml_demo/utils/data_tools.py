import numpy as np


def split_data(data: np.array, train_test: bool = False) -> dict:
    """Split numpy array containing data and targets into separate arrays and optionally into training
    and testing data."""
    X = data[:, :-1]
    y = data[:, -1]

    if train_test:
        np.random.seed(0)
        indices = np.random.permutation(len(X))
        train_size = int(np.round(X.shape[0] / 100 * 15))
        print(train_size)
        X_train = X[indices[:-train_size]]
        y_train = y[indices[:-train_size]]
        X_test = X[indices[-train_size:]]
        y_test = y[indices[-train_size:]]

        return {'X': X_train, 'y': y_train, 'X_test': X_test, 'y_test': y_test}
    else:
        return {'X': X, 'y': y}
