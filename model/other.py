import numpy as np

class Nonparam_reg:
    def __init__(self, window_type, n_neighbors, window_width,
                 metric ='euclidean', kernel = 'uniform'):

        distance_func_set = {'euclidean': euqlidian_distance,
                             'manhattan': manhattan_distance,
                             'chebyshev': chebyshev_distance}

        kernel_func_set = {'uniform' : uniform_kernel,
                           'triangular' : triangular_kernel,
                           'epanechnikov' : epanechnikov_kernel,
                           'quartic' : quartic_kernel}

        window_type_set = ('fixed', 'unfixed')

        self.distance_func = distance_func_set[metric]
        self.kernel_func = kernel_func_set[kernel]

        self.window_type = window_type
        if self.window_type == 'fixed':
            self.window_width = window_width
        elif self.window_type == 'unfixed':
            self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        if len(X.shape) > 1:
            y_pred = [self._predict(x) for x in X]
        else:
            y_pred = self._predict(X)
        return np.array(y_pred)

    def _predict(self, x):
        distances = [self.distance_func(x, x_train) for x_train in self.X_train]

        if self.window_type == 'unfixed':
            self.window_width = sorted(distances)[self.n_neighbors - 1]

        kernel_func_values = np.array([self.kernel_func(d / self.window_width)
                                       for d in distances])

        return (np.sum([y_train * k_func_value for (y_train, k_func_value)
                in zip(self.y_train, kernel_func_values)], axis = 0)
                / np.sum(kernel_func_values))
