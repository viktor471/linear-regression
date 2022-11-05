import numpy as np

def euclidian_distance(x, xi):
    return np.sqrt(np.sum((x - xi) ** 2))


def manhattan_distance(x, xi):
    return np.sum(abs(x - xi))


def chebyshev_distance(x, xi):
    return np.max(abs(x - xi))
