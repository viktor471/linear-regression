import numpy as np

# def uniform_kernel(x):
#     return 0.5

def uniform_kernel(x):
    if np.abs(x) < 1:
        return 0.5
    else:
        return 0.0


def triangular_kernel(x):
    return abs(1 - x)

def epanechnikov_kernel(x):
    return 0.75 * (1 - x**2)

def quartic_kernel(x):
    return 15/16 * (1 - x**2)**2
