import numpy as np

def activation_sigmoid(X):
    return 1 / (1 + np.exp(-X))

def activation_tanh(X):
    return np.tanh(X)

def activation_softmax(X):
    return np.exp(X) / np.sum(np.exp(X), axis=1).reshape(-1, 1)

def tanh_derivative(X):
    return 1 - (X**2)
