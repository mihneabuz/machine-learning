import numpy as np

def sigmoid(Z):
    return 1 / (1.01 + np.exp(-Z)) + 0.001

def sigmoid_true(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_der(dA, cache):
    sig = sigmoid(cache)
    return dA * sig * (1 - sig)

def relu(Z):
    return np.maximum(Z, 0)

def relu_der(dA, cache):
    dZ = np.array(dA, copy=True)
    dZ[cache <= 0] = 0
    return dZ


