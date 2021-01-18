import numpy as np
import matplotlib
from utils import *

class NN:

    def __init__(self, layers, rate=0.1, lambd=0, dropout=0):
        self.parameters = {}
        self.layers = np.array(layers)
        self.learn_rate = rate
        self.random_initialization()
        self.normalization_u = 0
        self.normalization_sigma = 1
        self.lambd = lambd
        self.keep_prob = 1 - dropout

    def random_initialization(self):
        L = len(self.layers)
        for i in range(1, L):
            self.parameters['W' + str(i)] = (
                np.random.randn(self.layers[i], self.layers[i - 1]) *
                np.sqrt(2 / self.layers[i - 1]))
            self.parameters['b' + str(i)] = (
                np.random.randn(self.layers[i], 1) *
                np.sqrt(2 / self.layers[i - 1]))

    def activation_fw(self, A_prev, W, b, activation, dropout=True):
        Z = np.dot(W, A_prev) + b
        linear_cache = (A_prev, W, b)
        A = activation(Z)

        if (dropout and self.keep_prob < 1 ):
            D = np.random.rand(*A.shape) < self.keep_prob
            A = (A * D) / self.keep_prob
            linear_cache = (A_prev, W, b)
        else:
            D = np.array([0])

        activation_cache = Z
        cache = (linear_cache, activation_cache, D)

        assert(A.shape == (W.shape[0], A_prev.shape[1]))
        return A, cache

    def model_fw(self, X, dropout=True):
        caches = []
        L = len(self.layers) - 1
        A = X

        for i in range(1, L):
            A_prev = A
            A, cache = self.activation_fw(A_prev, self.parameters['W' + str(i)],
                                        self.parameters['b' + str(i)], relu, dropout)
            caches.append(cache)

        AL, cache = self.activation_fw(A, self.parameters['W' + str(L)],
                                     self.parameters['b' + str(L)], sigmoid, False)
        caches.append(cache)

        assert(AL.shape == (self.layers[L], X.shape[1]))
        return AL, caches

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -(np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), (1 - Y))).sum()/m
        if (self.lambd):
            L = len(self.layers)
            params_sum = 0
            for i in range(1, L):
                params_sum += np.sum(np.square(self.parameters["W" + str(i)]))
            cost += self.lambd / (2 * m) * params_sum
        assert(cost.shape == ())
        return cost

    def linear_bw(self, dZ, linear_cache):
        A_prev, W, b = linear_cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, A_prev.T) / m
        if (self.lambd):
            dW += (self.lambd / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        return dA_prev, dW, db

    def activation_bw(self, dA, cache, activation):
        linear_cache, activation_cache, D = cache

        if (D.any()):
            dA = (dA * D) / self.keep_prob
        dZ = activation(dA, activation_cache)
        dA_prev, dW, db = self.linear_bw(dZ, linear_cache)

        return dA_prev, dW, db

    def model_bw(self, AL, Y, caches, ):
        grads = {}
        L = len(self.layers) - 1
        m = AL.shape[1]

        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = (
            self.activation_bw(dAL, caches[L - 1], sigmoid_der))

        for i in reversed(range(L - 1)):
            grads["dA"+str(i + 1)], grads["dW"+str(i + 1)], grads["db"+str(i + 1)] = (
                self.activation_bw(grads["dA" + str(i + 2)], caches[i], relu_der))

        return grads

    def update_params(self, grads):
        L = len(self.layers)
        for i in range(1, L):
            self.parameters['W' + str(i)] -= self.learn_rate * grads['dW' + str(i)]
            self.parameters['b' + str(i)] -= self.learn_rate * grads['db' + str(i)]

    def params_mean(self):
        s = 0
        for i in self.parameters:
            s += np.mean(self.parameters[i])
        return s

    def learn(self, X, Y, iter=100, suppress=False):
        self.normalization_u = np.mean(X, axis=1).reshape(X.shape[0], 1)
        X_norm = X - self.normalization_u
        self.normalization_sigma = (np.std(X_norm, axis=1).reshape(X.shape[0], 1) +
                                    np.finfo(np.float64).eps)
        X_norm /= self.normalization_sigma

        costs = []
        AL, cache = self.model_fw(X_norm)
        grads = self.model_bw(AL, Y, cache)
        self.update_params(grads)
        cost = self.compute_cost(AL, Y)
        costs.append(cost)
        if (not suppress):
            print("Iters: ", 0, " Cost:", cost)

        for i in range(1, iter + 1):
            AL, cache = self.model_fw(X_norm)
            grads = self.model_bw(AL, Y, cache)
            self.update_params(grads)
            if (not suppress and not i % 10):
                cost = self.compute_cost(AL, Y)
                costs.append(cost)
                print("Iters:", i, " Cost:", cost)
        return costs

    def predict(self, X):
        X_norm = X - self.normalization_u
        X_norm /= self.normalization_sigma

        AL, cache = self.model_fw(X_norm, dropout=False)
        if (len(AL) > 1):
            return np.argmax(AL), np.max(AL)
        return AL

    def calculate_accuracy(self, X_test, Y_test):
        m = X_test.shape[1]
        count = 0
        for i in range(m):
            prediction = self.predict(X_test[:, i].reshape(X_test.shape[0], 1))[0]
            if (prediction == Y_test[i]):
                count += 1
        return count / m * 100

    def save_state(self, filename):
        file = open(filename, 'wb')
        np.array(len(self.layers)).astype(np.uint16).tofile(file)
        self.layers.astype(np.uint32).tofile(file)
        np.array(self.lambd).astype(np.float32).tofile(file)
        np.array(self.keep_prob).astype(np.float32).tofile(file)
        np.array(self.normalization_u).astype(np.float64).tofile(file)
        np.array(self.normalization_sigma).astype(np.float64).tofile(file)
        for i in range(1, len(self.layers)):
            self.parameters['W' + str(i)].tofile(file)
            self.parameters['b' + str(i)].tofile(file)

    def load_state(self, filename):
        file = open(filename, 'rb')
        layers = np.fromfile(file, dtype=np.uint16, count=1).squeeze()
        self.layers = np.fromfile(file, dtype=np.uint32, count=layers)
        self.lambd = np.fromfile(file, dtype=np.float32, count=1).squeeze()
        self.keep_prob = np.fromfile(file, dtype=np.float32, count=1).squeeze()
        self.normalization_u = (np.fromfile(file, dtype=np.float64, count=self.layers[0])
                                .reshape(self.layers[0], 1))
        self.normalization_sigma = (np.fromfile(file, dtype=np.float64, count=self.layers[0])
                                    .reshape(self.layers[0], 1))
        for i in range(1, len(self.layers)):
            self.parameters['W' + str(i)] = (
                np.fromfile(file, dtype=np.float64, count=self.layers[i]*self.layers[i-1])
                    .reshape(self.layers[i], self.layers[i-1]))
            self.parameters['b' + str(i)] = (
                np.fromfile(file, dtype=np.float64, count=self.layers[i])
                    .reshape(self.layers[i], 1))

    def info(self):
        print("Layers:", self.layers)
        print("Regularization lambda:", self.lambd)
        print("Dropout chance:", np.around(1 - self.keep_prob, 2), '\n')

    def debug(self):
        x = np.array([-1, -0.5, 0, 0.5, 1])
        print("Testing activation functions: ", x)
        print("Sigmoid:", list(sigmoid(x)))
        print("Sigmoid Grad:", list(sigmoid_der(x, np.ones(5))))
        print("ReLu:" , list(relu(x)))
        print("ReLu Grad:", list(relu_der(x, np.ones(5))))
        print("Layer sizez:", self.layers)
        print("Parameter matrix sizes")
        for key, value in self.parameters.items():
            print(key, value.shape)
        print("Forward pass on one example")
        AL1, cache1 = self.model_fw(np.random.randn(self.layers[0], 1))
        print("Done!")
        print("Forward pass on 100 examples")
        AL2, cache2 = self.model_fw(np.random.randn(self.layers[0], 100))
        print("Done!")
        print("Cost function:")
        self.compute_cost(AL2, np.ones(AL2.shape))
        print("Done!")
        print("Backward pass on one example")
        grads1 = self.model_bw(AL1, np.ones(AL1.shape), cache1)
        print("Done!")
        print("Backward pass on 100 examples")
        grads2 = self.model_bw(AL2, np.ones(AL2.shape), cache2)
        print("Done!")
        for key in self.parameters.keys():
            assert('d' + key in grads1 and 'd' + key in grads2)
        print("Update parameters")
        self.update_params(grads1)
        self.update_params(grads2)
        print("Done!")
        print("Learning")
        self.learn(np.random.randn(self.layers[0], 100), np.ones((self.layers[-1], 100)),
                   suppress=True)
        print("Done!")
        print("Prediction")
        self.predict(np.random.rand(self.layers[0], 1))
        print("Done!")
        print("Saving parameters")
        self.info()
        self.save_state("test.bin")
        print("Done!")
        print("Loading parameters")
        self.load_state("test.bin")
        self.info()
        print("Done!")
