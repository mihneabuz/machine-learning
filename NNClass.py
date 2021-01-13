import numpy as np
import matplotlib

def sigmoid(Z):
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

class NN:

    def __init__(self, layers, rate=0.1):
        self.parameters = {}
        self.layers = np.array(layers)
        self.learn_rate = rate
        self.random_initialization()

    def random_initialization(self):
        L = len(self.layers)
        for i in range(1, L):
            self.parameters['W' + str(i)] = (
                np.random.randn(self.layers[i], self.layers[i - 1]) * 0.01)
            self.parameters['b' + str(i)] = (
                np.random.randn(self.layers[i], 1))

    def activation_fw(A_prev, W, b, activation):
        Z = np.dot(W, A_prev) + b
        linear_cache = (A_prev, W, b)
        A = activation(Z)
        activation_cache = Z
        cache = (linear_cache, activation_cache)

        assert(A.shape == (W.shape[0], A_prev.shape[1]))
        return A, cache

    def model_fw(self, X):
        caches = []
        L = len(self.layers) - 1
        A = X

        for i in range(1, L):
            A_prev = A
            A, cache = NN.activation_fw(A_prev, self.parameters['W' + str(i)],
                                        self.parameters['b' + str(i)], relu)
            caches.append(cache)

        AL, cache = NN.activation_fw(A, self.parameters['W' + str(L)],
                                     self.parameters['b' + str(L)], sigmoid)
        caches.append(cache)

        assert(AL.shape == (self.layers[L], X.shape[1]))
        return AL, caches

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(np.sum(
            np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), (1 - Y))))

        assert(cost.shape == ())
        return cost

    def linear_bw(dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        return dA_prev, dW, db

    def activation_bw(dA, cache, activation):
        linear_cache, activation_cache = cache

        dZ = activation(dA, activation_cache)
        dA_prev, dW, db = NN.linear_bw(dZ, linear_cache)

        return dA_prev, dW, db

    def model_bw(self, AL, Y, caches):
        grads = {}
        L = len(self.layers) - 1
        m = AL.shape[1]

        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = (
            NN.activation_bw(dAL, caches[L - 1], sigmoid_der))

        for i in reversed(range(L - 1)):
            grads["dA"+str(i+1)], grads["dW"+str(i+1)], grads["db"+str(i+1)] = (
                NN.activation_bw(grads["dA" + str(i+2)], caches[i], relu_der))

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
        costs = []
        for _ in range(iter):
            AL, cache = self.model_fw(X)
            grads = self.model_bw(AL, Y, cache)
            self.update_params(grads)
            if (not suppress and not _ % 10):
                cost = self.compute_cost(AL, Y)
                costs.append(cost)
                if (not suppress):
                    print("Iters:", _, " Cost:", cost)
        return costs

    def predict(self, X):
        AL, cache = self.model_fw(X)
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
        self.layers.astype(np.uint16).tofile(file)
        for i in range(1, len(self.layers)):
            self.parameters['W' + str(i)].tofile(file)
            self.parameters['b' + str(i)].tofile(file)

    def load_state(self, filename):
        file = open(filename, 'rb')
        layers = np.fromfile(file, dtype=np.uint16, count=1).squeeze()
        self.layers = np.fromfile(file, dtype = np.uint16, count=layers)
        for i in range(1, len(self.layers)):
            self.parameters['W' + str(i)] = (
                np.fromfile(file, dtype=np.float64, count=self.layers[i]*self.layers[i-1])\
                    .reshape(self.layers[i], self.layers[i-1]))
            self.parameters['b' + str(i)] = (
                np.fromfile(file, dtype=np.float64, count=self.layers[i])
                    .reshape(self.layers[i], 1))

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
        self.save_state("test.bin")
        print("Done!")
        print("Loading parameters")
        self.load_state("test.bin")
        print("Done!")
