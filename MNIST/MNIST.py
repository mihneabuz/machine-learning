from sys import path
path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from NNClass import NN
from time import time

traindatafile = "train-images-idx3-ubyte"
trainlabelfile = "train-labels-idx1-ubyte"
testdatafile = "t10k-images-idx3-ubyte"
testlabeslfile = "t10k-labels-idx1-ubyte"
order = "big"

magic1 = np.fromfile(traindatafile, dtype=np.int32, count=1).byteswap().squeeze()
assert(magic1 == 2051)
dims = np.fromfile(traindatafile, dtype=np.int32, count=3, offset=4).byteswap()
X_train = np.memmap(traindatafile, dtype=np.dtype('>u1'), mode='r',
                    offset=16).reshape(dims) / 255

print("Train data:", type(X_train), X_train.shape)

magic2 = np.fromfile(trainlabelfile, dtype=np.int32, count=1).byteswap().squeeze()
assert(magic2 == 2049)
Y_train_aux = np.memmap(trainlabelfile, dtype=np.dtype('>u1'), mode='r', offset=8)
Y_train = np.zeros((dims[0], 10))
for i in range(dims[0]):
    Y_train[i, Y_train_aux[i] % 10] = 1
print("Train labels:", type(Y_train), Y_train.shape)

choice = input("See some examples? Y/N\n")
if (choice.lower() == 'y'):
    for i in range(5):
        ex = np.random.randint(0, dims[0])
        print("Example: ", Y_train[ex])
        plt.imshow(X_train[ex])
        plt.show()

choice = input("Train nn? Y/N\n")
if (choice.lower() == 'y'):
    layers = (dims[1] * dims[2], 56, 28, 10)
    nn = NN(layers, 0.15)
    print("Training...")
    begin = time()
    costs = nn.learn(X_train.reshape(dims[0], dims[1] * dims[2]).T, Y_train.T, iter=1000)
    print("Done!")
    plt.plot(range(len(costs)), costs)
    print("Train time: {:.2f} minutes".format((time() - begin) / 60))
    plt.show()


magic1 = np.fromfile(testdatafile, dtype=np.int32, count=1).byteswap().squeeze()
assert(magic1 == 2051)
dims = np.fromfile(testdatafile, dtype=np.int32, count=3, offset=4).byteswap()
X_test = np.memmap(testdatafile, dtype=np.dtype('>u1'), mode='r',
                    offset=16).reshape(dims) / 255

#choice = input("See some examples? Y/N\n")
choice = 'y'
if (choice.lower() == 'y'):
    for i in range(20):
        ex = np.random.randint(0, dims[0])
        print("Prediction: ", np.argmax(
            nn.predict(X_test[ex].reshape(dims[1] * dims[2], 1))))
        plt.imshow(X_test[ex])
        plt.show()


