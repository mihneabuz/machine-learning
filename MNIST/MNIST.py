from sys import path
path.append('../')

from time import time
import numpy as np
import matplotlib.pyplot as plt
from NNClass import NN
from drawimage import drawImage

traindatafile = "train-images-idx3-ubyte"
trainlabelfile = "train-labels-idx1-ubyte"
testdatafile = "t10k-images-idx3-ubyte"
testlabelfile = "t10k-labels-idx1-ubyte"
order = "big"

print("Loading train data...")
magic1 = np.fromfile(traindatafile, dtype=np.int32, count=1).byteswap().squeeze()
assert magic1 == 2051
dims = np.fromfile(traindatafile, dtype=np.int32, count=3, offset=4).byteswap()
X_train = np.memmap(traindatafile, dtype=np.dtype('>u1'), mode='r',
                    offset=16).reshape(dims)

print("Train data:", type(X_train), X_train.shape)

magic2 = np.fromfile(trainlabelfile, dtype=np.int32, count=1).byteswap().squeeze()
assert magic2 == 2049
Y_train_aux = np.memmap(trainlabelfile, dtype=np.dtype('>u1'), mode='r', offset=8)
Y_train = np.zeros((dims[0], 10))
for i in range(dims[0]):
    Y_train[i, Y_train_aux[i] % 10] = 1
print("Train labels:", type(Y_train), Y_train.shape)

choice = input("See some examples? Y/N\n")
while choice.lower() == 'y':
    for i in range(5):
        ex = np.random.randint(0, dims[0])
        print("Example: ", np.argmax(Y_train[ex]))
        plt.imshow(X_train[ex], cmap='gray')
        plt.show()
    choice = input("See some more examples? Y/N\n")

nn = None
choice = input("Load parameters? Y/N\n")
if choice.lower() == 'y':
    nn = NN([1, 1])
    param_file = input("File name: ")
    nn.load_state(param_file)
    print("Parameters loaded from " + param_file)
    nn.info()
    choice = input("Change hyper-parameters? Y/N\n")
    if choice.lower() == 'y':
        lambd = float(input("Input regularization lambda: "))
        drop_chance = float(input("Input dropout chance: "))
        nn.lambd = lambd
        nn.keep_prob = 1 - drop_chance
        nn.info()

costs = []
choice = input("Train model? Y/N\n")
while choice.lower() == 'y':
    if not nn:
        string = input("Input size of hidden layers:\n")
        hidden_layers = [int(x) for x in string.split()]
        layers = (dims[1] * dims[2], *hidden_layers, 10)
        lambd = float(input("Input regularization lambda: "))
        drop_chance = float(input("Input dropout chance: "))
        nn = NN(layers, 0.1, lambd, drop_chance)
        print("Created neural network with:")
        nn.info()
    learning_rate = float(input("Learning rate: "))
    iters = int(input("Number of iterations: "))
    nn.learn_rate = learning_rate
    print("Training...")
    begin = time()
    costs += nn.learn(X_train.reshape(dims[0], dims[1] * dims[2]).T, Y_train.T, iter=iters)
    print("Done!")
    print("Train time: {:.2f} minutes".format((time() - begin) / 60))

    print("Showing cost function plot")
    plt.plot(range(len(costs)), costs)
    plt.show()
    choice = input("Train some more? Y/N\n")

if not nn:
    exit()

print("Calculating accuracy...")

print("Train accuracy: {:.1f}%".format(
    nn.calculate_accuracy(X_train.reshape(dims[0], dims[1] * dims[2]).T,
                          Y_train_aux.reshape(dims[0], 1))))

magic1 = np.fromfile(testdatafile, dtype=np.int32, count=1).byteswap().squeeze()
assert magic1 == 2051
dims = np.fromfile(testdatafile, dtype=np.int32, count=3, offset=4).byteswap()
X_test = np.memmap(testdatafile, dtype=np.dtype('>u1'), mode='r',
                    offset=16).reshape(dims)

magic2 = np.fromfile(testlabelfile, dtype=np.int32, count=1).byteswap().squeeze()
assert magic2 == 2049
Y_test = (np.memmap(testlabelfile, dtype=np.dtype('>u1'), mode='r', offset=8)
            .reshape(dims[0], 1))

print("Test accuracy: {:.1f}%".format(
    nn.calculate_accuracy(X_test.reshape(dims[0], dims[1] * dims[2]).T, Y_test)))

choice = input("Save parameters? Y/N\n")
if choice.lower() == 'y':
    param_file = input("File name: ")
    nn.save_state(param_file)
    print("Parameters saved in " + param_file)
    nn.info()

choice = input("Try some predictions? Y/N\n")
while choice.lower() == 'y':
    for i in range(10):
        ex = np.random.randint(0, dims[0])
        print("Prediction: ", nn.predict(X_test[ex].reshape(1, dims[1] * dims[2]).T)[0])
        plt.imshow(X_test[ex], cmap='gray')
        plt.show()
    choice = input("Try some more predictions? Y/N\n")

choice = input("Try to draw some examples? Y/N\n")
if choice.lower() == 'y':
    print("Draw a digit and press enter!")
while choice.lower() == 'y':
    X_draw = np.array(drawImage())
    prediction, chance = nn.predict(X_draw.reshape(1, dims[1]* dims[2]).T)
    if (chance > 0.75):
        print("That's a {} for sure!\n".format(prediction))
    elif (chance < 0.5):
        print("Not quite sure, maybe a {}?\n".format(prediction))
    else:
        print("That's a {}!, i think".format(prediction, chance))
    choice = input("Another one? Y/N\n")
