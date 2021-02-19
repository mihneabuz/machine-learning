from sys import path
path.append('../')

from time import time
import numpy as np
import matplotlib.pyplot as plt
from NNClass import NN
from load_data import load_train, load_test

print("Loading train data...")
X_train, Y_train, dims = load_train()

print("Train data:", X_train.shape)
print("Train labels:", Y_train.shape)

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
    costs += nn.learn(X_train.reshape(dims[0], dims[1]*dims[2]).T, Y_train.T, iters=iters)
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
                          np.argmax(Y_train, axis=1).T)))

X_test, Y_test, dims = load_test()

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
    from drawimage import drawImage
    X_draw = np.array(drawImage())
    prediction, chance = nn.predict(X_draw.reshape(1, dims[1]* dims[2]).T)
    if chance > 0.75:
        print("That's a {} for sure!\n".format(prediction))
    elif chance < 0.5:
        print("Not quite sure, maybe a {}?\n".format(prediction))
    else:
        print("That's a {}!, i think".format(prediction))
    choice = input("Another one? Y/N\n")
