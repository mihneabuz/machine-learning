import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from time import time

traindatafile = "train-images-idx3-ubyte"
trainlabelfile = "train-labels-idx1-ubyte"
testdatafile = "t10k-images-idx3-ubyte"
testlabelfile = "t10k-labels-idx1-ubyte"

print("Loading data...")
magic1 = np.fromfile(traindatafile, dtype=np.int32, count=1).byteswap().squeeze()
assert(magic1 == 2051)
dims = np.fromfile(traindatafile, dtype=np.int32, count=3, offset=4).byteswap()
dims = (dims[0], dims[1] * dims[2])
X_train = np.fromfile(traindatafile, dtype=np.dtype('>u1'), count=dims[0]*dims[1],
                      offset=16).reshape(dims)
print("Train set:", X_train.shape)

magic2 = np.fromfile(trainlabelfile, dtype=np.int32, count=1).byteswap().squeeze()
assert(magic2 == 2049)
Y_train_aux = np.fromfile(trainlabelfile, dtype=np.dtype('>u1'), count=dims[0], offset=8)
Y_train = np.zeros((dims[0], 10))
for i in range(dims[0]):
    Y_train[i, Y_train_aux[i] % 10] = 1
print("Train labels:", Y_train.shape)

clothes_dict = {0:"T-shirt", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat",
                5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle boot"}

choice = input("See some examples? Y\\N\n")
while (choice.lower() == "y"):
    for i in range(5):
        ex = np.random.randint(0, dims[0])
        print("Example:", clothes_dict[np.argmax(Y_train[ex])])
        plt.imshow(X_train[ex].reshape([28, 28]))
        plt.show()
    choice = input("See some more examples? Y\\N\n")

