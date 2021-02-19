import numpy as np

def load_train():

    traindatafile = "train-images-idx3-ubyte"
    trainlabelfile = "train-labels-idx1-ubyte"

    magic1 = np.fromfile(traindatafile, dtype=np.int32, count=1).byteswap().squeeze()
    assert magic1 == 2051
    dims = np.fromfile(traindatafile, dtype=np.int32, count=3, offset=4).byteswap()
    X_train = np.fromfile(traindatafile, dtype=np.dtype('>u1'), count=np.prod(dims),
                          offset=16).reshape(dims)

    magic2 = np.fromfile(trainlabelfile, dtype=np.int32, count=1).byteswap().squeeze()
    assert magic2 == 2049
    Y_train = np.fromfile(trainlabelfile, dtype=np.dtype('>u1'), count=dims[0], offset=8)

    return X_train, Y_train, dims

def load_test():

    testdatafile = "t10k-images-idx3-ubyte"
    testlabelfile = "t10k-labels-idx1-ubyte"

    magic1 = np.fromfile(testdatafile, dtype=np.int32, count=1).byteswap().squeeze()
    assert magic1 == 2051
    dims = np.fromfile(testdatafile, dtype=np.int32, count=3, offset=4).byteswap()
    X_test = np.fromfile(testdatafile, dtype=np.dtype('>u1'), count=np.prod(dims),
                         offset=16).reshape(dims)

    magic2 = np.fromfile(testlabelfile, dtype=np.int32, count=1).byteswap().squeeze()
    assert magic2 == 2049
    Y_test = (np.fromfile(testlabelfile, dtype=np.dtype('>u1'), count=dims[0], offset=8)
              .reshape(dims[0], 1))

    return X_test, Y_test, dims
