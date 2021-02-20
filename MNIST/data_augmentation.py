import numpy as np
import random
from scipy import ndimage
from drawimage import drawImage
from matplotlib import pyplot as plt

def random_rotate(data, degrees=10):
    results = np.zeros(data.shape)
    for idx, im in enumerate(data):
        sign = 1
        if random.randint(0, 9) < 5:
            sign = -1

        results[idx] = ndimage.rotate(im, sign * degrees, axes=(1, 2), reshape=False)

    return results

def mean_normalize(data):
    mean = data.mean()
    std = data.std()
    return (data - mean) / std
