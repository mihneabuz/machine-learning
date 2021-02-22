import numpy as np
import random
from scipy import ndimage

def random_rotate(data, degrees=10):
    results = np.zeros(data.shape)
    for idx, im in enumerate(data):
        sign = 1
        if random.randint(0, 9) < 5:
            sign = -1

        results[idx] = ndimage.rotate(im, sign * degrees, axes=(1, 2), reshape=False)

    return results

def compute_mean_std(data):
    return data.mean(), data.std()

def mean_normalize(data, std, mean):
    return (data - mean) / std
