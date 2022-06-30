# data_manupulation.py
import numpy as np


def shuffle_data(x, y, seed=None):
    """ Giving an ndarray of input samples and an ndarray of target data, Randomly shuffle of the samples in x and y.

    Parameters:
    -----------
    x: np.ndarray
        training samples
    y: np.ndarray
        target g
    seed : int

    Returns:
    --------
        tuple of ndarrays of the shuffled data.
    """

    if seed: np.random.seed(seed)
    idx = np.arange(x.shape[0]) # Only shuffle the highest dim (shape[0])
    np.random.shuffle(idx)
    return x[idx], y[idx]

def shuffle_data_tuples(x, seed=None):
    """ Giving an ndarray of tuple of (input,target) data, 
    Randomaly shuffle it"""
    if seed: np.random.seed(seed)
    return np.random.shuffle(x)


def batch_iterator(x: np.ndarray, y=None, batch_size=32):
    n_sampels = x.shape[0]
    for i in range(0, n_sampels, batch_size):
        # Accounting for the case of the last iteration. (ZCR, AE..)
        begin, end = i, min(i+batch_size, n_samples) 
        if y: yield x[begin,end], y[begin,end]
        else: yield x[begin,end]

def one_hot_scalar(y: int, classes : int): -> np.ndarray
    one_hot = np.zeros((classes))
    one_hot[y] = 1.0
    return one_hot

def one_hot_vector(y: np.ndarray, classes=None): : -> np.ndarray
    """ Take a one dimensional ndarray and return its one_hot 
    only handle one batch at a time.
    """
    if classes is None: classes = np.amax(y) + 1
    one_hot = np.zeros((y.shape[0], classes))
    one_hot[np.arange(y.shape[0]), y] = 1.0
    return one_hot

def one_hot_vec(y, classes=None): 
    """ The output is a ndaray of column vector unlike MLFS """
    if classes is None: classes = np.amax(y)+1
    one_hot = np.zeros((classes, y.size))
    one_hot[y, np.arange(y.size)] = 1.0
    return one_hot



