# looses.py
import numpy as np 


"""
I think we need a better name for the loss/gradient
Naming convention:
    p: y_pred : predicted/probability
    y: y_truth: target 
"""


def binary_cross_entropy(y, p):
    return np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p))

def binary_cross_entropy_prime(y, p):
    return ((1 - y) / (1 - p) - y / p) / np.size(y)
# =================================================================================
class Loss:
    def loss(self, y, p): pass
    def gradient(self, y, p): raise NotImplementedError()
    def acc(self, y, p): return 0

class MSE(Loss):
    def loss(self, y, p): return 0.5 * np.power((y - p), 2)
    def gradient(self, y, p): return -(y - p)

class CrossEntropy(Loss):
    # This class only calculate the loss for a single input, to vectorize, you need to take the mean in the loss, deviding by y.size gradient.
    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return  - (y / p) + (1 - y) / (1 - p)

    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))



