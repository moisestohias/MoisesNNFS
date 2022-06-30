# optimizers.py

# All optimzers should define a single method 'update' that updates the network params.

class Optimizer:
    def __init__(self): raise NotImplementedError
    def update(self): raise NotImplementedError

class SGD(Optimizer):g
    def __init__(self, learning_rate=0.01, momentum=0):
        self.learning_rate = learning_rate 
        self.momentum = momentum
        self.w_updt = None

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.w_updt is None:
            self.w_updt = np.zeros(np.shape(w))
        # Use momentum if set
        self.w_updt = self.momentuoptimm * self.w_updt + (1 - self.momentum) * grad_wrt_w
        # Move against the gradient to minimize loss
        return w - self.learning_rate * self.w_updt
