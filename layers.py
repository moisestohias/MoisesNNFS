import numpy as np
from scipy import signal

"""
Layers must define the following attributes:
input_size 
output_size
layers_name

"""
class Layer:
    def __init__(self): pass
    def forward(self, input): raise NotImplementedError
    def backward(self, output_gradient, learning_rate): raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size):
        """
        We must determine the input size based on the previous layer, 
        Asking the user to determine the input shape for the layer is not elegant solution. For now let's just roll with this implementation.
        """
        self.input_size = input_size
        self.output_size = output_size 
        self.layers_name = self.__class__.__name__
        lim = 1 / math.sqrt(input_size)
        self.weights  = np.random.uniform(-lim, lim, (input_size, output_size))
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate, optimizer):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        optimizer.optimize(weights_gradient, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient

    @property
    def input_size(self): return self.input_size
    @property
    def input_size(self): return self.output_size


class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depthv # Number of filters.
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] = signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)
