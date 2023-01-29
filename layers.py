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
        Asking the user to determine the input shape for the layer is not super elegant solution, To do that we must determine the input size based on the previous layer. For now let's just roll with this implementation.
        """
        self.input_size = input_size
        self.output_size = output_size 
        self.layers_name = self.__class__.__name__

        # Hard coding the Initializer for now.
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

    # Using property and define only Getters
    @property
    def input_size(self): return self.input_size
    @property
    def input_size(self): return self.output_size
    @property
    def shape(self): return [self.input_size, self.output_size]


class Conv2d(Layer):
    def __init__(self, input_shape, depth, kernel_size):
        # stride & dialation arguments to be added
        # Only symetric kernel_size is allowed for now.
        self.input_depth, self.input_width, self.input_height = input_shape
        self.depth = depth # Number of filters.
        self.output_shape = (depth, self.input_height - kernel_size + 1, self.input_width - kernel_size + 1)
        self.kernels_shape = (depth, self.input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def __call__(self, input): return self.forward(input)
    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] = signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output, self.output_shape

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
    """Inefficient for Dense models, but To flatten a layer, this is the only solution. """
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input): return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)

if __name__ == "__main__":
    x = np.random.randn(3, 28, 28)
    conv2d = Conv2d(x.shape, 10, 3)
    x = conv2d(x)
    print(out.shape)
    x = conv2d(x)
