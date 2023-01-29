import numpy as np
from scipy import signal
import math



"""
+ To be added:
    + all layer must define a method or attribute to return output shape
    + Layers:
        + trainable_parameters, input_shape, output_shape attributes must be stored in the base class Layers
        +
    + MaxPool2d:
    + Conv2d:
        + Stride & Dilation to the Conv2d
        + Non-symmetric kernel_size

+ def __call__ vs __call__ :
    + def __call__(self, input): return self.forward(input) # This doesn't requires forward before
    + __call__ = forward # this line must come after the definition of forward method

+ (N,C,H,W)
    N: Input Samples (aka batch)
    C: Channels
    H: Height
    W: Width

+ The input shape (not only the number of channels but width & height as well) is needed  to be able to calculate the bias shape, which is the same as the output shape.
+ The name 'input' is used rather than 'x' in the forward pass, because during the backward pass we need to refer to it.
"""


class Layer:
    def __init__(self): pass
    def __call__(self, x): return self.forward(x)
    def forward(self, input): raise NotImplementedError
    def backward(self, output_gradient, learning_rate): raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_shape, output_shape):
        # shape means size in the case of Dense
        self.input_shape = input_shape
        self.output_shape = output_shape, 1
        self.layers_name = self.__class__.__name__

        lim = 1 / math.sqrt(input_shape)
        # Like Mike - Each row represents the weights of the current neuron to the neurons in the previous layer. So that in the forward pass we don't transpose, allowing the weight matrix comes first.
        self.weights  = np.random.uniform(-lim, lim, (output_shape, input_shape))
        self.bias = np.random.randn(output_shape, 1)

    def forward(self, input):
        self.input = input # The input must be 1-column vector.
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        """
        To make it easier for you to remember/understand:
            + Both output_gradient & input are column vector
            + Output gradient has the same size as the output.
            + The weights_gradient matrix has the same size as the weights, thus
                + The number of rows must be equal to the length in the output_gradient.
                + The number of columns must be equal to the length in the input.
            + input_gradient has the same size as the input.
        """
        weights_gradient = np.dot(output_gradient, self.input.T) # Nx1•1xM = NxM
        input_gradient = np.dot(self.weights.T, output_gradient) # NxM•Mx1 = Nx1
        # optimizer.optimize(weights_gradient, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient

class Conv2d(Layer):
    """
    BackProp in CNN Explanation & How To: youtu.be/Lakz2MoHy6o
    *: Convolution =
    ⋆: Correlation
    !Note: X*K = X⋆rot180(K)
    """

    def __init__(self, input_shape, depth, kernel_size):
        #  stride & dilation to be added
        # Only symmetric kernel_size is allowed for now.

        """
        We need to know the input_shape: channels, height & width once the layer is created,
        because we need to Create & Initialize the layer's weight & biases
            + weights shape is calculated using the kernel and output depth
            + biases shape is calculated using the input shape and kernel size & output depth.
        Input channels is also needed during the forward/backward pass
        Ouput Shape can be calculated from the output of the convolution
        """

        self.input_shape = input_shape
        self.channels, self.input_height, self.input_width = input_shape
        self.depth = depth # Number of filters.
        self.output_shape = (depth, self.input_height - kernel_size + 1, self.input_width - kernel_size + 1)
        self.kernels_shape = (depth, self.channels, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape) # (depth, channels,height, width)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, x):
        self.input = x # Storing the input for the backward pass.
        self.output = np.copy(self.biases) # copy the bias instead zero_like avoiding sum in the loop
        for i in range(self.depth): # loop over depth first, each out channel is independent
            for j in range(self.channels): # loop over in_channel second, all channels must be summed.
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Using the output_gradient, we calculate the kernels_gradient, and the input_gradient.
        # kernels_gradient = ∂E/∂K_ij = X_j⋅∂E/Y_i = X⋆∂E/Y_i
        # inpt_gradient = output_gradient*K = output_gradient⋆rot180(K) !Note: both *&⋆ full version
        # biases_gradient = output_gradient
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.channels):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        # Here: Define a function responsible for updating the params be be able to freeze layers.
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

class Reshape(Layer):
    """Inefficient for Dense models, but To flatten a layer, this is the only solution. """
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input): return np.reshape(input, (self.output_shape,1))
    # def __call__(self, input): return self.forward(input)
    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)

class Dropout(Layer):
    def __init__(self, input_shape, p=0.1):
        self.p = p # Probability to Drop
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.mask = None

    def forward(self, input):
        self.mask = np.random.rand(*self.input_shape) < self.p
        output = np.copy(input)
        output[self.mask] = 0
        return output

    def backward(self, output_gradient):
        input_gradient = np.ones(self.input_shape)
        input_gradient[self.mask] = 0
        return input_gradient, None

if __name__ == "__main__":
    lr = 0.001
    x = np.random.randn(3, 28, 28)

    C1 = Conv2d(x.shape, 10, 3)
    C2 = Conv2d(C1.output_shape, 8, 3)
    ConvDrop = Dropout(C2.output_shape, 0.5)
    C3 = Conv2d(ConvDrop.output_shape, 6, 3)
    R = Reshape(C3.output_shape, 6*22*22)
    D1 = Dense(R.output_shape, 100)
    DenseDrop = Dropout((100,1), 0.2)
    D2 = Dense(100, 10)

    x1 = C1(x)
    x2 = C2(x1)
    x2 = ConvDrop(x2)
    x3 = C3(x2)
    x3 = C3(x2)
    x4 = R(x3)
    x5 = D1(x4)
    x5 = DenseDrop(x5)
    x6 = D2(x5)

    print("# Forward===========")
    print("Input:",x.shape)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    print(x4.shape)
    print(x5.shape)
    print(x6.shape)

    x_back1 = D2.backward(x6, lr)
    x_back2 = D1.backward(x_back1, lr)
    x_back3 = R.backward( x_back2, lr)
    x_back4 = C3.backward(x_back3, lr)
    x_back5 = C2.backward(x_back4, lr)
    x_back6 = C1.backward(x_back5, lr)

    print("# Backward===========")
    print("# Dense---")
    print(x_back1.shape)
    print(x_back2.shape)
    print("# Reshape---")
    print(x_back3.shape)
    print("# Conv---")
    print(x_back4.shape)
    print(x_back5.shape)
    print(x_back6.shape)
