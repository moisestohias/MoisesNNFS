import numpy as np
from scipy import signal

"""
Calculate the size of the output layer:
    + I (ixi)    : The input dimensions of the image
    + k (kxk)    : The size of filter/kernel
    + S (integer): Strides
    + P (integer): Padding
    + D (integer): Depth/Number of feature maps/activation maps
Conv = [(I - K +2 *P) / S] +1 x D
Pool = [(I - K) / S] + 1 x D
"""



class Layer:
    def __init__(self): pass
    def __call__(self, x): return self.forward(x)
    def forward(self, input): raise NotImplementedError
    def backward(self, output_gradient, learning_rate): raise NotImplementedError

class MaxPool2d(Layer):
    def __init__(self, input_shape, kernel_size):
        self.input_shape = input_shape
        self.channels, self.Xh, self.Xw = input_shape
        self.Kh, self.Kw = kernel_size
        self.output_shape = self.channels, self.Xh//self.Kh, self.Xw//self.Kw
    def forward(self, input):
        out = np.zeros(self.output_shape)
        dtypeSize = input.itemsize # default stride along the first axis (column)
        for indx, channel in enumerate(input):
            out[indx] = np.lib.stride_tricks.as_strided(channel, shape=(self.Xh//self.Kh, self.Xw//self.Kw, self.Kh,self.Kw),
                       strides=(self.Xw*self.Kh*dtypeSize, self.Kw*dtypeSize,
                                self.Xw//dtypeSize, dtypeSize)).max(axis=(-2,-1))
        return out
    def backward(self, output_gradient): return np.zeros(self.input_shape)

class StridedMaxPool2d(Layer):
    def __init__(self, input_shape, kernel_size, stride):
        self.input_shape = input_shape
        self.channels, self.Xh, self.Xw = input_shape
        self.Kh, self.Kw = kernel_size
        self.output_shape = self.channels, self.Xh//self.Kh, self.Xw//self.Kw

        # Calculate the shape & stride outside the Loop
        dtypeSize = input.itemsize # default stride along the first axis (column)
        stride = stride*dtypeSize # If the strid is used not one.
        self.strides = self.Xw*self.Kh*stride, self.Kw*stride, self.Xw//stride, stride
        self.shape = self.Xh//self.Kh, self.Xw//self.Kw, self.Kh,self.Kw
    def forward(self, input):
        out = np.zeros(self.output_shape)
        for indx, channel in enumerate(input):
            out[indx] = np.lib.stride_tricks.as_strided(channel, shape=self.shape, strides=self.strides).max(axis=(-2,-1))
        return out
    def backward(self, output_gradient): return np.zeros(self.input_shape)

class Dense(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape, 1
        self.layers_name = self.__class__.__name__

        lim = 1 / np.sqrt(input_shape)
        self.weights  = np.random.uniform(-lim, lim, (output_shape, input_shape))
        self.bias = np.random.randn(output_shape, 1)

    def forward(self, input):
        self.input = input # The input must be 1-column vector.
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T) # Nx1•1xM = NxM
        input_gradient = np.dot(self.weights.T, output_gradient) # NxM•Mx1 = Nx1
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient

class Conv2d(Layer):
    def __init__(self, input_shape, depth, kernel_size):
        self.input_shape = input_shape
        self.channels, self.input_height, self.input_width = input_shape
        self.depth = depth # Number of filters.
        self.output_shape = (depth, self.input_height - kernel_size + 1, self.input_width - kernel_size + 1)
        self.kernels_shape = (depth, self.channels, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape) # (depth, channels,height, width)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, x):
        self.input = x
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.channels):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.channels):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    def forward(self, input): return np.reshape(input, (self.output_shape,1))
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
    MP = MaxPool2d(C2.output_shape, (2,2))
    C3 = Conv2d(MP.output_shape, 6, 3)
    R = Reshape(C3.output_shape, np.prod(C3.output_shape))
    D1 = Dense(R.output_shape, 100)
    D2 = Dense(100, 10)

    x1 = C1(x)
    x2 = C2(x1)
    x2mp = MP(x2)
    x3 = C3(x2mp)
    x4 = R(x3)
    x5 = D1(x4)
    x6 = D2(x5)

    print("# Forward===========")
    print("Input:",x.shape)
    print(x1.shape)
    print(x2.shape)
    print(x2mp.shape)
    print(x3.shape)
    print(x4.shape)
    print(x5.shape)
    print(x6.shape)
    x_back1 = D2.backward(x6, lr)
    x_back2 = D1.backward(x_back1, lr)
    x_back3 = R.backward( x_back2, lr)
    x_back4 = C3.backward(x_back3, lr)
    x_back4mp = MP.backward(x_back4)
    x_back5 = C2.backward(x_back4mp, lr)
    x_back6 = C1.backward(x_back5, lr)

    print("Backward===========")
    # print("Dense---")
    print(x_back1.shape)
    print(x_back2.shape)
    # print("Reshape---")
    print(x_back3.shape)
    # print("# Conv---")
    print(x_back4.shape)
    # print("Unpooling",x_back4mp.shape)
    print(x_back5.shape)
    print(x_back6.shape)
