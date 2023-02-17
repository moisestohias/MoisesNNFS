import numpy as np
from scipy import signal
from numpy.lib.stride_tricks import as_strided

class Layer:
  def __init__(self): self.layers_name = self.__class__.__name__
  def __call__(self, x): return self.forward(x)
  def forward(self, x): raise NotImplementedError
  def backward(self, output_gradient, learning_rate): raise NotImplementedError

class Dense(Layer):
  def __init__(self, input_shape, output_shape):
    self.input_shape = input_shape
    self.output_shape = output_shape
    lim = 1/np.sqrt(input_shape)
    self.weights = np.random.uniform(-lim, lim, (output_shape, input_shape))
    self.biases = np.random.uniform(-1, 1, (output_shape,1))
    self.layers_name = self.__class__.__name__
  def forward(self, x):
    self.input = x.reshape(-1, 1) # Save the input for the backward passk, make sure x: Mx1
    return self.weights.dot(self.input) + self.biases # NxM•Mx1 + Nx1 = Nx1
  def backward(self, output_gradient, learning_rate): # Nx1
    weights_gradient = output_gradient.dot(self.input.T) # Nx1•1xM = NxM
    input_gradient = self.weights.T.dot(output_gradient) # MxN•Nx1 = Mx1
    self.weights -= weights_gradient*learning_rate
    self.biases -= self.biases*learning_rate
    return input_gradient

class Conv2d(Layer):
  def __init__(self, input_shape, OutCh, KS, stride=1):
    self.input_shape = input_shape
    self.OutCh = OutCh # Output channels aka depth
    self.InCh, self.InH, self.InW = input_shape
    self.KS = KS # KernelSize
    self.stride = stride
    self.output_shape = OutCh, int((self.InH-KS)/stride)+1, int((self.InW-KS)/stride)+1
    self.kernels = np.random.randn(self.OutCh, self.InCh, KS, KS)
    self.biases = np.random.randn(*self.output_shape)
    self.layers_name = self.__class__.__name__

  # Must be jitted
  def forward(self, x):
    self.input = x
    self.output = np.copy(self.biases)
    for i in range(self.OutCh):
      for j in range(self.InCh):
        self.output[i] = signal.correlate2d(self.input[j], self.kernels[i,j], "valid")
    return self.output

  # Must be jitted
  def backward(self, output_gradient, learning_rate):
    kernels_gradient = np.zeros(self.kernels.shape)
    input_gradient = np.zeros(self.input_shape)

    for i in range(self.OutCh):
      for j in range(self.InCh):
        kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
        input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

    self.kernels -= learning_rate * kernels_gradient
    self.biases -= learning_rate * output_gradient
    return input_gradient

class MaxPool2d(Layer):
  def __init__(self, input_shape, kernel_size):
    self.input_shape = input_shape
    self.channels, self.IH, self.IW = input_shape # InputHeight/Width
    self.KH, self.KW = kernel_size, kernel_size
    # Only pair H/W  or cropped otherwise
    self.output_shape = self.channels, self.IH//self.KH, self.IW//self.KW
  def forward(self, input):
    out = np.zeros(self.output_shape)
    dtypeSize = input.itemsize # default stride along the first axis (column)
    for indx, channel in enumerate(input): # Indexing a bit faster than zip(input, out)
      out[indx] = as_strided(channel,
                             shape=(self.IH//self.KH, self.IW//self.KW, self.KH,self.KW),
                             strides=(self.IW*self.KH*dtypeSize,
                                      self.KW*dtypeSize,
                                      self.IW//dtypeSize,
                                      dtypeSize)).max(axis=(-2,-1))
    return out
  def backward(self, output_gradient): return np.zeros(self.input_shape)


class Dropout(Layer):
  def __init__(self, input_shape, p):
    self.input_shape, self.output_shape = input_shape, input_shape
    self.p = p
    self.mask = None # New mask must be created for each call.
  def forward(self, x):
    self.mask = np.random.randn(*self.input_shape) < self.p
    x[self.mask] = 0
    return self.x
  def backward(self, output_gradient):
    input_gradient = np.copy(output_gradient)
    input_gradient[self.mask] = 0
    return input_gradient

class Reshape(Layer):
  def __init__(self, input_shape, output_shape):
    self.input_shape, self.output_shape = input_shape, output_shape
    self.layers_name = self.__class__.__name__
  def forward(self, x): return x.reshape(self.output_shape)
  def backward(self, output_gradient): return output_gradient.reshape(*self.input_shape)

class Flatten(Layer):
  """Special case of the Reshape where the output is one-dim vector"""
  def __init__(self, input_shape):
    self.input_shape, self.output_shape = input_shape, np.prod(input_shape)
    self.layers_name = self.__class__.__name__
  def forward(self, x): return x.reshape(self.output_shape, 1)
  def backward(self, output_gradient): return output_gradient.reshape(*self.input_shape)

if __name__ == '__main__':
  lr = 0.1
  x = np.random.randn(1,28,28)
  # C1 = Conv2d(x.shape, 8, 4)
  # MP1 = MaxPool2d(C1.output_shape, 2)
  # C2 = Conv2d(MP1.output_shape, 8, 4)
  # MP2 = MaxPool2d(C2.output_shape, 2)
  # R1 = Flatten(MP2.output_shape )
  # D1 = Dense(R1.output_shape, 100)
  # D2 = Dense(D1.output_shape, 10)

  C1 = Conv2d(x.shape, 8, 3)
  MP1 = MaxPool2d(C1.output_shape, 2)
  C2 = Conv2d(MP1.output_shape, 8, 4)
  MP2 = MaxPool2d(C2.output_shape, 2)
  R1 = Flatten(MP2.output_shape, )
  D1 = Dense(R1.output_shape, 100)
  D2 = Dense(D1.output_shape, 10)

  x1 = C1(x)
  x2 = MP1(x1)
  x3 = C2(x2)
  x4 = MP2(x3)
  x5 = R1(x4)
  x6 = D1(x5)
  x7 = D2(x6)

  print("Forward")
  print("Input", x.shape)
  print("Conv2d", x1.shape)
  print("MaxPool2d", x2.shape)
  print("Conv2d", x3.shape)
  print("MaxPool2d", x4.shape)
  print("Flatten", x5.shape)
  print("Dense", x6.shape)
  print("Dense", x7.shape)

  xOutBack = np.random.randn(*x7.shape)
  D2Back = D2.backward(xOutBack, lr)
  D1Back = D1.backward(D2Back, lr)
  R1Back = R1.backward(D1Back)
  MP2Back = MP2.backward(R1Back)
  C2Back = C2.backward(MP2Back, lr)
  MP1Back = MP1.backward(C2Back)
  C1Back = C1.backward(MP1Back, lr)

  print("Backward")
  print(xOutBack.shape)
  print(D2Back.shape)
  print(D1Back.shape)
  print(R1Back.shape)
  print(MP2Back.shape)
  print(C2Back.shape)
  print(MP1Back.shape)
  print(C1Back.shape)


