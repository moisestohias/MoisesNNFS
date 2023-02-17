# ActivationsReWrite.py
import numpy as np
# ReLU, LeakyReLU, ELU enable faster and better convergence than sigmoids.
# GELU: Gaussian Error Linear Unit used in most Transformers(GPT-3, BERT): paperswithcode.com/method/gelu
# Hard-Swish: paperswithcode.com/method/hard-swish

def sigmoid(x): return np.reciprocal((1.0+np.exp(-x)))
def sigmoid_prime(x): s = np.reciprocal((1.0+np.exp(-x))); return s * (1 - s) # σ(x)*(1-σ(x))
def relu(x): return np.where(x>= 0, x, 0)
def relu_prime(x): return np.where(x>= 0, 1, 0)
def leaky_relu(x, alpha=0.01): return np.where(x>= 0, x, alpha*x)
def leaky_relu_prime(x, alpha=0.01): return np.where(x>= 0, 1, alpha)
def elu(x, alpha=0.01): return np.where(x>= 0, x, alpha*(np.exp(x)-1))
def elu_prime(x, alpha=0.01): return np.where(x>= 0, 1, alpha*np.exp(x))
def swish(x): return x * np.reciprocal((1.0+np.exp(-x))) # x*σ(x) σ(x)+σ'(x)x : σ(x)+σ(x)*(1-σ(x))*x
def swish_prime(x): s = np.reciprocal((1.0+np.exp(-x))); return s+s*(1-s)*x #σ(x)+σ(x)*(1-σ(x))*x
silu, silu_prime = swish, swish_prime # The SiLU function is also known as the swish function.
def tanh(x): return np.tanh(x) # or 2.0*(σ((2.0 * x)))-1.0
def tanh_prime(x): return 1 - np.tanh(x) ** 2
def gelu(x): return 0.5*x*(1+np.tanh(0.7978845608*(x+0.044715*np.power(x,3)))) # sqrt(2/pi)=0.7978845608
def gelu_prime(x): return NotImplemented#Yet Error
def quick_gelu(x): return x*sigmoid(x*1.702) # faster version but inacurate
def quick_gelu_prime(x): return 1.702*sigmoid_prime(x*1.702)
def hardswish(x): return x*relu(x+3.0)/6.0
def hardswish_prime(x): return 1.0/6.0 *relu(x+3)*(x+1.0)
def softplus(x, limit=20.0, beta=1.0): return (1.0/beta) * np.log(1 + np.exp(x*beta))
def softplus_prime(limit=20, beta=1.0): _s = np.exp(x*beta) ; return (beta*_s)/(1+_s)
def relu6(x): return relu(x)-relu(x-6)
def relu6_prime(x): return relu_prime(x)-relu_prime(x-6)


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def __call__(self, input): return self.forward(input)
    def forward(self, input):
        self.input = input # save input for the backward pass
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Sigmoid(Activation):
  def __init__(self): super().__init__(sigmoid, sigmoid_prime)
class Relu(Activation):
  def __init__(self): super().__init__(relu, relu_prime)
class Relu6(Activation):
  def __init__(self): super().__init__(relu6, relu6_prime)
class LeakyRelu(Activation):
  def __init__(self, alpha=0.01): super().__init__(leaky_relu, leaky_relu_prime)
class Elu(Activation):
  def __init__(self, alpha=0.01): super().__init__(elu, elu_prime)
class Swish(Activation):
  def __init__(self): super().__init__(swish, swish_prime)
class Tanh(Activation):
  def __init__(self): super().__init__(tanh, tanh_prime)
class Gelu(Activation):
  def __init__(self): super().__init__(gelu, gelu_prime)
class Quick_gelu(Activation):
  def __init__(self): super().__init__(quick_gelu, quick_gelu_prime)
class Hardswish(Activation):
  def __init__(self): super().__init__(hardswish, hardswish_prime)
class Softplus(Activation):
  def __init__(self, limit=20.0, beta=1.0): super().__init__(softplus, softplus_prime)

class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # This version is faster than the one presented in the video
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
        # Original formula:
        # tmp = np.tile(self.output, n)
        # return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)

"""
# Plot
import matplotlib.pyplot as plt
x = np.arange(-6, 7, 0.1)
plt.title("ReLU and Swish functions", fontsize = 16)
plt.plot(sigmoid(x), label="Sigmoid(x)")
plt.legend(prop={'size': 10})
plt.grid()
plt.axes()
plt.show()
"""
