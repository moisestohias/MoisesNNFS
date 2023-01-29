# MoisesNNFS
### This is my attempt to write a Deep Learning framework from scratch using only pure Numpy. 

## Warning: In progress.

### Inspiration:
+ The awesome [ML-From-scratch](https://github.com/eriklindernoren/ML-From-Scratch) by [Erik Linder-Norén](https://github.com/eriklindernoren)
+ The super elegant and simple implementation [python-neural-networks
](https://github.com/OmarAflak/python-neural-networks) by [Omar Aflak](https://github.com/OmarAflak?tab=repositories)

### Challenges:
+ 😶 The biggest challenge is not just how to implement operations, but how to organize the code, such that the implementation won't come back to bite you later down the line.

# Design philosophy:
+ 🥇 Modularity: everything should be modular and independent, this goes for layers, activation functions, losses, optimizers...
+ Simplicity: the code must remain simple, easy to read as much as possible, 
+ Efficiency: is not the start of the show here, but I've trying to optimize the code as much as possible without violation the second design philosophy (Simplicity)

## Main Components:
+ Network: A container for the layers, takes in: optimizes, loss
+ Layers: The building block for NN:
    * Usual NN layers: Dense, Conv2d, MaxPool2d ... 
    * Activation layers: Relu, LeakyRelu, ...
    !Note: the reason behind formulating the activation as layers, to make the gradient calculation much simpler and cleaner.
+ Losses: for measuring the error.
+ Optimzers: Responsible for updating layers parameters.

## Layers:
Layers are the main building block of any NN framework, this one is no exception, everything is treated as layer, whether it's concrete NN layer (Dense/Conv2D/Pool...), utility layer (Dropout/Reshape) or Activation Function (The non-linearity part of the NN layer). 

+ NN concrete layers should be simple, they should only perform basic linear operations, the activation layer is the part responsible for the non-linearity during the forward pass. 
Each NN concrete layer, must define two fundamental methods:
    + forward : Take an input perform linear part returns output
    + backward: Take the Accumulated Gradient (AG) from next layer, calculate the gradient w.r.t its weight and biases and passes them to the optimizer.
    return accumulated gradient (aka error). 

## Activation function: are treated as layer. They are responsible for:
    + The non-linearity during the backward pass
    + Calculate the accumulated gradient during the backward pass.

## Weights:
Doesn't matter how they are represented, you're going to perform the transpose during the backward pass anyway. I Like Micheal's implementation, basically rows represent nodes in the current layer.

## Optimizers:
The optimization part is performed by the optimizer (updating the net parameters )


## Naming convention:
all names are underscore separated.
    losses: 
    Activation: mse/mse_prime, cross_entropy/cross_entropy_prime/
    p: y_pred : predicted/probability
    y: y_truth: target 

    input_size
    output_size
    layers_name

## Guidlines:
+ \__init\__ methods should never raise NotImplmentedErrro 
+ I don't wanna bother too much with type hinting, only use it where it's easy. 
+ The code must remain simple, I don't want to deal with exception and error handling.., and also makes the translate of code to anothr library easy.

## TIPs:
+ Use method instead of function call whenever possible for faster computation.

## Helper function:
As it's always the case you're going to need helper functions, since there will be many utils helper function, they should be organized into data manipulation utils, and misc utils.


```python
from MoisesNNFS.layers import Dense, Reshape
from MoisesNNFS.activatios import Tanh
from MoisesNNFS.losses import MSE
from MoisesNNFS.opimizers import SGD
from MoisesNNFS.Network import Network

MLP = Network(optimizer=SGD(learning_rate=0.001, epochs=20, batch_size=32), loss=MSE())
MLP.add(Reshape(np.product(data.shape),1), (data.shape))
MLP.add(Dense(100))
MLP.add(Tanh())
MLP.add(Dense(100))
MLP.add(Tanh())

MLP.fit(training_data, learning_rate=0.001, epochs=20, batch_size=32)

```

```python
optimizer = SGD() 
layers = [Reshape((1, 784), input_shape=(28, 28)),
    Dense(30),
    LeakyReLU(0.2),
    Dense(16),
    LeakyReLU(0.2),
    Dense(30),
    LeakyReLU(0.2),
    Dense(784),
    Reshape((28, 28)) ])
MLP = Network(layers=layer, optimizer=optimizer, loss=MSE())

MLP.fit(training_data, learning_rate=0.001, epochs=20, batch_size=32)
```
