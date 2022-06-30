# layers.py


"""
https://www.youtube.com/watch?v=8Tg1vk9SUBU
https://www.youtube.com/watch?v=2-Ol7ZB0MmU

https://github.com/JaeDukSeo/Only_Numpy_Basic
https://towardsdatascience.com/only-numpy-understanding-back-propagation-for-transpose-convolution-in-multi-layer-cnn-with-c0a07d191981

https://stackoverflow.com/questions/34254679/how-can-i-implement-deconvolution-layer-for-a-cnn-in-numpy
https://stackoverflow.com/questions/41699513/how-to-update-the-weights-of-a-deconvolutional-layer
https://github.com/many-facedgod/Numpy-Atrous-Transposed-CNN
https://github.com/many-facedgod
"""

class Layer:
    def __call__(slef): raise NotImplementedError()
    def backward(slef): raise NotImplementedError()

class Upsampling(Layer):
    def __call__(slef): raise NotImplementedError()
    def backward(slef): raise NotImplementedError()

class Dense(Layer):
    """ The weights are stored in Micheal's implementation,
    each row represent a neuron in the current layer,
    the columns represent the weights to the previews input layer.

    MLFS stores the weights differently, where
    each column represent neuron in the current layer,
    the rows represent the weights to the previews input layer.

    """
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None # NotreallyneedetobeInitializedHere, OnlyUsed in for/backward
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W  = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.B = np.zeros((1, self.n_units))

    def forward(slef, X ):
        self.layer_input = X # save the input for later use in the BP pass.
        return X.dot(self.W) + self.w0 #
    def backward(slef, accum_grad):
        W = self.W # Save weights used during forwards pass, to be updated after BP pass.
        if self.trainable:
            # Calculate gradient w.r.t layer weights
            grad_w = self.layer_input.T.dot(accum_grad)
            grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)


            # Update the layer weights
            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)

        # Return accumu
class Conv2D(Layer):
    def forward(slef): raise NotImplementedError()
    def backward(slef): raise NotImplementedError()