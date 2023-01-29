"""
Resources:

Sliding window in loop:
+ opensourceoptions.com/blog/vectorize-moving-window-grid-operations-on-numpy-arrays/

Unpooling:
+ oreilly.com/library/view/hands-on-convolutional-neural/9781789130331/6476c4d5-19f2-455f-8590-c6f99504b7a5.xhtml
+ towardsdatascience.com/review-deconvnet-unpooling-layer-semantic-segmentation-55cf8a6e380e
"""
import numpy as np
"""
def max_pooling(x : np.ndarray, k: tuple):
    # If input doesn't fits,  pad it.
    if x.shape[0] % k[0] != 0: 
        x = np.pad(x, ((0, k[0] - x.shape[0] % k[0]), (0, 0), (0,0)), 'constant')
    if x.shape[1] % k[1] != 0:
        x = np.pad(x, ((0, 0), (0, k[1] - x.shape[1] % k[1]), (0,0)), 'constant')
    return x.reshape(x.shape[0] // k[0], k[0], x.shape[1] // k[1], k[1] ).max(axis=(1, 3))
"""




# def pool2D(x, kernel=(2, 2), stride=(1, 1)):
#     Nh, Nw = x.shape # input size
#     Kh, Kw = kernel # Kernel size (along height and width)
#     sh, sw = stride # strides along height and width

#     Oh = (Nh-Kh)//sh + 1 # output height
#     Ow = (Nw-Kw)//sw + 1 # output width
#     # creating appropriate strides
#     strides = (sh*Nw, sw, Nw, 1)
#     strides = tuple(i * x.itemsize for i in strides)
#     return np.lib.stride_tricks.as_strided(x, shape=(Oh, Ow, Kh, Kw), strides=strides)

def pooling(mat,ksize,method='max',pad=False):
    '''Non-overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize

    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,ky)
        nx=_ceil(n,kx)
        size=(ny*ky, nx*kx)+mat.shape[2:]
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
    else:
        ny=m//ky
        nx=n//kx
        mat_pad=mat[:ny*ky, :nx*kx, ...]

    new_shape=(ny,ky,nx,kx)+mat.shape[2:]

    if method=='max': result=np.nanmax(mat_pad.reshape(new_shape),axis=(1,3))
    else: result=np.nanmean(mat_pad.reshape(new_shape),axis=(1,3))

    return result

def pool2D(x, kernel=(2, 2), stride=(1, 1)):
    Xheight, Xwidth = x.shape
    Kheight, Kwidth = kernel
    if Xheight % Kheight != 0: x = np.pad(x, ((0, Kheight - Xheight % Kheight), (0, 0), (0,0)), 'constant')
    if Xwidth  % Kwidth  != 0: x = np.pad(x, ((0, 0), (0, Kwidth - Xwidth % Kwidth), (0,0)), 'constant')

    Oheight, Owidth = Xheight // Kheight, Xwidth//Kwidth
    return x[:Oheight*Kheight, :Owidth*Kwidth].reshape(Oheight, Kheight, Owidth, Kwidth).max(axis=(1, 3))

x = np.random.rand(15,15)
# out = pool2D(x, kernel=(2,2))
out = pooling(x, ksize=(2,2))
print(out.shape)



a = np.arange(128)
