# MaxPooling.py

# https://towardsdatascience.com/advanced-numpy-master-stride-tricks-with-25-illustrated-exercises-923a9393ab20
# https://remykarem.medium.com/
# https://medium.com/analytics-vidhya/simple-cnn-using-numpy-part-iii-relu-max-pooling-softmax-c03a3377eaf2
# https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy
# https://nbviewer.org/github/craffel/crucialpython/blob/master/week3/stride_tricks.ipynb

import numpy as np
as_strided = np.lib.stride_tricks.as_strided
"""
# x = np.random.rand(6,6).astype(dtype=np.float16)
x = np.arange(1,37).reshape(6,6).astype(dtype=np.int8)
out = as_strided(x, shape=(3,3,2,2), strides=(12,2,6,1))
print(out.shape)
"""

"""
Kh, Kw = 2, 4 # ksize = 3,3
x = np.random.rand(24,26).astype(dtype=np.float16)
dtypeSize = x.itemsize
Xh, Xw = x.shape
print(dtypeSize)
print(Xh, Xw)
out = as_strided(x, shape=(Xh//Kh, Xw//Kw, Kh,Kw), strides=(Xw*Kh*dtypeSize,Kw*dtypeSize,Xw//dtypeSize,dtypeSize))
print(out.shape)
"""

def pool(x, ksize=(2,2)):
    Xh, Xw = x.shape
    Kh, Kw = ksize
    dtypeSize = x.itemsize # default stride along the first axis (column)
    return as_strided(x, shape=(Xh//Kh, Xw//Kw, Kh,Kw),
                      strides=(Xw*Kh*dtypeSize, Kw*dtypeSize,
                               Xw//dtypeSize, dtypeSize)).max(axis=(-2,-1))

# x = np.random.rand(24,24)
# x = np.arange(1,37).astype(np.int8).reshape(6,6)
x = np.random.randint(-9,10,(8,6)).astype(np.int8)
pooled = pool(x, (3,2))
print(x)
print(pooled)
# print(pooled.shape)
