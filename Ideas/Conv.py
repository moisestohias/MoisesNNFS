import numpy as np
def conv(x, w, b, stride, pad):
  # N, C = x.shape[-4:-2]
  H, W = x.shape[-2:]
  F, _, HH, WW = w.shape

  # Check dimensions
  assert (W + 2 * pad - WW) % stride == 0, 'width does not work'
  assert (H + 2 * pad - HH) % stride == 0, 'height does not work'

  # Pad the input
  p = pad
  x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

  # Figure out output dimensions
  H += 2 * pad
  W += 2 * pad
  out_h = (H - HH) / stride + 1
  out_w = (W - WW) / stride + 1

  # Perform an im2col operation by picking clever strides
  shape = (C, HH, WW, N, out_h, out_w)
  strides = (H * W, W, 1, C * H * W, stride * W, stride)
  strides = x.itemsize * np.array(strides)
  x_stride = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
  x_cols = np.ascontiguousarray(x_stride)
  x_cols.shape = (C * HH * WW, N * out_h * out_w)

  # Now all our convolutions are a big matrix multiply
  res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)

  # Reshape the output
  res.shape = (F, N, out_h, out_w)
  out = res.transpose(1, 0, 2, 3)
  out = np.ascontiguousarray(out)
  return out


x = np.random.randn(24,24)
out = conv(x, (2,2))
