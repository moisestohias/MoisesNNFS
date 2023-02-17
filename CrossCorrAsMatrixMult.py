import numpy as np

def conv2d(image, kernel):
    # Get the dimensions of the image and kernel
    IH, IW = image.shape
    KH, KW = kernel.shape
    
    # Determine the dimensions of the output image
    OutH = IH - KH + 1
    OutW = IW - KW + 1
    
    # Convert the image into a 1D vector
    image_vector = image.reshape(-1, 1)
    
    # Create the Double block Toeplitz matrix of the kernel
    toeplitz_matrix = np.zeros((IH * IW, OutH * OutW))
    for i in range(OutH * OutW):
        row = i // OutW
        col = i % OutW
        res = np.flip(kernel.reshape(-1))
        print(res.shape)
        print(toeplitz_matrix[row:row + KH, i].shape)
        raise SystemExit
        toeplitz_matrix[row:row + KH, i] = res

    print(toeplitz_matrix.shape)

    # Multiply the Toeplitz matrix and the image vector
    result = np.dot(toeplitz_matrix, image_vector)
    
    # Reshape the result back into a 2D image
    result = result.reshape(OutH, OutW)
    
    return result

if __name__ == '__main__':
    image = np.random.randint(1,10, (8,8)).astype(np.int32)
    kernel = np.random.randint(1, 6, (2,2)).astype(np.int32)
    print(image.shape)
    print(kernel.shape)
    res = conv2d(image, kernel)
