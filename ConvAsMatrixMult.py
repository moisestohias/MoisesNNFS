import numpy as np

import numpy as np

def conv2d(image, kernel):
    # Get the dimensions of the image and kernel
    i_h, i_w = image.shape[0], image.shape[1]
    k_h, k_w = kernel.shape[0], kernel.shape[1]

    # Determine the dimensions of the output image
    o_h = i_h - k_h + 1
    o_w = i_w - k_w + 1

    # Convert the image into a 1D vector
    image_vec = image.flatten()

    # Reshape the kernel into a 1D vector
    kernel_vec = kernel.flatten()

    # Create the Toeplitz matrix using the kernel vector
    toeplitz_matrix = np.zeros((i_h * i_w, k_h * k_w))
    for i in range(k_h * k_w):
        toeplitz_matrix[i:, i] = kernel_vec[:-i]

    # Multiply the Toeplitz matrix and the image vector
    result = np.dot(toeplitz_matrix, image_vec)

    # Reshape the result back into a 2D image
    result = result.reshape((o_h, o_w))

    return result

if __name__ == '__main__':
    image = np.random.randint(1,10, (4,4)).astype(np.int32)
    kernel = np.random.randint(1, 6, (2,2)).astype(np.int32)
    print(image.shape)
    print(kernel.shape)
    res = conv2d(image, kernel)


"""
convolution as matrix multiplication
Create python code that implements the cross-correlation as matrix multiplcation, without using loops,
follow this pseudocode:
# Get the dimensions of the image and kernel
# Determine the dimensions of the output image
# Convert the image into a 1D vector
# Create the Double block Toeplitz matrix of the kernel
# Reshape the resulting Toeplitz matrix of the kernel into a matrix, where the number of column equalt to the lenght of the resulting image vector.
# Multiply the Toeplitz matrix and the image vector
# Reshape the result back into a 2D image


https://www.youtube.com/@robinhorn6767/videos
"""
