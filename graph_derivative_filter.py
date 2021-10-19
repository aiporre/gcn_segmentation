import matplotlib.pyplot as plt
import numpy as np
import skimage

def kernel_function(x, sigma=1):
    return np.sign(-x)*(1- np.exp(-sigma*x**2))/(2*np.pi*sigma**2)

def convolve2D(img, kernel, padding=0, strides=1, cropping=True):
    # computes the inversion of the convolution kernel
    print("kernel shape shape: " + str(kernel.shape))
    kernel = np.flipud(np.fliplr(kernel))
    if isinstance(padding, int):
        padding = (padding, padding)
    else:
        assert len(padding) == 2, "padding should be long of 2 d"
    kernel_width = kernel.shape[0]
    kernel_height = kernel.shape[1]
    x_width = img.shape[0]
    x_height = img.shape[0]
    padding_x, padding_y = padding[0], padding[1]
    # Shape of Output Convolution
    x_output = int(((x_width - kernel_width + 2 * padding_x) / strides) + 1)
    y_output = int(((x_height - kernel_height + 2 * padding_y) / strides) + 1)
    output = np.zeros((x_output, y_output))

    # Apply Equal Padding to All Sides
    if padding_x != 0 and padding_y != 0:
        imagePadded = np.zeros((img.shape[0] + padding_x * 2, img.shape[1] + padding_y* 2))
        print('image padded shape: ' + str(imagePadded.shape))
        imagePadded[int(padding_x):int(-1 * padding_x), int(padding_y):int(-1 * padding_y)] = img
    elif padding_x != 0:
        imagePadded = np.zeros((img.shape[0] + padding_x * 2, img.shape[1] + padding_y * 2))
        print('image padded shape: ' + str(imagePadded.shape))
        imagePadded[int(padding_x):int(-1 * padding_x), : ] = img
    elif padding_y != 0:
        imagePadded = np.zeros((img.shape[0] + padding_x * 2, img.shape[1] + padding_y* 2))
        print('image padded shape: ' + str(imagePadded.shape))
        imagePadded[:, int(padding_y):int(-1 * padding_y)] = img
    else:
        imagePadded = img

    # Iterate through image
    for y in range(img.shape[1]):
        # Exit Convolution
        if y > img.shape[1] - kernel_height:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(img.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > img.shape[0] - kernel_width:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + kernel_width, y: y + kernel_height]).sum()
                except:
                    break
    # if cropping:
    #     print("output: " + str(output.shape))
    #     if padding_x != 0 and padding_y != 0:
    #         output = output[int(padding_x):int(-1 * padding_x), int(padding_y):int(-1 * padding_y)]
    #     elif padding_x != 0:
    #         output = output[int(padding_x):int(-1 * padding_x), :]
    #     elif padding_y != 0:
    #         output = output[:, int(padding_y):int(-1 * padding_y)]
    return output

def gradient_directional(img, delta=1, size=3, direction='x'):
    assert isinstance(direction, str)  and direction in ['x', 'y'], 'Input direction must be string options \'x\' and \'y\''
    weights = np.array([[kernel_function(z) for z in np.arange(-size, size + delta, delta)]])
    print("the shape of image is ", img.shape)
    print(f'the kernel in direction {direction} is {weights}')
    if direction == 'x':
        return convolve2D(img, weights, padding=(0, size))
    else:
        weights = weights.T
        return convolve2D(img, weights, padding=(size, 0))
def main():
    img = skimage.data.camera()
    grad_x = gradient_directional(img, direction='x')
    grad_y = gradient_directional(img, direction='y')
    grad_norm = np.sqrt(grad_x**2 + grad_y**2)
    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(grad_x)
    plt.title(' gradient x')
    plt.subplot(1,4,2)
    plt.imshow(grad_y)
    plt.title(' gadient y')
    plt.subplot(1,4,3)
    plt.imshow(grad_norm)
    plt.title(' gradient image')
    plt.subplot(1,4,4)
    plt.imshow(img)
    plt.title(' original image')
    plt.show()

if __name__ == '__main__':
    main()