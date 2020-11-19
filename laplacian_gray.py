"""
Author: Capilnean Adrian Vasile
Date: 4/11/2020
"""
import argparse
import logging
import imageio
import numpy as np
import math
import sys
from PIL import Image, ImageOps


def load_image(filename):
    im = Image.open(filename)
    logging.info('Converting to grayscale')
    im = ImageOps.grayscale(im)
    imageio.imwrite('grayscale.jpg', im)
    return np.array(im)


def l_o_g(x, y, sigma):
    nom = ((y**2)+(x**2)-2*(sigma**2))
    denom = ((2*math.pi*(sigma**6)))
    expo = math.exp(-((x**2)+(y**2))/(2*(sigma**2)))
    return nom*expo/denom


def create_log_kernel(sigma=1.0, size=5):
    w = math.ceil(float(size)*float(sigma))

    if(w % 2 == 0):

        w = w + 1

    l_o_g_mask = []
    w_range = int(math.floor(w/2))
    logging.info("Kernel range is (" + str(-w_range) + ", " + str(w_range)+')')
    for i in range(-w_range, w_range+1):
        for j in range(-w_range, w_range+1):
            l_o_g_mask.append(l_o_g(i, j, sigma))
    l_o_g_mask = np.array(l_o_g_mask)
    l_o_g_mask = l_o_g_mask.reshape(w, w)
    return l_o_g_mask


def z_c_test(l_o_g_image):
    z_c_image = np.zeros(l_o_g_image.shape)
    for i in range(1, l_o_g_image.shape[0]-1):
        for j in range(1, l_o_g_image.shape[1]-1):
            neg_count = 0
            pos_count = 0
            for a in range(-1, 2):
                for b in range(-1, 2):
                    if(a != 0 and b != 0):
                        if(l_o_g_image[i+a, j+b] <= 0):
                            neg_count += 1
                        elif(l_o_g_image[i+a, j+b] > 0):
                            pos_count += 1
            z_c = ((neg_count > 0) and (pos_count > 0))
            if(z_c):
                z_c_image[i, j] = 1
    return z_c_image


def convolve_pixel(img, kernel, i, j):
    k = kernel.shape[0] // 2
    if i < k or j < k or i >= img.shape[0]-k or j >= img.shape[1]-k:
        return img[i, j]
    else:
        value = 0
        for u in np.arange(-k, k+1):
            for v in np.arange(-k, k+1):
                value += img[i-u, j-v] * kernel[k+u, k+v]
        return value


def convolve(img, kernel):
    new_img = np.array(img)
    for i in np.arange(0, img.shape[0]):
        for j in np.arange(0, img.shape[1]):
            new_img[i, j] = convolve_pixel(img, kernel, i, j)
    img = new_img
    return img


if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Edge detection using laplacian operator')
    parser.add_argument('input', type=str, help='The input image file')
    parser.add_argument('output', type=str, help='Where to save the result')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='The standard deviation to use for the LoG kernel')
    parser.add_argument('--k', type=int, default=5,
                        help='The size of the kernel.')
    args = parser.parse_args()

    logging.info('Loading input image %s' % (args.input))
    inputImage = load_image(args.input)

    logging.info('Computing a LoG kernel with size %d and sigma %.2f' %
                 (args.k, args.sigma))
    kernel = create_log_kernel(args.k, args.sigma)

    logging.info('Convoluting image with laplacian kernel')
    resultImage = convolve(inputImage, kernel)
    imageio.imwrite('convolvedNotChecked.jpg', resultImage)

    logging.info('Checking for zero crossings')
    resultImage = z_c_test(resultImage)
    imageio.imwrite('zeroChecked.jpg', resultImage)

    logging.info('Converting image type from float64 to uint8')
    resultImage *= 255
    resultImage = resultImage.astype(np.uint8)

    logging.info('Saving result to %s' % (args.output))
    imageio.imwrite(args.output, resultImage)
