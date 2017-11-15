#!/usr/bin/python3

import pdb
import sys
import argparse
import matplotlib.pyplot as plt

import skimage.color
import skimage.filters

import numpy as np

from scipy import ndarray
from scipy.ndimage import convolve

sys.path.append('../')
from util import *

HSOBEL_WEIGHTS = np.array([[ 1, 2, 1],
                           [ 0, 0, 0],
                           [-1,-2,-1]]) / 4.0
VSOBEL_WEIGHTS = HSOBEL_WEIGHTS.T

def image_energy(img):
    grayscale = skimage.color.rgb2gray(img)

    def sobel_h(image):
        return convolve(image, HSOBEL_WEIGHTS, mode='wrap')

    def sobel_v(image):
        return convolve(image, VSOBEL_WEIGHTS, mode='wrap')

    return np.sqrt(
            sobel_h(grayscale)**2 + sobel_v(grayscale)**2
            ) / np.sqrt(2)

def get_best_column_path(energy):
    (n, m) = energy.shape[0:2]
    def neighbours(i, j):
        ret = []
        def add_if_good(x, y):
            def good(x, y):
                return x >= 0 and y >= 0 and x < n and y < m
            if good(x, y):
                ret.append((x, y))
        add_if_good(i - 1, j)
        add_if_good(i - 1, j - 1)
        add_if_good(i - 1, j + 1)
        return ret

    best = ndarray(energy.shape, dtype=energy.dtype)
    prev = [[None for _ in range(m)] for _ in range(n)]

    for j in range(m):
        best[0][j] = energy[0][j]

    for i in range(1, n):
        for j in range(m):
            best_prev = best[i - 1][j]
            prv = (i - 1, j)

            for (ni, nj) in neighbours(i, j):
                if best[ni][nj] < best_prev:
                    best_prev = best[ni][nj]
                    prv = (ni, nj)

            best[i][j] = best_prev + energy[i][j]
            prev[i][j] = prv

    starting = 0
    for j in range(m):
        if (best[n - 1][j] < best[n - 1][starting]):
            starting = j

    def get_prev(i, j):
        return prev[i][j]

    path = {}
    def reconstruct_path(i, j):
        path[i] = j
        if i > 0:
            reconstruct_path(*get_prev(i, j))

    reconstruct_path(n - 1, starting)

    return path

def remove_column_path(img, path):
    (n, m) = img.shape[0:2]
    ret = ndarray((n, m - 1, 3), dtype=img.dtype)
    for i in range(n):
        assert i in path
        j = path[i]
        ret[i] = numpy.concatenate((img[i][0:j], img[i][j + 1:m]), axis=0)
    return ret

def remove_column(img):
    energy = image_energy(img)
    path = get_best_column_path(energy)
    img = remove_column_path(img, path)
    return img

def remove_columns(img, cnt):
    if cnt == 0:
        return img
    return remove_columns(remove_column(img), cnt - 1)

def main():
    parser = argparse.ArgumentParser(description = 'Smartly resize imges.')
    parser.add_argument('target', type=str, help='Target picture to resize')
    parser.add_argument('output', type=str, help='Where to save the picture')
    parser.add_argument('--columns', type=int, help='How many columns to remove.')
    parser.add_argument('--lines', type=int, help='How many lines to remove.')
    args = parser.parse_args()

    img = read_image(args.target)

    fin_img = img

    if args.columns:
        fin_img = remove_columns(img, args.columns)

    misc.imsave(args.output, fin_img)

if __name__ == '__main__':
    sys.exit(main())
