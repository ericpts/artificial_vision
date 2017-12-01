import numpy
import time
import skimage.io
from scipy.spatial.distance import euclidean
from scipy.spatial import KDTree
from scipy import ndarray
from scipy import misc
from pathlib import Path
import math
import sys
import matplotlib.pyplot as plt
import os
import pdb
import random
import numpy as np

from scipy.ndimage import convolve

INF = 2**60

def read_image(file_name):
    return normalize_image(skimage.io.imread(file_name))
    # return normalize_image(plt.imread(file_name))

def normalize_image(img):
    ret = misc.imresize(img, 100)
    if len(ret.shape) == 3 and ret.shape[2] == 4:
        # Transform from RGBA to RGB.
        ret = np.delete(ret, axis=2, obj=3)
    return ret


def neighbours_with_limits(n, m, i, j):
    assert i > 0

    if j == 0:
        return [(i - 1, j), (i - 1, j + 1)]

    if j == m - 1:
        return [(i - 1, j - 1), (i - 1, j)]

    return [
        (i - 1, j - 1),
        (i - 1, j),
        (i - 1, j + 1)
    ]


def get_column_path_dynamicprogramming(energy):
    """ Returns the path, a dict from line to column. """
    (n, m) = energy.shape[0:2]

    def neighbours(i, j):
        return neighbours_with_limits(n, m, i, j)

    best = ndarray(energy.shape, dtype=energy.dtype)

    for j in range(m):
        best[0][j] = energy[0][j]

    for i in range(1, n):
        shift_left = np.append(best[i - 1][1:], INF)
        shift_right = np.append([INF], best[i - 1][:-1])

        best[i] = energy[i] + np.minimum(
            best[i - 1],
            np.minimum(shift_left,
                       shift_right)
        )

    starting = 0
    for j in range(m):
        if (best[n - 1][j] < best[n - 1][starting]):
            starting = j

    def get_prev(i, j):
        for nbr in neighbours(i, j):
            if best[nbr] + energy[i][j] == best[i][j]:
                return nbr

    path = {}

    def reconstruct_path(i, j):
        path[i] = j
        while i > 0:
            (i, j) = get_prev(i, j)
            path[i] = j

    reconstruct_path(n - 1, starting)

    return path


def get_column_path_random(energy):
    (n, m) = energy.shape[0:2]

    def neighbours(i, j):
        return neighbours_with_limits(n, m, i, j)

    path = {}
    i = n - 1
    j = random.randint(0, m - 1)

    while i > 0:
        path[i] = j
        (i, j) = random.choice(neighbours(i, j))
    path[i] = j
    return path


def get_column_path_greedy(energy):
    (n, m) = energy.shape[0:2]

    def neighbours(i, j):
        return neighbours_with_limits(n, m, i, j)

    path = {}
    i = n - 1
    j = np.argmin(energy[i])

    while i > 0:
        path[i] = j
        nbrs = neighbours(i, j)

        best_nbr = None
        for n in nbrs:
            if best_nbr is None or energy[n] < energy[best_nbr]:
                best_nbr = n

        (i, j) = best_nbr

    path[i] = j
    return path


def transpose(img):
    if len(img.shape) == 3:
        return np.transpose(img, axes=(1, 0, 2))
    elif len(img.shape) == 2:
        return np.transpose(img, axes=(1, 0))

    assert False


HSOBEL_WEIGHTS = np.array([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]]) / 4.0
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

